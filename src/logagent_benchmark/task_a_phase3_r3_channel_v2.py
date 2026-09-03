"""Task A Phase 3-R3: channel-wise DeBERTa evidence and interpretable reranking.

This development stage keeps the frozen A2 candidate universe immutable. It
reconstructs model-visible operational features from the sanitized Phase-2
trace partition, verbalizes Trace/Operation/HTTP/Role evidence independently,
and scores every available channel with a frozen local DeBERTa NLI backend.
Evaluator labels are joined only after every model-side score is materialized.

The NLI result is a ranking feature, never a hard veto. Scientific utility is
measured against both exact-size A2-only and exact-size operational-evidence
controls so that shortlist-size gains are not misattributed to DeBERTa.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .onnx_deberta import OnnxDebertaNLIBackend
from .phase3_contract import stable_case_split
from . import task_a_phase3_r2 as r2
from .task_a_phase3_r2_compat import evaluator_flags


CANDIDATE_KEY = ("subject", "predicate", "object")
CELL_KEY = ("incident_token", "seed", "mask_id")
CHANNELS = ("trace", "operation", "http", "role")
EVALUATOR_COLUMNS = {
    "case",
    "fault",
    "role",
    "is_masked_target",
    "is_silver_matched",
}


class Phase3R3Error(RuntimeError):
    """Raised when the frozen R3 experiment contract cannot be satisfied."""


@dataclass(frozen=True)
class TriStateConfig:
    corroborate_entailment_min: float = 0.45
    contradict_probability_min: float = 0.65
    evidence_margin_min: float = 0.05
    contradiction_margin_min: float = 0.10

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class R3Policy:
    retention_fraction: float
    minimum_keep: int
    operational_weight: float
    nli_weight: float

    def __post_init__(self) -> None:
        if not 0.0 < self.retention_fraction <= 1.0:
            raise ValueError("retention_fraction must be in (0,1]")
        if isinstance(self.minimum_keep, bool) or self.minimum_keep <= 0:
            raise ValueError("minimum_keep must be a positive integer")
        for name, value in (
            ("operational_weight", self.operational_weight),
            ("nli_weight", self.nli_weight),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.nli_weight <= 0.0:
            raise ValueError("R3 requires a positive NLI weight")
        if self.operational_weight + self.nli_weight >= 1.0:
            raise ValueError("A2 prior must retain positive weight")

    @property
    def a2_weight(self) -> float:
        return 1.0 - self.operational_weight - self.nli_weight


@dataclass(frozen=True)
class GateConfig:
    recall_macro_min: float = 0.95
    recall_pooled_min: float = 0.95
    recall_each_cell_min: float = 0.90
    selected_count_ratio_max: float = 0.95
    p_lb_delta_vs_full_min: float = 0.0
    mrr_delta_vs_full_min: float = 0.0
    matched_a2_recall_tolerance: float = 0.0
    matched_a2_p_lb_delta_min: float = 0.0
    matched_a2_mrr_delta_min: float = 0.0
    matched_r2_recall_tolerance: float = 0.0
    matched_r2_p_lb_delta_min: float = 0.0
    matched_r2_mrr_delta_min: float = 0.0
    nli_additive_gain_min: float = 1e-6
    nli_candidate_coverage_min: float = 0.95
    nli_score_std_min: float = 1e-6


@dataclass(frozen=True)
class ChannelEvidence:
    available: bool
    premise: str
    confidence: float


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _display(value: Any) -> str:
    return str(value).rsplit(":", 1)[-1]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _finite(value)))


def _normal_probabilities(raw: Mapping[str, Any]) -> dict[str, float]:
    result = {
        label: _finite(raw.get(label), default=math.nan)
        for label in ("entailment", "contradiction", "neutral")
    }
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in result.values()
    ):
        raise Phase3R3Error(f"invalid NLI probabilities: {result}")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise Phase3R3Error(f"NLI probabilities do not sum to one: {result}")
    return result


def _rank_normalize(group: pd.DataFrame, score_column: str) -> pd.Series:
    order = group.sort_values(
        [score_column, "a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).index
    denominator = max(1, len(group) - 1)
    output = pd.Series(0.0, index=group.index, dtype=float)
    for rank, index in enumerate(order):
        output.loc[index] = 1.0 - rank / denominator
    return output


def _channel_evidence(record: Mapping[str, Any]) -> dict[str, ChannelEvidence]:
    subject = _display(record["subject"])
    object_id = _display(record["object"])

    supporting = max(0, int(_finite(record.get("supporting_traces"))))
    reverse_supporting = max(
        0, int(_finite(record.get("reverse_supporting_traces")))
    )
    boundaries = max(0, int(_finite(record.get("boundary_spans"))))
    reverse_boundaries = max(
        0, int(_finite(record.get("reverse_boundary_spans")))
    )
    alignment = _clip01(record.get("boundary_alignment"))
    direction_score = _clip01(record.get("direction_score", 0.5))
    trace_available = supporting > 0 or boundaries > 0
    trace_premise = (
        "Distributed-trace direction evidence only. "
        f"Service {subject} temporally encloses spans of service {object_id} "
        f"at {boundaries} reconstructed boundaries across {supporting} whole traces. "
        f"The reverse orientation has {reverse_boundaries} boundaries across "
        f"{reverse_supporting} traces. Boundary reconstruction alignment is "
        f"{alignment:.3f}, and the normalized forward direction score is "
        f"{direction_score:.3f}."
    )

    pair_count = max(
        0, int(_finite(record.get("reconstructed_boundary_pairs")))
    )
    operation_overlap = _clip01(record.get("operation_jaccard_mean"))
    pair_concentration = _clip01(record.get("operation_pair_concentration"))
    operation_role = _clip01(record.get("operation_role_score", 0.5))
    endpoint = _clip01(record.get("endpoint_compatibility_score"))
    operation_available = pair_count > 0
    operation_premise = (
        "Operation-name evidence only. "
        f"For candidate {subject} to {object_id}, {pair_count} boundary operation "
        f"pairs were reconstructed. Their mean operation-token overlap is "
        f"{operation_overlap:.3f}, the dominant operation-pair concentration is "
        f"{pair_concentration:.3f}, the parent/child operation-role compatibility "
        f"is {operation_role:.3f}, and the aggregate endpoint compatibility is "
        f"{endpoint:.3f}."
    )

    method_coverage = _clip01(record.get("method_coverage"))
    method_match = _clip01(record.get("method_match_rate", 0.5))
    route_coverage = _clip01(record.get("route_coverage"))
    route_exact = _clip01(record.get("route_exact_rate"))
    route_overlap = _clip01(record.get("route_jaccard_mean"))
    http_available = method_coverage > 0.0 or route_coverage > 0.0
    http_premise = (
        "HTTP method and normalized-route evidence only. "
        f"For candidate {subject} to {object_id}, method coverage is "
        f"{method_coverage:.3f} with method agreement {method_match:.3f}; route "
        f"coverage is {route_coverage:.3f} with exact-route agreement "
        f"{route_exact:.3f} and route-token overlap {route_overlap:.3f}. "
        "Unavailable HTTP attributes are not imputed."
    )

    graph_role = _clip01(record.get("graph_role_score", 0.5))
    source_out = max(0.0, _finite(record.get("source_out_degree")))
    source_in = max(0.0, _finite(record.get("source_in_degree")))
    target_out = max(0.0, _finite(record.get("target_out_degree")))
    target_in = max(0.0, _finite(record.get("target_in_degree")))
    span_coverage = _clip01(record.get("span_kind_coverage"))
    span_compat = _clip01(
        record.get("span_kind_compatibility_score", 0.5)
    )
    workload_coverage = _clip01(record.get("workload_coverage"))
    workload_match = _clip01(record.get("workload_match_score", 0.5))
    role_available = any(
        (source_out, source_in, target_out, target_in)
    ) or operation_available
    role_premise = (
        "Runtime role evidence only. "
        f"Observed CALLS topology gives {subject} out-degree {source_out:.0f} and "
        f"in-degree {source_in:.0f}; {object_id} has out-degree {target_out:.0f} "
        f"and in-degree {target_in:.0f}. The caller-to-callee role score is "
        f"{graph_role:.3f}. Span-kind coverage is {span_coverage:.3f} with "
        f"CLIENT-to-SERVER compatibility {span_compat:.3f}; workload coverage is "
        f"{workload_coverage:.3f} with source/destination match {workload_match:.3f}. "
        "Missing direct attributes remain unavailable rather than inferred as facts."
    )

    return {
        "trace": ChannelEvidence(trace_available, trace_premise, alignment),
        "operation": ChannelEvidence(
            operation_available, operation_premise, min(1.0, pair_count / 5.0)
        ),
        "http": ChannelEvidence(
            http_available, http_premise, max(method_coverage, route_coverage)
        ),
        "role": ChannelEvidence(role_available, role_premise, graph_role),
    }


def _hypotheses(record: Mapping[str, Any]) -> tuple[str, str]:
    subject = _display(record["subject"])
    object_id = _display(record["object"])
    return (
        f"Within this runtime system, service {subject} directly calls service {object_id}.",
        f"Within this runtime system, service {object_id} directly calls service {subject}.",
    )


def classify_channel(
    forward: Mapping[str, Any],
    reverse: Mapping[str, Any],
    config: TriStateConfig,
) -> dict[str, Any]:
    f = _normal_probabilities(forward)
    r = _normal_probabilities(reverse)
    direction_margin = f["entailment"] - r["entailment"]
    label_margin = f["entailment"] - f["contradiction"]
    reverse_dominates = (
        r["entailment"] >= config.contradict_probability_min
        and r["entailment"] - f["entailment"]
        >= config.contradiction_margin_min
    )
    contradiction_dominates = (
        f["contradiction"] >= config.contradict_probability_min
        and f["contradiction"] - f["entailment"]
        >= config.contradiction_margin_min
    )
    corroborates = (
        f["entailment"] >= config.corroborate_entailment_min
        and direction_margin >= config.evidence_margin_min
        and label_margin >= config.evidence_margin_min
    )
    if reverse_dominates or contradiction_dominates:
        state = "contradicts"
    elif corroborates:
        state = "corroborates"
    else:
        state = "ambiguous"
    raw_score = (
        0.50 * direction_margin
        + 0.35 * label_margin
        + 0.15 * (1.0 - f["neutral"])
    )
    return {
        "state": state,
        "score": max(-1.0, min(1.0, raw_score)),
        "forward_entailment": f["entailment"],
        "reverse_entailment": r["entailment"],
        "forward_contradiction": f["contradiction"],
        "forward_neutral": f["neutral"],
        "direction_margin": direction_margin,
        "label_margin": label_margin,
    }


def score_channel_nli(
    model_frame: pd.DataFrame,
    *,
    backend: OnnxDebertaNLIBackend,
    tri_state: TriStateConfig,
    channel_weights: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forbidden = EVALUATOR_COLUMNS.intersection(model_frame.columns)
    if forbidden:
        raise Phase3R3Error(
            "NLI scoring received evaluator columns: "
            + ", ".join(sorted(forbidden))
        )
    missing_channels = set(CHANNELS).difference(channel_weights)
    if missing_channels:
        raise Phase3R3Error(
            f"channel_weights missing {sorted(missing_channels)}"
        )

    frame = model_frame.copy().reset_index(drop=True)
    pair_rows: list[tuple[str, str]] = []
    locations: list[tuple[int, str, str]] = []
    evidence_cache: dict[int, dict[str, ChannelEvidence]] = {}
    for index, record in enumerate(frame.to_dict(orient="records")):
        evidence_by_channel = _channel_evidence(record)
        evidence_cache[index] = evidence_by_channel
        forward_hypothesis, reverse_hypothesis = _hypotheses(record)
        for channel in CHANNELS:
            evidence = evidence_by_channel[channel]
            frame.loc[index, f"nli_{channel}_available"] = bool(
                evidence.available
            )
            frame.loc[index, f"nli_{channel}_confidence"] = float(
                evidence.confidence
            )
            if not evidence.available:
                frame.loc[index, f"nli_{channel}_state"] = "unavailable"
                frame.loc[index, f"nli_{channel}_score"] = 0.0
                continue
            pair_rows.extend(
                (
                    (evidence.premise, forward_hypothesis),
                    (evidence.premise, reverse_hypothesis),
                )
            )
            locations.extend(
                ((index, channel, "forward"), (index, channel, "reverse"))
            )

    availability = backend.availability()
    if availability.status != "READY":
        raise Phase3R3Error(
            f"DeBERTa backend unavailable: {availability.reason_code} "
            f"{availability.detail}"
        )
    token_lengths = (
        backend.pair_token_lengths(tuple(pair_rows)) if pair_rows else ()
    )
    if token_lengths and max(token_lengths) > backend.max_length:
        raise Phase3R3Error(
            "channel NLI input exceeds token budget: "
            f"{max(token_lengths)} > {backend.max_length}"
        )
    raw_scores = tuple(backend.score_pairs(tuple(pair_rows))) if pair_rows else ()
    if len(raw_scores) != len(locations):
        raise Phase3R3Error(
            f"NLI result count mismatch: expected={len(locations)} "
            f"got={len(raw_scores)}"
        )

    probabilities: dict[tuple[int, str, str], Mapping[str, Any]] = {
        location: score for location, score in zip(locations, raw_scores)
    }
    state_counts: dict[str, dict[str, int]] = {
        channel: {
            name: 0
            for name in (
                "corroborates",
                "ambiguous",
                "contradicts",
                "unavailable",
            )
        }
        for channel in CHANNELS
    }
    weighted_scores: list[float] = []
    available_counts: list[int] = []
    for index in range(len(frame)):
        numerator = 0.0
        denominator = 0.0
        available_count = 0
        for channel in CHANNELS:
            evidence = evidence_cache[index][channel]
            if not evidence.available:
                state_counts[channel]["unavailable"] += 1
                continue
            classified = classify_channel(
                probabilities[(index, channel, "forward")],
                probabilities[(index, channel, "reverse")],
                tri_state,
            )
            for name, value in classified.items():
                frame.loc[index, f"nli_{channel}_{name}"] = value
            state_counts[channel][str(classified["state"])] += 1
            weight = max(0.0, float(channel_weights[channel])) * max(
                0.05, float(evidence.confidence)
            )
            numerator += weight * float(classified["score"])
            denominator += weight
            available_count += 1
        weighted_scores.append(
            numerator / denominator if denominator > 0.0 else 0.0
        )
        available_counts.append(available_count)

    frame["nli_evidence_score"] = weighted_scores
    frame["nli_available_channel_count"] = available_counts
    ranked_groups = []
    for _key, group in frame.groupby(
        list(CELL_KEY), sort=True, dropna=False
    ):
        group = group.copy()
        group["nli_rank_normalized"] = _rank_normalize(
            group, "nli_evidence_score"
        )
        ranked_groups.append(group)
    frame = (
        pd.concat(ranked_groups, ignore_index=True)
        if ranked_groups
        else frame
    )

    candidate_coverage = (
        sum(count > 0 for count in available_counts) / len(available_counts)
        if available_counts
        else 0.0
    )
    score_std = (
        float(frame["nli_evidence_score"].std(ddof=0))
        if len(frame)
        else 0.0
    )
    channel_coverage = {
        channel: float(
            frame[f"nli_{channel}_available"].map(_truthy).mean()
        )
        for channel in CHANNELS
    }
    return frame, {
        "backend": backend.metadata(),
        "candidate_count": len(frame),
        "pair_count": len(pair_rows),
        "pairs_per_available_channel": 2,
        "minimum_pair_tokens": min(token_lengths) if token_lengths else None,
        "maximum_pair_tokens": max(token_lengths) if token_lengths else None,
        "truncation_count": 0,
        "candidate_coverage": candidate_coverage,
        "channel_coverage": channel_coverage,
        "state_counts": state_counts,
        "nli_score_mean": (
            float(frame["nli_evidence_score"].mean())
            if len(frame)
            else None
        ),
        "nli_score_std": score_std,
        "cache": asdict(backend.cache_info()),
    }


def _incident_token(revision: str, case: str) -> str:
    return hashlib.sha256(
        f"{revision}|r3|{case}".encode("utf-8")
    ).hexdigest()[:20]


def _cell_root(phase2_root: Path, row: Mapping[str, Any]) -> Path:
    summary = phase2_root / str(row["run_summary"])
    return summary.parent / "masks" / str(row["mask_id"])


def _load_model_cell(
    phase2_root: Path,
    row: Mapping[str, Any],
    *,
    revision: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    cell_root = _cell_root(phase2_root, row)
    records = r2._parse_a2(cell_root / "predictions" / "A2.parquet")
    trace_path = cell_root / "model_input" / "traces.parquet"
    edge_path = cell_root / "model_input" / "observed_edges.parquet"
    if not trace_path.is_file() or not edge_path.is_file():
        raise Phase3R3Error(
            f"missing heavy Phase-2 cell artifacts: {cell_root}"
        )
    traces, availability = r2._canonical_trace_frame(
        pd.read_parquet(trace_path)
    )
    observed = pd.read_parquet(edge_path)
    features, diagnostics = r2._candidate_feature_rows(
        records, traces, observed
    )
    features = r2.add_profile_scores(features)
    token = _incident_token(revision, str(row["case"]))
    features["incident_token"] = token
    features["seed"] = int(row["seed"])
    features["mask_id"] = str(row["mask_id"])
    features["mask_ratio"] = float(row["mask_ratio"])
    features["subject_label"] = features["subject"].map(_display)
    features["object_label"] = features["object"].map(_display)
    diagnostics.update(
        {
            "incident_token": token,
            "seed": int(row["seed"]),
            "mask_id": str(row["mask_id"]),
            "candidate_count": len(features),
            "trace_rows": len(traces),
            "field_availability": availability,
        }
    )
    return features, diagnostics, cell_root


def _load_model_cells(
    phase2_root: Path,
    cells_frame: pd.DataFrame,
    *,
    revision: str,
    max_workers: int,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    dict[tuple[str, int, str], tuple[Path, Mapping[str, Any]]],
]:
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    cell_registry: dict[
        tuple[str, int, str], tuple[Path, Mapping[str, Any]]
    ] = {}
    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _load_model_cell,
                phase2_root,
                row,
                revision=revision,
            ): row
            for row in cells_frame.to_dict(orient="records")
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            frame, diagnostic, cell_root = future.result()
            key = (
                str(frame.iloc[0]["incident_token"]),
                int(row["seed"]),
                str(row["mask_id"]),
            )
            if key in cell_registry:
                raise Phase3R3Error(f"duplicate cell key: {key}")
            cell_registry[key] = (cell_root, row)
            frames.append(frame)
            diagnostics.append(diagnostic)
            print(
                f"[{index}/{len(futures)}] incident={key[0]} "
                f"seed={key[1]} mask={key[2]} "
                "operational-features=READY",
                flush=True,
            )
    model = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    if EVALUATOR_COLUMNS.intersection(model.columns):
        raise Phase3R3Error("model handoff contains evaluator columns")
    return model, diagnostics, cell_registry


def _attach_evaluator_labels(
    scored_model: pd.DataFrame,
    *,
    cell_registry: Mapping[
        tuple[str, int, str], tuple[Path, Mapping[str, Any]]
    ],
    calibration_cases: set[str],
) -> pd.DataFrame:
    label_frames: list[pd.DataFrame] = []
    for key, (cell_root, row) in sorted(cell_registry.items()):
        group = scored_model.loc[
            scored_model["incident_token"].astype(str).eq(key[0])
            & scored_model["seed"].astype(int).eq(key[1])
            & scored_model["mask_id"].astype(str).eq(key[2])
        ]
        candidate_keys = {
            tuple(map(str, values))
            for values in group[list(CANDIDATE_KEY)].itertuples(
                index=False, name=None
            )
        }
        flags = evaluator_flags(cell_root, candidate_keys)
        flags["incident_token"] = key[0]
        flags["seed"] = key[1]
        flags["mask_id"] = key[2]
        flags["case"] = str(row["case"])
        flags["fault"] = str(row["fault"])
        flags["role"] = (
            "calibration"
            if str(row["case"]) in calibration_cases
            else "heldout"
        )
        label_frames.append(flags)
    labels = pd.concat(label_frames, ignore_index=True)
    merged = scored_model.merge(
        labels,
        on=[*CELL_KEY, *CANDIDATE_KEY],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(scored_model):
        raise Phase3R3Error("evaluator label join changed candidate count")
    return merged


def apply_policy(frame: pd.DataFrame, policy: R3Policy) -> pd.DataFrame:
    scored = frame.copy()
    scored["a3_r3_score"] = (
        policy.a2_weight * scored["a2_rank_normalized"].astype(float)
        + policy.operational_weight
        * scored["combined_rank_normalized"].astype(float)
        + policy.nli_weight
        * scored["nli_rank_normalized"].astype(float)
    )
    keep = min(
        len(scored),
        max(
            int(policy.minimum_keep),
            int(math.ceil(policy.retention_fraction * len(scored))),
        ),
    )
    scored["selected"] = False
    direct = set(
        scored.index[scored["direct_evidence"].map(_truthy)]
    )
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        [
            "a3_r3_score",
            "a2_score",
            "proposal_rank",
            "subject",
            "object",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    selected = set(direct)
    for index in ranked.index:
        if len(selected) >= keep:
            break
        selected.add(index)
    if selected:
        scored.loc[list(selected), "selected"] = True
    return scored


def _control(
    frame: pd.DataFrame,
    selected_count: int,
    *,
    kind: str,
    policy: R3Policy,
) -> pd.DataFrame:
    scored = frame.copy()
    if kind == "a2":
        scored["a3_r3_score"] = scored["a2_rank_normalized"].astype(float)
    elif kind == "r2":
        denominator = policy.a2_weight + policy.operational_weight
        if denominator <= 0.0:
            raise Phase3R3Error("invalid operational control denominator")
        scored["a3_r3_score"] = (
            policy.a2_weight
            * scored["a2_rank_normalized"].astype(float)
            + policy.operational_weight
            * scored["combined_rank_normalized"].astype(float)
        ) / denominator
    else:
        raise ValueError(f"unknown control kind: {kind}")
    scored["selected"] = False
    direct = set(
        scored.index[scored["direct_evidence"].map(_truthy)]
    )
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        [
            "a3_r3_score",
            "a2_score",
            "proposal_rank",
            "subject",
            "object",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    selected = set(direct)
    for index in ranked.index:
        if len(selected) >= selected_count:
            break
        selected.add(index)
    if selected:
        scored.loc[list(selected), "selected"] = True
    return scored


def evaluate_cell(scored: pd.DataFrame) -> dict[str, Any]:
    selected = scored.loc[scored["selected"].map(_truthy)]
    target_keys = {
        tuple(map(str, row))
        for row in scored.loc[
            scored["is_masked_target"].map(_truthy),
            list(CANDIDATE_KEY),
        ].itertuples(index=False, name=None)
    }
    silver_keys = {
        tuple(map(str, row))
        for row in scored.loc[
            scored["is_silver_matched"].map(_truthy),
            list(CANDIDATE_KEY),
        ].itertuples(index=False, name=None)
    }
    selected_keys = {
        tuple(map(str, row))
        for row in selected[list(CANDIDATE_KEY)].itertuples(
            index=False, name=None
        )
    }
    recovered = selected_keys & target_keys
    by_query: dict[tuple[str, str], list[Any]] = {}
    by_key: dict[tuple[str, str, str], Any] = {}
    for row in selected.itertuples(index=False):
        key = (str(row.subject), str(row.predicate), str(row.object))
        by_key[key] = row
        by_query.setdefault((key[0], key[1]), []).append(row)
    reciprocal: list[float] = []
    ranks: list[int | None] = []
    epsilon = 1e-12
    for target in sorted(target_keys):
        item = by_key.get(target)
        if item is None:
            reciprocal.append(0.0)
            ranks.append(None)
            continue
        target_score = float(item.a3_r3_score)
        competitors = []
        for candidate in by_query.get((target[0], target[1]), ()):
            key = (
                str(candidate.subject),
                str(candidate.predicate),
                str(candidate.object),
            )
            if key == target or key in silver_keys:
                continue
            competitors.append(candidate)
        higher = sum(
            float(candidate.a3_r3_score) > target_score + epsilon
            for candidate in competitors
        )
        tied = sum(
            abs(float(candidate.a3_r3_score) - target_score) <= epsilon
            for candidate in competitors
        )
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    return {
        "selected_count": len(selected_keys),
        "target_count": len(target_keys),
        "recovered_target_count": len(recovered),
        "recall": (
            len(recovered) / len(target_keys) if target_keys else None
        ),
        "silver_matched_count": len(selected_keys & silver_keys),
        "silver_precision_lower_bound": (
            len(selected_keys & silver_keys) / len(selected_keys)
            if selected_keys
            else None
        ),
        "mrr": statistics.fmean(reciprocal) if reciprocal else None,
        "ranks": json.dumps(ranks, separators=(",", ":")),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    recalls = [
        float(row["recall"])
        for row in rows
        if row.get("recall") is not None
    ]
    p_lbs = [
        float(row["silver_precision_lower_bound"])
        for row in rows
        if row.get("silver_precision_lower_bound") is not None
    ]
    mrrs = [
        float(row["mrr"])
        for row in rows
        if row.get("mrr") is not None
    ]
    counts = [int(row["selected_count"]) for row in rows]
    target_count = sum(int(row["target_count"]) for row in rows)
    recovered_count = sum(
        int(row["recovered_target_count"]) for row in rows
    )
    return {
        "cell_count": len(rows),
        "recall_macro": statistics.fmean(recalls),
        "recall_min": min(recalls),
        "recall_pooled": (
            recovered_count / target_count if target_count else None
        ),
        "selected_count_mean": statistics.fmean(counts),
        "selected_count_median": statistics.median(counts),
        "selected_count_max": max(counts),
        "silver_precision_lower_bound_macro": statistics.fmean(p_lbs),
        "silver_precision_lower_bound_min": min(p_lbs),
        "mrr_macro": statistics.fmean(mrrs),
        "mrr_min": min(mrrs),
    }


def _delta(
    enhanced: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "recall_macro": float(enhanced["recall_macro"])
        - float(baseline["recall_macro"]),
        "selected_count_mean": float(enhanced["selected_count_mean"])
        - float(baseline["selected_count_mean"]),
        "silver_precision_lower_bound_macro": float(
            enhanced["silver_precision_lower_bound_macro"]
        )
        - float(baseline["silver_precision_lower_bound_macro"]),
        "mrr_macro": float(enhanced["mrr_macro"])
        - float(baseline["mrr_macro"]),
    }


def _evaluate_policy(
    cells: Mapping[tuple[str, int, str], pd.DataFrame],
    policy: R3Policy,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]:
    proposed_rows: list[dict[str, Any]] = []
    a2_rows: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for key, group in sorted(cells.items()):
        group = group.reset_index(drop=True)
        proposed = apply_policy(group, policy)
        proposed_metric = evaluate_cell(proposed)
        a2_control = _control(
            group,
            int(proposed_metric["selected_count"]),
            kind="a2",
            policy=policy,
        )
        r2_control = _control(
            group,
            int(proposed_metric["selected_count"]),
            kind="r2",
            policy=policy,
        )
        a2_metric = evaluate_cell(a2_control)
        r2_metric = evaluate_cell(r2_control)
        first = group.iloc[0]
        common = {
            "incident_token": key[0],
            "case": str(first["case"]),
            "fault": str(first["fault"]),
            "role": str(first["role"]),
            "seed": int(key[1]),
            "mask_id": str(key[2]),
            "mask_ratio": float(first["mask_ratio"]),
        }
        proposed_rows.append({**common, **proposed_metric})
        a2_rows.append({**common, **a2_metric})
        r2_rows.append({**common, **r2_metric})
        for name, value in common.items():
            proposed[name] = value
        scored_frames.append(proposed)
    proposed_frame = pd.DataFrame.from_records(proposed_rows)
    a2_frame = pd.DataFrame.from_records(a2_rows)
    r2_frame = pd.DataFrame.from_records(r2_rows)
    scored_all = (
        pd.concat(scored_frames, ignore_index=True)
        if scored_frames
        else pd.DataFrame()
    )
    return (
        proposed_frame,
        _aggregate(proposed_rows),
        a2_frame,
        _aggregate(a2_rows),
        r2_frame,
        _aggregate(r2_rows),
        scored_all,
    )


def _baseline(
    cells: Mapping[tuple[str, int, str], pd.DataFrame]
) -> dict[str, Any]:
    rows = []
    for group in cells.values():
        pseudo = group.copy()
        pseudo["selected"] = True
        pseudo["a3_r3_score"] = pseudo["a2_rank_normalized"].astype(float)
        rows.append(evaluate_cell(pseudo))
    return _aggregate(rows)


def _policy_grid(config: Mapping[str, Any]) -> list[R3Policy]:
    search = config["policy_search"]
    policies = []
    for retention in search["retention_fractions"]:
        for minimum_keep in search["minimum_keep"]:
            for operational_weight in search["operational_weights"]:
                for nli_weight in search["nli_weights"]:
                    if float(operational_weight) + float(nli_weight) >= float(
                        search.get("maximum_auxiliary_weight", 0.8)
                    ):
                        continue
                    policies.append(
                        R3Policy(
                            float(retention),
                            int(minimum_keep),
                            float(operational_weight),
                            float(nli_weight),
                        )
                    )
    if not policies:
        raise Phase3R3Error("policy grid is empty")
    return policies


def _policy_conditions(
    proposed: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    a2_control: Mapping[str, Any],
    r2_control: Mapping[str, Any],
    gate: GateConfig,
) -> tuple[dict[str, bool], dict[str, float]]:
    selected_ratio = float(proposed["selected_count_mean"]) / float(
        baseline["selected_count_mean"]
    )
    delta_full = _delta(proposed, baseline)
    delta_a2 = _delta(proposed, a2_control)
    delta_r2 = _delta(proposed, r2_control)
    additive_gain = max(
        delta_r2["silver_precision_lower_bound_macro"],
        delta_r2["mrr_macro"],
    )
    conditions = {
        "recall_macro": (
            float(proposed["recall_macro"]) >= gate.recall_macro_min
        ),
        "recall_each_cell": (
            float(proposed["recall_min"]) >= gate.recall_each_cell_min
        ),
        "candidate_count_reduced": (
            selected_ratio <= gate.selected_count_ratio_max
        ),
        "mrr_noninferior_to_full_a2": (
            delta_full["mrr_macro"] >= -0.01
        ),
        "matched_a2_recall": (
            delta_a2["recall_macro"]
            >= -gate.matched_a2_recall_tolerance
        ),
        "matched_a2_p_lb": (
            delta_a2["silver_precision_lower_bound_macro"]
            >= gate.matched_a2_p_lb_delta_min
        ),
        "matched_a2_mrr": (
            delta_a2["mrr_macro"] >= gate.matched_a2_mrr_delta_min
        ),
        "matched_r2_recall": (
            delta_r2["recall_macro"]
            >= -gate.matched_r2_recall_tolerance
        ),
        "matched_r2_p_lb": (
            delta_r2["silver_precision_lower_bound_macro"]
            >= gate.matched_r2_p_lb_delta_min
        ),
        "matched_r2_mrr": (
            delta_r2["mrr_macro"] >= gate.matched_r2_mrr_delta_min
        ),
        "nli_additive_gain": (
            additive_gain >= gate.nli_additive_gain_min
        ),
    }
    diagnostics = {
        "selected_count_ratio": selected_ratio,
        "delta_vs_full_p_lb": delta_full[
            "silver_precision_lower_bound_macro"
        ],
        "delta_vs_full_mrr": delta_full["mrr_macro"],
        "matched_a2_recall_delta": delta_a2["recall_macro"],
        "matched_a2_p_lb_delta": delta_a2[
            "silver_precision_lower_bound_macro"
        ],
        "matched_a2_mrr_delta": delta_a2["mrr_macro"],
        "matched_r2_recall_delta": delta_r2["recall_macro"],
        "matched_r2_p_lb_delta": delta_r2[
            "silver_precision_lower_bound_macro"
        ],
        "matched_r2_mrr_delta": delta_r2["mrr_macro"],
        "nli_additive_gain": additive_gain,
    }
    return conditions, diagnostics


def _select_policy(
    calibration: Mapping[tuple[str, int, str], pd.DataFrame],
    *,
    config: Mapping[str, Any],
    gate: GateConfig,
) -> tuple[R3Policy, pd.DataFrame, bool]:
    baseline = _baseline(calibration)
    rows = []
    for policy in _policy_grid(config):
        (
            _proposed_rows,
            proposed,
            _a2_rows,
            a2_control,
            _r2_rows,
            r2_control,
            _scored,
        ) = _evaluate_policy(calibration, policy)
        conditions, diagnostics = _policy_conditions(
            proposed,
            baseline=baseline,
            a2_control=a2_control,
            r2_control=r2_control,
            gate=gate,
        )
        rows.append(
            {
                **asdict(policy),
                **proposed,
                **diagnostics,
                "feasible": all(conditions.values()),
                "violation_count": sum(
                    not value for value in conditions.values()
                ),
            }
        )
    grid = pd.DataFrame.from_records(rows)
    feasible = grid.loc[grid["feasible"].map(_truthy)]
    pool = feasible if not feasible.empty else grid
    order = (
        [
            "selected_count_mean",
            "nli_additive_gain",
            "silver_precision_lower_bound_macro",
            "mrr_macro",
            "nli_weight",
            "operational_weight",
        ]
        if not feasible.empty
        else [
            "violation_count",
            "nli_additive_gain",
            "recall_macro",
            "silver_precision_lower_bound_macro",
            "mrr_macro",
        ]
    )
    ascending = (
        [True, False, False, False, True, True]
        if not feasible.empty
        else [True, False, False, False, False]
    )
    chosen = pool.sort_values(
        order, ascending=ascending, kind="mergesort"
    ).iloc[0]
    policy = R3Policy(
        float(chosen.retention_fraction),
        int(chosen.minimum_keep),
        float(chosen.operational_weight),
        float(chosen.nli_weight),
    )
    grid["selected"] = (
        grid["retention_fraction"].astype(float).eq(
            policy.retention_fraction
        )
        & grid["minimum_keep"].astype(int).eq(policy.minimum_keep)
        & grid["operational_weight"].astype(float).eq(
            policy.operational_weight
        )
        & grid["nli_weight"].astype(float).eq(policy.nli_weight)
    )
    return policy, grid, not feasible.empty


def _render_report(summary: Mapping[str, Any]) -> str:
    heldout = summary["heldout"]
    base = heldout["baseline_a2_full"]
    proposed = heldout["proposed_a3_r3"]
    delta_full = heldout["delta_vs_full_a2"]
    delta_a2 = heldout["delta_vs_equal_size_a2"]
    delta_r2 = heldout["delta_vs_equal_size_r2"]
    reasons = ", ".join(summary["gate"]["reason_codes"]) or "없음"
    return f"""# Task A Phase 3-R3 결과 — Evidence별 DeBERTa Tri-state

- 최종 과학적 Gate: **{summary['status']}**
- Calibration feasible 정책: **{summary['calibration']['feasible_policy_count']} / {summary['calibration']['searched_policy_count']}**
- 선택 정책: `{summary['selected_policy']}`
- 미통과 조건: `{reasons}`
- 프로토콜: **개발 재검증** — 신규 독립 Incident 확인시험이 별도로 필요함

## Held-out 40 Cell

| 지표 | A2 전체 | A3-R3 | A2 전체 대비 | 동일 크기 A2 대비 | 동일 크기 R2 대비 |
|---|---:|---:|---:|---:|---:|
| Recall Macro | {base['recall_macro']:.4f} | {proposed['recall_macro']:.4f} | {delta_full['recall_macro']:+.4f} | {delta_a2['recall_macro']:+.4f} | {delta_r2['recall_macro']:+.4f} |
| Recall Minimum | {base['recall_min']:.4f} | {proposed['recall_min']:.4f} | - | - | - |
| 후보 수 평균 | {base['selected_count_mean']:.3f} | {proposed['selected_count_mean']:.3f} | {delta_full['selected_count_mean']:+.3f} | {delta_a2['selected_count_mean']:+.3f} | {delta_r2['selected_count_mean']:+.3f} |
| P-LB Macro | {base['silver_precision_lower_bound_macro']:.4f} | {proposed['silver_precision_lower_bound_macro']:.4f} | {delta_full['silver_precision_lower_bound_macro']:+.4f} | {delta_a2['silver_precision_lower_bound_macro']:+.4f} | {delta_r2['silver_precision_lower_bound_macro']:+.4f} |
| MRR Macro | {base['mrr_macro']:.4f} | {proposed['mrr_macro']:.4f} | {delta_full['mrr_macro']:+.4f} | {delta_a2['mrr_macro']:+.4f} | {delta_r2['mrr_macro']:+.4f} |

## NLI 채널 진단

```json
{json.dumps(summary['nli_diagnostics'], ensure_ascii=False, indent=2, sort_keys=True)}
```

## 해석 규칙

- Trace, Operation, HTTP, Role은 각각 독립 Premise로 평가한다.
- 채널이 실제 데이터에 없으면 `unavailable`로 남기고 사실을 생성하지 않는다.
- `contradicts`는 순위 Feature일 뿐 후보를 단독 삭제하지 않는다.
- PASS는 동일 후보 수의 A2-only뿐 아니라 **동일 후보 수의 R2 operational control보다도** P-LB 또는 MRR이 실제 개선됐다는 뜻이다.
- `CALLS` 복원은 runtime 구조 관계이며 causal `CAUSES` 복원을 의미하지 않는다.
"""


def run_phase3_r3(
    *,
    phase2_root: Path,
    model_dir: Path,
    output: Path,
    config_path: Path,
    max_workers: int = 2,
) -> Path:
    phase2_root = phase2_root.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if output.exists():
        raise Phase3R3Error(f"refusing to overwrite output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != (
        "rcaeval-task-a-phase3-r3-channel-nli-v2"
    ):
        raise Phase3R3Error("unexpected experiment_id")
    phase2_summary = json.loads(
        (phase2_root / "summary.json").read_text(encoding="utf-8")
    )
    if not phase2_summary.get("gate", {}).get("passed"):
        raise Phase3R3Error("Phase-2 D3 gate must pass before R3")
    cells_frame = pd.read_csv(phase2_root / "cells.csv")
    source = config["source_contract"]
    if len(cells_frame) != int(source["required_cells"]):
        raise Phase3R3Error("unexpected Phase-2 cell count")
    if cells_frame["case"].nunique() != int(
        source["required_incidents"]
    ):
        raise Phase3R3Error("unexpected Phase-2 incident count")
    if int(cells_frame["a2_proposal_count"].sum()) != int(
        source["required_candidate_rows"]
    ):
        raise Phase3R3Error("unexpected A2 candidate count")

    revision = str(config["dataset_revision"])
    calibration_cases, heldout_cases, case_hashes = stable_case_split(
        cells_frame["case"].astype(str),
        revision=revision,
        calibration_incidents=int(
            config["split_contract"]["calibration_incidents"]
        ),
    )
    calibration_set = set(calibration_cases)

    model_features, feature_diagnostics, registry = _load_model_cells(
        phase2_root,
        cells_frame,
        revision=revision,
        max_workers=max_workers,
    )
    expected_candidates = int(source["required_candidate_rows"])
    if len(model_features) != expected_candidates:
        raise Phase3R3Error(
            f"model feature count mismatch: {len(model_features)} "
            f"!= {expected_candidates}"
        )

    backend_config = config["backend"]
    backend = OnnxDebertaNLIBackend(
        model_dir,
        expected_sha256=str(backend_config["onnx_sha256"]),
        revision=str(backend_config["revision"]),
        batch_size=1,
        performance_mode=False,
        max_length=int(backend_config.get("max_length", 512)),
    )
    tri_state = TriStateConfig(**dict(config["tri_state"]))
    scored_model, nli_diagnostics = score_channel_nli(
        model_features,
        backend=backend,
        tri_state=tri_state,
        channel_weights=config["channel_weights"],
    )
    if EVALUATOR_COLUMNS.intersection(scored_model.columns):
        raise Phase3R3Error("evaluator leakage detected after NLI scoring")

    frame = _attach_evaluator_labels(
        scored_model,
        cell_registry=registry,
        calibration_cases=calibration_set,
    )
    calibration_frame = frame.loc[
        frame["role"].eq("calibration")
    ].copy()
    heldout_frame = frame.loc[frame["role"].eq("heldout")].copy()
    split = config["split_contract"]
    calibration_cells = calibration_frame.groupby(list(CELL_KEY)).ngroups
    heldout_cells = heldout_frame.groupby(list(CELL_KEY)).ngroups
    if calibration_cells != int(split["calibration_cells"]):
        raise Phase3R3Error("unexpected calibration cell count")
    if heldout_cells != int(split["heldout_cells"]):
        raise Phase3R3Error("unexpected heldout cell count")

    def cell_map(
        selected: pd.DataFrame,
    ) -> dict[tuple[str, int, str], pd.DataFrame]:
        return {
            tuple(key): group.copy()
            for key, group in selected.groupby(list(CELL_KEY), sort=True)
        }

    calibration = cell_map(calibration_frame)
    heldout = cell_map(heldout_frame)
    gate = GateConfig(**dict(config["gate"]))
    policy, grid, calibration_feasible = _select_policy(
        calibration, config=config, gate=gate
    )

    (
        calibration_rows,
        calibration_aggregate,
        _calibration_a2_rows,
        calibration_a2,
        _calibration_r2_rows,
        calibration_r2,
        _calibration_scored,
    ) = _evaluate_policy(calibration, policy)
    (
        heldout_rows,
        heldout_aggregate,
        heldout_a2_rows,
        heldout_a2,
        heldout_r2_rows,
        heldout_r2,
        heldout_scored,
    ) = _evaluate_policy(heldout, policy)
    baseline_calibration = _baseline(calibration)
    baseline_heldout = _baseline(heldout)
    delta_full = _delta(heldout_aggregate, baseline_heldout)
    delta_a2 = _delta(heldout_aggregate, heldout_a2)
    delta_r2 = _delta(heldout_aggregate, heldout_r2)
    selected_ratio = (
        heldout_aggregate["selected_count_mean"]
        / baseline_heldout["selected_count_mean"]
    )
    additive_gain = max(
        delta_r2["silver_precision_lower_bound_macro"],
        delta_r2["mrr_macro"],
    )
    runtime_meta = nli_diagnostics["backend"]
    conditions = {
        "calibration_policy_feasible": calibration_feasible,
        "heldout_complete": (
            len(heldout_rows) == int(split["heldout_cells"])
        ),
        "recall_macro": (
            heldout_aggregate["recall_macro"] >= gate.recall_macro_min
        ),
        "recall_pooled": (
            heldout_aggregate["recall_pooled"] >= gate.recall_pooled_min
        ),
        "recall_each_cell": (
            heldout_aggregate["recall_min"] >= gate.recall_each_cell_min
        ),
        "candidate_count_reduced": (
            selected_ratio <= gate.selected_count_ratio_max
        ),
        "p_lb_noninferior_to_full_a2": (
            delta_full["silver_precision_lower_bound_macro"]
            >= gate.p_lb_delta_vs_full_min
        ),
        "mrr_noninferior_to_full_a2": (
            delta_full["mrr_macro"] >= gate.mrr_delta_vs_full_min
        ),
        "matched_a2_recall_noninferior": (
            delta_a2["recall_macro"]
            >= -gate.matched_a2_recall_tolerance
        ),
        "matched_a2_p_lb_noninferior": (
            delta_a2["silver_precision_lower_bound_macro"]
            >= gate.matched_a2_p_lb_delta_min
        ),
        "matched_a2_mrr_noninferior": (
            delta_a2["mrr_macro"] >= gate.matched_a2_mrr_delta_min
        ),
        "matched_r2_recall_noninferior": (
            delta_r2["recall_macro"]
            >= -gate.matched_r2_recall_tolerance
        ),
        "matched_r2_p_lb_noninferior": (
            delta_r2["silver_precision_lower_bound_macro"]
            >= gate.matched_r2_p_lb_delta_min
        ),
        "matched_r2_mrr_noninferior": (
            delta_r2["mrr_macro"] >= gate.matched_r2_mrr_delta_min
        ),
        "nli_additive_gain": (
            additive_gain >= gate.nli_additive_gain_min
        ),
        "nli_weight_active": policy.nli_weight > 0.0,
        "nli_candidate_coverage": (
            nli_diagnostics["candidate_coverage"]
            >= gate.nli_candidate_coverage_min
        ),
        "nli_score_has_variance": (
            nli_diagnostics["nli_score_std"] >= gate.nli_score_std_min
        ),
        "backend_research_valid": bool(
            runtime_meta.get("research_valid")
        ),
        "backend_batch_size_one": (
            int(runtime_meta.get("batch_size", 0)) == 1
        ),
        "no_silent_truncation": (
            nli_diagnostics["truncation_count"] == 0
        ),
        "a2_candidates_preserved": len(frame) == expected_candidates,
    }
    passed = all(conditions.values())

    output.mkdir(parents=True, exist_ok=False)
    model_output = output / "model_output"
    evaluator_private = output / "evaluator_private"
    published = output / "published"
    model_output.mkdir()
    evaluator_private.mkdir()
    published.mkdir()

    scored_model.to_parquet(
        model_output / "a3_r3_channel_nli_features.parquet",
        index=False,
    )
    heldout_scored.to_parquet(
        evaluator_private / "heldout_candidate_analysis.parquet",
        index=False,
    )
    heldout_a2_rows.to_csv(
        evaluator_private / "equal_size_a2_control_cells.csv",
        index=False,
    )
    heldout_r2_rows.to_csv(
        evaluator_private / "equal_size_r2_control_cells.csv",
        index=False,
    )
    pd.DataFrame.from_records(feature_diagnostics).to_json(
        evaluator_private / "operational_feature_diagnostics.json",
        orient="records",
        indent=2,
    )
    calibration_rows.to_csv(
        published / "task_a_phase3_r3_calibration_cells.csv",
        index=False,
    )
    heldout_rows.to_csv(
        published / "task_a_phase3_r3_heldout_cells.csv",
        index=False,
    )
    grid.to_csv(
        published / "task_a_phase3_r3_policy_grid.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "PASS" if passed else "FAIL",
        "gate_id": "D4C_A3_R3_CHANNEL_NLI_DEVELOPMENT",
        "protocol_status": dict(config["protocol_status"]),
        "source": {
            "phase2_root": str(phase2_root),
            "phase2_cells_sha256": hashlib.sha256(
                (phase2_root / "cells.csv").read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "candidate_rows": len(frame),
            "candidate_cells": frame.groupby(list(CELL_KEY)).ngroups,
        },
        "split": {
            "calibration_cases": list(calibration_cases),
            "heldout_cases": list(heldout_cases),
            "case_hashes": case_hashes,
            "calibration_cells": calibration_cells,
            "heldout_cells": heldout_cells,
        },
        "leakage_boundary": {
            "feature_inputs": (
                "sanitized Phase-2 traces, observed graph and A2 candidates only"
            ),
            "evaluator_columns_removed_before_nli": True,
            "evaluator_labels_joined_after_nli_freeze": True,
            "fault_or_root_label_used_for_scoring": False,
            "policy_selection": "calibration incidents only",
            "heldout_labels_used_for_policy_selection": False,
        },
        "selected_policy": asdict(policy),
        "channel_weights": dict(config["channel_weights"]),
        "tri_state": asdict(tri_state),
        "nli_diagnostics": nli_diagnostics,
        "calibration": {
            "feasible": calibration_feasible,
            "searched_policy_count": len(grid),
            "feasible_policy_count": int(
                grid["feasible"].map(_truthy).sum()
            ),
            "baseline_a2_full": baseline_calibration,
            "proposed_a3_r3": calibration_aggregate,
            "equal_size_a2_control": calibration_a2,
            "equal_size_r2_control": calibration_r2,
        },
        "heldout": {
            "baseline_a2_full": baseline_heldout,
            "proposed_a3_r3": heldout_aggregate,
            "equal_size_a2_control": heldout_a2,
            "equal_size_r2_control": heldout_r2,
            "delta_vs_full_a2": delta_full,
            "delta_vs_equal_size_a2": delta_a2,
            "delta_vs_equal_size_r2": delta_r2,
            "selected_count_ratio": selected_ratio,
            "nli_additive_gain": additive_gain,
        },
        "gate": {
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "conditions": conditions,
            "reason_codes": [
                name.upper()
                for name, value in conditions.items()
                if not value
            ],
            "required": asdict(gate),
        },
        "claim_limit": (
            "Development-only channel-wise DeBERTa reranking of runtime "
            "CALLS candidates on six previously inspected RCAEval TrainTicket "
            "incidents. No causal-edge, RCA, LLM, production-generalization, "
            "or confirmatory claim."
        ),
    }
    result_json = published / "task_a_phase3_r3_results.json"
    result_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (published / "task_a_phase3_r3_results.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    (published / "task_a_phase3_r3_status.txt").write_text(
        summary["status"] + "\n", encoding="utf-8"
    )
    manifest = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(published.iterdir())
        if path.is_file()
    }
    (published / "task_a_phase3_r3_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "ChannelEvidence",
    "GateConfig",
    "Phase3R3Error",
    "R3Policy",
    "TriStateConfig",
    "apply_policy",
    "classify_channel",
    "evaluate_cell",
    "run_phase3_r3",
    "score_channel_nli",
    "_channel_evidence",
]
