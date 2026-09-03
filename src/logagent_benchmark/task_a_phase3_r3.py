"""Task A Phase 3-R3: channel-specific DeBERTa evidence and linear reranking.

R3 consumes the frozen 1,250-candidate A3-R2 handoff.  It verbalizes Trace
Direction, Operation, HTTP, and Runtime Role evidence independently, scores
forward/reverse hypotheses with a frozen local DeBERTa NLI backend, and tests
whether those scores add held-out utility beyond an equal-size operational-only
ranker.  Tri-state labels are diagnostics only; no NLI state is a hard veto.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .onnx_deberta import (
    NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
    OnnxDebertaNLIBackend,
)


CELL_KEY = ("incident_token", "seed", "mask_id")
CANDIDATE_KEY = ("subject", "predicate", "object")
JOIN_KEY = (*CELL_KEY, *CANDIDATE_KEY)
EVALUATOR_COLUMNS = {
    "case",
    "fault",
    "role",
    "is_masked_target",
    "is_silver_matched",
}
CHANNELS = ("direction", "operation", "http", "role")
OPERATIONAL_PROFILES = (
    "direction_role",
    "operation_endpoint",
    "method_route",
    "combined",
)
NLI_PROFILE_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "direction_operation": {"direction": 0.55, "operation": 0.45},
    "operation_http": {"operation": 0.60, "http": 0.40},
    "direction_role": {"direction": 0.60, "role": 0.40},
    "all_available": {
        "direction": 0.35,
        "operation": 0.30,
        "http": 0.20,
        "role": 0.15,
    },
}


class Phase3R3Error(RuntimeError):
    """Raised when the R3 experiment contract is violated."""


@dataclass(frozen=True)
class TriStateConfig:
    corroborate_entailment_min: float = 0.40
    contradict_probability_min: float = 0.60
    direction_margin_min: float = 0.03
    label_margin_min: float = -0.05


@dataclass(frozen=True)
class R3Policy:
    operational_profile: str
    nli_profile: str
    retention_fraction: float
    minimum_keep: int
    operational_weight: float
    nli_weight: float

    def __post_init__(self) -> None:
        if self.operational_profile not in OPERATIONAL_PROFILES:
            raise ValueError(
                f"unsupported operational profile: {self.operational_profile}"
            )
        if self.nli_profile not in NLI_PROFILE_WEIGHTS:
            raise ValueError(f"unsupported NLI profile: {self.nli_profile}")
        if not 0.0 < self.retention_fraction <= 1.0:
            raise ValueError("retention_fraction must be in (0, 1]")
        if isinstance(self.minimum_keep, bool) or self.minimum_keep <= 0:
            raise ValueError("minimum_keep must be a positive integer")
        if not 0.0 <= self.operational_weight <= 1.0:
            raise ValueError("operational_weight must be in [0, 1]")
        if not 0.0 < self.nli_weight <= 1.0:
            raise ValueError("nli_weight must be in (0, 1]")
        if self.operational_weight + self.nli_weight > 0.80 + 1e-12:
            raise ValueError(
                "operational_weight + nli_weight must leave at least 0.20 A2 prior"
            )


@dataclass(frozen=True)
class GateConfig:
    recall_macro_min: float = 0.95
    recall_pooled_min: float = 0.95
    recall_each_cell_min: float = 0.90
    selected_count_ratio_max: float = 0.95
    p_lb_delta_vs_full_min: float = 0.0
    mrr_delta_vs_full_min: float = 0.0
    matched_operational_recall_tolerance: float = 0.0
    matched_operational_p_lb_delta_min: float = 0.0
    matched_operational_mrr_delta_min: float = 0.0
    matched_a2_recall_tolerance: float = 0.0
    matched_a2_p_lb_delta_min: float = 0.0
    matched_a2_mrr_delta_min: float = 0.0
    nli_additive_gain_min: float = 1e-6
    scored_candidate_coverage_min: float = 1.0
    nli_score_std_min: float = 1e-5


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_text(value: Any, *, limit: int = 180) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())[:limit]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _percent(value: Any) -> str:
    return f"{100.0 * max(0.0, min(1.0, _finite(value))):.0f}%"


def _count(value: Any) -> int:
    return max(0, int(round(_finite(value))))


def _rank_normalize(frame: pd.DataFrame, score_column: str) -> pd.Series:
    order = frame.sort_values(
        [score_column, "a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).index
    denominator = max(1, len(frame) - 1)
    output = pd.Series(0.0, index=frame.index, dtype=float)
    for rank, index in enumerate(order):
        output.loc[index] = 1.0 - rank / denominator
    return output


def _channel_premise(channel: str, row: Mapping[str, Any]) -> str | None:
    """Build one relation-specific premise without evaluator labels or service IDs."""

    header = (
        "Service A and Service B are distinct services in one distributed runtime. "
    )
    if channel == "direction":
        return header + (
            f"A span from Service A enclosed a span from Service B in "
            f"{_count(row.get('supporting_traces'))} distributed traces and at "
            f"{_count(row.get('boundary_spans'))} candidate boundaries. "
            f"The reversed enclosure was observed in "
            f"{_count(row.get('reverse_supporting_traces'))} traces and at "
            f"{_count(row.get('reverse_boundary_spans'))} boundaries. "
            "The direct parent identifiers at the candidate boundaries were missing."
        )

    if channel == "operation":
        parent = _safe_text(row.get("representative_parent_operation"))
        child = _safe_text(row.get("representative_child_operation"))
        available = bool(parent or child) or _finite(
            row.get("reconstructed_boundary_pairs")
        ) > 0
        if not available:
            return None
        return header + (
            f"A representative outer operation from Service A was '{parent or 'unavailable'}'. "
            f"A representative inner operation from Service B was '{child or 'unavailable'}'. "
            f"Their average operation-token overlap was "
            f"{_percent(row.get('operation_jaccard_mean'))}. "
            f"The most frequent operation pair accounted for "
            f"{_percent(row.get('operation_pair_concentration'))} of candidate boundaries. "
            f"Service A's operation appeared in a parent role at rate "
            f"{_percent(row.get('source_operation_parent_prior'))}; "
            f"Service B's operation appeared in a child role at rate "
            f"{_percent(row.get('target_operation_child_prior'))}."
        )

    if channel == "http":
        method_coverage = _finite(row.get("method_coverage"))
        route_coverage = _finite(row.get("route_coverage"))
        parent_method = _safe_text(
            row.get("representative_parent_http_method")
        )
        child_method = _safe_text(
            row.get("representative_child_http_method")
        )
        parent_route = _safe_text(row.get("representative_parent_route"))
        child_route = _safe_text(row.get("representative_child_route"))
        if not (
            method_coverage > 0
            or route_coverage > 0
            or parent_method
            or child_method
            or parent_route
            or child_route
        ):
            return None
        return header + (
            f"HTTP method evidence was available at "
            f"{_percent(method_coverage)} of candidate boundaries and the known "
            f"methods matched at rate {_percent(row.get('method_match_rate'))}. "
            f"The representative methods were Service A='{parent_method or 'unavailable'}' "
            f"and Service B='{child_method or 'unavailable'}'. "
            f"Normalized route evidence covered {_percent(route_coverage)} of boundaries; "
            f"route-token overlap was {_percent(row.get('route_jaccard_mean'))} and exact "
            f"route match was {_percent(row.get('route_exact_rate'))}. "
            f"The representative routes were Service A='{parent_route or 'unavailable'}' "
            f"and Service B='{child_route or 'unavailable'}'."
        )

    if channel == "role":
        return header + (
            f"In the observed dependency graph with the candidate edge absent, Service A "
            f"had {_count(row.get('source_out_degree'))} outgoing and "
            f"{_count(row.get('source_in_degree'))} incoming service neighbors. "
            f"Service B had {_count(row.get('target_out_degree'))} outgoing and "
            f"{_count(row.get('target_in_degree'))} incoming neighbors. "
            f"Their caller-to-callee graph-role compatibility was "
            f"{_percent(row.get('graph_role_score'))}. "
            f"Direct span-kind evidence covered {_percent(row.get('span_kind_coverage'))} "
            f"and its CLIENT-to-SERVER compatibility was "
            f"{_percent(row.get('span_kind_compatibility_score'))}. "
            f"Workload-routing evidence covered {_percent(row.get('workload_coverage'))} "
            f"and matched the proposed direction at rate "
            f"{_percent(row.get('workload_match_score'))}."
        )

    raise ValueError(f"unsupported evidence channel: {channel}")


def _hypotheses() -> tuple[str, str]:
    return (
        "Service A directly calls Service B in the runtime dependency graph.",
        "Service B directly calls Service A in the runtime dependency graph.",
    )


def _normal_probabilities(raw: Mapping[str, Any]) -> dict[str, float]:
    values = {
        name: float(raw[name])
        for name in ("entailment", "contradiction", "neutral")
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise Phase3R3Error(f"invalid NLI probabilities: {values}")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise Phase3R3Error(f"NLI probabilities do not sum to one: {values}")
    return values


def _tri_state(
    forward: Mapping[str, float],
    reverse: Mapping[str, float],
    config: TriStateConfig,
) -> tuple[str, float]:
    """Return a diagnostic state and continuous forward-direction support."""

    forward_entailment = float(forward["entailment"])
    reverse_entailment = float(reverse["entailment"])
    forward_contradiction = float(forward["contradiction"])
    reverse_contradiction = float(reverse["contradiction"])
    direction_margin = forward_entailment - reverse_entailment
    label_margin = forward_entailment - forward_contradiction
    score = max(
        -1.0,
        min(
            1.0,
            0.50 * (forward_entailment - reverse_entailment)
            + 0.50 * (reverse_contradiction - forward_contradiction),
        ),
    )
    if (
        forward_entailment >= config.corroborate_entailment_min
        and direction_margin >= config.direction_margin_min
        and label_margin >= config.label_margin_min
    ):
        state = "corroborates"
    elif (
        reverse_entailment >= config.contradict_probability_min
        and reverse_entailment - forward_entailment
        >= config.direction_margin_min
    ) or (
        forward_contradiction >= config.contradict_probability_min
        and forward_contradiction - forward_entailment
        >= config.direction_margin_min
    ):
        state = "contradicts"
    else:
        state = "ambiguous"
    return state, score


def score_evidence_channels(
    model_frame: pd.DataFrame,
    *,
    backend: OnnxDebertaNLIBackend,
    tri_state: TriStateConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score independent evidence channels; no evaluator column is accepted."""

    leaked = EVALUATOR_COLUMNS.intersection(model_frame.columns)
    if leaked:
        raise Phase3R3Error(
            "NLI scoring received evaluator columns: " + ", ".join(sorted(leaked))
        )
    required = {
        *JOIN_KEY,
        "a2_score",
        "proposal_rank",
        "direct_evidence",
        "direction_role_rank_normalized",
        "operation_endpoint_rank_normalized",
        "method_route_rank_normalized",
        "combined_rank_normalized",
    }
    missing = sorted(required.difference(model_frame.columns))
    if missing:
        raise Phase3R3Error(f"R3 model input is missing columns: {missing}")
    frame = model_frame.reset_index(drop=True).copy()
    forward_hypothesis, reverse_hypothesis = _hypotheses()
    pairs: list[tuple[str, str]] = []
    locations: list[tuple[int, str, str]] = []
    premise_counts: dict[str, int] = {channel: 0 for channel in CHANNELS}

    for index, row in frame.iterrows():
        record = row.to_dict()
        for channel in CHANNELS:
            premise = _channel_premise(channel, record)
            available = premise is not None
            frame.loc[index, f"nli_{channel}_available"] = available
            if not available:
                continue
            premise_counts[channel] += 1
            pairs.extend(
                (
                    (premise, forward_hypothesis),
                    (premise, reverse_hypothesis),
                )
            )
            locations.extend(
                ((index, channel, "forward"), (index, channel, "reverse"))
            )

    token_lengths = backend.pair_token_lengths(tuple(pairs)) if pairs else ()
    if token_lengths and max(token_lengths) > backend.max_length:
        raise Phase3R3Error(
            "channel-specific NLI input exceeds token budget: "
            f"max={max(token_lengths)} budget={backend.max_length}"
        )
    raw_scores = backend.score_pairs(tuple(pairs)) if pairs else ()
    if len(raw_scores) != len(locations):
        raise Phase3R3Error(
            f"expected {len(locations)} NLI outputs, received {len(raw_scores)}"
        )
    score_maps: dict[tuple[int, str, str], dict[str, float]] = {}
    for location, raw in zip(locations, raw_scores):
        score_maps[location] = _normal_probabilities(raw)

    state_counts: dict[str, dict[str, int]] = {
        channel: {"corroborates": 0, "ambiguous": 0, "contradicts": 0}
        for channel in CHANNELS
    }
    for index in frame.index:
        for channel in CHANNELS:
            if not _truthy(frame.loc[index, f"nli_{channel}_available"]):
                for name in ("entailment", "contradiction", "neutral"):
                    frame.loc[index, f"nli_{channel}_forward_{name}"] = math.nan
                    frame.loc[index, f"nli_{channel}_reverse_{name}"] = math.nan
                frame.loc[index, f"nli_{channel}_score"] = 0.0
                frame.loc[index, f"nli_{channel}_state"] = "unavailable"
                continue
            forward = score_maps[(index, channel, "forward")]
            reverse = score_maps[(index, channel, "reverse")]
            for name, value in forward.items():
                frame.loc[index, f"nli_{channel}_forward_{name}"] = value
            for name, value in reverse.items():
                frame.loc[index, f"nli_{channel}_reverse_{name}"] = value
            state, score = _tri_state(forward, reverse, tri_state)
            frame.loc[index, f"nli_{channel}_score"] = score
            frame.loc[index, f"nli_{channel}_state"] = state
            state_counts[channel][state] += 1

    cache = backend.cache_info()
    diagnostics = {
        "candidate_count": len(frame),
        "candidate_channel_pairs": len(locations) // 2,
        "nli_pair_count": len(pairs),
        "channel_candidate_coverage": {
            channel: premise_counts[channel] / len(frame) if len(frame) else 0.0
            for channel in CHANNELS
        },
        "tri_state_counts": state_counts,
        "minimum_pair_tokens": min(token_lengths) if token_lengths else None,
        "maximum_pair_tokens": max(token_lengths) if token_lengths else None,
        "truncation_count": 0,
        "cache": asdict(cache),
        "backend": dict(backend.metadata()),
    }
    return frame, diagnostics


def add_nli_profile_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output_groups: list[pd.DataFrame] = []
    for _key, group in frame.groupby(list(CELL_KEY), sort=True, dropna=False):
        group = group.copy().reset_index(drop=True)
        group["a2_rank_normalized"] = _rank_normalize(group, "a2_score")
        for profile, weights in NLI_PROFILE_WEIGHTS.items():
            numerators = pd.Series(0.0, index=group.index, dtype=float)
            denominators = pd.Series(0.0, index=group.index, dtype=float)
            for channel, weight in weights.items():
                available = group[f"nli_{channel}_available"].map(_truthy)
                numerators += (
                    group[f"nli_{channel}_score"].astype(float)
                    * float(weight)
                    * available.astype(float)
                )
                denominators += float(weight) * available.astype(float)
            raw = numerators / denominators.where(denominators > 0, 1.0)
            group[f"nli_{profile}_raw"] = raw
            group[f"nli_{profile}_available_weight"] = denominators
            group[f"nli_{profile}_rank_normalized"] = _rank_normalize(
                group, f"nli_{profile}_raw"
            )
        output_groups.append(group)
    return pd.concat(output_groups, ignore_index=True) if output_groups else frame.copy()


def apply_policy(frame: pd.DataFrame, policy: R3Policy) -> pd.DataFrame:
    scored = frame.copy()
    a2_weight = 1.0 - policy.operational_weight - policy.nli_weight
    operational_rank = scored[
        f"{policy.operational_profile}_rank_normalized"
    ].astype(float)
    nli_rank = scored[f"nli_{policy.nli_profile}_rank_normalized"].astype(float)
    scored["r3_operational_component"] = operational_rank
    scored["r3_nli_component"] = nli_rank
    scored["a3_r3_score"] = (
        a2_weight * scored["a2_rank_normalized"].astype(float)
        + policy.operational_weight * operational_rank
        + policy.nli_weight * nli_rank
    )
    keep = min(
        len(scored),
        max(
            policy.minimum_keep,
            int(math.ceil(policy.retention_fraction * len(scored))),
        ),
    )
    scored["selected"] = False
    direct = set(scored.index[scored["direct_evidence"].map(_truthy)])
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        ["a3_r3_score", "a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    selected = set(direct)
    for index in ranked.index:
        if len(selected) >= keep:
            break
        selected.add(index)
    scored.loc[list(selected), "selected"] = True
    return scored


def _control(
    frame: pd.DataFrame,
    *,
    selected_count: int,
    operational_profile: str | None = None,
    operational_weight: float = 0.0,
) -> pd.DataFrame:
    scored = frame.copy()
    if operational_profile is None or operational_weight <= 0.0:
        scored["a3_r3_score"] = scored["a2_rank_normalized"].astype(float)
    else:
        scored["a3_r3_score"] = (
            (1.0 - operational_weight)
            * scored["a2_rank_normalized"].astype(float)
            + operational_weight
            * scored[f"{operational_profile}_rank_normalized"].astype(float)
        )
    scored["selected"] = False
    direct = set(scored.index[scored["direct_evidence"].map(_truthy)])
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        ["a3_r3_score", "a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    selected = set(direct)
    for index in ranked.index:
        if len(selected) >= selected_count:
            break
        selected.add(index)
    scored.loc[list(selected), "selected"] = True
    return scored


def evaluate_cell(scored: pd.DataFrame) -> dict[str, Any]:
    selected = scored.loc[scored["selected"].map(_truthy)]
    target_keys = {
        tuple(map(str, row))
        for row in scored.loc[
            scored["is_masked_target"].map(_truthy), list(CANDIDATE_KEY)
        ].itertuples(index=False, name=None)
    }
    silver_keys = {
        tuple(map(str, row))
        for row in scored.loc[
            scored["is_silver_matched"].map(_truthy), list(CANDIDATE_KEY)
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
        target_item = by_key.get(target)
        if target_item is None:
            reciprocal.append(0.0)
            ranks.append(None)
            continue
        target_score = float(target_item.a3_r3_score)
        competitors = []
        for item in by_query.get((target[0], target[1]), ()):
            key = (str(item.subject), str(item.predicate), str(item.object))
            if key == target or key in silver_keys:
                continue
            competitors.append(item)
        higher = sum(
            float(item.a3_r3_score) > target_score + epsilon
            for item in competitors
        )
        tied = sum(
            abs(float(item.a3_r3_score) - target_score) <= epsilon
            for item in competitors
        )
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    return {
        "selected_count": len(selected_keys),
        "target_count": len(target_keys),
        "recovered_target_count": len(recovered),
        "recall": len(recovered) / len(target_keys) if target_keys else None,
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
    recalls = [float(row["recall"]) for row in rows if row["recall"] is not None]
    p_lbs = [
        float(row["silver_precision_lower_bound"])
        for row in rows
        if row["silver_precision_lower_bound"] is not None
    ]
    mrrs = [float(row["mrr"]) for row in rows if row["mrr"] is not None]
    counts = [int(row["selected_count"]) for row in rows]
    targets = sum(int(row["target_count"]) for row in rows)
    recovered = sum(int(row["recovered_target_count"]) for row in rows)
    return {
        "cell_count": len(rows),
        "recall_macro": statistics.fmean(recalls),
        "recall_min": min(recalls),
        "recall_pooled": recovered / targets if targets else None,
        "selected_count_mean": statistics.fmean(counts),
        "selected_count_median": statistics.median(counts),
        "selected_count_max": max(counts),
        "silver_precision_lower_bound_macro": statistics.fmean(p_lbs),
        "silver_precision_lower_bound_min": min(p_lbs),
        "mrr_macro": statistics.fmean(mrrs),
        "mrr_min": min(mrrs),
    }


def _delta(enhanced: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
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
    rows: list[dict[str, Any]] = []
    operational_rows: list[dict[str, Any]] = []
    a2_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for key, group in sorted(cells.items()):
        group = group.reset_index(drop=True)
        scored = apply_policy(group, policy)
        metric = evaluate_cell(scored)
        operational = _control(
            group,
            selected_count=int(metric["selected_count"]),
            operational_profile=policy.operational_profile,
            operational_weight=policy.operational_weight,
        )
        operational_metric = evaluate_cell(operational)
        a2 = _control(group, selected_count=int(metric["selected_count"]))
        a2_metric = evaluate_cell(a2)
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
        rows.append({**common, **metric})
        operational_rows.append({**common, **operational_metric})
        a2_rows.append({**common, **a2_metric})
        scored = scored.copy()
        for name, value in common.items():
            scored[name] = value
        scored_frames.append(scored)
    return (
        pd.DataFrame.from_records(rows),
        _aggregate(rows),
        pd.DataFrame.from_records(operational_rows),
        _aggregate(operational_rows),
        pd.DataFrame.from_records(a2_rows),
        _aggregate(a2_rows),
        pd.concat(scored_frames, ignore_index=True),
    )


def _baseline(cells: Mapping[tuple[str, int, str], pd.DataFrame]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in cells.values():
        pseudo = group.copy()
        pseudo["selected"] = True
        pseudo["a3_r3_score"] = pseudo["a2_rank_normalized"].astype(float)
        rows.append(evaluate_cell(pseudo))
    return _aggregate(rows)


def _policy_grid(config: Mapping[str, Any]) -> list[R3Policy]:
    search = config["policy_search"]
    policies: list[R3Policy] = []
    for operational_profile in search["operational_profiles"]:
        for nli_profile in search["nli_profiles"]:
            for retention in search["retention_fractions"]:
                for minimum_keep in search["minimum_keep"]:
                    for operational_weight in search["operational_weights"]:
                        for nli_weight in search["nli_weights"]:
                            if float(operational_weight) + float(nli_weight) > 0.80:
                                continue
                            policies.append(
                                R3Policy(
                                    str(operational_profile),
                                    str(nli_profile),
                                    float(retention),
                                    int(minimum_keep),
                                    float(operational_weight),
                                    float(nli_weight),
                                )
                            )
    return policies


def _select_policy(
    calibration: Mapping[tuple[str, int, str], pd.DataFrame],
    config: Mapping[str, Any],
    gate: GateConfig,
) -> tuple[R3Policy, pd.DataFrame, bool]:
    baseline = _baseline(calibration)
    rows: list[dict[str, Any]] = []
    for policy in _policy_grid(config):
        (
            _candidate_rows,
            aggregate,
            _operational_rows,
            operational,
            _a2_rows,
            a2,
            _scored,
        ) = _evaluate_policy(calibration, policy)
        delta_operational = _delta(aggregate, operational)
        delta_a2 = _delta(aggregate, a2)
        selected_ratio = (
            aggregate["selected_count_mean"] / baseline["selected_count_mean"]
        )
        additive_gain = max(
            delta_operational["silver_precision_lower_bound_macro"],
            delta_operational["mrr_macro"],
        )
        conditions = {
            "recall_macro": aggregate["recall_macro"] >= gate.recall_macro_min,
            "recall_each_cell": aggregate["recall_min"] >= gate.recall_each_cell_min,
            "candidate_count_reduced": selected_ratio <= gate.selected_count_ratio_max,
            "mrr_noninferior_to_full_a2": aggregate["mrr_macro"]
            >= baseline["mrr_macro"] - 0.01,
            "matched_operational_recall": delta_operational["recall_macro"]
            >= -gate.matched_operational_recall_tolerance,
            "matched_operational_p_lb": delta_operational[
                "silver_precision_lower_bound_macro"
            ]
            >= gate.matched_operational_p_lb_delta_min,
            "matched_operational_mrr": delta_operational["mrr_macro"]
            >= gate.matched_operational_mrr_delta_min,
            "matched_a2_recall": delta_a2["recall_macro"]
            >= -gate.matched_a2_recall_tolerance,
            "matched_a2_p_lb": delta_a2["silver_precision_lower_bound_macro"]
            >= gate.matched_a2_p_lb_delta_min,
            "matched_a2_mrr": delta_a2["mrr_macro"]
            >= gate.matched_a2_mrr_delta_min,
            "nli_additive_gain": additive_gain >= gate.nli_additive_gain_min,
        }
        rows.append(
            {
                **asdict(policy),
                **aggregate,
                "baseline_mrr_macro": baseline["mrr_macro"],
                "selected_count_ratio": selected_ratio,
                "operational_control_recall_macro": operational["recall_macro"],
                "operational_control_p_lb_macro": operational[
                    "silver_precision_lower_bound_macro"
                ],
                "operational_control_mrr_macro": operational["mrr_macro"],
                "a2_control_recall_macro": a2["recall_macro"],
                "a2_control_p_lb_macro": a2[
                    "silver_precision_lower_bound_macro"
                ],
                "a2_control_mrr_macro": a2["mrr_macro"],
                "matched_operational_recall_delta": delta_operational["recall_macro"],
                "matched_operational_p_lb_delta": delta_operational[
                    "silver_precision_lower_bound_macro"
                ],
                "matched_operational_mrr_delta": delta_operational["mrr_macro"],
                "matched_a2_recall_delta": delta_a2["recall_macro"],
                "matched_a2_p_lb_delta": delta_a2[
                    "silver_precision_lower_bound_macro"
                ],
                "matched_a2_mrr_delta": delta_a2["mrr_macro"],
                "nli_additive_gain": additive_gain,
                "feasible": all(conditions.values()),
                "violation_count": sum(not value for value in conditions.values()),
            }
        )
    grid = pd.DataFrame.from_records(rows)
    feasible = grid.loc[grid["feasible"].map(_truthy)]
    pool = feasible if not feasible.empty else grid
    sort_columns = (
        [
            "selected_count_mean",
            "nli_additive_gain",
            "silver_precision_lower_bound_macro",
            "mrr_macro",
            "nli_weight",
        ]
        if not feasible.empty
        else [
            "violation_count",
            "nli_additive_gain",
            "recall_macro",
            "selected_count_mean",
            "nli_weight",
        ]
    )
    chosen = pool.sort_values(
        sort_columns,
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    policy = R3Policy(
        str(chosen.operational_profile),
        str(chosen.nli_profile),
        float(chosen.retention_fraction),
        int(chosen.minimum_keep),
        float(chosen.operational_weight),
        float(chosen.nli_weight),
    )
    grid["selected"] = (
        grid["operational_profile"].astype(str).eq(policy.operational_profile)
        & grid["nli_profile"].astype(str).eq(policy.nli_profile)
        & grid["retention_fraction"].astype(float).eq(policy.retention_fraction)
        & grid["minimum_keep"].astype(int).eq(policy.minimum_keep)
        & grid["operational_weight"].astype(float).eq(policy.operational_weight)
        & grid["nli_weight"].astype(float).eq(policy.nli_weight)
    )
    return policy, grid, not feasible.empty


def _render_report(summary: Mapping[str, Any]) -> str:
    held = summary["heldout"]
    base = held["baseline_a2_full"]
    proposed = held["proposed_a3_r3"]
    op_delta = held["delta_vs_equal_size_operational"]
    a2_delta = held["delta_vs_equal_size_a2"]
    full_delta = held["delta_vs_full_a2"]
    reasons = ", ".join(summary["gate"]["reason_codes"]) or "없음"
    return f"""# Task A Phase 3-R3 결과 — Evidence별 DeBERTa + 선형 재랭커

- 최종 Gate: **{summary['status']}**
- Calibration feasible 정책: **{summary['calibration']['feasible_policy_count']} / {summary['calibration']['searched_policy_count']}**
- 선택 정책: `{summary['selected_policy']}`
- 미통과 조건: `{reasons}`

## Held-out 40 Cell

| 지표 | A2 전체 | A3-R3 | A2 전체 대비 | 동일크기 Operational 대비 | 동일크기 A2 대비 |
|---|---:|---:|---:|---:|---:|
| Recall Macro | {base['recall_macro']:.4f} | {proposed['recall_macro']:.4f} | {full_delta['recall_macro']:+.4f} | {op_delta['recall_macro']:+.4f} | {a2_delta['recall_macro']:+.4f} |
| Recall Minimum | {base['recall_min']:.4f} | {proposed['recall_min']:.4f} | - | - | - |
| 후보 수 평균 | {base['selected_count_mean']:.3f} | {proposed['selected_count_mean']:.3f} | {full_delta['selected_count_mean']:+.3f} | {op_delta['selected_count_mean']:+.3f} | {a2_delta['selected_count_mean']:+.3f} |
| P-LB Macro | {base['silver_precision_lower_bound_macro']:.4f} | {proposed['silver_precision_lower_bound_macro']:.4f} | {full_delta['silver_precision_lower_bound_macro']:+.4f} | {op_delta['silver_precision_lower_bound_macro']:+.4f} | {a2_delta['silver_precision_lower_bound_macro']:+.4f} |
| MRR Macro | {base['mrr_macro']:.4f} | {proposed['mrr_macro']:.4f} | {full_delta['mrr_macro']:+.4f} | {op_delta['mrr_macro']:+.4f} | {a2_delta['mrr_macro']:+.4f} |

## 원칙

- Trace direction, Operation, HTTP, Runtime role을 하나의 긴 문장이 아닌 독립 Premise로 평가했다.
- DeBERTa의 tri-state는 진단용이며 어떤 후보도 단독으로 제거하지 않는다.
- A2 prior는 최소 20%를 항상 보존했다.
- NLI의 효과는 동일 후보 수의 Operational-only와 A2-only 대조군으로 분리했다.
- 이 결과는 개발 단계이며 새로운 독립 Incident에서 재확인이 필요하다.
"""


def run_phase3_r3(
    *,
    candidate_analysis: Path,
    model_dir: Path,
    output: Path,
    config_path: Path,
) -> Path:
    candidate_analysis = candidate_analysis.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if output.exists():
        raise Phase3R3Error(f"refusing to overwrite output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != (
        "rcaeval-task-a-phase3-r3-channel-nli"
    ):
        raise Phase3R3Error("unexpected experiment_id")

    raw = pd.read_parquet(candidate_analysis)
    required = {
        *JOIN_KEY,
        *EVALUATOR_COLUMNS,
        "mask_ratio",
        "a2_score",
        "proposal_rank",
        "direct_evidence",
        "direction_role_rank_normalized",
        "operation_endpoint_rank_normalized",
        "method_route_rank_normalized",
        "combined_rank_normalized",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise Phase3R3Error(f"candidate analysis is missing columns: {missing}")
    source = config["source_contract"]
    if len(raw) != int(source["required_candidate_rows"]):
        raise Phase3R3Error(
            f"unexpected candidate rows: expected={source['required_candidate_rows']} observed={len(raw)}"
        )
    if raw.groupby(list(CELL_KEY)).ngroups != int(source["required_cells"]):
        raise Phase3R3Error("unexpected candidate cell count")
    if raw.duplicated(list(JOIN_KEY)).any():
        raise Phase3R3Error("duplicate R3 candidate keys")
    if set(raw["role"].astype(str)) != {"calibration", "heldout"}:
        raise Phase3R3Error("calibration and heldout roles are required")

    evaluator = raw[list(JOIN_KEY) + sorted(EVALUATOR_COLUMNS)].copy()
    model = raw[[column for column in raw.columns if column not in EVALUATOR_COLUMNS]].copy()
    backend = OnnxDebertaNLIBackend(
        model_dir,
        expected_sha256=str(
            config["model"].get(
                "onnx_sha256", NLI_DEBERTA_V3_SMALL_AVX2_SHA256
            )
        ),
        batch_size=1,
        performance_mode=False,
        max_length=int(config["model"].get("max_length", 512)),
    )
    availability = backend.availability()
    if availability.status != "READY" or not backend.research_valid:
        raise Phase3R3Error(
            f"DeBERTa backend unavailable or research-invalid: {availability}"
        )
    scored_model, nli_diagnostics = score_evidence_channels(
        model,
        backend=backend,
        tri_state=TriStateConfig(**dict(config["tri_state"])),
    )
    scored_model = add_nli_profile_scores(scored_model)
    frame = scored_model.merge(
        evaluator, on=list(JOIN_KEY), how="inner", validate="one_to_one"
    )
    if len(frame) != len(raw):
        raise Phase3R3Error("model/evaluator rejoin changed candidate count")

    calibration_frame = frame.loc[frame["role"].astype(str).eq("calibration")].copy()
    heldout_frame = frame.loc[frame["role"].astype(str).eq("heldout")].copy()
    split = config["split_contract"]
    if calibration_frame.groupby(list(CELL_KEY)).ngroups != int(
        split["calibration_cells"]
    ):
        raise Phase3R3Error("unexpected calibration cell count")
    if heldout_frame.groupby(list(CELL_KEY)).ngroups != int(
        split["heldout_cells"]
    ):
        raise Phase3R3Error("unexpected heldout cell count")

    calibration_cells = {
        tuple(key): group.copy()
        for key, group in calibration_frame.groupby(
            list(CELL_KEY), sort=True, dropna=False
        )
    }
    heldout_cells = {
        tuple(key): group.copy()
        for key, group in heldout_frame.groupby(
            list(CELL_KEY), sort=True, dropna=False
        )
    }
    gate = GateConfig(**dict(config["gate"]))
    policy, grid, calibration_feasible = _select_policy(
        calibration_cells, config, gate
    )
    (
        calibration_rows,
        calibration_aggregate,
        calibration_operational_rows,
        calibration_operational,
        calibration_a2_rows,
        calibration_a2,
        _calibration_scored,
    ) = _evaluate_policy(calibration_cells, policy)
    (
        heldout_rows,
        heldout_aggregate,
        heldout_operational_rows,
        heldout_operational,
        heldout_a2_rows,
        heldout_a2,
        heldout_scored,
    ) = _evaluate_policy(heldout_cells, policy)
    baseline_calibration = _baseline(calibration_cells)
    baseline_heldout = _baseline(heldout_cells)
    delta_full = _delta(heldout_aggregate, baseline_heldout)
    delta_operational = _delta(heldout_aggregate, heldout_operational)
    delta_a2 = _delta(heldout_aggregate, heldout_a2)
    selected_ratio = (
        heldout_aggregate["selected_count_mean"]
        / baseline_heldout["selected_count_mean"]
    )
    nli_additive_gain = max(
        delta_operational["silver_precision_lower_bound_macro"],
        delta_operational["mrr_macro"],
    )
    nli_profile_scores = heldout_frame[
        f"nli_{policy.nli_profile}_raw"
    ].astype(float)
    scored_coverage = nli_diagnostics["candidate_count"] / len(raw)
    conditions = {
        "calibration_policy_feasible": calibration_feasible,
        "heldout_complete": len(heldout_rows) == int(split["heldout_cells"]),
        "recall_macro": heldout_aggregate["recall_macro"]
        >= gate.recall_macro_min,
        "recall_pooled": heldout_aggregate["recall_pooled"]
        >= gate.recall_pooled_min,
        "recall_each_cell": heldout_aggregate["recall_min"]
        >= gate.recall_each_cell_min,
        "candidate_count_reduced": selected_ratio
        <= gate.selected_count_ratio_max,
        "p_lb_noninferior_to_full_a2": delta_full[
            "silver_precision_lower_bound_macro"
        ]
        >= gate.p_lb_delta_vs_full_min,
        "mrr_noninferior_to_full_a2": delta_full["mrr_macro"]
        >= gate.mrr_delta_vs_full_min,
        "matched_operational_recall": delta_operational["recall_macro"]
        >= -gate.matched_operational_recall_tolerance,
        "matched_operational_p_lb": delta_operational[
            "silver_precision_lower_bound_macro"
        ]
        >= gate.matched_operational_p_lb_delta_min,
        "matched_operational_mrr": delta_operational["mrr_macro"]
        >= gate.matched_operational_mrr_delta_min,
        "matched_a2_recall": delta_a2["recall_macro"]
        >= -gate.matched_a2_recall_tolerance,
        "matched_a2_p_lb": delta_a2["silver_precision_lower_bound_macro"]
        >= gate.matched_a2_p_lb_delta_min,
        "matched_a2_mrr": delta_a2["mrr_macro"]
        >= gate.matched_a2_mrr_delta_min,
        "nli_additive_gain": nli_additive_gain >= gate.nli_additive_gain_min,
        "nli_weight_active": policy.nli_weight > 0.0,
        "a2_prior_preserved": (
            1.0 - policy.operational_weight - policy.nli_weight
        )
        >= 0.20 - 1e-12,
        "all_candidates_scored": scored_coverage
        >= gate.scored_candidate_coverage_min,
        "nli_score_discrimination": float(nli_profile_scores.std(ddof=0))
        >= gate.nli_score_std_min,
        "no_token_truncation": nli_diagnostics["truncation_count"] == 0,
        "backend_research_valid": bool(backend.research_valid),
        "a2_candidates_preserved": len(frame) == len(raw),
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
        model_output / "a3_r3_channel_nli_features.parquet", index=False
    )
    frame.to_parquet(
        evaluator_private / "all_candidate_analysis.parquet", index=False
    )
    heldout_scored.to_parquet(
        evaluator_private / "heldout_candidate_analysis.parquet", index=False
    )
    heldout_operational_rows.to_csv(
        evaluator_private / "equal_size_operational_control_cells.csv", index=False
    )
    heldout_a2_rows.to_csv(
        evaluator_private / "equal_size_a2_control_cells.csv", index=False
    )
    calibration_rows.to_csv(
        published / "task_a_phase3_r3_calibration_cells.csv", index=False
    )
    heldout_rows.to_csv(
        published / "task_a_phase3_r3_heldout_cells.csv", index=False
    )
    grid.to_csv(
        published / "task_a_phase3_r3_policy_grid.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "PASS" if passed else "FAIL",
        "gate_id": "D4C_A3_R3_CHANNEL_NLI_ADDITIVE_UTILITY",
        "protocol_status": dict(config.get("protocol_status", {})),
        "source": {
            "candidate_analysis": str(candidate_analysis),
            "candidate_analysis_sha256": hashlib.sha256(
                candidate_analysis.read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "candidate_rows": len(raw),
            "candidate_cells": raw.groupby(list(CELL_KEY)).ngroups,
        },
        "model": {
            **dict(backend.metadata()),
            "availability": availability.status,
            "artifact_sha256": backend.artifact_sha256,
        },
        "leakage_boundary": {
            "evaluator_columns_removed_before_nli": True,
            "anonymous_service_placeholders": True,
            "service_ids_used_in_nli_text": False,
            "fault_or_root_labels_used_in_nli": False,
            "tri_state_is_hard_veto": False,
            "policy_selection": "calibration incidents only",
            "heldout_labels_used_for_policy_selection": False,
            "model_evaluator_rejoin": "opaque incident token + candidate key, one_to_one",
        },
        "nli_diagnostics": nli_diagnostics,
        "selected_policy": asdict(policy),
        "calibration": {
            "feasible": calibration_feasible,
            "searched_policy_count": len(grid),
            "feasible_policy_count": int(
                grid["feasible"].map(_truthy).sum()
            ),
            "baseline_a2_full": baseline_calibration,
            "proposed_a3_r3": calibration_aggregate,
            "equal_size_operational_control": calibration_operational,
            "equal_size_a2_control": calibration_a2,
        },
        "heldout": {
            "baseline_a2_full": baseline_heldout,
            "proposed_a3_r3": heldout_aggregate,
            "equal_size_operational_control": heldout_operational,
            "equal_size_a2_control": heldout_a2,
            "delta_vs_full_a2": delta_full,
            "delta_vs_equal_size_operational": delta_operational,
            "delta_vs_equal_size_a2": delta_a2,
            "selected_count_ratio": selected_ratio,
            "nli_additive_gain": nli_additive_gain,
            "nli_profile_score_std": float(nli_profile_scores.std(ddof=0)),
        },
        "gate": {
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "conditions": conditions,
            "reason_codes": [
                name.upper() for name, value in conditions.items() if not value
            ],
            "required": asdict(gate),
        },
        "claim_limit": (
            "Development-only CALLS candidate reranking on six previously inspected "
            "RCAEval TrainTicket incidents. No causal-edge, RCA, LLM, or production "
            "generalization claim; a fresh independent incident set is required."
        ),
    }
    result_path = published / "task_a_phase3_r3_results.json"
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (published / "task_a_phase3_r3_results.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    (published / "task_a_phase3_r3_status.txt").write_text(
        summary["status"] + "\n", encoding="utf-8"
    )
    return output


__all__ = [
    "GateConfig",
    "Phase3R3Error",
    "R3Policy",
    "TriStateConfig",
    "add_nli_profile_scores",
    "apply_policy",
    "evaluate_cell",
    "run_phase3_r3",
    "score_evidence_channels",
    "_channel_premise",
    "_tri_state",
]
