"""Task A Phase 3-R1: model-visible structural evidence re-ranking.

This experiment addresses the highest-priority A3 failure before reusing NLI.
It derives relation-specific CALLS evidence directly from the sanitized trace
partition, keeps every A2 proposal immutable, and tests whether positional and
operation evidence improves an equal-size A2-only shortlist on held-out cases.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .phase3_contract import (
    Phase3Error,
    REQUIRED_CELL_COLUMNS,
    _json_mapping,
    _json_sequence,
    _load_json,
    stable_case_split,
)
from .phase3_nli import _load_evaluator_sets


DEFAULT_STRUCTURAL_CONFIG = Path(
    "configs/experiment_task_a_rcaeval_phase3_structural.json"
)
EXPERIMENT_ID = "rcaeval-task-a-phase3-structural-evidence"
PROFILE_FIELDS: Mapping[str, Mapping[str, float]] = {
    "temporal_directness": {
        "tightness_mean": 0.35,
        "alternative_margin_mean": 0.25,
        "container_uniqueness_mean": 0.15,
        "parent_span_diversity": 0.15,
        "trace_independence": 0.10,
    },
    "directional_structure": {
        "direction_asymmetry": 0.45,
        "tightness_mean": 0.25,
        "alternative_margin_mean": 0.15,
        "support_alignment": 0.15,
    },
    "operation_compatibility": {
        "operation_overlap_mean": 0.45,
        "method_match_rate": 0.30,
        "operation_pair_concentration": 0.25,
    },
    "hybrid": {
        "temporal_directness_score": 0.50,
        "directional_structure_score": 0.30,
        "operation_compatibility_score": 0.20,
    },
}

_OPERATION_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*", re.I)
_OPERATION_GENERIC = {
    "api",
    "client",
    "controller",
    "delete",
    "get",
    "handler",
    "head",
    "http",
    "https",
    "options",
    "patch",
    "post",
    "put",
    "request",
    "response",
    "server",
    "service",
}
_NULL_TEXT = {"", "<na>", "nan", "nat", "none", "null"}


@dataclass(frozen=True)
class StructuralPolicy:
    profile_id: str
    retention_fraction: float
    minimum_keep: int
    structural_weight: float

    def __post_init__(self) -> None:
        if self.profile_id not in PROFILE_FIELDS:
            raise ValueError(f"unknown structural profile: {self.profile_id}")
        if (
            not math.isfinite(self.retention_fraction)
            or not 0.0 < self.retention_fraction <= 1.0
        ):
            raise ValueError("retention_fraction must be in (0,1]")
        if (
            isinstance(self.minimum_keep, bool)
            or not isinstance(self.minimum_keep, int)
            or self.minimum_keep <= 0
        ):
            raise ValueError("minimum_keep must be a positive integer")
        if (
            not math.isfinite(self.structural_weight)
            or not 0.0 <= self.structural_weight <= 1.0
        ):
            raise ValueError("structural_weight must be in [0,1]")


@dataclass(frozen=True)
class _Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service_id: str
    operation_name: str
    method_name: str
    start: float
    end: float

    @property
    def width(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class _BoundaryEvent:
    key: tuple[str, str, str]
    trace_id: str
    parent_span_id: str
    child_span_id: str
    tightness: float
    alternative_margin: float
    container_uniqueness: float
    operation_overlap: float
    method_match: float
    operation_pair: str


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).split())
    return "" if text.strip().lower() in _NULL_TEXT else text


def _parent_id(value: Any) -> str | None:
    text = _text_or_empty(value).strip()
    return None if text.lower() in _NULL_TEXT else text


def _operation_tokens(operation: str, method: str) -> frozenset[str]:
    text = f"{operation} {method}".lower()
    tokens = {
        token
        for token in _OPERATION_TOKEN_RE.findall(text)
        if token not in _OPERATION_GENERIC
        and not token.isdigit()
        and len(token) > 1
    }
    return frozenset(tokens)


def _operation_overlap(parent: _Span, child: _Span) -> float:
    left = _operation_tokens(parent.operation_name, parent.method_name)
    right = _operation_tokens(child.operation_name, child.method_name)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _operation_signature(span: _Span) -> str:
    operation = _text_or_empty(span.operation_name).lower()
    method = _text_or_empty(span.method_name).lower()
    material = f"{method}|{operation}".strip("|")
    return material or "unknown"


def _method_match(parent: _Span, child: _Span) -> float:
    left = _text_or_empty(parent.method_name).lower()
    right = _text_or_empty(child.method_name).lower()
    return float(bool(left and right and left == right))


def _cell_root(phase2_root: Path, row: Mapping[str, Any]) -> Path:
    summary = phase2_root / str(row["run_summary"])
    return summary.parent / "masks" / str(row["mask_id"])


def _load_candidates(
    phase2_root: Path,
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    a2_path = _cell_root(phase2_root, row) / "predictions" / "A2.parquet"
    if not a2_path.is_file():
        raise Phase3Error(f"A2 predictions are missing: {a2_path}")
    frame = pd.read_parquet(a2_path)
    required = {
        "subject",
        "predicate",
        "object",
        "score",
        "decision",
        "evidence_ids_json",
        "stage_scores_json",
        "reason_codes_json",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Phase3Error(f"A2 parquet missing columns {missing}: {a2_path}")

    records: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        key = (str(item.subject), str(item.predicate), str(item.object))
        if str(item.decision) != "accepted":
            raise Phase3Error(f"A2 proposal is not accepted: {key}")
        stage = _json_mapping(
            item.stage_scores_json,
            field_name="stage_scores_json",
        )
        reasons = {
            str(value)
            for value in _json_sequence(
                item.reason_codes_json,
                field_name="reason_codes_json",
            )
        }
        evidence_ids = tuple(
            str(value)
            for value in _json_sequence(
                item.evidence_ids_json,
                field_name="evidence_ids_json",
            )
        )
        records.append(
            {
                "subject": key[0],
                "predicate": key[1],
                "object": key[2],
                "a2_score": float(item.score),
                "supporting_traces": int(
                    float(stage.get("supporting_traces", 0))
                ),
                "boundary_spans": int(float(stage.get("boundary_spans", 0))),
                "proposal_rank": int(float(stage.get("proposal_rank", 0))),
                "direct_evidence": "DIRECT_EVIDENCE" in reasons,
                "a2_evidence_span_count": len(set(evidence_ids)),
            }
        )
    if len(records) != int(row["a2_proposal_count"]):
        raise Phase3Error(
            f"A2 proposal count mismatch: csv={row['a2_proposal_count']} "
            f"parquet={len(records)}"
        )
    return records


def _trace_spans(path: Path) -> dict[str, list[_Span]]:
    if not path.is_file():
        raise Phase3Error(f"sanitized trace artifact is missing: {path}")
    frame = pd.read_parquet(path)
    required = {
        "trace_id",
        "span_id",
        "parent_span_id",
        "service_id",
        "start_time_us",
        "end_time_us",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Phase3Error(f"trace parquet missing columns {missing}: {path}")
    for optional in ("operation_name", "method_name"):
        if optional not in frame.columns:
            frame[optional] = ""

    by_trace: dict[str, list[_Span]] = {}
    selected = frame[
        [
            "trace_id",
            "span_id",
            "parent_span_id",
            "service_id",
            "operation_name",
            "method_name",
            "start_time_us",
            "end_time_us",
        ]
    ]
    for row in selected.itertuples(index=False, name=None):
        trace_id, span_id, parent, service, operation, method, start, end = row
        start_value = float(start)
        end_value = float(end)
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or end_value < start_value
        ):
            continue
        span = _Span(
            trace_id=str(trace_id),
            span_id=str(span_id),
            parent_span_id=_parent_id(parent),
            service_id=str(service),
            operation_name=_text_or_empty(operation),
            method_name=_text_or_empty(method),
            start=start_value,
            end=end_value,
        )
        by_trace.setdefault(span.trace_id, []).append(span)
    return by_trace


def _extract_boundary_events(
    traces_by_id: Mapping[str, list[_Span]],
    candidate_keys: set[tuple[str, str, str]],
    *,
    include_null_parent: bool,
) -> Mapping[tuple[str, str, str], list[_BoundaryEvent]]:
    output: dict[tuple[str, str, str], list[_BoundaryEvent]] = {
        key: [] for key in candidate_keys
    }
    for trace_id, raw_spans in traces_by_id.items():
        span_ids = {span.span_id for span in raw_spans}
        trace_spans = sorted(
            raw_spans,
            key=lambda item: (item.start, -item.end, item.span_id),
        )
        active: list[_Span] = []
        for child in trace_spans:
            active = [parent for parent in active if parent.end >= child.start]
            parent_unmatched = (
                child.parent_span_id is not None
                and child.parent_span_id not in span_ids
            ) or (include_null_parent and child.parent_span_id is None)
            if not parent_unmatched:
                active.append(child)
                continue
            containers = [
                parent
                for parent in active
                if parent.span_id != child.span_id
                and parent.service_id != child.service_id
                and parent.end >= child.end
                and (parent.start < child.start or parent.end > child.end)
            ]
            if not containers:
                active.append(child)
                continue
            containers.sort(
                key=lambda item: (item.width, item.start, item.span_id)
            )
            minimum_width = containers[0].width
            tied_services = {
                item.service_id
                for item in containers
                if math.isclose(
                    item.width,
                    minimum_width,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            }
            if len(tied_services) != 1:
                active.append(child)
                continue
            parent = containers[0]
            key = (parent.service_id, "CALLS", child.service_id)
            if key not in output:
                active.append(child)
                continue

            alternative = next(
                (
                    item
                    for item in containers[1:]
                    if item.service_id != parent.service_id
                ),
                None,
            )
            if alternative is None:
                alternative_margin = 1.0
            else:
                denominator = max(alternative.width, 1e-12)
                alternative_margin = _clip01(
                    (alternative.width - parent.width) / denominator
                )
            parent_width = max(parent.width, 1e-12)
            tightness = _clip01(child.width / parent_width)
            service_count = len({item.service_id for item in containers})
            container_uniqueness = 1.0 / max(1, service_count)
            output[key].append(
                _BoundaryEvent(
                    key=key,
                    trace_id=trace_id,
                    parent_span_id=parent.span_id,
                    child_span_id=child.span_id,
                    tightness=tightness,
                    alternative_margin=alternative_margin,
                    container_uniqueness=container_uniqueness,
                    operation_overlap=_operation_overlap(parent, child),
                    method_match=_method_match(parent, child),
                    operation_pair=(
                        f"{_operation_signature(parent)}=>"
                        f"{_operation_signature(child)}"
                    ),
                )
            )
            active.append(child)
    return output


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    material = list(values)
    return statistics.fmean(material) if material else default


def _profile_value(
    record: Mapping[str, Any],
    weights: Mapping[str, float],
) -> float:
    denominator = sum(float(value) for value in weights.values())
    if denominator <= 0.0:
        return 0.0
    return _clip01(
        sum(
            float(record.get(name, 0.0)) * float(weight)
            for name, weight in weights.items()
        )
        / denominator
    )


def attach_structural_evidence(
    candidates: Sequence[dict[str, Any]],
    traces: pd.DataFrame | Path,
    *,
    include_null_parent: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach trace-position and operation evidence without evaluator labels."""

    if isinstance(traces, Path):
        traces_by_id = _trace_spans(traces)
        trace_path = str(traces)
    else:
        temporary = traces.copy()
        required = {
            "trace_id",
            "span_id",
            "parent_span_id",
            "service_id",
            "start_time_us",
            "end_time_us",
        }
        missing = sorted(required.difference(temporary.columns))
        if missing:
            raise Phase3Error(f"trace frame missing columns: {missing}")
        for optional in ("operation_name", "method_name"):
            if optional not in temporary.columns:
                temporary[optional] = ""
        traces_by_id: dict[str, list[_Span]] = {}
        for row in temporary[
            [
                "trace_id",
                "span_id",
                "parent_span_id",
                "service_id",
                "operation_name",
                "method_name",
                "start_time_us",
                "end_time_us",
            ]
        ].itertuples(index=False, name=None):
            trace_id, span_id, parent, service, operation, method, start, end = row
            span = _Span(
                str(trace_id),
                str(span_id),
                _parent_id(parent),
                str(service),
                _text_or_empty(operation),
                _text_or_empty(method),
                float(start),
                float(end),
            )
            traces_by_id.setdefault(span.trace_id, []).append(span)
        trace_path = "<in-memory>"

    keys = {
        (
            str(item["subject"]),
            str(item["predicate"]),
            str(item["object"]),
        )
        for item in candidates
    }
    events_by_key = _extract_boundary_events(
        traces_by_id,
        keys,
        include_null_parent=include_null_parent,
    )

    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            str(candidate["subject"]),
            str(candidate["predicate"]),
            str(candidate["object"]),
        )
        events = events_by_key.get(key, [])
        operation_pairs = Counter(event.operation_pair for event in events)
        pair_count = len(events)
        trace_count = len({event.trace_id for event in events})
        parent_count = len({event.parent_span_id for event in events})
        boundary_expected = int(candidate.get("boundary_spans", 0))
        trace_expected = int(candidate.get("supporting_traces", 0))
        support_alignment = (
            1.0
            if bool(candidate.get("direct_evidence")) and boundary_expected == 0
            else _clip01(
                1.0
                - abs(pair_count - boundary_expected)
                / max(1, boundary_expected)
            )
        )
        trace_alignment = (
            1.0
            if bool(candidate.get("direct_evidence")) and trace_expected == 0
            else _clip01(
                1.0
                - abs(trace_count - trace_expected)
                / max(1, trace_expected)
            )
        )
        record = {
            **candidate,
            "reconstructed_boundary_spans": pair_count,
            "reconstructed_supporting_traces": trace_count,
            "boundary_count_delta": pair_count - boundary_expected,
            "trace_count_delta": trace_count - trace_expected,
            "support_alignment": min(support_alignment, trace_alignment),
            "tightness_mean": _mean(event.tightness for event in events),
            "alternative_margin_mean": _mean(
                event.alternative_margin for event in events
            ),
            "container_uniqueness_mean": _mean(
                event.container_uniqueness for event in events
            ),
            "operation_overlap_mean": _mean(
                event.operation_overlap for event in events
            ),
            "method_match_rate": _mean(
                event.method_match for event in events
            ),
            "operation_pair_concentration": (
                max(operation_pairs.values()) / pair_count
                if pair_count
                else 0.0
            ),
            "parent_span_diversity": (
                parent_count / pair_count if pair_count else 0.0
            ),
            "trace_independence": (
                trace_count / pair_count if pair_count else 0.0
            ),
            "operation_pair_unique_count": len(operation_pairs),
        }
        aggregates[key] = record

    for key, record in aggregates.items():
        reverse = aggregates.get((key[2], key[1], key[0]))
        reverse_boundaries = int(
            reverse["reconstructed_boundary_spans"] if reverse else 0
        )
        forward_boundaries = int(record["reconstructed_boundary_spans"])
        record["reverse_reconstructed_boundary_spans"] = reverse_boundaries
        record["direction_asymmetry"] = _clip01(
            0.5
            * (
                1.0
                + (forward_boundaries - reverse_boundaries)
                / max(1.0, forward_boundaries + reverse_boundaries)
            )
        )
        record["temporal_directness_score"] = _profile_value(
            record,
            PROFILE_FIELDS["temporal_directness"],
        )
        record["directional_structure_score"] = _profile_value(
            record,
            PROFILE_FIELDS["directional_structure"],
        )
        record["operation_compatibility_score"] = _profile_value(
            record,
            PROFILE_FIELDS["operation_compatibility"],
        )
        record["hybrid_score"] = _profile_value(
            record,
            PROFILE_FIELDS["hybrid"],
        )

    ordered = [
        aggregates[
            (
                str(item["subject"]),
                str(item["predicate"]),
                str(item["object"]),
            )
        ]
        for item in candidates
    ]
    abductive = [item for item in ordered if not item["direct_evidence"]]
    diagnostics = {
        "trace_source": trace_path,
        "trace_count": len(traces_by_id),
        "candidate_count": len(ordered),
        "abductive_candidate_count": len(abductive),
        "aligned_abductive_candidates": sum(
            item["boundary_count_delta"] == 0
            and item["trace_count_delta"] == 0
            for item in abductive
        ),
        "boundary_mismatch_count": sum(
            item["boundary_count_delta"] != 0 for item in abductive
        ),
        "trace_mismatch_count": sum(
            item["trace_count_delta"] != 0 for item in abductive
        ),
        "feature_profiles": sorted(PROFILE_FIELDS),
        "model_visible_only": True,
    }
    return ordered, diagnostics


def _a2_rank_norm(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[tuple[str, str, str], float]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item["a2_score"]),
            -int(item["supporting_traces"]),
            -int(item["boundary_spans"]),
            int(item["proposal_rank"]),
            str(item["subject"]),
            str(item["object"]),
        ),
    )
    denominator = max(1, len(ordered) - 1)
    return {
        (
            str(item["subject"]),
            str(item["predicate"]),
            str(item["object"]),
        ): 1.0
        - index / denominator
        for index, item in enumerate(ordered)
    }


def _dense_percentile(
    values: Mapping[tuple[str, str, str], float],
) -> Mapping[tuple[str, str, str], float]:
    unique = sorted(set(float(value) for value in values.values()))
    if len(unique) <= 1:
        return {key: 0.5 for key in values}
    rank = {
        value: index / (len(unique) - 1)
        for index, value in enumerate(unique)
    }
    return {key: rank[float(value)] for key, value in values.items()}


def apply_structural_policy(
    candidates: Sequence[dict[str, Any]],
    policy: StructuralPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not candidates:
        return [], []
    a2_norm = _a2_rank_norm(candidates)
    raw = {
        (
            str(item["subject"]),
            str(item["predicate"]),
            str(item["object"]),
        ): float(item[f"{policy.profile_id}_score"])
        for item in candidates
    }
    structure_norm = _dense_percentile(raw)
    scored: list[dict[str, Any]] = []
    for item in candidates:
        key = (
            str(item["subject"]),
            str(item["predicate"]),
            str(item["object"]),
        )
        score = (
            (1.0 - policy.structural_weight) * a2_norm[key]
            + policy.structural_weight * structure_norm[key]
        )
        scored.append(
            {
                **item,
                "structural_profile": policy.profile_id,
                "structural_profile_score": raw[key],
                "structural_rank_normalized": structure_norm[key],
                "a2_rank_normalized": a2_norm[key],
                "a3s_score": score,
                "selected": False,
            }
        )

    keep = min(
        len(scored),
        max(
            policy.minimum_keep,
            int(math.ceil(policy.retention_fraction * len(scored))),
        ),
    )
    direct = [item for item in scored if item["direct_evidence"]]
    ranked = sorted(
        (item for item in scored if not item["direct_evidence"]),
        key=lambda item: (
            -float(item["a3s_score"]),
            -float(item["a2_score"]),
            int(item["proposal_rank"]),
            str(item["subject"]),
            str(item["object"]),
        ),
    )
    selected_keys = {
        (item["subject"], item["predicate"], item["object"]) for item in direct
    }
    for item in ranked:
        if len(selected_keys) >= keep:
            break
        selected_keys.add(
            (item["subject"], item["predicate"], item["object"])
        )
    for item in scored:
        item["selected"] = (
            item["subject"],
            item["predicate"],
            item["object"],
        ) in selected_keys
    return [item for item in scored if item["selected"]], scored


def evaluate_structural_shortlist(
    selected: Sequence[Mapping[str, Any]],
    *,
    targets: set[tuple[str, str, str]],
    silver: set[tuple[str, str, str]],
) -> dict[str, Any]:
    keys = {
        (str(item["subject"]), str(item["predicate"]), str(item["object"]))
        for item in selected
    }
    recovered = keys & targets
    matched = keys & silver
    by_key = {
        (str(item["subject"]), str(item["predicate"]), str(item["object"])): item
        for item in selected
    }
    by_query: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for key, item in by_key.items():
        by_query.setdefault((key[0], key[1]), []).append(item)

    reciprocal: list[float] = []
    ranks: list[int | None] = []
    epsilon = 1e-12
    for target in sorted(targets):
        target_item = by_key.get(target)
        if target_item is None:
            reciprocal.append(0.0)
            ranks.append(None)
            continue
        target_score = float(target_item["a3s_score"])
        competitors = []
        for item in by_query.get((target[0], target[1]), ()):
            key = (
                str(item["subject"]),
                str(item["predicate"]),
                str(item["object"]),
            )
            if key == target or key in silver:
                continue
            competitors.append(item)
        higher = sum(
            float(item["a3s_score"]) > target_score + epsilon
            for item in competitors
        )
        tied = sum(
            abs(float(item["a3s_score"]) - target_score) <= epsilon
            for item in competitors
        )
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    return {
        "selected_count": len(keys),
        "target_count": len(targets),
        "recovered_target_count": len(recovered),
        "recall": len(recovered) / len(targets) if targets else None,
        "silver_matched_count": len(matched),
        "silver_precision_lower_bound": (
            len(matched) / len(keys) if keys else None
        ),
        "mrr": statistics.fmean(reciprocal) if reciprocal else None,
        "ranks": ranks,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recalls = [
        float(item["recall"])
        for item in rows
        if item["recall"] is not None
    ]
    p_lbs = [
        float(item["silver_precision_lower_bound"])
        for item in rows
        if item["silver_precision_lower_bound"] is not None
    ]
    mrrs = [
        float(item["mrr"]) for item in rows if item["mrr"] is not None
    ]
    counts = [int(item["selected_count"]) for item in rows]
    total_targets = sum(int(item["target_count"]) for item in rows)
    total_recovered = sum(
        int(item["recovered_target_count"]) for item in rows
    )
    return {
        "cell_count": len(rows),
        "recall_macro": statistics.fmean(recalls),
        "recall_min": min(recalls),
        "recall_pooled": (
            total_recovered / total_targets if total_targets else None
        ),
        "selected_count_mean": statistics.fmean(counts),
        "selected_count_median": statistics.median(counts),
        "selected_count_max": max(counts),
        "silver_precision_lower_bound_macro": statistics.fmean(p_lbs),
        "silver_precision_lower_bound_min": min(p_lbs),
        "mrr_macro": statistics.fmean(mrrs),
        "mrr_min": min(mrrs),
    }


def _evaluate_cells(
    cells: Sequence[dict[str, Any]],
    policy: StructuralPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        selected, _scored = apply_structural_policy(
            cell["candidates"],
            policy,
        )
        metric = evaluate_structural_shortlist(
            selected,
            targets=cell["targets"],
            silver=cell["silver"],
        )
        rows.append(
            {
                "case": cell["case"],
                "fault": cell["fault"],
                "role": cell["role"],
                "seed": cell["seed"],
                "mask_id": cell["mask_id"],
                "mask_ratio": cell["mask_ratio"],
                **metric,
            }
        )
    return rows, _aggregate(rows)


def _baseline_aggregate(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cell_count": len(cells),
        "recall_macro": statistics.fmean(
            float(cell["a2_recall"]) for cell in cells
        ),
        "recall_min": min(float(cell["a2_recall"]) for cell in cells),
        "selected_count_mean": statistics.fmean(
            int(cell["a2_count"]) for cell in cells
        ),
        "selected_count_median": statistics.median(
            int(cell["a2_count"]) for cell in cells
        ),
        "selected_count_max": max(int(cell["a2_count"]) for cell in cells),
        "silver_precision_lower_bound_macro": statistics.fmean(
            float(cell["a2_p_lb"]) for cell in cells
        ),
        "silver_precision_lower_bound_min": min(
            float(cell["a2_p_lb"]) for cell in cells
        ),
        "mrr_macro": statistics.fmean(
            float(cell["a2_mrr"]) for cell in cells
        ),
        "mrr_min": min(float(cell["a2_mrr"]) for cell in cells),
    }


def _delta(
    enhanced: Mapping[str, Any],
    baseline: Mapping[str, Any],
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


def select_structural_policy(
    calibration_cells: Sequence[dict[str, Any]],
    *,
    search: Mapping[str, Any],
    gate: Mapping[str, Any],
    allow_diagnostic_fallback: bool = True,
) -> tuple[StructuralPolicy, list[dict[str, Any]]]:
    baseline_mrr = statistics.fmean(
        float(cell["a2_mrr"]) for cell in calibration_cells
    )
    rows: list[dict[str, Any]] = []
    for profile_id in search["profiles"]:
        for retention in search["retention_fractions"]:
            for minimum_keep in search["minimum_keep"]:
                control = StructuralPolicy(
                    str(profile_id),
                    float(retention),
                    int(minimum_keep),
                    0.0,
                )
                control_rows, control_aggregate = _evaluate_cells(
                    calibration_cells,
                    control,
                )
                control_sets = {
                    (row["case"], row["seed"], row["mask_id"]): row
                    for row in control_rows
                }
                for weight in search["structural_weights"]:
                    policy = StructuralPolicy(
                        str(profile_id),
                        float(retention),
                        int(minimum_keep),
                        float(weight),
                    )
                    candidate_rows, aggregate = _evaluate_cells(
                        calibration_cells,
                        policy,
                    )
                    matched_delta = _delta(aggregate, control_aggregate)
                    additive_gain = (
                        matched_delta[
                            "silver_precision_lower_bound_macro"
                        ]
                        > 1e-12
                        or matched_delta["mrr_macro"] > 1e-12
                    )
                    selection_changed = any(
                        (
                            row["silver_matched_count"],
                            row["recovered_target_count"],
                            row["mrr"],
                        )
                        != (
                            control_sets[
                                (
                                    row["case"],
                                    row["seed"],
                                    row["mask_id"],
                                )
                            ]["silver_matched_count"],
                            control_sets[
                                (
                                    row["case"],
                                    row["seed"],
                                    row["mask_id"],
                                )
                            ]["recovered_target_count"],
                            control_sets[
                                (
                                    row["case"],
                                    row["seed"],
                                    row["mask_id"],
                                )
                            ]["mrr"],
                        )
                        for row in candidate_rows
                    )
                    conditions = {
                        "recall_macro": aggregate["recall_macro"]
                        >= float(gate["recall_macro_min"]),
                        "recall_each_cell": aggregate["recall_min"]
                        >= float(gate["recall_each_cell_min"]),
                        "mrr_noninferiority": aggregate["mrr_macro"]
                        >= baseline_mrr
                        - float(gate["mrr_noninferiority_tolerance"]),
                        "matched_budget_recall": matched_delta[
                            "recall_macro"
                        ]
                        >= -float(gate["matched_budget_recall_tolerance"]),
                        "matched_budget_p_lb": matched_delta[
                            "silver_precision_lower_bound_macro"
                        ]
                        >= float(gate["matched_budget_p_lb_delta_min"]),
                        "matched_budget_mrr": matched_delta["mrr_macro"]
                        >= float(gate["matched_budget_mrr_delta_min"]),
                        "matched_budget_additive_gain": additive_gain,
                        "selection_changed": selection_changed,
                    }
                    feasible = all(conditions.values())
                    rows.append(
                        {
                            **asdict(policy),
                            **aggregate,
                            "baseline_mrr_macro": baseline_mrr,
                            "control_recall_macro": control_aggregate[
                                "recall_macro"
                            ],
                            "control_p_lb_macro": control_aggregate[
                                "silver_precision_lower_bound_macro"
                            ],
                            "control_mrr_macro": control_aggregate["mrr_macro"],
                            "matched_budget_recall_delta": matched_delta[
                                "recall_macro"
                            ],
                            "matched_budget_p_lb_delta": matched_delta[
                                "silver_precision_lower_bound_macro"
                            ],
                            "matched_budget_mrr_delta": matched_delta[
                                "mrr_macro"
                            ],
                            "matched_budget_additive_gain": additive_gain,
                            "selection_changed": selection_changed,
                            **{
                                f"condition_{name}": value
                                for name, value in conditions.items()
                            },
                            "violation_count": sum(
                                not value for value in conditions.values()
                            ),
                            "feasible": feasible,
                            "selected": False,
                            "selection_status": "NOT_SELECTED",
                        }
                    )
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        pool = feasible
        status = "FEASIBLE_POLICY"
    elif allow_diagnostic_fallback:
        pool = rows
        status = "DIAGNOSTIC_FALLBACK_NO_FEASIBLE_POLICY"
    else:
        raise Phase3Error(
            "no structural policy satisfies the calibration gate"
        )
    if not pool:
        raise Phase3Error("structural policy search produced no policies")
    chosen = sorted(
        pool,
        key=lambda row: (
            int(row["violation_count"]),
            float(row["selected_count_mean"]),
            -float(row["matched_budget_p_lb_delta"]),
            -float(row["matched_budget_mrr_delta"]),
            -float(row["recall_min"]),
            -float(row["recall_macro"]),
            float(row["structural_weight"]),
            str(row["profile_id"]),
        ),
    )[0]
    identity = (
        str(chosen["profile_id"]),
        float(chosen["retention_fraction"]),
        int(chosen["minimum_keep"]),
        float(chosen["structural_weight"]),
    )
    for row in rows:
        if (
            str(row["profile_id"]),
            float(row["retention_fraction"]),
            int(row["minimum_keep"]),
            float(row["structural_weight"]),
        ) == identity:
            row["selected"] = True
            row["selection_status"] = status
    return StructuralPolicy(*identity), rows


def validate_structural_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise Phase3Error(
            "structural Phase-3 config schema_version must be 1"
        )
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise Phase3Error("unexpected structural Phase-3 experiment_id")
    split = config.get("calibration_split", {})
    if (
        split.get("method") != "sha256_case_order"
        or int(split.get("calibration_incidents", 0)) != 2
    ):
        raise Phase3Error(
            "structural Phase-3 requires the frozen 2-incident hash split"
        )
    search = config.get("policy_search", {})
    profiles = tuple(str(value) for value in search.get("profiles", ()))
    if not profiles or any(value not in PROFILE_FIELDS for value in profiles):
        raise Phase3Error(
            "policy_search.profiles contains an unsupported profile"
        )
    for field in (
        "retention_fractions",
        "minimum_keep",
        "structural_weights",
    ):
        if not tuple(search.get(field, ())):
            raise Phase3Error(f"policy_search.{field} cannot be empty")
    if any(
        float(value) <= 0.0 or float(value) > 1.0
        for value in search["retention_fractions"]
    ):
        raise Phase3Error("retention fractions must be in (0,1]")
    if any(int(value) <= 0 for value in search["minimum_keep"]):
        raise Phase3Error("minimum_keep values must be positive")
    if any(
        float(value) <= 0.0 or float(value) > 1.0
        for value in search["structural_weights"]
    ):
        raise Phase3Error("structural_weights must be in (0,1]")


def _render_report(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline"]["heldout"]
    proposed = summary["proposed"]["heldout"]
    delta = summary["proposed"]["heldout_delta_vs_a2"]
    matched = summary["matched_budget_control"][
        "heldout_delta_proposed_minus_control"
    ]
    policy = summary["selected_policy"]
    gate = summary["gate"]
    return "\n".join(
        [
            "# Task A Phase 3-R1 결과 — 구조 Evidence 재랭킹",
            "",
            f"- 최종 Gate: **{gate['status']}**",
            (
                "- Calibration feasible 정책: "
                f"**{summary['calibration_selection']['feasible_policy_count']} / "
                f"{summary['calibration_selection']['searched_policy_count']}**"
            ),
            f"- 선택 Profile: **{policy['profile_id']}**",
            f"- Structural weight: **{policy['structural_weight']:.4f}**",
            f"- 미통과 조건: `{', '.join(gate['reason_codes']) or '없음'}`",
            "",
            "## Held-out 결과",
            "",
            "| 지표 | A2 전체 | 구조 Shortlist | 변화 |",
            "|---|---:|---:|---:|",
            (
                f"| Recall Macro | {baseline['recall_macro']:.4f} | "
                f"{proposed['recall_macro']:.4f} | "
                f"{delta['recall_macro']:+.4f} |"
            ),
            (
                f"| Recall Minimum | {baseline['recall_min']:.4f} | "
                f"{proposed['recall_min']:.4f} | - |"
            ),
            (
                f"| 후보 수 평균 | {baseline['selected_count_mean']:.2f} | "
                f"{proposed['selected_count_mean']:.2f} | "
                f"{delta['selected_count_mean']:+.2f} |"
            ),
            (
                "| P-LB Macro | "
                f"{baseline['silver_precision_lower_bound_macro']:.4f} | "
                f"{proposed['silver_precision_lower_bound_macro']:.4f} | "
                f"{delta['silver_precision_lower_bound_macro']:+.4f} |"
            ),
            (
                f"| MRR Macro | {baseline['mrr_macro']:.4f} | "
                f"{proposed['mrr_macro']:.4f} | "
                f"{delta['mrr_macro']:+.4f} |"
            ),
            "",
            "## 동일 후보 수 A2-only 대비 구조 Evidence 고유 효과",
            "",
            "| 지표 | 구조 - A2 matched-budget |",
            "|---|---:|",
            f"| Recall Macro | {matched['recall_macro']:+.4f} |",
            (
                "| P-LB Macro | "
                f"{matched['silver_precision_lower_bound_macro']:+.4f} |"
            ),
            f"| MRR Macro | {matched['mrr_macro']:+.4f} |",
            "",
            "## 해석",
            "",
            (
                "- 이 단계는 DeBERTa를 사용하지 않고 Trace 위치·방향·"
                "Operation Evidence 자체의 추가 판별력을 검증한다."
            ),
            (
                "- 구조 Evidence가 equal-size A2 control보다 개선돼야 이후 "
                "NLI를 독립 Evidence 채널로 재도입한다."
            ),
            (
                "- `CALLS`는 runtime dependency이며 causal `CAUSES`를 "
                "의미하지 않는다."
            ),
            "",
        ]
    )


def run_structural_phase3(
    *,
    phase2_root: Path,
    output: Path,
    config_path: Path = DEFAULT_STRUCTURAL_CONFIG,
) -> Path:
    phase2_root = phase2_root.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if output.exists():
        raise Phase3Error(f"refusing to overwrite existing output: {output}")
    config = _load_json(config_path)
    validate_structural_config(config)

    cells_path = phase2_root / "cells.csv"
    frame = pd.read_csv(cells_path)
    missing = sorted(REQUIRED_CELL_COLUMNS.difference(frame.columns))
    if missing:
        raise Phase3Error(
            f"Phase-2 cells.csv is missing columns: {missing}"
        )
    if frame.duplicated(["case", "seed", "mask_id"]).any():
        raise Phase3Error("Phase-2 cells.csv contains duplicate cells")
    phase2_summary = _load_json(phase2_root / "summary.json")
    if not phase2_summary.get("gate", {}).get("passed"):
        raise Phase3Error("Phase-2 D3 gate must pass before structural A3")
    contract = config["phase2_contract"]
    if len(frame) != int(contract["required_cells"]):
        raise Phase3Error(
            "Phase-2 cell count differs from the structural A3 contract"
        )
    if frame["case"].nunique() != int(contract["required_incidents"]):
        raise Phase3Error(
            "Phase-2 incident count differs from the structural A3 contract"
        )
    if set(frame["seed"].astype(int)) != {
        int(value) for value in contract["required_seeds"]
    }:
        raise Phase3Error(
            "Phase-2 seeds differ from the structural A3 contract"
        )
    if set(frame["mask_ratio"].astype(float)) != {
        float(value) for value in contract["required_mask_ratios"]
    }:
        raise Phase3Error(
            "Phase-2 mask ratios differ from the structural A3 contract"
        )

    revision = str(config["dataset_revision"])
    calibration_cases, heldout_cases, case_hashes = stable_case_split(
        frame["case"].astype(str),
        revision=revision,
        calibration_incidents=int(
            config["calibration_split"]["calibration_incidents"]
        ),
    )
    role_by_case = {case: "calibration" for case in calibration_cases}
    role_by_case.update({case: "heldout" for case in heldout_cases})

    cells: list[dict[str, Any]] = []
    source_rows: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    extraction_diagnostics: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        key = (
            str(row["case"]),
            int(row["seed"]),
            str(row["mask_id"]),
        )
        candidates = _load_candidates(phase2_root, row)
        trace_path = (
            _cell_root(phase2_root, row)
            / "model_input"
            / "traces.parquet"
        )
        candidates, diagnostics = attach_structural_evidence(
            candidates,
            trace_path,
            include_null_parent=bool(
                config["structural_evidence"].get(
                    "include_null_parent",
                    True,
                )
            ),
        )
        source_rows[key] = row
        extraction_diagnostics.append(
            {
                "case": key[0],
                "seed": key[1],
                "mask_id": key[2],
                **diagnostics,
            }
        )
        cells.append(
            {
                "case": key[0],
                "fault": str(row["fault"]),
                "seed": key[1],
                "mask_id": key[2],
                "mask_ratio": float(row["mask_ratio"]),
                "role": role_by_case[key[0]],
                "a2_count": int(row["a2_proposal_count"]),
                "a2_recall": float(row["candidate_recall"]),
                "a2_mrr": float(row["mrr_within_a2"]),
                "a2_p_lb": float(row["silver_precision_lower_bound"]),
                "candidates": candidates,
            }
        )

    calibration_cells = [
        cell for cell in cells if cell["role"] == "calibration"
    ]
    heldout_cells = [cell for cell in cells if cell["role"] == "heldout"]
    for cell in calibration_cells:
        row = source_rows[
            (cell["case"], cell["seed"], cell["mask_id"])
        ]
        candidate_keys = {
            (item["subject"], item["predicate"], item["object"])
            for item in cell["candidates"]
        }
        cell["targets"], cell["silver"] = _load_evaluator_sets(
            phase2_root,
            row,
            candidate_keys,
        )

    policy, calibration_grid = select_structural_policy(
        calibration_cells,
        search=config["policy_search"],
        gate=config["calibration_gate"],
        allow_diagnostic_fallback=True,
    )
    selected_rows = [row for row in calibration_grid if row["selected"]]
    if len(selected_rows) != 1:
        raise Phase3Error("calibration grid must mark exactly one policy")
    selected_row = selected_rows[0]
    calibration_feasible = bool(selected_row["feasible"])
    control_policy = StructuralPolicy(
        policy.profile_id,
        policy.retention_fraction,
        policy.minimum_keep,
        0.0,
    )

    for cell in heldout_cells:
        row = source_rows[
            (cell["case"], cell["seed"], cell["mask_id"])
        ]
        candidate_keys = {
            (item["subject"], item["predicate"], item["object"])
            for item in cell["candidates"]
        }
        cell["targets"], cell["silver"] = _load_evaluator_sets(
            phase2_root,
            row,
            candidate_keys,
        )

    proposed_rows, proposed_all = _evaluate_cells(cells, policy)
    proposed_cal_rows, proposed_cal = _evaluate_cells(
        calibration_cells,
        policy,
    )
    proposed_held_rows, proposed_held = _evaluate_cells(
        heldout_cells,
        policy,
    )
    control_rows, control_all = _evaluate_cells(cells, control_policy)
    _control_held_rows, control_held = _evaluate_cells(
        heldout_cells,
        control_policy,
    )
    baseline_all = _baseline_aggregate(cells)
    baseline_cal = _baseline_aggregate(calibration_cells)
    baseline_held = _baseline_aggregate(heldout_cells)
    matched_delta = _delta(proposed_held, control_held)
    gate_cfg = config["heldout_gate"]
    count_ratio = (
        proposed_held["selected_count_mean"]
        / baseline_held["selected_count_mean"]
    )
    additive_gain = (
        matched_delta["silver_precision_lower_bound_macro"] > 1e-12
        or matched_delta["mrr_macro"] > 1e-12
    )
    all_aligned = all(
        item["boundary_mismatch_count"] == 0
        and item["trace_mismatch_count"] == 0
        for item in extraction_diagnostics
    )
    conditions = {
        "calibration_policy_feasible": calibration_feasible,
        "heldout_complete": len(proposed_held_rows) == len(heldout_cells),
        "feature_alignment": all_aligned,
        "recall_macro": proposed_held["recall_macro"]
        >= float(gate_cfg["recall_macro_min"]),
        "recall_pooled": proposed_held["recall_pooled"]
        >= float(gate_cfg["recall_pooled_min"]),
        "recall_each_cell": proposed_held["recall_min"]
        >= float(gate_cfg["recall_each_cell_min"]),
        "candidate_count_reduced": count_ratio
        <= float(gate_cfg["selected_count_ratio_max"]),
        "p_lb_improved": (
            proposed_held["silver_precision_lower_bound_macro"]
            >= baseline_held["silver_precision_lower_bound_macro"]
            + float(gate_cfg["p_lb_macro_delta_min"])
        ),
        "mrr_improved": (
            proposed_held["mrr_macro"]
            >= baseline_held["mrr_macro"]
            + float(gate_cfg["mrr_macro_delta_min"])
        ),
        "structural_weight_active": policy.structural_weight > 0.0,
        "matched_budget_recall_noninferior": (
            matched_delta["recall_macro"]
            >= -float(gate_cfg["matched_budget_recall_tolerance"])
        ),
        "matched_budget_p_lb_noninferior": (
            matched_delta["silver_precision_lower_bound_macro"]
            >= float(gate_cfg["matched_budget_p_lb_delta_min"])
        ),
        "matched_budget_mrr_noninferior": (
            matched_delta["mrr_macro"]
            >= float(gate_cfg["matched_budget_mrr_delta_min"])
        ),
        "matched_budget_additive_gain": additive_gain,
        "a2_candidates_preserved": sum(
            len(cell["candidates"]) for cell in cells
        )
        == int(frame["a2_proposal_count"].sum()),
    }
    passed = all(conditions.values())

    detailed: list[dict[str, Any]] = []
    selected_counts = {
        (row["case"], int(row["seed"]), row["mask_id"]): int(
            row["selected_count"]
        )
        for row in proposed_rows
    }
    for cell in cells:
        _selected, scored = apply_structural_policy(
            cell["candidates"],
            policy,
        )
        for record in scored:
            edge = (
                record["subject"],
                record["predicate"],
                record["object"],
            )
            detailed.append(
                {
                    "incident_id": hashlib.sha256(
                        f"{revision}|a3s|{cell['case']}".encode("utf-8")
                    ).hexdigest()[:24],
                    "case": cell["case"],
                    "fault": cell["fault"],
                    "role": cell["role"],
                    "seed": cell["seed"],
                    "mask_id": cell["mask_id"],
                    "mask_ratio": cell["mask_ratio"],
                    **record,
                    "is_masked_target": edge in cell["targets"],
                    "is_silver_matched": edge in cell["silver"],
                    "cell_selected_count": selected_counts[
                        (cell["case"], cell["seed"], cell["mask_id"])
                    ],
                }
            )

    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "model_output"
    private_dir = output / "evaluator_private"
    published_dir = output / "published"
    model_dir.mkdir()
    private_dir.mkdir()
    published_dir.mkdir()
    detail_frame = pd.DataFrame.from_records(detailed)
    private_columns = {
        "case",
        "fault",
        "role",
        "is_masked_target",
        "is_silver_matched",
    }
    model_columns = [
        column for column in detail_frame.columns if column not in private_columns
    ]
    detail_frame[model_columns].to_parquet(
        model_dir / "a3s_structural_evidence.parquet",
        index=False,
    )
    detail_frame.to_parquet(
        private_dir / "a3s_candidate_analysis.parquet",
        index=False,
    )
    pd.DataFrame.from_records(extraction_diagnostics).to_csv(
        private_dir / "feature_alignment.csv",
        index=False,
    )
    pd.DataFrame.from_records(proposed_rows).to_csv(
        private_dir / "a3s_cells.csv",
        index=False,
    )
    pd.DataFrame.from_records(control_rows).to_csv(
        private_dir / "a2_budget_control_cells.csv",
        index=False,
    )
    pd.DataFrame.from_records(calibration_grid).to_csv(
        private_dir / "calibration_grid.csv",
        index=False,
    )
    pd.DataFrame.from_records(proposed_cal_rows).to_csv(
        private_dir / "calibration_cells.csv",
        index=False,
    )
    pd.DataFrame.from_records(proposed_held_rows).to_csv(
        private_dir / "heldout_cells.csv",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if passed else "FAIL",
        "config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "phase2_cells_sha256": hashlib.sha256(
            cells_path.read_bytes()
        ).hexdigest(),
        "split": {
            "method": config["calibration_split"]["method"],
            "calibration_cases": list(calibration_cases),
            "heldout_cases": list(heldout_cases),
            "case_hashes": case_hashes,
            "calibration_cells": len(calibration_cells),
            "heldout_cells": len(heldout_cells),
        },
        "feature_contract": {
            "relation": "CALLS",
            "source": "sanitized model traces only",
            "profiles": PROFILE_FIELDS,
            "include_null_parent": bool(
                config["structural_evidence"].get(
                    "include_null_parent",
                    True,
                )
            ),
            "evaluator_labels_used_for_feature_extraction": False,
        },
        "feature_alignment": {
            "all_aligned": all_aligned,
            "boundary_mismatch_cells": sum(
                item["boundary_mismatch_count"] > 0
                for item in extraction_diagnostics
            ),
            "trace_mismatch_cells": sum(
                item["trace_mismatch_count"] > 0
                for item in extraction_diagnostics
            ),
        },
        "calibration_selection": {
            "status": selected_row["selection_status"],
            "selected_policy_feasible": calibration_feasible,
            "feasible_policy_count": sum(
                bool(row["feasible"]) for row in calibration_grid
            ),
            "searched_policy_count": len(calibration_grid),
            "selected_policy_row": selected_row,
            "heldout_labels_used_for_selection": False,
        },
        "selected_policy": asdict(policy),
        "matched_budget_control_policy": asdict(control_policy),
        "baseline": {
            "all": baseline_all,
            "calibration": baseline_cal,
            "heldout": baseline_held,
        },
        "proposed": {
            "all": proposed_all,
            "calibration": proposed_cal,
            "heldout": proposed_held,
            "heldout_delta_vs_a2": _delta(
                proposed_held,
                baseline_held,
            ),
        },
        "matched_budget_control": {
            "all": control_all,
            "heldout": control_held,
            "heldout_delta_proposed_minus_control": matched_delta,
        },
        "gate": {
            "gate_id": gate_cfg["gate_id"],
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "conditions": conditions,
            "reason_codes": [
                name.upper() for name, value in conditions.items() if not value
            ],
            "required": gate_cfg,
            "observed_selected_count_ratio": count_ratio,
        },
        "leakage_boundary": {
            "structural_features_before_any_evaluator_labels": True,
            "heldout_labels_loaded_after_policy_freeze": True,
            "model_output_separated_from_evaluator_private": True,
        },
        "claim_limit": (
            "Phase 3-R1 tests structural shortlisting of A2 runtime CALLS "
            "proposals. It does not establish causal-edge recovery or "
            "LLM/RCA improvement."
        ),
    }
    summary_text = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    (output / "summary.json").write_text(
        summary_text,
        encoding="utf-8",
    )
    report = _render_report(summary)
    (published_dir / "task_a_phase3_structural_results.md").write_text(
        report,
        encoding="utf-8",
    )
    (
        published_dir / "task_a_phase3_structural_results.json"
    ).write_text(summary_text, encoding="utf-8")
    pd.DataFrame.from_records(proposed_held_rows).to_csv(
        published_dir / "task_a_phase3_structural_heldout_cells.csv",
        index=False,
    )
    pd.DataFrame.from_records(calibration_grid).to_csv(
        published_dir / "task_a_phase3_structural_policy_grid.csv",
        index=False,
    )
    return output


__all__ = [
    "DEFAULT_STRUCTURAL_CONFIG",
    "EXPERIMENT_ID",
    "PROFILE_FIELDS",
    "StructuralPolicy",
    "apply_structural_policy",
    "attach_structural_evidence",
    "evaluate_structural_shortlist",
    "run_structural_phase3",
    "select_structural_policy",
    "validate_structural_config",
]
