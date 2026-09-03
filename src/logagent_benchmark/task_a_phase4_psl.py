"""Task A Phase 4: multi-evidence PSL validation over frozen A2 candidates.

The experiment consumes the leakage-controlled 1,250-candidate handoff produced
from RCAEval Phase 2/R2.  A2 remains the high-recall candidate generator.  PSL
combines independent structural and operational evidence without any DeBERTa
score and without hard-vetoing candidates.  Rule profiles and shortlist policy
are selected on calibration incidents only, then frozen for held-out incidents.
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

from .psl_multi_evidence import (
    CANDIDATE_KEY,
    CELL_KEY,
    FORBIDDEN_EVALUATOR_COLUMNS,
    MODEL_EVIDENCE_COLUMNS,
    PslMultiEvidenceBackendV1,
    PslRuleWeights,
)


SOURCE_CELL_KEY = ("incident_token", "seed", "mask_id")
EVALUATOR_COLUMNS = frozenset(
    {
        "case",
        "fault",
        "role",
        "is_masked_target",
        "is_silver_matched",
        "root_cause_service",
    }
)
REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        *SOURCE_CELL_KEY,
        "mask_ratio",
        *CANDIDATE_KEY,
        "a2_score",
        "a2_rank_normalized",
        "proposal_rank",
        "supporting_traces",
        "boundary_spans",
        "reverse_supporting_traces",
        "reverse_boundary_spans",
        "direct_evidence",
        "boundary_alignment",
        "direction_score",
        "operation_role_score",
        "operation_pair_concentration",
        "method_coverage",
        "method_match_rate",
        "route_coverage",
        "route_exact_rate",
        "route_jaccard_mean",
        "operation_jaccard_mean",
        "endpoint_compatibility_score",
        "graph_role_score",
        *EVALUATOR_COLUMNS.difference({"root_cause_service"}),
    }
)


class Phase4PslError(RuntimeError):
    """Raised when the Phase-4 PSL experiment contract is violated."""


@dataclass(frozen=True)
class ShortlistPolicy:
    profile_id: str
    retention_fraction: float
    minimum_keep: int

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        if not 0.0 < self.retention_fraction <= 1.0:
            raise ValueError("retention_fraction must be in (0,1]")
        if isinstance(self.minimum_keep, bool) or self.minimum_keep <= 0:
            raise ValueError("minimum_keep must be a positive integer")


@dataclass(frozen=True)
class GateConfig:
    recall_macro_min: float = 0.95
    recall_pooled_min: float = 0.95
    recall_each_cell_min: float = 0.90
    selected_count_ratio_max: float = 0.90
    matched_a2_recall_tolerance: float = 0.0
    matched_a2_p_lb_delta_min: float = 0.01
    matched_a2_mrr_delta_min: float = 0.0
    psl_score_std_min: float = 1e-6
    operation_evidence_coverage_min: float = 0.20
    ablation_gain_min: float = 1e-6
    permutation_gain_min: float = 1e-6


@dataclass(frozen=True)
class InferenceVariant:
    variant_id: str
    disabled_rules: tuple[str, ...] = ()
    permute_nonprior_evidence: bool = False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _finite(value)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cell_id(incident_token: Any, seed: Any, mask_id: Any) -> str:
    material = f"{incident_token}|{int(seed)}|{mask_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def validate_source_frame(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(frame.columns))
    if missing:
        raise Phase4PslError(f"candidate analysis is missing columns: {missing}")
    expected = config["source_contract"]
    if len(frame) != int(expected["candidate_rows"]):
        raise Phase4PslError(
            f"candidate row count differs: {len(frame)} != {expected['candidate_rows']}"
        )
    cells = frame.groupby(list(SOURCE_CELL_KEY), dropna=False).ngroups
    if cells != int(expected["candidate_cells"]):
        raise Phase4PslError(
            f"candidate cell count differs: {cells} != {expected['candidate_cells']}"
        )
    if frame["case"].astype(str).nunique() != int(expected["incidents"]):
        raise Phase4PslError("incident count differs from the frozen source contract")
    if set(frame["role"].astype(str)) != {"calibration", "heldout"}:
        raise Phase4PslError("source must contain calibration and heldout roles")
    calibration_cells = frame.loc[
        frame["role"].astype(str).eq("calibration")
    ].groupby(list(SOURCE_CELL_KEY), dropna=False).ngroups
    heldout_cells = frame.loc[
        frame["role"].astype(str).eq("heldout")
    ].groupby(list(SOURCE_CELL_KEY), dropna=False).ngroups
    if calibration_cells != int(expected["calibration_cells"]):
        raise Phase4PslError("calibration cell count differs from the contract")
    if heldout_cells != int(expected["heldout_cells"]):
        raise Phase4PslError("heldout cell count differs from the contract")
    duplicate = frame.duplicated([*SOURCE_CELL_KEY, *CANDIDATE_KEY], keep=False)
    if bool(duplicate.any()):
        raise Phase4PslError("candidate analysis contains duplicate candidate keys")
    if set(frame["predicate"].astype(str).str.upper()) != {"CALLS"}:
        raise Phase4PslError("Phase 4 PSL v1 accepts CALLS candidates only")


def _group_log_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    logged = values.map(math.log1p)
    maximum = float(logged.max()) if len(logged) else 0.0
    if maximum <= 0.0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return logged / maximum


def _weighted_available(
    frame: pd.DataFrame,
    components: Sequence[tuple[str, str | None, float]],
) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index, dtype=float)
    denominator = pd.Series(0.0, index=frame.index, dtype=float)
    for value_column, coverage_column, weight in components:
        value = pd.to_numeric(frame[value_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        coverage = (
            pd.to_numeric(frame[coverage_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            if coverage_column
            else pd.Series(1.0, index=frame.index, dtype=float)
        )
        effective = float(weight) * coverage
        numerator += value * effective
        denominator += effective
    return (numerator / denominator.where(denominator > 0.0, 1.0)).clip(0.0, 1.0)


def build_model_evidence(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build PSL truths before any evaluator label is attached."""

    model = source.drop(
        columns=[column for column in FORBIDDEN_EVALUATOR_COLUMNS if column in source.columns]
    ).copy()
    forbidden = FORBIDDEN_EVALUATOR_COLUMNS.intersection(model.columns)
    if forbidden:
        raise Phase4PslError(f"evaluator columns survived model split: {sorted(forbidden)}")

    model[CELL_KEY] = [
        _cell_id(row.incident_token, row.seed, row.mask_id)
        for row in model.itertuples(index=False)
    ]
    grouped = model.groupby(CELL_KEY, sort=False, dropna=False)
    model["trace_support"] = grouped["supporting_traces"].transform(_group_log_normalize)
    model["boundary_support"] = grouped["boundary_spans"].transform(_group_log_normalize)
    forward_trace = pd.to_numeric(model["supporting_traces"], errors="coerce").fillna(0.0).clip(lower=0.0)
    reverse_trace = pd.to_numeric(model["reverse_supporting_traces"], errors="coerce").fillna(0.0).clip(lower=0.0)
    forward_boundary = pd.to_numeric(model["boundary_spans"], errors="coerce").fillna(0.0).clip(lower=0.0)
    reverse_boundary = pd.to_numeric(model["reverse_boundary_spans"], errors="coerce").fillna(0.0).clip(lower=0.0)
    trace_total = forward_trace + reverse_trace
    boundary_total = forward_boundary + reverse_boundary
    trace_direction = (forward_trace / trace_total.where(trace_total > 0.0, 1.0)).where(trace_total > 0.0, 0.5)
    boundary_direction = (
        forward_boundary / boundary_total.where(boundary_total > 0.0, 1.0)
    ).where(boundary_total > 0.0, 0.5)
    alignment = pd.to_numeric(model["boundary_alignment"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    model["candidate"] = 1.0
    a2_score = pd.to_numeric(model["a2_score"], errors="raise").clip(0.0, 1.0)
    a2_rank = pd.to_numeric(model["a2_rank_normalized"], errors="raise").clip(0.0, 1.0)
    model["a2_prior"] = (0.60 * a2_score + 0.40 * a2_rank).clip(0.0, 1.0)
    model["repeated_support"] = (forward_trace / 3.0).clip(0.0, 1.0)
    explicit_direction = pd.to_numeric(model["direction_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    model["direction_support"] = (
        alignment * (0.45 * trace_direction + 0.35 * boundary_direction + 0.20 * explicit_direction)
    ).clip(0.0, 1.0)
    model["reverse_support"] = (
        0.60 * (reverse_trace / trace_total.where(trace_total > 0.0, 1.0)).where(trace_total > 0.0, 0.0)
        + 0.40
        * (reverse_boundary / boundary_total.where(boundary_total > 0.0, 1.0)).where(boundary_total > 0.0, 0.0)
    ).clip(0.0, 1.0)
    model["direction_conflict"] = (
        0.60
        * ((reverse_trace - forward_trace).clip(lower=0.0) / trace_total.where(trace_total > 0.0, 1.0))
        + 0.40
        * ((reverse_boundary - forward_boundary).clip(lower=0.0) / boundary_total.where(boundary_total > 0.0, 1.0))
    ).clip(0.0, 1.0)

    model["operation_match"] = _weighted_available(
        model,
        (
            ("operation_jaccard_mean", None, 0.30),
            ("operation_pair_concentration", None, 0.20),
            ("operation_role_score", None, 0.25),
            ("endpoint_compatibility_score", None, 0.25),
        ),
    )
    model["endpoint_match"] = _weighted_available(
        model,
        (
            ("method_match_rate", "method_coverage", 0.45),
            ("route_jaccard_mean", "route_coverage", 0.35),
            ("route_exact_rate", "route_coverage", 0.20),
        ),
    )
    model["role_compatibility"] = (
        0.60
        * pd.to_numeric(model["graph_role_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        + 0.40
        * pd.to_numeric(model["operation_role_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    model["direct_observed"] = model["direct_evidence"].map(_truthy).astype(float)
    model["self_loop"] = model["subject"].astype(str).eq(model["object"].astype(str)).astype(float)

    evidence = model[[CELL_KEY, *CANDIDATE_KEY, *MODEL_EVIDENCE_COLUMNS]].copy()
    for column in MODEL_EVIDENCE_COLUMNS:
        evidence[column] = pd.to_numeric(evidence[column], errors="raise").clip(0.0, 1.0)

    diagnostics = {
        "candidate_rows": len(evidence),
        "candidate_cells": int(evidence[CELL_KEY].nunique()),
        "model_evidence_columns": list(MODEL_EVIDENCE_COLUMNS),
        "evaluator_columns_removed_before_psl": True,
        "operation_evidence_coverage": float((evidence["operation_match"] > 0.0).mean()),
        "endpoint_evidence_coverage": float((evidence["endpoint_match"] > 0.0).mean()),
        "direction_evidence_coverage": float((evidence["direction_support"] > 0.0).mean()),
        "reverse_evidence_coverage": float((evidence["reverse_support"] > 0.0).mean()),
        "direct_observed_count": int((evidence["direct_observed"] > 0.0).sum()),
        "self_loop_count": int((evidence["self_loop"] > 0.0).sum()),
        "feature_std": {
            column: float(evidence[column].std(ddof=0)) for column in MODEL_EVIDENCE_COLUMNS
        },
    }
    return evidence, diagnostics


def _permute_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Break evidence-to-edge alignment while preserving each cell distribution."""

    output = frame.copy()
    permuted_columns = [
        "trace_support",
        "boundary_support",
        "repeated_support",
        "direction_support",
        "operation_match",
        "endpoint_match",
        "role_compatibility",
        "reverse_support",
        "direction_conflict",
    ]
    groups = []
    for _cell, group in output.groupby(CELL_KEY, sort=True, dropna=False):
        group = group.sort_values(["subject", "predicate", "object"], kind="mergesort").copy()
        if len(group) > 1:
            values = group[permuted_columns].to_numpy(copy=True)
            group.loc[:, permuted_columns] = pd.DataFrame(
                values[-1:].tolist() + values[:-1].tolist(),
                index=group.index,
                columns=permuted_columns,
            )
        groups.append(group)
    return pd.concat(groups, ignore_index=True) if groups else output


def _profile_weights(config: Mapping[str, Any]) -> dict[str, PslRuleWeights]:
    profiles = config["rule_profiles"]
    if len(profiles) < 1:
        raise Phase4PslError("at least one PSL rule profile is required")
    return {
        str(profile_id): PslRuleWeights.from_mapping(values)
        for profile_id, values in profiles.items()
    }


def _score_frame(
    evidence: pd.DataFrame,
    *,
    backend: PslMultiEvidenceBackendV1,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = backend.infer(evidence)
    records = [
        {
            CELL_KEY: cell,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            score_column: score,
        }
        for (cell, subject, predicate, obj), score in result.scores.items()
    ]
    frame = pd.DataFrame.from_records(records)
    expected = evidence[[CELL_KEY, *CANDIDATE_KEY]]
    if len(frame) != len(expected):
        raise Phase4PslError("PSL score frame changed candidate count")
    return frame, {
        "grounded_rule_count": result.grounded_rule_count,
        "grounded_atom_count": result.grounded_atom_count,
        "metadata": dict(result.metadata),
        "score_std": float(frame[score_column].std(ddof=0)),
        "score_min": float(frame[score_column].min()),
        "score_max": float(frame[score_column].max()),
        "unique_score_count": int(frame[score_column].round(12).nunique()),
    }


def _labels(source: pd.DataFrame) -> pd.DataFrame:
    columns = [*SOURCE_CELL_KEY, "mask_ratio", *CANDIDATE_KEY, *sorted(EVALUATOR_COLUMNS.intersection(source.columns))]
    labels = source[columns].copy()
    labels[CELL_KEY] = [
        _cell_id(row.incident_token, row.seed, row.mask_id)
        for row in labels.itertuples(index=False)
    ]
    return labels


def _analysis_frame(
    source: pd.DataFrame,
    evidence: pd.DataFrame,
    score_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    model_metadata = source.drop(
        columns=[column for column in EVALUATOR_COLUMNS if column in source.columns]
    )[[*SOURCE_CELL_KEY, "mask_ratio", *CANDIDATE_KEY, "a2_score", "a2_rank_normalized", "proposal_rank", "direct_evidence"]].copy()
    model_metadata[CELL_KEY] = [
        _cell_id(row.incident_token, row.seed, row.mask_id)
        for row in model_metadata.itertuples(index=False)
    ]
    merged = model_metadata.merge(
        evidence,
        on=[CELL_KEY, *CANDIDATE_KEY],
        how="inner",
        validate="one_to_one",
    )
    for frame in score_frames:
        merged = merged.merge(
            frame,
            on=[CELL_KEY, *CANDIDATE_KEY],
            how="inner",
            validate="one_to_one",
        )
    merged = merged.merge(
        _labels(source),
        on=[CELL_KEY, *SOURCE_CELL_KEY, "mask_ratio", *CANDIDATE_KEY],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(source):
        raise Phase4PslError("evaluator label join changed candidate count")
    return merged


def apply_shortlist(
    frame: pd.DataFrame,
    policy: ShortlistPolicy,
    *,
    score_column: str,
) -> pd.DataFrame:
    scored = frame.copy().reset_index(drop=True)
    keep = min(
        len(scored),
        max(int(policy.minimum_keep), int(math.ceil(policy.retention_fraction * len(scored)))),
    )
    scored["ranking_score"] = pd.to_numeric(scored[score_column], errors="raise")
    scored["selected"] = False
    direct = set(scored.index[scored["direct_evidence"].map(_truthy)])
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        ["ranking_score", "a2_score", "proposal_rank", "subject", "object"],
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


def _a2_control(frame: pd.DataFrame, selected_count: int) -> pd.DataFrame:
    scored = frame.copy().reset_index(drop=True)
    scored["ranking_score"] = pd.to_numeric(scored["a2_rank_normalized"], errors="raise")
    scored["selected"] = False
    direct = set(scored.index[scored["direct_evidence"].map(_truthy)])
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        ["a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    selected = set(direct)
    for index in ranked.index:
        if len(selected) >= int(selected_count):
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
        for row in selected[list(CANDIDATE_KEY)].itertuples(index=False, name=None)
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
        target_score = float(item.ranking_score)
        competitors = []
        for candidate in by_query.get((target[0], target[1]), ()):
            key = (str(candidate.subject), str(candidate.predicate), str(candidate.object))
            if key == target or key in silver_keys:
                continue
            competitors.append(candidate)
        higher = sum(float(candidate.ranking_score) > target_score + epsilon for candidate in competitors)
        tied = sum(abs(float(candidate.ranking_score) - target_score) <= epsilon for candidate in competitors)
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    p_lb = len(selected_keys & silver_keys) / len(selected_keys) if selected_keys else None
    return {
        "selected_count": len(selected_keys),
        "target_count": len(target_keys),
        "recovered_target_count": len(recovered),
        "recall": len(recovered) / len(target_keys) if target_keys else None,
        "silver_matched_count": len(selected_keys & silver_keys),
        "silver_precision_lower_bound": p_lb,
        "false_edge_rate_upper_bound": 1.0 - p_lb if p_lb is not None else None,
        "mrr": statistics.fmean(reciprocal) if reciprocal else None,
        "ranks": json.dumps(ranks, separators=(",", ":")),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    recalls = [float(row["recall"]) for row in rows if row.get("recall") is not None]
    p_lbs = [float(row["silver_precision_lower_bound"]) for row in rows if row.get("silver_precision_lower_bound") is not None]
    mrrs = [float(row["mrr"]) for row in rows if row.get("mrr") is not None]
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
        "false_edge_rate_upper_bound_macro": 1.0 - statistics.fmean(p_lbs),
        "mrr_macro": statistics.fmean(mrrs),
        "mrr_min": min(mrrs),
    }


def _delta(enhanced: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    fields = (
        "recall_macro",
        "selected_count_mean",
        "silver_precision_lower_bound_macro",
        "false_edge_rate_upper_bound_macro",
        "mrr_macro",
    )
    return {field: float(enhanced[field]) - float(baseline[field]) for field in fields}


def _baseline(cells: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    rows = []
    for group in cells.values():
        pseudo = group.copy()
        pseudo["selected"] = True
        pseudo["ranking_score"] = pseudo["a2_rank_normalized"].astype(float)
        rows.append(evaluate_cell(pseudo))
    return _aggregate(rows)


def _evaluate_policy(
    cells: Mapping[str, pd.DataFrame],
    policy: ShortlistPolicy,
    *,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    for cell_id, group in sorted(cells.items()):
        scored = apply_shortlist(group, policy, score_column=score_column)
        metric = evaluate_cell(scored)
        control = _a2_control(group, metric["selected_count"])
        control_metric = evaluate_cell(control)
        first = group.iloc[0]
        common = {
            CELL_KEY: cell_id,
            "case": str(first["case"]),
            "fault": str(first["fault"]),
            "role": str(first["role"]),
            "seed": int(first["seed"]),
            "mask_id": str(first["mask_id"]),
            "mask_ratio": float(first["mask_ratio"]),
        }
        rows.append({**common, **metric})
        control_rows.append({**common, **control_metric})
        scored = scored.copy()
        for name, value in common.items():
            scored[name] = value
        candidate_frames.append(scored)
    result_rows = pd.DataFrame.from_records(rows)
    control_frame = pd.DataFrame.from_records(control_rows)
    return (
        result_rows,
        _aggregate(rows),
        control_frame,
        _aggregate(control_rows),
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
    )


def _cells(frame: pd.DataFrame, role: str) -> dict[str, pd.DataFrame]:
    selected = frame.loc[frame["role"].astype(str).eq(role)].copy()
    return {
        str(cell): group.reset_index(drop=True)
        for cell, group in selected.groupby(CELL_KEY, sort=True, dropna=False)
    }


def _shortlist_grid(config: Mapping[str, Any], profile_ids: Iterable[str]) -> list[ShortlistPolicy]:
    search = config["shortlist_policy_search"]
    return [
        ShortlistPolicy(str(profile_id), float(retention), int(minimum_keep))
        for profile_id in sorted(profile_ids)
        for retention in search["retention_fractions"]
        for minimum_keep in search["minimum_keep"]
    ]


def _policy_conditions(
    proposed: Mapping[str, Any],
    control: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gate: GateConfig,
) -> dict[str, bool]:
    ratio = float(proposed["selected_count_mean"]) / float(baseline["selected_count_mean"])
    delta_control = _delta(proposed, control)
    return {
        "recall_macro": float(proposed["recall_macro"]) >= gate.recall_macro_min,
        "recall_pooled": float(proposed["recall_pooled"]) >= gate.recall_pooled_min,
        "recall_each_cell": float(proposed["recall_min"]) >= gate.recall_each_cell_min,
        "candidate_count_reduced": ratio <= gate.selected_count_ratio_max,
        "matched_a2_recall_noninferior": delta_control["recall_macro"] >= -gate.matched_a2_recall_tolerance,
        "matched_a2_p_lb_improved": delta_control["silver_precision_lower_bound_macro"] >= gate.matched_a2_p_lb_delta_min,
        "matched_a2_mrr_noninferior": delta_control["mrr_macro"] >= gate.matched_a2_mrr_delta_min,
    }


def select_policy(
    calibration_cells: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    profile_ids: Iterable[str],
    gate: GateConfig,
) -> tuple[ShortlistPolicy, pd.DataFrame, bool]:
    baseline = _baseline(calibration_cells)
    rows = []
    policies: dict[tuple[str, float, int], ShortlistPolicy] = {}
    for policy in _shortlist_grid(config, profile_ids):
        score_column = f"psl_score__{policy.profile_id}"
        _cell_rows, proposed, _control_rows, control, _candidates = _evaluate_policy(
            calibration_cells, policy, score_column=score_column
        )
        conditions = _policy_conditions(proposed, control, baseline, gate)
        feasible = all(conditions.values())
        delta_control = _delta(proposed, control)
        key = (policy.profile_id, policy.retention_fraction, policy.minimum_keep)
        policies[key] = policy
        rows.append(
            {
                **asdict(policy),
                "feasible": feasible,
                "condition_pass_count": sum(conditions.values()),
                "condition_count": len(conditions),
                **{f"condition__{name}": value for name, value in conditions.items()},
                "recall_macro": proposed["recall_macro"],
                "recall_min": proposed["recall_min"],
                "recall_pooled": proposed["recall_pooled"],
                "selected_count_mean": proposed["selected_count_mean"],
                "selected_count_ratio": proposed["selected_count_mean"] / baseline["selected_count_mean"],
                "p_lb_macro": proposed["silver_precision_lower_bound_macro"],
                "mrr_macro": proposed["mrr_macro"],
                "matched_a2_recall_delta": delta_control["recall_macro"],
                "matched_a2_p_lb_delta": delta_control["silver_precision_lower_bound_macro"],
                "matched_a2_mrr_delta": delta_control["mrr_macro"],
            }
        )
    grid = pd.DataFrame.from_records(rows)
    feasible_rows = grid.loc[grid["feasible"].map(_truthy)].copy()
    if not feasible_rows.empty:
        ordered = feasible_rows.sort_values(
            [
                "matched_a2_p_lb_delta",
                "matched_a2_mrr_delta",
                "recall_min",
                "recall_macro",
                "selected_count_mean",
                "profile_id",
                "retention_fraction",
                "minimum_keep",
            ],
            ascending=[False, False, False, False, True, True, True, True],
            kind="mergesort",
        )
        feasible = True
    else:
        ordered = grid.sort_values(
            [
                "condition_pass_count",
                "recall_min",
                "recall_macro",
                "matched_a2_p_lb_delta",
                "matched_a2_mrr_delta",
                "selected_count_mean",
                "profile_id",
                "retention_fraction",
                "minimum_keep",
            ],
            ascending=[False, False, False, False, False, True, True, True, True],
            kind="mergesort",
        )
        feasible = False
    chosen = ordered.iloc[0]
    key = (str(chosen["profile_id"]), float(chosen["retention_fraction"]), int(chosen["minimum_keep"]))
    return policies[key], grid, feasible


def _gate_config(config: Mapping[str, Any]) -> GateConfig:
    return GateConfig(**{name: float(value) for name, value in config["gate"].items()})


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    held = summary["heldout"]
    proposed = held["proposed_psl"]
    baseline = held["baseline_a2_full"]
    control = held["equal_size_a2_control"]
    delta = held["delta_vs_equal_size_a2"]
    lines = [
        "# Task A Phase 4 — Multi-evidence PSL v1",
        "",
        f"- Scientific status: **{summary['status']}**",
        f"- Gate: `{summary['gate_id']}`",
        f"- Selected policy: `{json.dumps(summary['selected_policy'], sort_keys=True)}`",
        f"- Gate reasons: `{', '.join(summary['gate']['reason_codes']) or 'none'}`",
        "",
        "## Held-out result",
        "",
        "| Metric | A2 full | Equal-size A2 | PSL v1 | PSL vs equal-size A2 |",
        "|---|---:|---:|---:|---:|",
        f"| Recall macro | {baseline['recall_macro']:.6f} | {control['recall_macro']:.6f} | {proposed['recall_macro']:.6f} | {delta['recall_macro']:+.6f} |",
        f"| Recall minimum | {baseline['recall_min']:.6f} | {control['recall_min']:.6f} | {proposed['recall_min']:.6f} | - |",
        f"| Recall pooled | {baseline['recall_pooled']:.6f} | {control['recall_pooled']:.6f} | {proposed['recall_pooled']:.6f} | - |",
        f"| Mean candidates | {baseline['selected_count_mean']:.3f} | {control['selected_count_mean']:.3f} | {proposed['selected_count_mean']:.3f} | {delta['selected_count_mean']:+.3f} |",
        f"| P-LB macro | {baseline['silver_precision_lower_bound_macro']:.6f} | {control['silver_precision_lower_bound_macro']:.6f} | {proposed['silver_precision_lower_bound_macro']:.6f} | {delta['silver_precision_lower_bound_macro']:+.6f} |",
        f"| MRR macro | {baseline['mrr_macro']:.6f} | {control['mrr_macro']:.6f} | {proposed['mrr_macro']:.6f} | {delta['mrr_macro']:+.6f} |",
        "",
        "## Rule ablation",
        "",
        "| Variant | Recall | P-LB | MRR | Mean candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, values in summary["ablations"].items():
        lines.append(
            f"| {variant} | {values['recall_macro']:.6f} | "
            f"{values['silver_precision_lower_bound_macro']:.6f} | "
            f"{values['mrr_macro']:.6f} | {values['selected_count_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            summary["claim_limit"],
            "",
            "PSL output is a probability-ranked runtime `CALLS` hypothesis set. "
            "It is not a causal `CAUSES` graph and does not establish RCA/LLM improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase4_psl(
    *,
    candidate_analysis_path: Path,
    config_path: Path,
    output: Path,
) -> Path:
    candidate_analysis_path = candidate_analysis_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise Phase4PslError(f"refusing to overwrite existing output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise Phase4PslError("PSL v1 config schema_version must be 1")
    source_sha = _sha256_file(candidate_analysis_path)
    expected_sha = str(config["source_contract"].get("candidate_analysis_sha256", ""))
    if expected_sha and source_sha != expected_sha:
        raise Phase4PslError(
            f"candidate analysis SHA-256 mismatch: {source_sha} != {expected_sha}"
        )
    source = pd.read_parquet(candidate_analysis_path)
    validate_source_frame(source, config)
    output.mkdir(parents=True, exist_ok=False)
    published = output / "published"
    model_output = output / "model_output"
    evaluator_private = output / "evaluator_private"
    for directory in (published, model_output, evaluator_private):
        directory.mkdir(parents=True, exist_ok=True)

    evidence, evidence_diagnostics = build_model_evidence(source)
    evidence.to_parquet(model_output / "psl_evidence.parquet", index=False)

    profiles = _profile_weights(config)
    score_frames: list[pd.DataFrame] = []
    profile_diagnostics: dict[str, Any] = {}
    random_seed = int(config["psl_runtime"]["random_seed"])
    jvm_options = tuple(str(value) for value in config["psl_runtime"]["jvm_options"])
    for profile_id, weights in profiles.items():
        backend = PslMultiEvidenceBackendV1(
            weights=weights,
            profile_id=profile_id,
            random_seed=random_seed,
            jvm_options=jvm_options,
        )
        score_column = f"psl_score__{profile_id}"
        score_frame, diagnostics = _score_frame(
            evidence, backend=backend, score_column=score_column
        )
        score_frames.append(score_frame)
        profile_diagnostics[profile_id] = diagnostics
        print(
            f"PSL profile={profile_id} candidates={len(score_frame)} "
            f"score_std={diagnostics['score_std']:.6f}",
            flush=True,
        )

    analysis = _analysis_frame(source, evidence, score_frames)
    analysis.to_parquet(evaluator_private / "candidate_analysis.parquet", index=False)
    calibration_cells = _cells(analysis, "calibration")
    heldout_cells = _cells(analysis, "heldout")
    gate_config = _gate_config(config)
    selected_policy, policy_grid, calibration_feasible = select_policy(
        calibration_cells, config, profiles, gate_config
    )
    policy_grid.to_csv(published / "task_a_phase4_psl_policy_grid.csv", index=False)
    selected_score_column = f"psl_score__{selected_policy.profile_id}"

    calibration_rows, calibration_summary, calibration_control_rows, calibration_control, _ = _evaluate_policy(
        calibration_cells,
        selected_policy,
        score_column=selected_score_column,
    )
    heldout_rows, heldout_summary, heldout_control_rows, heldout_control, selected_candidates = _evaluate_policy(
        heldout_cells,
        selected_policy,
        score_column=selected_score_column,
    )
    calibration_rows.to_csv(
        published / "task_a_phase4_psl_calibration_cells.csv", index=False
    )
    heldout_rows.to_csv(
        published / "task_a_phase4_psl_heldout_cells.csv", index=False
    )
    calibration_control_rows.to_csv(
        evaluator_private / "calibration_equal_size_a2.csv", index=False
    )
    heldout_control_rows.to_csv(
        evaluator_private / "heldout_equal_size_a2.csv", index=False
    )
    selected_candidates.to_parquet(
        model_output / "psl_selected_candidates.parquet", index=False
    )

    selected_weights = profiles[selected_policy.profile_id]
    variants = (
        InferenceVariant(
            "prior_only",
            (
                "TRACE_SUPPORT",
                "BOUNDARY_SUPPORT",
                "REPEATED_SUPPORT",
                "DIRECTION_SUPPORT",
                "OPERATION_MATCH",
                "ENDPOINT_MATCH",
                "ROLE_COMPATIBILITY",
                "REVERSE_SUPPORT_NEGATIVE",
                "DIRECTION_CONFLICT_NEGATIVE",
            ),
        ),
        InferenceVariant(
            "no_negative",
            ("REVERSE_SUPPORT_NEGATIVE", "DIRECTION_CONFLICT_NEGATIVE"),
        ),
        InferenceVariant(
            "no_operation",
            ("OPERATION_MATCH", "ENDPOINT_MATCH"),
        ),
        InferenceVariant(
            "no_structure",
            (
                "TRACE_SUPPORT",
                "BOUNDARY_SUPPORT",
                "REPEATED_SUPPORT",
                "DIRECTION_SUPPORT",
                "ROLE_COMPATIBILITY",
                "REVERSE_SUPPORT_NEGATIVE",
                "DIRECTION_CONFLICT_NEGATIVE",
            ),
        ),
        InferenceVariant("permuted_evidence", (), True),
    )
    ablation_summaries: dict[str, Any] = {"full": heldout_summary}
    ablation_diagnostics: dict[str, Any] = {}
    ablation_rows = []
    for variant in variants:
        variant_evidence = _permute_evidence(evidence) if variant.permute_nonprior_evidence else evidence
        backend = PslMultiEvidenceBackendV1(
            weights=selected_weights,
            profile_id=f"{selected_policy.profile_id}-{variant.variant_id}",
            disabled_rules=variant.disabled_rules,
            random_seed=random_seed,
            jvm_options=jvm_options,
        )
        score_column = f"psl_score__{variant.variant_id}"
        score_frame, diagnostics = _score_frame(
            variant_evidence, backend=backend, score_column=score_column
        )
        variant_analysis = _analysis_frame(source, evidence, [score_frame])
        variant_cells = _cells(variant_analysis, "heldout")
        _rows, aggregate, _controls, _control_aggregate, _candidate_output = _evaluate_policy(
            variant_cells,
            selected_policy,
            score_column=score_column,
        )
        ablation_summaries[variant.variant_id] = aggregate
        ablation_diagnostics[variant.variant_id] = diagnostics
        ablation_rows.append({"variant": variant.variant_id, **aggregate})
    pd.DataFrame.from_records(ablation_rows).to_csv(
        published / "task_a_phase4_psl_ablation_results.csv", index=False
    )

    baseline_calibration = _baseline(calibration_cells)
    baseline_heldout = _baseline(heldout_cells)
    calibration_delta = _delta(calibration_summary, calibration_control)
    heldout_delta = _delta(heldout_summary, heldout_control)
    selected_ratio = heldout_summary["selected_count_mean"] / baseline_heldout["selected_count_mean"]
    full_vs_prior = _delta(heldout_summary, ablation_summaries["prior_only"])
    full_vs_permuted = _delta(heldout_summary, ablation_summaries["permuted_evidence"])
    selected_score_std = float(profile_diagnostics[selected_policy.profile_id]["score_std"])

    conditions = {
        "calibration_policy_feasible": calibration_feasible,
        "candidate_handoff_complete": len(source) == int(config["source_contract"]["candidate_rows"]),
        "psl_score_has_variance": selected_score_std >= gate_config.psl_score_std_min,
        "operation_evidence_coverage": evidence_diagnostics["operation_evidence_coverage"] >= gate_config.operation_evidence_coverage_min,
        "recall_macro": heldout_summary["recall_macro"] >= gate_config.recall_macro_min,
        "recall_pooled": heldout_summary["recall_pooled"] >= gate_config.recall_pooled_min,
        "recall_each_cell": heldout_summary["recall_min"] >= gate_config.recall_each_cell_min,
        "candidate_count_reduced": selected_ratio <= gate_config.selected_count_ratio_max,
        "matched_a2_recall_noninferior": heldout_delta["recall_macro"] >= -gate_config.matched_a2_recall_tolerance,
        "matched_a2_p_lb_improved": heldout_delta["silver_precision_lower_bound_macro"] >= gate_config.matched_a2_p_lb_delta_min,
        "matched_a2_mrr_noninferior": heldout_delta["mrr_macro"] >= gate_config.matched_a2_mrr_delta_min,
        "rule_ablation_has_effect": max(
            full_vs_prior["silver_precision_lower_bound_macro"],
            full_vs_prior["mrr_macro"],
            full_vs_prior["recall_macro"],
        ) >= gate_config.ablation_gain_min,
        "permuted_evidence_is_worse": max(
            full_vs_permuted["silver_precision_lower_bound_macro"],
            full_vs_permuted["mrr_macro"],
            full_vs_permuted["recall_macro"],
        ) >= gate_config.permutation_gain_min,
        "evaluator_labels_joined_after_psl": True,
        "deberta_not_used": True,
    }
    passed = all(conditions.values())
    reason_codes = [name.upper() for name, value in conditions.items() if not value]
    gate = {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "reason_codes": reason_codes,
        "conditions": conditions,
        "required": asdict(gate_config),
    }

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": gate["status"],
        "gate_id": config["gate_id"],
        "gate": gate,
        "source": {
            "candidate_analysis": str(candidate_analysis_path),
            "candidate_analysis_sha256": source_sha,
            "candidate_rows": len(source),
            "candidate_cells": int(source.groupby(list(SOURCE_CELL_KEY)).ngroups),
            "incidents": int(source["case"].nunique()),
            "calibration_cells": len(calibration_cells),
            "heldout_cells": len(heldout_cells),
        },
        "selected_policy": asdict(selected_policy),
        "selected_rule_weights": asdict(selected_weights),
        "calibration": {
            "feasible": calibration_feasible,
            "searched_policy_count": len(policy_grid),
            "feasible_policy_count": int(policy_grid["feasible"].map(_truthy).sum()),
            "baseline_a2_full": baseline_calibration,
            "equal_size_a2_control": calibration_control,
            "proposed_psl": calibration_summary,
            "delta_vs_equal_size_a2": calibration_delta,
        },
        "heldout": {
            "baseline_a2_full": baseline_heldout,
            "equal_size_a2_control": heldout_control,
            "proposed_psl": heldout_summary,
            "delta_vs_equal_size_a2": heldout_delta,
            "selected_count_ratio": selected_ratio,
        },
        "ablations": ablation_summaries,
        "ablation_deltas": {
            "full_vs_prior_only": full_vs_prior,
            "full_vs_permuted_evidence": full_vs_permuted,
        },
        "psl_profiles": profile_diagnostics,
        "psl_ablation_diagnostics": ablation_diagnostics,
        "evidence_diagnostics": evidence_diagnostics,
        "leakage_boundary": {
            "evaluator_columns_removed_before_feature_build": True,
            "evaluator_columns_removed_before_psl": True,
            "evaluator_labels_joined_after_psl_score_freeze": True,
            "fault_or_root_label_used_for_scoring": False,
            "mask_or_silver_label_used_for_scoring": False,
            "deberta_score_used": False,
        },
        "protocol_status": {
            "phase": "DEVELOPMENT_PSL_V1",
            "confirmatory_holdout_required": True,
            "reason": (
                "The six RCAEval incidents were previously inspected. This experiment "
                "can validate the PSL mechanism and held-out incident split but cannot "
                "serve as final cross-system confirmation."
            ),
        },
        "claim_limit": config["claim_limit"],
    }
    result_json = published / "task_a_phase4_psl_results.json"
    result_md = published / "task_a_phase4_psl_results.md"
    _write_json(result_json, summary)
    _write_markdown(result_md, summary)
    (published / "task_a_phase4_psl_status.txt").write_text(gate["status"] + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256_file(config_path),
        "candidate_analysis_sha256": source_sha,
        "output_files": {},
    }
    for path in sorted(published.iterdir()):
        if path.is_file() and path.name != "task_a_phase4_psl_manifest.json":
            manifest["output_files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _write_json(published / "task_a_phase4_psl_manifest.json", manifest)
    print(f"Task A Phase 4 PSL gate: {gate['status']} -> {output}", flush=True)
    return output


__all__ = [
    "EVALUATOR_COLUMNS",
    "GateConfig",
    "InferenceVariant",
    "Phase4PslError",
    "REQUIRED_SOURCE_COLUMNS",
    "ShortlistPolicy",
    "apply_shortlist",
    "build_model_evidence",
    "evaluate_cell",
    "run_phase4_psl",
    "select_policy",
    "validate_source_frame",
]
