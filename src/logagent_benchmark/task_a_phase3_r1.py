"""Task A Phase 3-R1: leakage-safe structured evidence reranking.

This stage tests the highest-priority A3 failure hypothesis before another NLI
experiment: model-visible directional evidence must first create candidate-level
discrimination beyond the frozen A2 order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import pandas as pd


MODEL_KEY_COLUMNS = (
    "incident_id",
    "seed",
    "mask_id",
    "subject",
    "predicate",
    "object",
)
EVALUATOR_COLUMNS = {"case", "fault", "role", "is_masked_target", "is_silver_matched"}
REQUIRED_COLUMNS = {
    *MODEL_KEY_COLUMNS,
    "mask_ratio",
    "a2_score",
    "proposal_rank",
    "supporting_traces",
    "boundary_spans",
    "reverse_supporting_traces",
    "reverse_boundary_spans",
    "direct_evidence",
    "is_masked_target",
    "is_silver_matched",
    "role",
}


class Phase3R1Error(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredPolicy:
    retention_fraction: float
    minimum_keep: int
    structure_weight: float
    direction_weight: float
    density_weight: float

    def __post_init__(self) -> None:
        if not 0.0 < self.retention_fraction <= 1.0:
            raise ValueError("retention_fraction must be in (0, 1]")
        if self.minimum_keep < 1:
            raise ValueError("minimum_keep must be positive")
        if not 0.0 <= self.structure_weight <= 1.0:
            raise ValueError("structure_weight must be in [0, 1]")
        if self.direction_weight < 0.0 or self.density_weight < 0.0:
            raise ValueError("feature weights must be non-negative")
        if self.direction_weight + self.density_weight <= 0.0:
            raise ValueError("at least one feature weight must be positive")


@dataclass(frozen=True)
class GateConfig:
    recall_macro_min: float = 0.95
    recall_pooled_min: float = 0.95
    recall_each_cell_min: float = 0.90
    selected_count_ratio_max: float = 0.95
    matched_budget_recall_tolerance: float = 0.0
    matched_budget_p_lb_delta_min: float = 0.0
    matched_budget_mrr_delta_min: float = 0.0
    additive_gain_min: float = 1e-12


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _rank_normalize(
    values: Sequence[float], tie_breakers: Sequence[tuple[Any, ...]]
) -> list[float]:
    if len(values) != len(tie_breakers):
        raise ValueError("values and tie_breakers must have the same length")
    if not values:
        return []
    order = sorted(
        range(len(values)),
        key=lambda index: (-float(values[index]), tie_breakers[index]),
    )
    denominator = max(1, len(values) - 1)
    output = [0.0] * len(values)
    for rank, index in enumerate(order):
        output[index] = 1.0 - rank / denominator
    return output


def add_structured_features(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute CALLS features from model-visible A2 evidence only."""

    forbidden = EVALUATOR_COLUMNS.intersection(model_frame.columns)
    if forbidden:
        raise Phase3R1Error(
            "model scoring received evaluator columns: "
            + ", ".join(sorted(forbidden))
        )
    required = REQUIRED_COLUMNS.difference(EVALUATOR_COLUMNS)
    missing = sorted(required.difference(model_frame.columns))
    if missing:
        raise Phase3R1Error(f"model frame missing columns: {missing}")

    frame = model_frame.copy()
    numeric = (
        "a2_score",
        "proposal_rank",
        "supporting_traces",
        "boundary_spans",
        "reverse_supporting_traces",
        "reverse_boundary_spans",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    count_columns = [
        "supporting_traces",
        "boundary_spans",
        "reverse_supporting_traces",
        "reverse_boundary_spans",
    ]
    if (frame[count_columns] < 0).any().any():
        raise Phase3R1Error("support counts must be non-negative")

    forward_trace = frame["supporting_traces"].astype(float)
    reverse_trace = frame["reverse_supporting_traces"].astype(float)
    forward_boundary = frame["boundary_spans"].astype(float)
    reverse_boundary = frame["reverse_boundary_spans"].astype(float)

    frame["trace_direction_margin"] = (
        forward_trace - reverse_trace
    ) / (forward_trace + reverse_trace + 1.0)
    frame["boundary_direction_margin"] = (
        forward_boundary - reverse_boundary
    ) / (forward_boundary + reverse_boundary + 1.0)
    frame["trace_direction_ratio"] = (forward_trace + 1.0) / (
        forward_trace + reverse_trace + 2.0
    )
    frame["boundary_direction_ratio"] = (forward_boundary + 1.0) / (
        forward_boundary + reverse_boundary + 2.0
    )
    frame["forward_support_density"] = (forward_trace + 1.0) / (
        forward_boundary + 1.0
    )
    frame["reverse_support_density"] = (reverse_trace + 1.0) / (
        reverse_boundary + 1.0
    )
    frame["density_margin"] = (
        frame["forward_support_density"] - frame["reverse_support_density"]
    )
    frame["self_loop"] = (
        frame["subject"].astype(str) == frame["object"].astype(str)
    ).astype(int)

    outputs: list[pd.DataFrame] = []
    for _key, group in frame.groupby(
        ["incident_id", "seed", "mask_id"], sort=True, dropna=False
    ):
        group = group.copy().reset_index(drop=True)
        tie_breakers = [
            (
                int(group.loc[index, "proposal_rank"]),
                str(group.loc[index, "subject"]),
                str(group.loc[index, "object"]),
            )
            for index in range(len(group))
        ]
        a2_values = [float(value) for value in group["a2_score"]]
        direction_values = [
            0.55 * float(group.loc[index, "trace_direction_margin"])
            + 0.45 * float(group.loc[index, "boundary_direction_margin"])
            - float(group.loc[index, "self_loop"])
            for index in range(len(group))
        ]
        density_values = [
            math.log1p(
                max(0.0, float(group.loc[index, "forward_support_density"]))
            )
            - math.log1p(
                max(0.0, float(group.loc[index, "reverse_support_density"]))
            )
            for index in range(len(group))
        ]
        group["a2_rank_normalized"] = _rank_normalize(
            a2_values, tie_breakers
        )
        group["direction_rank_normalized"] = _rank_normalize(
            direction_values, tie_breakers
        )
        group["density_rank_normalized"] = _rank_normalize(
            density_values, tie_breakers
        )
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True) if outputs else frame


def apply_structured_policy(
    feature_frame: pd.DataFrame, policy: StructuredPolicy
) -> pd.DataFrame:
    if feature_frame.empty:
        return feature_frame.copy()
    frame = feature_frame.copy()
    denominator = policy.direction_weight + policy.density_weight
    structured_score = (
        policy.direction_weight
        * frame["direction_rank_normalized"].astype(float)
        + policy.density_weight
        * frame["density_rank_normalized"].astype(float)
    ) / denominator
    frame["structured_evidence_score"] = structured_score
    frame["a3_r1_score"] = (
        (1.0 - policy.structure_weight)
        * frame["a2_rank_normalized"].astype(float)
        + policy.structure_weight * structured_score
    )

    keep = min(
        len(frame),
        max(
            policy.minimum_keep,
            int(math.ceil(policy.retention_fraction * len(frame))),
        ),
    )
    frame["selected"] = False
    direct_mask = frame["direct_evidence"].map(_truthy)
    selected = set(frame.index[direct_mask])
    ranked = frame.loc[~direct_mask].sort_values(
        ["a3_r1_score", "a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    for index in ranked.index:
        if len(selected) >= keep:
            break
        selected.add(index)
    frame.loc[list(selected), "selected"] = True
    return frame


def _evaluate_cell(scored: pd.DataFrame) -> dict[str, Any]:
    selected = scored.loc[scored["selected"].map(_truthy)]
    target_keys = {
        (str(row.subject), str(row.predicate), str(row.object))
        for row in scored.loc[
            scored["is_masked_target"].map(_truthy)
        ].itertuples(index=False)
    }
    silver_keys = {
        (str(row.subject), str(row.predicate), str(row.object))
        for row in scored.loc[
            scored["is_silver_matched"].map(_truthy)
        ].itertuples(index=False)
    }
    selected_keys = {
        (str(row.subject), str(row.predicate), str(row.object))
        for row in selected.itertuples(index=False)
    }
    recovered = selected_keys & target_keys
    recall = len(recovered) / len(target_keys) if target_keys else None
    p_lb = (
        len(selected_keys & silver_keys) / len(selected_keys)
        if selected_keys
        else None
    )

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
        target_score = float(target_item.a3_r1_score)
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
            float(candidate.a3_r1_score) > target_score + epsilon
            for candidate in competitors
        )
        tied = sum(
            abs(float(candidate.a3_r1_score) - target_score) <= epsilon
            for candidate in competitors
        )
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    return {
        "selected_count": len(selected_keys),
        "target_count": len(target_keys),
        "recovered_target_count": len(recovered),
        "recall": recall,
        "silver_matched_count": len(selected_keys & silver_keys),
        "silver_precision_lower_bound": p_lb,
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
    target_count = sum(int(row["target_count"]) for row in rows)
    recovered_count = sum(int(row["recovered_target_count"]) for row in rows)
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


def _evaluate_policy(
    frame: pd.DataFrame, policy: StructuredPolicy
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    cell_rows: list[dict[str, Any]] = []
    scored_groups: list[pd.DataFrame] = []
    for key, group in frame.groupby(
        ["incident_id", "seed", "mask_id"], sort=True, dropna=False
    ):
        scored = apply_structured_policy(group.reset_index(drop=True), policy)
        metric = _evaluate_cell(scored)
        first = scored.iloc[0]
        cell_rows.append(
            {
                "incident_id": str(key[0]),
                "role": str(first["role"]),
                "fault": str(first.get("fault", "")),
                "seed": int(key[1]),
                "mask_id": str(key[2]),
                "mask_ratio": float(first["mask_ratio"]),
                **metric,
            }
        )
        scored_groups.append(scored)
    rows = pd.DataFrame.from_records(cell_rows)
    scored_all = (
        pd.concat(scored_groups, ignore_index=True)
        if scored_groups
        else pd.DataFrame()
    )
    return rows, _aggregate(cell_rows), scored_all


def _baseline_aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _key, group in frame.groupby(
        ["incident_id", "seed", "mask_id"], sort=True, dropna=False
    ):
        target_count = int(group["is_masked_target"].map(_truthy).sum())
        silver_count = int(group["is_silver_matched"].map(_truthy).sum())
        pseudo = group.copy()
        pseudo["selected"] = True
        pseudo["a3_r1_score"] = -pd.to_numeric(
            pseudo["proposal_rank"], errors="raise"
        ).astype(float)
        metric = _evaluate_cell(pseudo)
        rows.append(
            {
                "selected_count": len(group),
                "target_count": target_count,
                "recovered_target_count": target_count,
                "recall": 1.0 if target_count else None,
                "silver_precision_lower_bound": (
                    silver_count / len(group) if len(group) else None
                ),
                "mrr": metric["mrr"],
            }
        )
    return _aggregate(rows)


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


def _policy_grid(config: Mapping[str, Any]) -> list[StructuredPolicy]:
    search = config["policy_search"]
    return [
        StructuredPolicy(
            float(retention),
            int(minimum_keep),
            float(structure_weight),
            float(direction_weight),
            float(density_weight),
        )
        for retention in search["retention_fractions"]
        for minimum_keep in search["minimum_keep"]
        for structure_weight in search["structure_weights"]
        for direction_weight, density_weight in search["feature_weight_pairs"]
    ]


def _select_policy(
    calibration: pd.DataFrame,
    policies: Sequence[StructuredPolicy],
    gate: GateConfig,
) -> tuple[StructuredPolicy, pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    baseline = _baseline_aggregate(calibration)
    for policy in policies:
        _cell_rows, aggregate, _scored = _evaluate_policy(calibration, policy)
        control_policy = StructuredPolicy(
            policy.retention_fraction,
            policy.minimum_keep,
            0.0,
            policy.direction_weight,
            policy.density_weight,
        )
        _control_rows, control, _ = _evaluate_policy(
            calibration, control_policy
        )
        delta = _delta(aggregate, control)
        additive_gain = max(
            delta["silver_precision_lower_bound_macro"], delta["mrr_macro"]
        )
        conditions = {
            "recall_macro": aggregate["recall_macro"] >= gate.recall_macro_min,
            "recall_each_cell": aggregate["recall_min"]
            >= gate.recall_each_cell_min,
            "mrr_noninferior_to_full_a2": aggregate["mrr_macro"]
            >= baseline["mrr_macro"] - 0.01,
            "matched_budget_recall": delta["recall_macro"]
            >= -gate.matched_budget_recall_tolerance,
            "matched_budget_p_lb": delta[
                "silver_precision_lower_bound_macro"
            ]
            >= gate.matched_budget_p_lb_delta_min,
            "matched_budget_mrr": delta["mrr_macro"]
            >= gate.matched_budget_mrr_delta_min,
            "matched_budget_additive_gain": additive_gain
            >= gate.additive_gain_min,
        }
        rows.append(
            {
                **asdict(policy),
                **aggregate,
                "baseline_mrr_macro": baseline["mrr_macro"],
                "control_recall_macro": control["recall_macro"],
                "control_p_lb_macro": control[
                    "silver_precision_lower_bound_macro"
                ],
                "control_mrr_macro": control["mrr_macro"],
                "matched_budget_recall_delta": delta["recall_macro"],
                "matched_budget_p_lb_delta": delta[
                    "silver_precision_lower_bound_macro"
                ],
                "matched_budget_mrr_delta": delta["mrr_macro"],
                "matched_budget_additive_gain": additive_gain,
                "feasible": all(conditions.values()),
                "violation_count": sum(not value for value in conditions.values()),
            }
        )
    grid = pd.DataFrame.from_records(rows)
    feasible = grid.loc[grid["feasible"].map(_truthy)]
    if not feasible.empty:
        chosen = feasible.sort_values(
            [
                "selected_count_mean",
                "silver_precision_lower_bound_macro",
                "mrr_macro",
                "structure_weight",
            ],
            ascending=[True, False, False, True],
            kind="mergesort",
        ).iloc[0]
        return (
            StructuredPolicy(
                float(chosen.retention_fraction),
                int(chosen.minimum_keep),
                float(chosen.structure_weight),
                float(chosen.direction_weight),
                float(chosen.density_weight),
            ),
            grid,
            True,
        )
    chosen = grid.sort_values(
        [
            "violation_count",
            "matched_budget_additive_gain",
            "recall_macro",
            "selected_count_mean",
        ],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).iloc[0]
    return (
        StructuredPolicy(
            float(chosen.retention_fraction),
            int(chosen.minimum_keep),
            float(chosen.structure_weight),
            float(chosen.direction_weight),
            float(chosen.density_weight),
        ),
        grid,
        False,
    )


def run_phase3_r1(
    *, candidate_analysis: Path, output: Path, config_path: Path
) -> Path:
    candidate_analysis = candidate_analysis.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if output.exists():
        raise Phase3R1Error(f"refusing to overwrite output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != (
        "rcaeval-task-a-phase3-r1-structured-evidence"
    ):
        raise Phase3R1Error("unexpected experiment_id")

    raw = pd.read_parquet(candidate_analysis)
    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise Phase3R1Error(f"candidate analysis missing columns: {missing}")
    if raw.duplicated(list(MODEL_KEY_COLUMNS)).any():
        raise Phase3R1Error("duplicate candidate keys")
    if set(raw["role"].astype(str)) != {"calibration", "heldout"}:
        raise Phase3R1Error("calibration/heldout roles are required")

    evaluator = raw[list(MODEL_KEY_COLUMNS) + sorted(EVALUATOR_COLUMNS)].copy()
    model = raw[
        [column for column in raw.columns if column not in EVALUATOR_COLUMNS]
    ].copy()
    features = add_structured_features(model)
    frame = features.merge(
        evaluator,
        on=list(MODEL_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(raw):
        raise Phase3R1Error("model/evaluator rejoin changed candidate count")

    calibration = frame.loc[frame["role"] == "calibration"].copy()
    heldout = frame.loc[frame["role"] == "heldout"].copy()
    split = config["split_contract"]
    if calibration["incident_id"].nunique() != int(
        split["calibration_incidents"]
    ):
        raise Phase3R1Error("unexpected calibration incident count")
    if heldout["incident_id"].nunique() != int(split["heldout_incidents"]):
        raise Phase3R1Error("unexpected heldout incident count")
    calibration_cells = calibration.groupby(
        ["incident_id", "seed", "mask_id"]
    ).ngroups
    heldout_cells = heldout.groupby(
        ["incident_id", "seed", "mask_id"]
    ).ngroups
    if calibration_cells != int(split["calibration_cells"]):
        raise Phase3R1Error("unexpected calibration cell count")
    if heldout_cells != int(split["heldout_cells"]):
        raise Phase3R1Error("unexpected heldout cell count")

    gate = GateConfig(**dict(config["gate"]))
    policy, grid, calibration_feasible = _select_policy(
        calibration, _policy_grid(config), gate
    )
    control_policy = StructuredPolicy(
        policy.retention_fraction,
        policy.minimum_keep,
        0.0,
        policy.direction_weight,
        policy.density_weight,
    )

    calibration_rows, calibration_aggregate, _ = _evaluate_policy(
        calibration, policy
    )
    heldout_rows, heldout_aggregate, heldout_scored = _evaluate_policy(
        heldout, policy
    )
    _control_calibration_rows, control_calibration, _ = _evaluate_policy(
        calibration, control_policy
    )
    control_heldout_rows, control_heldout, _ = _evaluate_policy(
        heldout, control_policy
    )
    baseline_calibration = _baseline_aggregate(calibration)
    baseline_heldout = _baseline_aggregate(heldout)
    delta_full = _delta(heldout_aggregate, baseline_heldout)
    delta_control = _delta(heldout_aggregate, control_heldout)
    selected_ratio = (
        heldout_aggregate["selected_count_mean"]
        / baseline_heldout["selected_count_mean"]
    )
    additive_gain = max(
        delta_control["silver_precision_lower_bound_macro"],
        delta_control["mrr_macro"],
    )
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
        "p_lb_improved_vs_full_a2": delta_full[
            "silver_precision_lower_bound_macro"
        ]
        >= 0.0,
        "mrr_improved_vs_full_a2": delta_full["mrr_macro"] >= 0.0,
        "matched_budget_recall_noninferior": delta_control["recall_macro"]
        >= -gate.matched_budget_recall_tolerance,
        "matched_budget_p_lb_noninferior": delta_control[
            "silver_precision_lower_bound_macro"
        ]
        >= gate.matched_budget_p_lb_delta_min,
        "matched_budget_mrr_noninferior": delta_control["mrr_macro"]
        >= gate.matched_budget_mrr_delta_min,
        "matched_budget_additive_gain": additive_gain
        >= gate.additive_gain_min,
        "structured_weight_active": policy.structure_weight > 0.0,
        "a2_candidate_count_preserved_before_shortlist": len(frame) == len(raw),
    }
    passed = all(conditions.values())

    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "model_output"
    evaluator_dir = output / "evaluator_private"
    published_dir = output / "published"
    model_dir.mkdir()
    evaluator_dir.mkdir()
    published_dir.mkdir()

    features.to_parquet(
        model_dir / "a3_r1_structured_features.parquet", index=False
    )
    heldout_scored.to_parquet(
        evaluator_dir / "heldout_candidate_analysis.parquet", index=False
    )
    calibration_rows.to_csv(
        published_dir / "task_a_phase3_r1_calibration_cells.csv", index=False
    )
    heldout_rows.to_csv(
        published_dir / "task_a_phase3_r1_heldout_cells.csv", index=False
    )
    control_heldout_rows.to_csv(
        evaluator_dir / "a2_equal_size_control_cells.csv", index=False
    )
    grid.to_csv(
        published_dir / "task_a_phase3_r1_policy_grid.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "PASS" if passed else "FAIL",
        "gate_id": "D4A_A3_R1_STRUCTURED_EVIDENCE_UTILITY",
        "source": {
            "candidate_analysis": str(candidate_analysis),
            "candidate_analysis_sha256": hashlib.sha256(
                candidate_analysis.read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "candidate_rows": len(raw),
            "candidate_cells": frame.groupby(
                ["incident_id", "seed", "mask_id"]
            ).ngroups,
        },
        "leakage_boundary": {
            "evaluator_columns_removed_before_feature_computation": True,
            "model_evaluator_rejoin": "immutable candidate key, one_to_one",
            "calibration_only_policy_selection": True,
            "heldout_labels_not_used_for_policy_selection": True,
        },
        "selected_policy": asdict(policy),
        "equal_size_a2_control_policy": asdict(control_policy),
        "calibration": {
            "feasible": calibration_feasible,
            "searched_policy_count": len(grid),
            "feasible_policy_count": int(
                grid["feasible"].map(_truthy).sum()
            ),
            "baseline": baseline_calibration,
            "proposed": calibration_aggregate,
            "equal_size_control": control_calibration,
        },
        "heldout": {
            "baseline_a2_full": baseline_heldout,
            "proposed_a3_r1": heldout_aggregate,
            "equal_size_a2_control": control_heldout,
            "delta_vs_full_a2": delta_full,
            "delta_vs_equal_size_a2": delta_control,
            "selected_count_ratio": selected_ratio,
            "additive_gain": additive_gain,
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
        "interpretation": (
            "PASS means directional asymmetry and support density add held-out "
            "utility beyond selecting the same number of candidates by A2 order."
        ),
        "claim_limit": (
            "CALLS candidate shortlisting on six RCAEval TrainTicket incidents; "
            "not causal-edge or downstream RCA/LLM validation."
        ),
    }
    summary_path = published_dir / "task_a_phase3_r1_results.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (published_dir / "task_a_phase3_r1_results.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    (published_dir / "task_a_phase3_r1_status.txt").write_text(
        summary["status"] + "\n", encoding="utf-8"
    )
    return output


def _render_report(summary: Mapping[str, Any]) -> str:
    heldout = summary["heldout"]
    baseline = heldout["baseline_a2_full"]
    proposed = heldout["proposed_a3_r1"]
    control = heldout["equal_size_a2_control"]
    delta_full = heldout["delta_vs_full_a2"]
    delta_control = heldout["delta_vs_equal_size_a2"]
    gate = summary["gate"]
    policy = summary["selected_policy"]
    conditions = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in gate["conditions"].items()
    )
    return f"""# Task A Phase 3-R1 결과 — 구조 Evidence 재랭킹

- 최종 Gate: **{gate['status']}**
- Calibration feasible 정책: **{summary['calibration']['feasible_policy_count']} / {summary['calibration']['searched_policy_count']}**
- Held-out Cell: **{proposed['cell_count']}**
- 미통과 조건: `{', '.join(gate['reason_codes']) or '-'}`

## 우선 해결한 문제

기존 A3는 후보별 입력 차이를 만들지 못해 `corroborates=0`으로 붕괴했다. 본 실험은 DeBERTa를 제거하고 다음 model-visible Evidence가 A2 순위에 실제 추가 판별력을 주는지 먼저 검증한다.

- Forward/Reverse supporting trace 비대칭
- Forward/Reverse boundary span 비대칭
- Trace 대비 boundary support density
- Self-loop 구조 제약
- A2 prior 보존

## 선택 정책

| 항목 | 값 |
|---|---:|
| retention_fraction | {policy['retention_fraction']:.4f} |
| minimum_keep | {policy['minimum_keep']} |
| structure_weight | {policy['structure_weight']:.4f} |
| direction_weight | {policy['direction_weight']:.4f} |
| density_weight | {policy['density_weight']:.4f} |

## Held-out: A2 전체 대비

| 지표 | A2 전체 | A3-R1 | 변화 |
|---|---:|---:|---:|
| Candidate Recall Macro | {baseline['recall_macro']:.4f} | {proposed['recall_macro']:.4f} | {delta_full['recall_macro']:+.4f} |
| Candidate Recall Minimum | {baseline['recall_min']:.4f} | {proposed['recall_min']:.4f} | - |
| 후보 수 평균 | {baseline['selected_count_mean']:.3f} | {proposed['selected_count_mean']:.3f} | {delta_full['selected_count_mean']:+.3f} |
| P-LB Macro | {baseline['silver_precision_lower_bound_macro']:.4f} | {proposed['silver_precision_lower_bound_macro']:.4f} | {delta_full['silver_precision_lower_bound_macro']:+.4f} |
| MRR Macro | {baseline['mrr_macro']:.4f} | {proposed['mrr_macro']:.4f} | {delta_full['mrr_macro']:+.4f} |

## 동일 후보 수 A2-only 대조군 대비

| 지표 | A2 Equal-size | A3-R1 | 차이 |
|---|---:|---:|---:|
| Recall Macro | {control['recall_macro']:.4f} | {proposed['recall_macro']:.4f} | {delta_control['recall_macro']:+.4f} |
| P-LB Macro | {control['silver_precision_lower_bound_macro']:.4f} | {proposed['silver_precision_lower_bound_macro']:.4f} | {delta_control['silver_precision_lower_bound_macro']:+.4f} |
| MRR Macro | {control['mrr_macro']:.4f} | {proposed['mrr_macro']:.4f} | {delta_control['mrr_macro']:+.4f} |

## Gate 조건

| 조건 | 결과 |
|---|---|
{conditions}

## 해석

- **PASS**: 구조 Evidence가 동일 크기 A2-only보다 정답 관계를 더 앞에 두거나 unverified 후보를 더 잘 제거했다. 이후 Evidence별 NLI를 추가할 근거가 생긴다.
- **FAIL**: 현재 A2 handoff의 forward/reverse count만으로는 부족하다. 다음 우선순위는 Operation·HTTP span kind·Endpoint compatibility를 handoff schema에 명시적으로 추가하는 것이다.
- `CALLS`는 causal `CAUSES`로 해석하지 않는다.

## 범위 제한

{summary['claim_limit']}
"""


__all__ = [
    "GateConfig",
    "Phase3R1Error",
    "StructuredPolicy",
    "add_structured_features",
    "apply_structured_policy",
    "run_phase3_r1",
]
