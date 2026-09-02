"""Risk-limited post-calibration for Task A Phase 3 structural evidence.

Consumes the audit table emitted by ``task_a_phase3_structural``. Model-visible
features are scored before evaluator columns are used. Calibration incidents
choose one policy; the frozen policy is then evaluated on the existing
held-out incidents. Because those held-out incidents were already inspected in
a prior run, this module produces a development result, not final confirmation.
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


EXPERIMENT_ID = "rcaeval-task-a-phase3-risk-limited-structural-evidence"
DEFAULT_CONFIG = Path("configs/experiment_task_a_rcaeval_phase3_structural_safe.json")
EPS = 1e-12
PROFILES = (
    "temporal_directness",
    "directional_structure",
    "operation_compatibility",
    "hybrid",
)
CELL_COLUMNS = ("case", "seed", "mask_id")


class RiskLimitedError(RuntimeError):
    pass


@dataclass(frozen=True)
class RiskPolicy:
    profile_id: str
    max_drop_fraction: float
    query_keep_fraction: float
    minimum_keep: int
    structural_weight: float
    pareto_margin_min: float
    structural_spread_min: float

    def __post_init__(self) -> None:
        if self.profile_id not in PROFILES:
            raise ValueError(f"unknown profile: {self.profile_id}")
        for name, value in (
            ("max_drop_fraction", self.max_drop_fraction),
            ("query_keep_fraction", self.query_keep_fraction),
            ("structural_weight", self.structural_weight),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0,1]")
        if self.minimum_keep <= 0:
            raise ValueError("minimum_keep must be positive")
        for name, value in (
            ("pareto_margin_min", self.pareto_margin_min),
            ("structural_spread_min", self.structural_spread_min),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


def _edge(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["subject"]), str(row["predicate"]), str(row["object"])


def _cell(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["case"]), int(row["seed"]), str(row["mask_id"])


def _rank_norm(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], float]:
    ordered = sorted(
        records,
        key=lambda row: (
            -float(row["a2_score"]),
            -int(row["supporting_traces"]),
            -int(row["boundary_spans"]),
            int(row["proposal_rank"]),
            str(row["subject"]),
            str(row["object"]),
        ),
    )
    denominator = max(1, len(ordered) - 1)
    return {
        _edge(row): 1.0 - index / denominator
        for index, row in enumerate(ordered)
    }


def _dense_percentile(
    values: Mapping[tuple[str, str, str], float],
) -> dict[tuple[str, str, str], float]:
    unique = sorted(set(float(value) for value in values.values()))
    if len(unique) <= 1:
        return {key: 0.5 for key in values}
    ranks = {
        value: index / (len(unique) - 1)
        for index, value in enumerate(unique)
    }
    return {key: ranks[float(value)] for key, value in values.items()}


def apply_policy(
    records: Sequence[Mapping[str, Any]],
    policy: RiskPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Drop only same-query candidates dominated in both A2 and structure."""

    if not records:
        return [], [], {
            "candidate_count": 0,
            "selected_count": 0,
            "dropped_count": 0,
        }
    a2 = _rank_norm(records)
    raw = {
        _edge(row): float(row[f"{policy.profile_id}_score"])
        for row in records
    }
    structural = _dense_percentile(raw)
    groups: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for row in records:
        key = _edge(row)
        groups.setdefault((key[0], key[1]), []).append(key)

    scored: list[dict[str, Any]] = []
    for row in records:
        key = _edge(row)
        values = [
            structural[item]
            for item in groups[(key[0], key[1])]
        ]
        spread = max(values) - min(values) if values else 0.0
        combined = (
            (1.0 - policy.structural_weight) * a2[key]
            + policy.structural_weight * structural[key]
        )
        scored.append(
            {
                **dict(row),
                "a2_rank_normalized_safe": a2[key],
                "structural_rank_normalized_safe": structural[key],
                "structural_profile_score_safe": raw[key],
                "query_structural_spread_safe": spread,
                "risk_limited_score": combined,
                "pareto_margin_safe": 0.0,
                "drop_eligible_safe": False,
                "selected_safe": True,
                "drop_reason_safe": "RETAINED",
            }
        )
    by_key = {_edge(row): row for row in scored}
    for keys in groups.values():
        for key in keys:
            row = by_key[key]
            margin = 0.0
            for other_key in keys:
                if other_key == key:
                    continue
                other = by_key[other_key]
                a2_delta = (
                    float(other["a2_rank_normalized_safe"])
                    - float(row["a2_rank_normalized_safe"])
                )
                structural_delta = (
                    float(other["structural_rank_normalized_safe"])
                    - float(row["structural_rank_normalized_safe"])
                )
                if a2_delta > EPS and structural_delta > EPS:
                    margin = max(
                        margin,
                        min(a2_delta, structural_delta),
                    )
            row["pareto_margin_safe"] = margin
            row["drop_eligible_safe"] = bool(
                not bool(row["direct_evidence"])
                and margin > 0.0
                and margin + EPS >= policy.pareto_margin_min
                and float(row["query_structural_spread_safe"]) + EPS
                >= policy.structural_spread_min
            )

    total = len(scored)
    global_budget = min(
        max(0, total - min(total, policy.minimum_keep)),
        int(math.floor(policy.max_drop_fraction * total + EPS)),
    )
    query_budget: dict[tuple[str, str], int] = {}
    for query, keys in groups.items():
        direct_count = sum(
            bool(by_key[key]["direct_evidence"])
            for key in keys
        )
        minimum_query_keep = max(
            1,
            direct_count,
            int(math.ceil(policy.query_keep_fraction * len(keys))),
        )
        query_budget[query] = max(0, len(keys) - minimum_query_keep)

    eligible = sorted(
        (row for row in scored if row["drop_eligible_safe"]),
        key=lambda row: (
            float(row["risk_limited_score"]),
            -float(row["pareto_margin_safe"]),
            float(row["structural_rank_normalized_safe"]),
            float(row["a2_rank_normalized_safe"]),
            int(row["proposal_rank"]),
            str(row["subject"]),
            str(row["object"]),
        ),
    )
    dropped_by_query = {query: 0 for query in groups}
    dropped = 0
    for row in eligible:
        if dropped >= global_budget:
            break
        key = _edge(row)
        query = key[0], key[1]
        if dropped_by_query[query] >= query_budget[query]:
            row["drop_reason_safe"] = "QUERY_SAFETY_FLOOR"
            continue
        row["selected_safe"] = False
        row["drop_reason_safe"] = "PARETO_DOMINATED_LOW_RISK"
        dropped_by_query[query] += 1
        dropped += 1
    for row in scored:
        if (
            row["selected_safe"]
            and row["drop_eligible_safe"]
            and row["drop_reason_safe"] == "RETAINED"
        ):
            row["drop_reason_safe"] = (
                "GLOBAL_DROP_BUDGET"
                if dropped >= global_budget
                else "QUERY_SAFETY_FLOOR"
            )
    selected = [row for row in scored if row["selected_safe"]]
    diagnostics = {
        "candidate_count": total,
        "selected_count": len(selected),
        "global_drop_budget": global_budget,
        "dropped_count": dropped,
        "drop_eligible_count": len(eligible),
        "abstained_drop_capacity": max(0, global_budget - dropped),
        "query_count": len(groups),
        "query_drop_budget_total": sum(query_budget.values()),
    }
    return selected, scored, diagnostics


def exact_a2_budget(
    records: Sequence[Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Select exactly ``count`` candidates using only frozen A2 ordering."""

    if count < 0 or count > len(records):
        raise ValueError("invalid exact A2 budget")
    direct = [
        dict(row)
        for row in records
        if bool(row["direct_evidence"])
    ]
    if len(direct) > count:
        raise RiskLimitedError(
            "exact budget would remove direct evidence"
        )
    ranked = sorted(
        (
            dict(row)
            for row in records
            if not bool(row["direct_evidence"])
        ),
        key=lambda row: (
            -float(row["a2_score"]),
            -int(row["supporting_traces"]),
            -int(row["boundary_spans"]),
            int(row["proposal_rank"]),
            str(row["subject"]),
            str(row["object"]),
        ),
    )
    selected = direct + ranked[: max(0, count - len(direct))]
    a2 = _rank_norm(records)
    return [
        {
            **row,
            "risk_limited_score": a2[_edge(row)],
        }
        for row in selected
    ]


def evaluate_cell(
    selected: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_keys = {_edge(row) for row in selected}
    targets = {
        _edge(row)
        for row in all_records
        if bool(row["is_masked_target"])
    }
    silver = {
        _edge(row)
        for row in all_records
        if bool(row["is_silver_matched"])
    }
    recovered = selected_keys & targets
    matched = selected_keys & silver
    by_key = {_edge(row): row for row in selected}
    by_query: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in selected:
        key = _edge(row)
        by_query.setdefault((key[0], key[1]), []).append(row)

    reciprocal: list[float] = []
    ranks: list[int | None] = []
    for target in sorted(targets):
        target_row = by_key.get(target)
        if target_row is None:
            reciprocal.append(0.0)
            ranks.append(None)
            continue
        target_score = float(target_row["risk_limited_score"])
        competitors = []
        for row in by_query.get((target[0], target[1]), ()):
            key = _edge(row)
            if key == target or key in silver:
                continue
            competitors.append(row)
        higher = sum(
            float(row["risk_limited_score"])
            > target_score + EPS
            for row in competitors
        )
        tied = sum(
            abs(float(row["risk_limited_score"]) - target_score)
            <= EPS
            for row in competitors
        )
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    return {
        "selected_count": len(selected_keys),
        "target_count": len(targets),
        "recovered_target_count": len(recovered),
        "recall": (
            len(recovered) / len(targets)
            if targets
            else None
        ),
        "silver_matched_count": len(matched),
        "silver_precision_lower_bound": (
            len(matched) / len(selected_keys)
            if selected_keys
            else None
        ),
        "mrr": (
            statistics.fmean(reciprocal)
            if reciprocal
            else None
        ),
        "ranks": ranks,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recalls = [float(row["recall"]) for row in rows]
    p_lbs = [
        float(row["silver_precision_lower_bound"])
        for row in rows
    ]
    mrrs = [float(row["mrr"]) for row in rows]
    counts = [int(row["selected_count"]) for row in rows]
    targets = sum(int(row["target_count"]) for row in rows)
    recovered = sum(
        int(row["recovered_target_count"])
        for row in rows
    )
    return {
        "cell_count": len(rows),
        "recall_macro": statistics.fmean(recalls),
        "recall_min": min(recalls),
        "recall_pooled": recovered / targets,
        "selected_count_mean": statistics.fmean(counts),
        "selected_count_median": statistics.median(counts),
        "selected_count_min": min(counts),
        "selected_count_max": max(counts),
        "silver_precision_lower_bound_macro": statistics.fmean(p_lbs),
        "silver_precision_lower_bound_min": min(p_lbs),
        "mrr_macro": statistics.fmean(mrrs),
        "mrr_min": min(mrrs),
    }


def _delta(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "recall_macro": (
            float(left["recall_macro"])
            - float(right["recall_macro"])
        ),
        "selected_count_mean": (
            float(left["selected_count_mean"])
            - float(right["selected_count_mean"])
        ),
        "silver_precision_lower_bound_macro": (
            float(left["silver_precision_lower_bound_macro"])
            - float(right["silver_precision_lower_bound_macro"])
        ),
        "mrr_macro": (
            float(left["mrr_macro"])
            - float(right["mrr_macro"])
        ),
    }


def evaluate_policy(
    cells: Mapping[
        tuple[str, int, str],
        list[dict[str, Any]],
    ],
    policy: RiskPolicy,
) -> dict[str, Any]:
    proposed_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    selected_sets = {}
    control_sets = {}
    scored_records: list[dict[str, Any]] = []
    for cell_id, records in cells.items():
        selected, scored, diagnostics = apply_policy(
            records,
            policy,
        )
        control = exact_a2_budget(records, len(selected))
        proposed_metric = evaluate_cell(selected, records)
        control_metric = evaluate_cell(control, records)
        common = {
            "case": cell_id[0],
            "seed": cell_id[1],
            "mask_id": cell_id[2],
            "role": str(records[0]["role"]),
            "fault": str(records[0]["fault"]),
            "mask_ratio": float(records[0]["mask_ratio"]),
        }
        proposed_rows.append(
            {
                **common,
                **proposed_metric,
                **{
                    f"policy_{name}": value
                    for name, value in diagnostics.items()
                },
            }
        )
        control_rows.append({**common, **control_metric})
        selected_sets[cell_id] = frozenset(
            _edge(row) for row in selected
        )
        control_sets[cell_id] = frozenset(
            _edge(row) for row in control
        )
        for row in scored:
            scored_records.append(
                {
                    **row,
                    "safe_cell_selected_count": len(selected),
                }
            )
    return {
        "proposed_rows": proposed_rows,
        "control_rows": control_rows,
        "proposed": _aggregate(proposed_rows),
        "control": _aggregate(control_rows),
        "selected_sets": selected_sets,
        "control_sets": control_sets,
        "scored_records": scored_records,
    }


def _baseline(
    cells: Mapping[
        tuple[str, int, str],
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    rows = []
    for cell_id, records in cells.items():
        selected = exact_a2_budget(records, len(records))
        metric = evaluate_cell(selected, records)
        rows.append(
            {
                "case": cell_id[0],
                "seed": cell_id[1],
                "mask_id": cell_id[2],
                **metric,
            }
        )
    return _aggregate(rows)


def _policy_grid(
    config: Mapping[str, Any],
) -> list[RiskPolicy]:
    search = config["policy_search"]
    return [
        RiskPolicy(
            str(profile),
            float(drop_fraction),
            float(query_keep),
            int(minimum_keep),
            float(weight),
            float(margin),
            float(spread),
        )
        for profile in search["profiles"]
        for drop_fraction in search["max_drop_fractions"]
        for query_keep in search["query_keep_fractions"]
        for minimum_keep in search["minimum_keep"]
        for weight in search["structural_weights"]
        for margin in search["pareto_margin_min"]
        for spread in search["structural_spread_min"]
    ]


def select_policy(
    calibration_cells: Mapping[
        tuple[str, int, str],
        list[dict[str, Any]],
    ],
    config: Mapping[str, Any],
) -> tuple[RiskPolicy, list[dict[str, Any]]]:
    baseline = _baseline(calibration_cells)
    gate = config["calibration_gate"]
    grid_rows: list[dict[str, Any]] = []
    for policy in _policy_grid(config):
        result = evaluate_policy(calibration_cells, policy)
        proposed = result["proposed"]
        control = result["control"]
        matched = _delta(proposed, control)
        ratio = (
            proposed["selected_count_mean"]
            / baseline["selected_count_mean"]
        )
        changed = sum(
            result["selected_sets"][key]
            != result["control_sets"][key]
            for key in result["selected_sets"]
        )
        additive = bool(
            matched["silver_precision_lower_bound_macro"] > EPS
            or matched["mrr_macro"] > EPS
        )
        conditions = {
            "recall_macro": (
                proposed["recall_macro"]
                >= float(gate["recall_macro_min"])
            ),
            "recall_each_cell": (
                proposed["recall_min"]
                >= float(gate["recall_each_cell_min"])
            ),
            "mrr_noninferiority": (
                proposed["mrr_macro"]
                >= baseline["mrr_macro"]
                - float(gate["mrr_noninferiority_tolerance"])
            ),
            "candidate_count_reduced": (
                ratio
                <= float(gate["selected_count_ratio_max"])
            ),
            "matched_budget_recall": (
                matched["recall_macro"]
                >= -float(gate["matched_budget_recall_tolerance"])
            ),
            "matched_budget_p_lb": (
                matched["silver_precision_lower_bound_macro"]
                >= float(gate["matched_budget_p_lb_delta_min"])
            ),
            "matched_budget_mrr": (
                matched["mrr_macro"]
                >= float(gate["matched_budget_mrr_delta_min"])
            ),
            "matched_budget_additive_gain": additive,
            "selection_changed": changed > 0,
        }
        feasible = all(conditions.values())
        grid_rows.append(
            {
                **asdict(policy),
                **proposed,
                "selected_count_ratio": ratio,
                "control_recall_macro": control["recall_macro"],
                "control_p_lb_macro": control[
                    "silver_precision_lower_bound_macro"
                ],
                "control_mrr_macro": control["mrr_macro"],
                "matched_budget_recall_delta": matched[
                    "recall_macro"
                ],
                "matched_budget_p_lb_delta": matched[
                    "silver_precision_lower_bound_macro"
                ],
                "matched_budget_mrr_delta": matched["mrr_macro"],
                "matched_budget_additive_gain": additive,
                "selection_change_cell_count": changed,
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
    feasible_rows = [
        row for row in grid_rows if row["feasible"]
    ]
    pool = feasible_rows or grid_rows
    status = (
        "FEASIBLE_POLICY"
        if feasible_rows
        else "DIAGNOSTIC_FALLBACK_NO_FEASIBLE_POLICY"
    )
    chosen = sorted(
        pool,
        key=lambda row: (
            int(row["violation_count"]),
            -float(row["recall_min"]),
            -float(row["mrr_macro"]),
            -float(row["silver_precision_lower_bound_macro"]),
            -float(row["matched_budget_mrr_delta"]),
            -float(row["matched_budget_p_lb_delta"]),
            float(row["selected_count_mean"]),
            float(row["max_drop_fraction"]),
            -float(row["query_keep_fraction"]),
            float(row["structural_weight"]),
            str(row["profile_id"]),
            float(row["pareto_margin_min"]),
            float(row["structural_spread_min"]),
        ),
    )[0]
    fields = tuple(RiskPolicy.__dataclass_fields__)
    identity = tuple(chosen[field] for field in fields)
    for row in grid_rows:
        if tuple(row[field] for field in fields) == identity:
            row["selected"] = True
            row["selection_status"] = status
    return (
        RiskPolicy(
            str(identity[0]),
            float(identity[1]),
            float(identity[2]),
            int(identity[3]),
            float(identity[4]),
            float(identity[5]),
            float(identity[6]),
        ),
        grid_rows,
    )


def _render(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline_heldout"]
    proposed = summary["proposed_heldout"]
    delta = summary["delta_vs_full_a2"]
    additive = summary["delta_vs_exact_budget_a2"]
    return "\n".join(
        [
            "# Task A Phase 3-R2 — Risk-limited 구조 Evidence",
            "",
            f"- 개발 Gate: **{summary['gate']['status']}**",
            "- 프로토콜: **DEVELOPMENT_RETEST — fresh confirmatory holdout required**",
            (
                "- Calibration feasible 정책: "
                f"**{summary['calibration']['feasible_policy_count']} / "
                f"{summary['calibration']['searched_policy_count']}**"
            ),
            (
                "- 선택 정책: `"
                + json.dumps(
                    summary["selected_policy"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "`"
            ),
            (
                "- 미통과 조건: `"
                + (", ".join(summary["gate"]["reason_codes"]) or "없음")
                + "`"
            ),
            "",
            "| 지표 | A2 전체 | Risk-limited | 변화 |",
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
            "## 동일 후보 수 A2-only 대비",
            "",
            f"- Recall: **{additive['recall_macro']:+.4f}**",
            (
                "- P-LB: **"
                f"{additive['silver_precision_lower_bound_macro']:+.4f}**"
            ),
            f"- MRR: **{additive['mrr_macro']:+.4f}**",
            "",
            "- 같은 query에서 A2·구조 순위가 모두 열위인 후보만 제거한다.",
            "- Direct evidence와 판단이 모호한 후보는 유지한다.",
            "- 기존 Held-out을 이미 관찰했으므로 독립 Incident 확인 전에는 성공을 주장하지 않는다.",
            "",
        ]
    )


def run(
    *,
    candidate_analysis: Path,
    output: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> Path:
    candidate_analysis = candidate_analysis.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if output.exists():
        raise RiskLimitedError(f"refusing to overwrite {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RiskLimitedError("unexpected experiment_id")
    frame = pd.read_parquet(candidate_analysis)
    required = {
        "case",
        "fault",
        "role",
        "seed",
        "mask_id",
        "mask_ratio",
        "subject",
        "predicate",
        "object",
        "a2_score",
        "supporting_traces",
        "boundary_spans",
        "proposal_rank",
        "direct_evidence",
        "is_masked_target",
        "is_silver_matched",
        *(
            f"{profile}_score"
            for profile in PROFILES
        ),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RiskLimitedError(
            f"candidate analysis missing columns: {missing}"
        )
    if set(frame["role"].astype(str)) != {
        "calibration",
        "heldout",
    }:
        raise RiskLimitedError(
            "candidate analysis must contain calibration and heldout roles"
        )
    cells: dict[
        tuple[str, int, str],
        list[dict[str, Any]],
    ] = {}
    for record in frame.to_dict(orient="records"):
        cells.setdefault(_cell(record), []).append(record)
    calibration = {
        key: rows
        for key, rows in cells.items()
        if str(rows[0]["role"]) == "calibration"
    }
    heldout = {
        key: rows
        for key, rows in cells.items()
        if str(rows[0]["role"]) == "heldout"
    }
    if len(calibration) != 20 or len(heldout) != 40:
        raise RiskLimitedError(
            "expected 20/40 cells, found "
            f"{len(calibration)}/{len(heldout)}"
        )

    policy, grid = select_policy(calibration, config)
    chosen = next(row for row in grid if row["selected"])
    calibration_feasible = bool(chosen["feasible"])
    heldout_result = evaluate_policy(heldout, policy)
    all_result = evaluate_policy(cells, policy)
    baseline_heldout = _baseline(heldout)
    baseline_all = _baseline(cells)
    proposed = heldout_result["proposed"]
    control = heldout_result["control"]
    delta_full = _delta(proposed, baseline_heldout)
    delta_control = _delta(proposed, control)
    ratio = (
        proposed["selected_count_mean"]
        / baseline_heldout["selected_count_mean"]
    )
    gate_cfg = config["heldout_gate"]
    exact_budget = all(
        int(left["selected_count"])
        == int(right["selected_count"])
        for left, right in zip(
            heldout_result["proposed_rows"],
            heldout_result["control_rows"],
            strict=True,
        )
    )
    additive = bool(
        delta_control["silver_precision_lower_bound_macro"] > EPS
        or delta_control["mrr_macro"] > EPS
    )
    conditions = {
        "calibration_policy_feasible": calibration_feasible,
        "heldout_complete": (
            len(heldout_result["proposed_rows"]) == 40
        ),
        "recall_macro": (
            proposed["recall_macro"]
            >= float(gate_cfg["recall_macro_min"])
        ),
        "recall_pooled": (
            proposed["recall_pooled"]
            >= float(gate_cfg["recall_pooled_min"])
        ),
        "recall_each_cell": (
            proposed["recall_min"]
            >= float(gate_cfg["recall_each_cell_min"])
        ),
        "candidate_count_reduced": (
            ratio
            <= float(gate_cfg["selected_count_ratio_max"])
        ),
        "p_lb_improved": (
            proposed["silver_precision_lower_bound_macro"]
            >= baseline_heldout[
                "silver_precision_lower_bound_macro"
            ]
            + float(gate_cfg["p_lb_macro_delta_min"])
        ),
        "mrr_improved": (
            proposed["mrr_macro"]
            >= baseline_heldout["mrr_macro"]
            + float(gate_cfg["mrr_macro_delta_min"])
        ),
        "exact_matched_budget": exact_budget,
        "matched_budget_recall_noninferior": (
            delta_control["recall_macro"]
            >= -float(gate_cfg["matched_budget_recall_tolerance"])
        ),
        "matched_budget_p_lb_noninferior": (
            delta_control["silver_precision_lower_bound_macro"]
            >= float(gate_cfg["matched_budget_p_lb_delta_min"])
        ),
        "matched_budget_mrr_noninferior": (
            delta_control["mrr_macro"]
            >= float(gate_cfg["matched_budget_mrr_delta_min"])
        ),
        "matched_budget_additive_gain": additive,
    }
    passed = all(conditions.values())

    output.mkdir(parents=True)
    private = output / "evaluator_private"
    published = output / "published"
    private.mkdir()
    published.mkdir()
    pd.DataFrame(
        heldout_result["proposed_rows"]
    ).to_csv(private / "heldout_cells.csv", index=False)
    pd.DataFrame(
        heldout_result["control_rows"]
    ).to_csv(
        private / "heldout_exact_budget_a2.csv",
        index=False,
    )
    pd.DataFrame(grid).to_csv(
        private / "calibration_grid.csv",
        index=False,
    )
    pd.DataFrame(
        all_result["scored_records"]
    ).to_parquet(
        private / "candidate_analysis.parquet",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "DEVELOPMENT_PASS" if passed else "FAIL",
        "protocol_status": config["protocol_status"],
        "source_candidate_analysis_sha256": hashlib.sha256(
            candidate_analysis.read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "selected_policy": asdict(policy),
        "calibration": {
            "status": chosen["selection_status"],
            "selected_policy_feasible": calibration_feasible,
            "feasible_policy_count": sum(
                bool(row["feasible"])
                for row in grid
            ),
            "searched_policy_count": len(grid),
            "selected_policy_row": chosen,
            "heldout_labels_used_for_selection": False,
        },
        "baseline_all": baseline_all,
        "baseline_heldout": baseline_heldout,
        "proposed_all": all_result["proposed"],
        "proposed_heldout": proposed,
        "exact_budget_a2_heldout": control,
        "delta_vs_full_a2": delta_full,
        "delta_vs_exact_budget_a2": delta_control,
        "gate": {
            "gate_id": gate_cfg["gate_id"],
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "conditions": conditions,
            "reason_codes": [
                name.upper()
                for name, value in conditions.items()
                if not value
            ],
            "required": gate_cfg,
            "observed_selected_count_ratio": ratio,
            "confirmation_required": True,
        },
        "leakage_boundary": {
            "calibration_only_policy_selection": True,
            "heldout_labels_used_after_policy_freeze": True,
            "model_features_separated_from_evaluator_labels_in_source_artifact": True,
        },
        "known_limitation": (
            "Reverse-direction evidence in the source artifact is available "
            "only when the reverse edge was also an A2 candidate. A "
            "reverse-outside-candidate probe remains required before final "
            "confirmation."
        ),
        "claim_limit": config["claim_limit"],
    }
    text = (
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "summary.json").write_text(
        text,
        encoding="utf-8",
    )
    (
        published / "task_a_phase3_structural_safe_results.json"
    ).write_text(text, encoding="utf-8")
    (
        published / "task_a_phase3_structural_safe_results.md"
    ).write_text(_render(summary), encoding="utf-8")
    pd.DataFrame(
        heldout_result["proposed_rows"]
    ).to_csv(
        published / "task_a_phase3_structural_safe_heldout_cells.csv",
        index=False,
    )
    return output


__all__ = [
    "DEFAULT_CONFIG",
    "EXPERIMENT_ID",
    "RiskPolicy",
    "RiskLimitedError",
    "apply_policy",
    "exact_a2_budget",
    "run",
    "select_policy",
]
