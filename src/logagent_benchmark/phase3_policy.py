"""Calibration, shortlisting, and evaluator metrics for Task A Phase 3."""

from __future__ import annotations

from dataclasses import asdict
import math
import statistics
from typing import Any, Mapping, Sequence

from .phase3_contract import Phase3Error, ShortlistPolicy

def _a2_rank_norm(candidates: Sequence[dict[str, Any]]) -> Mapping[tuple[str, str, str], float]:
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
        (item["subject"], item["predicate"], item["object"]): 1.0 - index / denominator
        for index, item in enumerate(ordered)
    }


def apply_policy(
    candidates: Sequence[dict[str, Any]],
    policy: ShortlistPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selected and fully scored records without hard-veto deletion."""

    if not candidates:
        return [], []
    a2_norm = _a2_rank_norm(candidates)
    scored: list[dict[str, Any]] = []
    for record in candidates:
        key = (record["subject"], record["predicate"], record["object"])
        nli_norm = (float(record["nli_evidence_score"]) + 1.0) / 2.0
        score = (1.0 - policy.nli_weight) * a2_norm[key] + policy.nli_weight * nli_norm
        enriched = dict(record)
        enriched["a2_rank_normalized"] = a2_norm[key]
        enriched["a3_score"] = score
        enriched["selected"] = False
        scored.append(enriched)

    keep = min(
        len(scored),
        max(policy.minimum_keep, int(math.ceil(policy.retention_fraction * len(scored)))),
    )
    direct = [item for item in scored if item["direct_evidence"]]
    ranked = sorted(
        (item for item in scored if not item["direct_evidence"]),
        key=lambda item: (
            -float(item["a3_score"]),
            -float(item["a2_score"]),
            int(item["proposal_rank"]),
            str(item["subject"]),
            str(item["object"]),
        ),
    )
    selected_keys = {
        (item["subject"], item["predicate"], item["object"])
        for item in direct
    }
    for item in ranked:
        if len(selected_keys) >= keep:
            break
        selected_keys.add((item["subject"], item["predicate"], item["object"]))
    for item in scored:
        item["selected"] = (item["subject"], item["predicate"], item["object"]) in selected_keys
    selected = [item for item in scored if item["selected"]]
    return selected, scored


def evaluate_shortlist(
    selected: Sequence[Mapping[str, Any]],
    *,
    targets: set[tuple[str, str, str]],
    silver: set[tuple[str, str, str]],
) -> dict[str, Any]:
    keys = {(str(item["subject"]), str(item["predicate"]), str(item["object"])) for item in selected}
    recovered = keys & targets
    recall = len(recovered) / len(targets) if targets else None
    matched = keys & silver
    p_lb = len(matched) / len(keys) if keys else None

    by_query: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for item in selected:
        key = (str(item["subject"]), str(item["predicate"]), str(item["object"]))
        by_key[key] = item
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
        target_score = float(target_item["a3_score"])
        competitors = []
        for item in by_query.get((target[0], target[1]), ()):
            key = (str(item["subject"]), str(item["predicate"]), str(item["object"]))
            if key == target or key in silver:
                continue
            competitors.append(item)
        higher = sum(float(item["a3_score"]) > target_score + epsilon for item in competitors)
        tied = sum(abs(float(item["a3_score"]) - target_score) <= epsilon for item in competitors)
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    mrr = statistics.fmean(reciprocal) if reciprocal else None
    return {
        "selected_count": len(keys),
        "target_count": len(targets),
        "recovered_target_count": len(recovered),
        "recall": recall,
        "silver_matched_count": len(matched),
        "silver_precision_lower_bound": p_lb,
        "mrr": mrr,
        "ranks": ranks,
        "state_counts": {
            state: sum(str(item["nli_state"]) == state for item in selected)
            for state in ("corroborates", "ambiguous", "contradicts")
        },
    }


def _aggregate(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    recalls = [float(item["recall"]) for item in metrics if item["recall"] is not None]
    p_lbs = [float(item["silver_precision_lower_bound"]) for item in metrics if item["silver_precision_lower_bound"] is not None]
    mrrs = [float(item["mrr"]) for item in metrics if item["mrr"] is not None]
    counts = [int(item["selected_count"]) for item in metrics]
    total_targets = sum(int(item["target_count"]) for item in metrics)
    total_recovered = sum(int(item["recovered_target_count"]) for item in metrics)
    return {
        "cell_count": len(metrics),
        "recall_macro": statistics.fmean(recalls),
        "recall_min": min(recalls),
        "recall_pooled": total_recovered / total_targets if total_targets else None,
        "selected_count_mean": statistics.fmean(counts),
        "selected_count_median": statistics.median(counts),
        "selected_count_max": max(counts),
        "silver_precision_lower_bound_macro": statistics.fmean(p_lbs),
        "silver_precision_lower_bound_min": min(p_lbs),
        "mrr_macro": statistics.fmean(mrrs),
        "mrr_min": min(mrrs),
    }


def _evaluate_policy_on_cells(
    cells: Sequence[dict[str, Any]],
    policy: ShortlistPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        selected, _scored = apply_policy(cell["candidates"], policy)
        metric = evaluate_shortlist(selected, targets=cell["targets"], silver=cell["silver"])
        rows.append({
            "case": cell["case"],
            "fault": cell["fault"],
            "seed": cell["seed"],
            "mask_id": cell["mask_id"],
            "mask_ratio": cell["mask_ratio"],
            **metric,
        })
    return rows, _aggregate(rows)


def select_calibrated_policy(
    calibration_cells: Sequence[dict[str, Any]],
    *,
    search: Mapping[str, Any],
    calibration_gate: Mapping[str, Any],
) -> tuple[ShortlistPolicy, list[dict[str, Any]]]:
    baseline_mrr = statistics.fmean(float(cell["a2_mrr"]) for cell in calibration_cells)
    candidates: list[dict[str, Any]] = []
    for retention in search["retention_fractions"]:
        for minimum_keep in search["minimum_keep"]:
            control_policy = ShortlistPolicy(float(retention), int(minimum_keep), 0.0)
            _control_rows, control = _evaluate_policy_on_cells(
                calibration_cells, control_policy
            )
            for nli_weight in search["nli_weights"]:
                policy = ShortlistPolicy(float(retention), int(minimum_keep), float(nli_weight))
                _rows, aggregate = _evaluate_policy_on_cells(calibration_cells, policy)
                recall_delta = aggregate["recall_macro"] - control["recall_macro"]
                p_lb_delta = (
                    aggregate["silver_precision_lower_bound_macro"]
                    - control["silver_precision_lower_bound_macro"]
                )
                mrr_delta = aggregate["mrr_macro"] - control["mrr_macro"]
                additive_gain = p_lb_delta > 1e-12 or mrr_delta > 1e-12
                feasible = (
                    aggregate["recall_macro"] >= float(calibration_gate["recall_macro_min"])
                    and aggregate["recall_min"] >= float(calibration_gate["recall_each_cell_min"])
                    and aggregate["mrr_macro"]
                    >= baseline_mrr - float(calibration_gate["mrr_noninferiority_tolerance"])
                    and recall_delta
                    >= -float(calibration_gate["matched_budget_recall_tolerance"])
                    and p_lb_delta
                    >= float(calibration_gate["matched_budget_p_lb_delta_min"])
                    and mrr_delta
                    >= float(calibration_gate["matched_budget_mrr_delta_min"])
                    and (
                        additive_gain
                        if bool(calibration_gate.get("matched_budget_additive_gain_required", True))
                        else True
                    )
                )
                candidates.append({
                    **asdict(policy),
                    **aggregate,
                    "baseline_mrr_macro": baseline_mrr,
                    "control_recall_macro": control["recall_macro"],
                    "control_p_lb_macro": control["silver_precision_lower_bound_macro"],
                    "control_mrr_macro": control["mrr_macro"],
                    "matched_budget_recall_delta": recall_delta,
                    "matched_budget_p_lb_delta": p_lb_delta,
                    "matched_budget_mrr_delta": mrr_delta,
                    "matched_budget_additive_gain": additive_gain,
                    "feasible": feasible,
                })
    feasible_rows = [row for row in candidates if row["feasible"]]
    if not feasible_rows:
        raise Phase3Error(
            "no calibrated A3 policy satisfies recall/MRR floors and adds utility "
            "over its matched-budget A2-only control"
        )
    chosen = sorted(
        feasible_rows,
        key=lambda row: (
            float(row["selected_count_mean"]),
            -float(row["silver_precision_lower_bound_macro"]),
            -float(row["mrr_macro"]),
            float(row["nli_weight"]),
            float(row["retention_fraction"]),
            int(row["minimum_keep"]),
        ),
    )[0]
    policy = ShortlistPolicy(
        float(chosen["retention_fraction"]),
        int(chosen["minimum_keep"]),
        float(chosen["nli_weight"]),
    )
    return policy, candidates


def _baseline_aggregate(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cell_count": len(cells),
        "recall_macro": statistics.fmean(float(cell["a2_recall"]) for cell in cells),
        "recall_min": min(float(cell["a2_recall"]) for cell in cells),
        "selected_count_mean": statistics.fmean(int(cell["a2_count"]) for cell in cells),
        "selected_count_median": statistics.median(int(cell["a2_count"]) for cell in cells),
        "selected_count_max": max(int(cell["a2_count"]) for cell in cells),
        "silver_precision_lower_bound_macro": statistics.fmean(float(cell["a2_p_lb"]) for cell in cells),
        "silver_precision_lower_bound_min": min(float(cell["a2_p_lb"]) for cell in cells),
        "mrr_macro": statistics.fmean(float(cell["a2_mrr"]) for cell in cells),
        "mrr_min": min(float(cell["a2_mrr"]) for cell in cells),
    }


def _delta(enhanced: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "recall_macro": float(enhanced["recall_macro"]) - float(baseline["recall_macro"]),
        "selected_count_mean": float(enhanced["selected_count_mean"]) - float(baseline["selected_count_mean"]),
        "silver_precision_lower_bound_macro": (
            float(enhanced["silver_precision_lower_bound_macro"])
            - float(baseline["silver_precision_lower_bound_macro"])
        ),
        "mrr_macro": float(enhanced["mrr_macro"]) - float(baseline["mrr_macro"]),
    }


__all__ = [
    "_baseline_aggregate", "_delta", "_evaluate_policy_on_cells",
    "apply_policy", "evaluate_shortlist", "select_calibrated_policy",
]
