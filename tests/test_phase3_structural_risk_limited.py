from __future__ import annotations

from logagent_benchmark.phase3_structural_risk_limited import (
    RiskPolicy,
    apply_policy,
    exact_a2_budget,
    evaluate_cell,
    select_policy,
)


def candidate(
    subject: str,
    obj: str,
    rank: int,
    *,
    a2_score: float,
    structural: float,
    target: bool = False,
    silver: bool = False,
    direct: bool = False,
):
    return {
        "case": "case-1",
        "fault": "cpu",
        "role": "calibration",
        "seed": 11,
        "mask_id": "iid40",
        "mask_ratio": 0.4,
        "subject": subject,
        "predicate": "CALLS",
        "object": obj,
        "a2_score": a2_score,
        "supporting_traces": 10 - rank,
        "boundary_spans": 20 - rank,
        "proposal_rank": rank,
        "direct_evidence": direct,
        "is_masked_target": target,
        "is_silver_matched": silver,
        "temporal_directness_score": structural,
        "directional_structure_score": structural,
        "operation_compatibility_score": structural,
        "hybrid_score": structural,
    }


def policy(**overrides):
    values = {
        "profile_id": "hybrid",
        "max_drop_fraction": 0.5,
        "query_keep_fraction": 0.5,
        "minimum_keep": 1,
        "structural_weight": 0.5,
        "pareto_margin_min": 0.0,
        "structural_spread_min": 0.0,
    }
    values.update(overrides)
    return RiskPolicy(**values)


def test_drops_only_candidates_dominated_in_both_rankings():
    records = [
        candidate(
            "s",
            "true",
            1,
            a2_score=0.95,
            structural=0.9,
            target=True,
            silver=True,
        ),
        candidate(
            "s",
            "dominated",
            2,
            a2_score=0.80,
            structural=0.1,
        ),
        candidate(
            "s",
            "tradeoff",
            3,
            a2_score=0.70,
            structural=1.0,
        ),
    ]
    selected, scored, diagnostics = apply_policy(records, policy())
    selected_objects = {row["object"] for row in selected}
    assert "dominated" not in selected_objects
    assert "true" in selected_objects
    assert "tradeoff" in selected_objects
    assert diagnostics["dropped_count"] == 1
    dominated = next(
        row for row in scored if row["object"] == "dominated"
    )
    assert (
        dominated["drop_reason_safe"]
        == "PARETO_DOMINATED_LOW_RISK"
    )


def test_direct_evidence_and_per_query_floor_are_protected():
    records = [
        candidate(
            "s",
            "direct",
            1,
            a2_score=1.0,
            structural=0.0,
            direct=True,
        ),
        candidate("s", "a", 2, a2_score=0.9, structural=0.9),
        candidate("s", "b", 3, a2_score=0.8, structural=0.8),
        candidate("s", "c", 4, a2_score=0.7, structural=0.1),
    ]
    selected, _scored, diagnostics = apply_policy(
        records,
        policy(query_keep_fraction=0.75),
    )
    assert len(selected) >= 3
    assert any(row["object"] == "direct" for row in selected)
    assert diagnostics["dropped_count"] <= 1


def test_exact_a2_control_matches_requested_candidate_count():
    records = [
        candidate(
            "s",
            f"o{index}",
            index,
            a2_score=1.0 - index * 0.1,
            structural=0.5,
        )
        for index in range(1, 6)
    ]
    selected = exact_a2_budget(records, 3)
    assert len(selected) == 3
    assert [row["object"] for row in selected] == [
        "o1",
        "o2",
        "o3",
    ]


def hard_negative_records():
    """A2 drops a low-ranked true edge; structure drops a false middle edge."""

    return [
        candidate(
            "s",
            "true-top",
            1,
            a2_score=0.99,
            structural=0.9,
            target=True,
            silver=True,
        ),
        candidate(
            "s",
            "false-middle",
            2,
            a2_score=0.98,
            structural=0.1,
        ),
        candidate(
            "s",
            "false-bottom",
            3,
            a2_score=0.90,
            structural=0.0,
        ),
        candidate(
            "s",
            "true-low-a2",
            4,
            a2_score=0.80,
            structural=1.0,
            target=True,
            silver=True,
        ),
    ]


def test_risk_limited_selection_beats_exact_budget_a2_on_synthetic_cell():
    records = hard_negative_records()
    proposed, _scored, _diagnostics = apply_policy(
        records,
        policy(max_drop_fraction=0.25, query_keep_fraction=0.75),
    )
    control = exact_a2_budget(records, len(proposed))
    proposed_metric = evaluate_cell(proposed, records)
    control_metric = evaluate_cell(control, records)
    assert proposed_metric["recall"] > control_metric["recall"]
    assert (
        proposed_metric["silver_precision_lower_bound"]
        > control_metric["silver_precision_lower_bound"]
    )


def test_calibration_uses_worst_cell_recall_before_aggressive_compression():
    cells = {}
    for seed in (11, 17):
        cell_records = [
            {**row, "seed": seed}
            for row in hard_negative_records()
        ]
        cells[("case-1", seed, "iid40")] = cell_records
    config = {
        "policy_search": {
            "profiles": ["hybrid"],
            "max_drop_fractions": [0.25, 0.5],
            "query_keep_fractions": [0.5, 0.75],
            "minimum_keep": [1],
            "structural_weights": [0.5],
            "pareto_margin_min": [0.0],
            "structural_spread_min": [0.0],
        },
        "calibration_gate": {
            "recall_macro_min": 0.95,
            "recall_each_cell_min": 0.95,
            "mrr_noninferiority_tolerance": 0.0,
            "selected_count_ratio_max": 0.95,
            "matched_budget_recall_tolerance": 0.0,
            "matched_budget_p_lb_delta_min": 0.0,
            "matched_budget_mrr_delta_min": 0.0,
        },
    }
    chosen, grid = select_policy(cells, config)
    assert any(row["feasible"] for row in grid)
    assert chosen.max_drop_fraction in {0.25, 0.5}
    assert sum(row["selected"] for row in grid) == 1
    selected = next(row for row in grid if row["selected"])
    assert selected["recall_min"] == 1.0
    assert selected["matched_budget_additive_gain"] is True
