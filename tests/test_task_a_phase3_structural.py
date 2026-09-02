from __future__ import annotations

import pandas as pd

from logagent_benchmark.task_a_phase3_structural import (
    StructuralPolicy,
    apply_structural_policy,
    attach_structural_evidence,
    evaluate_structural_shortlist,
    select_structural_policy,
)


def _candidate(subject: str, obj: str, rank: int, *, score: float, boundary: int = 1):
    return {
        "subject": subject,
        "predicate": "CALLS",
        "object": obj,
        "a2_score": score,
        "supporting_traces": 1,
        "boundary_spans": boundary,
        "proposal_rank": rank,
        "direct_evidence": False,
        "a2_evidence_span_count": 2,
    }


def _scored(subject: str, obj: str, rank: int, hybrid: float, *, a2: float):
    record = _candidate(subject, obj, rank, score=a2)
    record.update(
        {
            "temporal_directness_score": hybrid,
            "directional_structure_score": hybrid,
            "operation_compatibility_score": hybrid,
            "hybrid_score": hybrid,
        }
    )
    return record


class TestStructuralEvidenceExtraction:
    def test_extracts_nearest_parent_position_and_operation_features(self):
        traces = pd.DataFrame(
            [
                {
                    "trace_id": "t1",
                    "span_id": "x",
                    "parent_span_id": None,
                    "service_id": "x",
                    "operation_name": "POST /api/orders",
                    "method_name": "createOrder",
                    "start_time_us": 0,
                    "end_time_us": 200,
                },
                {
                    "trace_id": "t1",
                    "span_id": "a",
                    "parent_span_id": None,
                    "service_id": "a",
                    "operation_name": "POST /api/orders",
                    "method_name": "createOrder",
                    "start_time_us": 10,
                    "end_time_us": 150,
                },
                {
                    "trace_id": "t1",
                    "span_id": "b",
                    "parent_span_id": "missing-parent",
                    "service_id": "b",
                    "operation_name": "POST /api/orders",
                    "method_name": "createOrder",
                    "start_time_us": 20,
                    "end_time_us": 120,
                },
            ]
        )
        candidates = [_candidate("a", "b", 1, score=0.9)]
        enriched, diagnostics = attach_structural_evidence(candidates, traces)
        assert diagnostics["boundary_mismatch_count"] == 0
        assert diagnostics["trace_mismatch_count"] == 0
        assert enriched[0]["reconstructed_boundary_spans"] == 1
        assert enriched[0]["reconstructed_supporting_traces"] == 1
        assert abs(enriched[0]["tightness_mean"] - (100 / 140)) < 1e-12
        assert abs(enriched[0]["alternative_margin_mean"] - 0.3) < 1e-12
        assert enriched[0]["operation_overlap_mean"] == 1.0
        assert enriched[0]["method_match_rate"] == 1.0
        assert enriched[0]["direction_asymmetry"] == 1.0


class TestStructuralPolicy:
    def test_structural_signal_changes_equal_size_selection(self):
        candidates = [
            _scored("s", "false-top", 1, 0.0, a2=0.99),
            _scored("s", "true-one", 2, 1.0, a2=0.98),
            _scored("s", "true-two", 3, 1.0, a2=0.97),
            _scored("s", "false-bottom", 4, 0.0, a2=0.96),
        ]
        control, _ = apply_structural_policy(
            candidates,
            StructuralPolicy("hybrid", 0.5, 2, 0.0),
        )
        enhanced, _ = apply_structural_policy(
            candidates,
            StructuralPolicy("hybrid", 0.5, 2, 0.6),
        )
        control_keys = {
            (item["subject"], item["predicate"], item["object"])
            for item in control
        }
        enhanced_keys = {
            (item["subject"], item["predicate"], item["object"])
            for item in enhanced
        }
        assert control_keys != enhanced_keys
        assert ("s", "CALLS", "true-two") in enhanced_keys

    def test_calibration_can_find_additive_structural_policy(self):
        candidates = [
            _scored("s", "false-top", 1, 0.0, a2=0.99),
            _scored("s", "true-one", 2, 1.0, a2=0.98),
            _scored("s", "true-two", 3, 1.0, a2=0.97),
            _scored("s", "false-bottom", 4, 0.0, a2=0.96),
        ]
        targets = {
            ("s", "CALLS", "true-one"),
            ("s", "CALLS", "true-two"),
        }
        cells = [
            {
                "case": f"c{index}",
                "fault": "cpu",
                "role": "calibration",
                "seed": index,
                "mask_id": "iid40",
                "mask_ratio": 0.4,
                "candidates": candidates,
                "targets": targets,
                "silver": set(targets),
                "a2_mrr": 0.75,
            }
            for index in range(2)
        ]
        policy, grid = select_structural_policy(
            cells,
            search={
                "profiles": ["hybrid"],
                "retention_fractions": [0.5],
                "minimum_keep": [2],
                "structural_weights": [0.6],
            },
            gate={
                "recall_macro_min": 0.95,
                "recall_each_cell_min": 0.9,
                "mrr_noninferiority_tolerance": 0.01,
                "matched_budget_recall_tolerance": 0.0,
                "matched_budget_p_lb_delta_min": 0.0,
                "matched_budget_mrr_delta_min": 0.0,
            },
        )
        assert policy.profile_id == "hybrid"
        assert policy.structural_weight == 0.6
        assert sum(bool(row["feasible"]) for row in grid) == 1
        assert sum(bool(row["selected"]) for row in grid) == 1

    def test_shortlist_metrics_use_structural_score(self):
        selected = [
            {
                **_scored("s", "t1", 1, 1.0, a2=0.9),
                "a3s_score": 0.9,
            },
            {
                **_scored("s", "t2", 2, 0.8, a2=0.8),
                "a3s_score": 0.8,
            },
        ]
        targets = {("s", "CALLS", "t1"), ("s", "CALLS", "t2")}
        metrics = evaluate_structural_shortlist(
            selected,
            targets=targets,
            silver=targets,
        )
        assert metrics["recall"] == 1.0
        assert metrics["silver_precision_lower_bound"] == 1.0
        assert metrics["mrr"] == 1.0
