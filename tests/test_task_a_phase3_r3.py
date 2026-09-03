from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from logagent_benchmark.task_a_phase3_r3 import (
    Phase3R3Error,
    R3Policy,
    TriStateConfig,
    _channel_premise,
    _tri_state,
    add_nli_profile_scores,
    apply_policy,
    evaluate_cell,
    score_evidence_channels,
)


@dataclass
class _CacheInfo:
    hits: int = 0
    misses: int = 0
    size: int = 0
    inference_batches: int = 0


class FakeBackend:
    max_length = 512
    research_valid = True

    def pair_token_lengths(self, pairs):
        return tuple(32 for _ in pairs)

    def score_pairs(self, pairs):
        rows = []
        for _premise, hypothesis in pairs:
            if hypothesis.startswith("Service A"):
                rows.append(
                    {"entailment": 0.75, "contradiction": 0.10, "neutral": 0.15}
                )
            else:
                rows.append(
                    {"entailment": 0.10, "contradiction": 0.75, "neutral": 0.15}
                )
        return tuple(rows)

    def cache_info(self):
        return _CacheInfo()

    def metadata(self):
        return {"backend": "FakeBackend", "research_valid": True}


def _candidate(object_id: str, rank: int, target: bool) -> dict:
    return {
        "incident_token": "incident:opaque",
        "seed": 11,
        "mask_id": "iid20_l2_s11",
        "mask_ratio": 0.2,
        "subject": "service:source",
        "predicate": "CALLS",
        "object": object_id,
        "a2_score": 1.0 - rank * 0.1,
        "proposal_rank": rank,
        "direct_evidence": False,
        "supporting_traces": 8 if target else 2,
        "boundary_spans": 10 if target else 3,
        "reverse_supporting_traces": 0,
        "reverse_boundary_spans": 0,
        "direction_role_rank_normalized": 1.0 if target else 0.0,
        "operation_endpoint_rank_normalized": 1.0 if target else 0.0,
        "method_route_rank_normalized": 1.0 if target else 0.0,
        "combined_rank_normalized": 1.0 if target else 0.0,
        "reconstructed_boundary_pairs": 2,
        "representative_parent_operation": "GET /api/orders",
        "representative_child_operation": (
            "GET /api/orders" if target else "GET /health"
        ),
        "representative_parent_http_method": "GET",
        "representative_child_http_method": "GET",
        "representative_parent_route": "/api/orders",
        "representative_child_route": (
            "/api/orders" if target else "/health"
        ),
        "operation_jaccard_mean": 1.0 if target else 0.0,
        "operation_pair_concentration": 1.0,
        "source_operation_parent_prior": 0.9,
        "target_operation_child_prior": 0.9 if target else 0.1,
        "method_coverage": 1.0,
        "method_match_rate": 1.0,
        "route_coverage": 1.0,
        "route_jaccard_mean": 1.0 if target else 0.0,
        "route_exact_rate": 1.0 if target else 0.0,
        "source_out_degree": 4,
        "source_in_degree": 1,
        "target_out_degree": 1,
        "target_in_degree": 4,
        "graph_role_score": 0.9 if target else 0.2,
        "span_kind_coverage": 0.0,
        "span_kind_compatibility_score": 0.5,
        "workload_coverage": 0.0,
        "workload_match_score": 0.5,
        "case": "private-case",
        "fault": "cpu",
        "role": "calibration",
        "is_masked_target": target,
        "is_silver_matched": target,
    }


def test_channel_premises_are_independent_and_anonymous():
    row = _candidate("service:target", 1, True)
    premises = {
        channel: _channel_premise(channel, row)
        for channel in ("direction", "operation", "http", "role")
    }
    assert all(premises.values())
    assert len(set(premises.values())) == 4
    for premise in premises.values():
        assert "service:source" not in premise
        assert "service:target" not in premise
        assert "private-case" not in premise


def test_tri_state_uses_forward_reverse_contrast_without_hard_veto():
    state, score = _tri_state(
        {"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1},
        {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1},
        TriStateConfig(),
    )
    assert state == "corroborates"
    assert score > 0


def test_nli_scoring_rejects_evaluator_columns():
    frame = pd.DataFrame([_candidate("service:target", 1, True)])
    with pytest.raises(Phase3R3Error):
        score_evidence_channels(
            frame, backend=FakeBackend(), tri_state=TriStateConfig()
        )


def test_channel_scoring_and_linear_policy_preserve_target():
    raw = pd.DataFrame(
        [
            _candidate("service:true", 1, True),
            _candidate("service:false", 2, False),
        ]
    )
    evaluator = raw[
        [
            "incident_token",
            "seed",
            "mask_id",
            "subject",
            "predicate",
            "object",
            "case",
            "fault",
            "role",
            "is_masked_target",
            "is_silver_matched",
        ]
    ].copy()
    model = raw.drop(
        columns=["case", "fault", "role", "is_masked_target", "is_silver_matched"]
    )
    scored_model, diagnostics = score_evidence_channels(
        model, backend=FakeBackend(), tri_state=TriStateConfig()
    )
    assert diagnostics["candidate_count"] == 2
    assert diagnostics["channel_candidate_coverage"]["http"] == 1.0
    feature = add_nli_profile_scores(scored_model).merge(
        evaluator,
        on=[
            "incident_token",
            "seed",
            "mask_id",
            "subject",
            "predicate",
            "object",
        ],
        validate="one_to_one",
    )
    selected = apply_policy(
        feature,
        R3Policy(
            operational_profile="combined",
            nli_profile="all_available",
            retention_fraction=0.5,
            minimum_keep=1,
            operational_weight=0.2,
            nli_weight=0.2,
        ),
    )
    metric = evaluate_cell(selected)
    assert metric["selected_count"] == 1
    assert metric["recall"] == 1.0
    assert metric["silver_precision_lower_bound"] == 1.0


def test_policy_requires_nonzero_nli_and_preserves_a2_prior():
    with pytest.raises(ValueError):
        R3Policy("combined", "all_available", 0.9, 8, 0.2, 0.0)
    with pytest.raises(ValueError):
        R3Policy("combined", "all_available", 0.9, 8, 0.6, 0.3)
