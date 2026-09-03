from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from logagent_benchmark.task_a_phase3_r3_channel_v2 import (
    EVALUATOR_COLUMNS,
    GateConfig,
    Phase3R3Error,
    R3Policy,
    TriStateConfig,
    _channel_evidence,
    _control,
    apply_policy,
    classify_channel,
    evaluate_cell,
    score_channel_nli,
)


class DummyBackend:
    max_length = 512

    def availability(self):
        return SimpleNamespace(status="READY", reason_code=None, detail=None)

    def pair_token_lengths(self, pairs):
        return tuple(32 for _ in pairs)

    def score_pairs(self, pairs):
        output = []
        for _premise, hypothesis in pairs:
            if "source directly calls service target" in hypothesis:
                output.append(
                    {"entailment": 0.72, "contradiction": 0.08, "neutral": 0.20}
                )
            else:
                output.append(
                    {"entailment": 0.12, "contradiction": 0.58, "neutral": 0.30}
                )
        return tuple(output)

    def metadata(self):
        return {
            "research_valid": True,
            "batch_size": 1,
            "model_id": "dummy",
        }

    def cache_info(self):
        return SimpleNamespace(hits=0, misses=0, size=0, inference_batches=0)


def candidate(object_id: str, rank: int, *, target=False, silver=False, direct=False):
    return {
        "incident_token": "incident",
        "seed": 11,
        "mask_id": "iid20",
        "mask_ratio": 0.2,
        "subject": "source",
        "predicate": "CALLS",
        "object": object_id,
        "a2_score": 1.0 - rank * 0.05,
        "proposal_rank": rank,
        "supporting_traces": 10 - rank,
        "boundary_spans": 20 - rank,
        "reverse_supporting_traces": 0,
        "reverse_boundary_spans": 0,
        "direct_evidence": direct,
        "boundary_alignment": 1.0,
        "direction_score": 0.9 if object_id == "target" else 0.4,
        "reconstructed_boundary_pairs": 5,
        "operation_jaccard_mean": 0.9 if object_id == "target" else 0.1,
        "operation_pair_concentration": 0.8 if object_id == "target" else 0.2,
        "operation_role_score": 0.9 if object_id == "target" else 0.3,
        "endpoint_compatibility_score": 0.9 if object_id == "target" else 0.2,
        "method_coverage": 1.0,
        "method_match_rate": 1.0 if object_id == "target" else 0.0,
        "route_coverage": 1.0,
        "route_exact_rate": 1.0 if object_id == "target" else 0.0,
        "route_jaccard_mean": 1.0 if object_id == "target" else 0.0,
        "graph_role_score": 0.9 if object_id == "target" else 0.2,
        "source_out_degree": 3,
        "source_in_degree": 1,
        "target_out_degree": 1,
        "target_in_degree": 4,
        "span_kind_coverage": 0.0,
        "span_kind_compatibility_score": 0.5,
        "workload_coverage": 0.0,
        "workload_match_score": 0.5,
        "a2_rank_normalized": 1.0 - rank * 0.1,
        "combined_rank_normalized": 1.0 if object_id == "target" else 0.3,
        "is_masked_target": target,
        "is_silver_matched": silver,
        "case": "case",
        "fault": "cpu",
        "role": "calibration",
    }


def test_http_channel_abstains_when_attributes_are_missing():
    row = candidate("target", 1)
    row["method_coverage"] = 0.0
    row["route_coverage"] = 0.0
    evidence = _channel_evidence(row)
    assert evidence["http"].available is False
    assert "not imputed" in evidence["http"].premise


def test_classify_channel_emits_corroboration_without_hard_veto():
    result = classify_channel(
        {"entailment": 0.8, "contradiction": 0.05, "neutral": 0.15},
        {"entailment": 0.1, "contradiction": 0.7, "neutral": 0.2},
        TriStateConfig(),
    )
    assert result["state"] == "corroborates"
    assert result["score"] > 0.0


def test_nli_scoring_rejects_evaluator_columns():
    frame = pd.DataFrame([candidate("target", 1)]).drop(
        columns=["is_masked_target", "is_silver_matched", "case", "fault"]
    )
    with pytest.raises(Phase3R3Error, match="evaluator columns"):
        score_channel_nli(
            frame,
            backend=DummyBackend(),
            tri_state=TriStateConfig(),
            channel_weights={"trace": 1, "operation": 1, "http": 1, "role": 1},
        )


def test_channel_scoring_is_complete_and_has_variance():
    raw = pd.DataFrame(
        [candidate("target", 1), candidate("other", 2)]
    ).drop(columns=sorted(EVALUATOR_COLUMNS))
    scored, diagnostics = score_channel_nli(
        raw,
        backend=DummyBackend(),
        tri_state=TriStateConfig(),
        channel_weights={"trace": 0.35, "operation": 0.3, "http": 0.2, "role": 0.15},
    )
    assert len(scored) == 2
    assert diagnostics["candidate_coverage"] == 1.0
    assert diagnostics["pair_count"] == 16
    assert "nli_rank_normalized" in scored


def test_policy_preserves_direct_evidence_and_exact_size_controls():
    frame = pd.DataFrame(
        [
            candidate("direct", 1, direct=True),
            candidate("target", 2, target=True, silver=True),
            candidate("false", 3),
        ]
    )
    frame["nli_rank_normalized"] = [0.2, 1.0, 0.0]
    policy = R3Policy(0.67, 2, 0.2, 0.3)
    proposed = apply_policy(frame, policy)
    assert proposed.loc[proposed["object"].eq("direct"), "selected"].item()
    metric = evaluate_cell(proposed)
    assert metric["selected_count"] == 3
    control = _control(frame, metric["selected_count"], kind="a2", policy=policy)
    assert evaluate_cell(control)["selected_count"] == metric["selected_count"]


def test_channel_nli_can_improve_equal_size_ranking_on_synthetic_cell():
    frame = pd.DataFrame(
        [
            candidate("false", 1),
            candidate("target", 2, target=True, silver=True),
            candidate("other", 3),
        ]
    )
    frame["nli_rank_normalized"] = [0.0, 1.0, 0.5]
    frame["combined_rank_normalized"] = frame["a2_rank_normalized"]
    policy = R3Policy(0.01, 1, 0.0, 0.7)
    proposed = apply_policy(frame, policy)
    proposed_metric = evaluate_cell(proposed)
    control = _control(
        frame, proposed_metric["selected_count"], kind="a2", policy=policy
    )
    control_metric = evaluate_cell(control)
    assert proposed_metric["recall"] > control_metric["recall"]
    assert proposed_metric["silver_precision_lower_bound"] > control_metric[
        "silver_precision_lower_bound"
    ]


def test_policy_requires_nli_and_keeps_a2_prior():
    with pytest.raises(ValueError):
        R3Policy(0.9, 8, 0.2, 0.0)
    with pytest.raises(ValueError):
        R3Policy(0.9, 8, 0.6, 0.4)
    assert GateConfig().nli_additive_gain_min > 0.0
