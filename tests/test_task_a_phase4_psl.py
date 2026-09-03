from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from logagent_benchmark.psl_multi_evidence import (
    FORBIDDEN_EVALUATOR_COLUMNS,
    MODEL_EVIDENCE_COLUMNS,
    PslMultiEvidenceBackendV1,
    PslMultiEvidenceError,
    PslRuleWeights,
    validate_model_evidence,
)
from logagent_benchmark.task_a_phase4_psl import (
    Phase4PslError,
    ShortlistPolicy,
    _permute_evidence,
    apply_shortlist,
    build_model_evidence,
    evaluate_cell,
    validate_source_frame,
)


HAS_PSLPYTHON = importlib.util.find_spec("pslpython") is not None


def source_row(
    *,
    incident: str,
    case: str,
    role: str,
    seed: int,
    mask: str,
    subject: str,
    obj: str,
    rank: int,
    target: bool = False,
    silver: bool = False,
    direct: bool = False,
    strong: bool = False,
) -> dict[str, object]:
    return {
        "incident_token": incident,
        "seed": seed,
        "mask_id": mask,
        "mask_ratio": 0.2,
        "subject": subject,
        "predicate": "CALLS",
        "object": obj,
        "a2_score": 0.95 - 0.05 * rank,
        "a2_rank_normalized": 1.0 - 0.1 * rank,
        "proposal_rank": rank,
        "supporting_traces": 8 if strong else 1,
        "boundary_spans": 16 if strong else 1,
        "reverse_supporting_traces": 0 if strong else 4,
        "reverse_boundary_spans": 0 if strong else 8,
        "direct_evidence": direct,
        "boundary_alignment": 1.0,
        "direction_score": 0.95 if strong else 0.15,
        "operation_role_score": 0.9 if strong else 0.1,
        "operation_pair_concentration": 0.85 if strong else 0.1,
        "method_coverage": 1.0,
        "method_match_rate": 1.0 if strong else 0.0,
        "route_coverage": 1.0,
        "route_exact_rate": 1.0 if strong else 0.0,
        "route_jaccard_mean": 1.0 if strong else 0.0,
        "operation_jaccard_mean": 0.9 if strong else 0.05,
        "endpoint_compatibility_score": 0.9 if strong else 0.1,
        "graph_role_score": 0.9 if strong else 0.1,
        "case": case,
        "fault": "cpu" if role == "calibration" else "delay",
        "role": role,
        "is_masked_target": target,
        "is_silver_matched": silver,
    }


def synthetic_source() -> pd.DataFrame:
    rows = []
    for role, incident, case in (
        ("calibration", "inc-cal", "case-cal"),
        ("heldout", "inc-held", "case-held"),
    ):
        rows.extend(
            [
                source_row(
                    incident=incident,
                    case=case,
                    role=role,
                    seed=11,
                    mask="iid20",
                    subject="a",
                    obj="target",
                    rank=2,
                    target=True,
                    silver=True,
                    strong=True,
                ),
                source_row(
                    incident=incident,
                    case=case,
                    role=role,
                    seed=11,
                    mask="iid20",
                    subject="a",
                    obj="false",
                    rank=1,
                ),
                source_row(
                    incident=incident,
                    case=case,
                    role=role,
                    seed=11,
                    mask="iid20",
                    subject="x",
                    obj="observed",
                    rank=3,
                    direct=True,
                    silver=True,
                    strong=True,
                ),
            ]
        )
    return pd.DataFrame(rows)


def source_config(frame: pd.DataFrame) -> dict[str, object]:
    calibration = frame.loc[frame["role"].eq("calibration")]
    heldout = frame.loc[frame["role"].eq("heldout")]
    return {
        "source_contract": {
            "candidate_rows": len(frame),
            "candidate_cells": frame.groupby(
                ["incident_token", "seed", "mask_id"]
            ).ngroups,
            "incidents": frame["case"].nunique(),
            "calibration_cells": calibration.groupby(
                ["incident_token", "seed", "mask_id"]
            ).ngroups,
            "heldout_cells": heldout.groupby(
                ["incident_token", "seed", "mask_id"]
            ).ngroups,
        }
    }


def test_source_contract_and_feature_builder_strip_evaluator_columns() -> None:
    source = synthetic_source()
    validate_source_frame(source, source_config(source))
    evidence, diagnostics = build_model_evidence(source)

    assert len(evidence) == len(source)
    assert diagnostics["evaluator_columns_removed_before_psl"] is True
    assert not FORBIDDEN_EVALUATOR_COLUMNS.intersection(evidence.columns)
    assert set(MODEL_EVIDENCE_COLUMNS).issubset(evidence.columns)
    assert evidence[list(MODEL_EVIDENCE_COLUMNS)].min().min() >= 0.0
    assert evidence[list(MODEL_EVIDENCE_COLUMNS)].max().max() <= 1.0

    target = evidence.loc[evidence["object"].eq("target")].iloc[0]
    false = evidence.loc[evidence["object"].eq("false")].iloc[0]
    assert target["direction_support"] > false["direction_support"]
    assert target["operation_match"] > false["operation_match"]
    assert target["reverse_support"] < false["reverse_support"]


def test_source_contract_rejects_duplicate_and_wrong_counts() -> None:
    source = synthetic_source()
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    config = source_config(duplicate)
    with pytest.raises(Phase4PslError, match="duplicate"):
        validate_source_frame(duplicate, config)

    config = source_config(source)
    config["source_contract"]["candidate_rows"] = 999
    with pytest.raises(Phase4PslError, match="row count"):
        validate_source_frame(source, config)


def test_psl_input_rejects_evaluator_columns_and_invalid_truth() -> None:
    source = synthetic_source()
    evidence, _ = build_model_evidence(source)
    contaminated = evidence.copy()
    contaminated["is_masked_target"] = False
    with pytest.raises(PslMultiEvidenceError, match="evaluator columns"):
        validate_model_evidence(contaminated)

    invalid = evidence.copy()
    invalid.loc[0, "a2_prior"] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        validate_model_evidence(invalid)


def test_shortlist_preserves_direct_evidence_and_evaluates_target() -> None:
    source = synthetic_source().loc[lambda frame: frame["role"].eq("heldout")].copy()
    source["psl_score"] = [0.95, 0.05, 0.10]
    scored = apply_shortlist(
        source,
        ShortlistPolicy("test", retention_fraction=0.34, minimum_keep=1),
        score_column="psl_score",
    )
    selected_objects = set(scored.loc[scored["selected"], "object"])
    assert selected_objects == {"target", "observed"}

    metric = evaluate_cell(scored)
    assert metric["recovered_target_count"] == 1
    assert metric["recall"] == 1.0
    assert metric["silver_precision_lower_bound"] == 1.0


def test_permutation_breaks_nonprior_evidence_but_preserves_prior() -> None:
    source = synthetic_source()
    evidence, _ = build_model_evidence(source)
    permuted = _permute_evidence(evidence)

    before = evidence.sort_values(["cell_id", "subject", "object"]).reset_index(drop=True)
    after = permuted.sort_values(["cell_id", "subject", "object"]).reset_index(drop=True)
    assert before["a2_prior"].equals(after["a2_prior"])
    assert not before["operation_match"].equals(after["operation_match"])


def test_rule_weights_are_complete_and_nonnegative() -> None:
    weights = PslRuleWeights()
    assert weights.a2_prior > 0
    assert weights.sparsity > 0
    with pytest.raises(ValueError):
        PslRuleWeights(a2_prior=-1.0)


@pytest.mark.skipif(not HAS_PSLPYTHON, reason="official pslpython is not installed")
def test_real_multi_evidence_psl_separates_supported_and_conflicted_edges() -> None:
    source = synthetic_source().loc[lambda frame: frame["role"].eq("heldout")].copy()
    evidence, _ = build_model_evidence(source)
    with tempfile.TemporaryDirectory(prefix="psl-v1-test-") as parent:
        backend = PslMultiEvidenceBackendV1(
            profile_id="integration",
            temporary_parent=parent,
            random_seed=7,
        )
        result = backend.infer(evidence)
        assert list(Path(parent).iterdir()) == []

    by_object = {key[3]: value for key, value in result.scores.items()}
    assert set(by_object) == {"target", "false", "observed"}
    assert by_object["target"] > by_object["false"]
    assert result.metadata["candidate_count"] == 3
    assert result.metadata["cell_count"] == 1
    assert result.metadata["temporary_data_cleaned"] is True
    assert result.grounded_rule_count > 0
