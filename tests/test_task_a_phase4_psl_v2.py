from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from logagent_benchmark.psl_direct_evidence_v2 import (
    DIRECT_EVIDENCE_COLUMNS,
    PslDirectEvidenceBackendV2,
    PslDirectEvidenceError,
    PslDirectRuleWeights,
    validate_direct_evidence,
)
from logagent_benchmark.psl_multi_evidence import (
    FORBIDDEN_EVALUATOR_COLUMNS,
)
from logagent_benchmark.task_a_phase4_psl_v2 import (
    DirectEvidenceAliases,
    DirectEvidencePolicy,
    Phase4PslV2Error,
    apply_abstention_policy,
    build_direct_evidence,
    evaluate_cell,
    run_phase4_psl_v2,
    validate_source_frame,
    weak_evidence_invariance,
)


HAS_PSLPYTHON = importlib.util.find_spec("pslpython") is not None
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


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
    direct_trace: float = 0.0,
    client_server: float = 0.0,
    workload: float = 0.0,
    weak_strength: float = 0.0,
) -> dict[str, object]:
    strong = float(weak_strength)
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
        "supporting_traces": 100 * strong,
        "boundary_spans": 200 * strong,
        "reverse_supporting_traces": 0,
        "reverse_boundary_spans": 0,
        "direct_evidence": direct_trace,
        "direct_trace_evidence": direct_trace,
        "client_server_evidence": client_server,
        "workload_evidence": workload,
        "boundary_alignment": strong,
        "direction_score": strong,
        "operation_role_score": strong,
        "operation_pair_concentration": strong,
        "method_coverage": strong,
        "method_match_rate": strong,
        "route_coverage": strong,
        "route_exact_rate": strong,
        "route_jaccard_mean": strong,
        "operation_jaccard_mean": strong,
        "endpoint_compatibility_score": strong,
        "graph_role_score": strong,
        "case": case,
        "fault": "cpu" if role == "calibration" else "delay",
        "role": role,
        "is_masked_target": target,
        "is_silver_matched": silver,
    }


def synthetic_source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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
                    subject="web",
                    obj="api",
                    rank=2,
                    target=True,
                    silver=True,
                    direct_trace=1.0,
                    weak_strength=0.0,
                ),
                source_row(
                    incident=incident,
                    case=case,
                    role=role,
                    seed=11,
                    mask="iid20",
                    subject="web",
                    obj="false-weak",
                    rank=1,
                    weak_strength=1.0,
                ),
                source_row(
                    incident=incident,
                    case=case,
                    role=role,
                    seed=11,
                    mask="iid20",
                    subject="api",
                    obj="db",
                    rank=3,
                    silver=True,
                    client_server=1.0,
                    workload=1.0,
                    weak_strength=0.0,
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


def full_config(frame: pd.DataFrame, source_sha: str = "") -> dict[str, object]:
    contract = source_config(frame)["source_contract"]
    contract.update(
        {
            "candidate_analysis_sha256": source_sha,
            "artifact_id": 0,
            "artifact_zip_sha256": "",
        }
    )
    return {
        "schema_version": 2,
        "experiment_id": "test-direct-evidence-v2",
        "gate_id": "TEST_DIRECT_EVIDENCE_V2",
        "source_contract": contract,
        "psl_runtime": {
            "implementation": "pslpython==2.4.0",
            "java_distribution": "temurin",
            "java_version": "17",
            "random_seed": 7,
            "jvm_options": ["-Xms128m", "-Xmx1024m"],
        },
        "direct_evidence_aliases": {
            "direct_trace": [
                "direct_trace_evidence",
                "trace_parent_child_evidence",
                "direct_evidence",
            ],
            "client_server": ["client_server_evidence"],
            "workload": ["workload_evidence"],
        },
        "direct_rule_weights": {
            "direct_trace": 10.0,
            "client_server": 12.0,
            "workload": 12.0,
            "trace_client_server": 4.0,
            "trace_workload": 4.0,
            "client_server_workload": 5.0,
            "all_direct": 8.0,
        },
        "abstention_policy": {
            "channel_truth_min": 0.9,
            "psl_score_min": 0.9,
            "minimum_direct_channels": 1,
        },
        "gate": {
            "direct_candidate_coverage_min": 0.001,
            "direct_target_coverage_min": 0.001,
            "confirmed_count_min": 1,
            "confirmed_precision_lower_bound_min": 0.9,
            "target_confirmation_recall_min": 0.5,
            "unsupported_confirmation_max": 0,
            "candidate_retention_min": 1.0,
        },
        "claim_limit": "test",
    }


def test_source_contract_and_direct_builder_strip_evaluator_columns() -> None:
    source = synthetic_source()
    validate_source_frame(source, source_config(source))
    evidence, metadata, diagnostics = build_direct_evidence(source)

    assert len(evidence) == len(source)
    assert len(metadata) == len(source)
    assert diagnostics["evaluator_columns_removed_before_psl"] is True
    assert not FORBIDDEN_EVALUATOR_COLUMNS.intersection(evidence.columns)
    assert set(DIRECT_EVIDENCE_COLUMNS).issubset(evidence.columns)

    target = evidence.loc[evidence["object"].eq("api")].iloc[0]
    weak = evidence.loc[evidence["object"].eq("false-weak")].iloc[0]
    direct_pair = evidence.loc[evidence["object"].eq("db")].iloc[0]
    assert target["direct_trace"] == 1.0
    assert weak[["direct_trace", "client_server", "workload"]].sum() == 0.0
    assert direct_pair["client_server"] == 1.0
    assert direct_pair["workload"] == 1.0
    assert diagnostics["a2_prior_used_for_confirmation"] is False
    assert diagnostics["reverse_or_direction_conflict_used"] is False


def test_weak_proxy_strength_cannot_create_direct_evidence() -> None:
    source = synthetic_source()
    evidence, _metadata, _diagnostics = build_direct_evidence(source)
    weak = evidence.loc[evidence["object"].eq("false-weak")]
    assert len(weak) == 2
    assert (
        weak[["direct_trace", "client_server", "workload"]]
        .to_numpy()
        .sum()
        == 0.0
    )
    assert weak_evidence_invariance(
        source, DirectEvidenceAliases(), evidence
    )


def test_abstention_policy_never_confirms_unsupported_candidate() -> None:
    source = synthetic_source().loc[
        lambda frame: frame["role"].eq("heldout")
    ].copy()
    _evidence, metadata, _diagnostics = build_direct_evidence(source)
    metadata["psl_score"] = [1.0, 1.0, 1.0]
    decided = apply_abstention_policy(
        metadata,
        DirectEvidencePolicy(),
        score_column="psl_score",
    )
    by_object = {
        row.object: row.decision_state
        for row in decided.itertuples(index=False)
    }
    assert by_object["api"] == "CONFIRMED"
    assert by_object["db"] == "CONFIRMED"
    assert by_object["false-weak"] == "ABSTAIN"
    assert set(decided["decision_state"]) == {"CONFIRMED", "ABSTAIN"}


def test_policy_abstains_when_direct_truth_or_psl_score_is_low() -> None:
    frame = pd.DataFrame(
        [
            {
                "a2_priority": 0.9,
                "direct_trace": 0.89,
                "client_server": 0.0,
                "workload": 0.0,
                "score": 1.0,
            },
            {
                "a2_priority": 0.8,
                "direct_trace": 1.0,
                "client_server": 0.0,
                "workload": 0.0,
                "score": 0.89,
            },
        ]
    )
    decided = apply_abstention_policy(
        frame,
        DirectEvidencePolicy(),
        score_column="score",
    )
    assert set(decided["decision_state"]) == {"ABSTAIN"}
    assert set(decided["decision_reason"]) == {
        "DIRECT_EVIDENCE_BELOW_CHANNEL_POLICY",
        "DIRECT_EVIDENCE_BELOW_PSL_POLICY",
    }


def test_evaluate_cell_reports_confirmation_and_abstention() -> None:
    source = synthetic_source().loc[
        lambda frame: frame["role"].eq("heldout")
    ].copy()
    _evidence, metadata, _diagnostics = build_direct_evidence(source)
    metadata = metadata.merge(
        source[
            [
                "incident_token",
                "seed",
                "mask_id",
                "subject",
                "predicate",
                "object",
                "is_masked_target",
                "is_silver_matched",
            ]
        ],
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
    metadata["score"] = [1.0, 1.0, 1.0]
    decided = apply_abstention_policy(
        metadata,
        DirectEvidencePolicy(),
        score_column="score",
    )
    metric = evaluate_cell(decided, DirectEvidencePolicy())
    assert metric["candidate_count"] == 3
    assert metric["confirmed_count"] == 2
    assert metric["abstained_count"] == 1
    assert metric["unsupported_confirmation_count"] == 0
    assert metric["target_confirmation_recall"] == 1.0
    assert metric["confirmed_precision_lower_bound"] == 1.0
    assert metric["no_negative_relation_state"] is True


def test_direct_evidence_validation_rejects_labels_and_invalid_truth() -> None:
    source = synthetic_source()
    evidence, _metadata, _diagnostics = build_direct_evidence(source)
    contaminated = evidence.copy()
    contaminated["is_masked_target"] = False
    with pytest.raises(PslDirectEvidenceError, match="evaluator columns"):
        validate_direct_evidence(contaminated)

    invalid = evidence.copy()
    invalid.loc[0, "direct_trace"] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        validate_direct_evidence(invalid)


def test_rule_set_has_no_negative_relation_or_a2_prior_rule() -> None:
    backend = PslDirectEvidenceBackendV2(
        weights=PslDirectRuleWeights()
    )
    rules = backend.active_rules()
    assert rules
    assert all("!" not in rule.body for rule in rules)
    assert all("A2Prior" not in rule.body for rule in rules)
    assert {rule.rule_id for rule in rules} == {
        "DIRECT_TRACE",
        "CLIENT_SERVER",
        "WORKLOAD",
        "TRACE_CLIENT_SERVER",
        "TRACE_WORKLOAD",
        "CLIENT_SERVER_WORKLOAD",
        "ALL_DIRECT",
    }


def test_source_contract_rejects_duplicates_and_wrong_counts() -> None:
    source = synthetic_source()
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(Phase4PslV2Error, match="duplicate"):
        validate_source_frame(duplicate, source_config(duplicate))

    config = source_config(source)
    config["source_contract"]["candidate_rows"] = 999
    with pytest.raises(Phase4PslV2Error, match="row count"):
        validate_source_frame(source, config)


@pytest.mark.skipif(
    not HAS_PSLPYTHON,
    reason="official pslpython is not installed",
)
def test_real_psl_v2_confirms_direct_and_abstains_unsupported() -> None:
    source = synthetic_source().loc[
        lambda frame: frame["role"].eq("heldout")
    ].copy()
    evidence, metadata, _diagnostics = build_direct_evidence(source)
    with tempfile.TemporaryDirectory(prefix="psl-v2-test-") as parent:
        backend = PslDirectEvidenceBackendV2(
            profile_id="integration",
            temporary_parent=parent,
            random_seed=7,
        )
        result = backend.infer(evidence)
        assert list(Path(parent).iterdir()) == []

    score = pd.DataFrame(
        [
            {
                "cell_id": key[0],
                "subject": key[1],
                "predicate": key[2],
                "object": key[3],
                "psl_score": value,
            }
            for key, value in result.scores.items()
        ]
    )
    merged = metadata.merge(
        score,
        on=["cell_id", "subject", "predicate", "object"],
        validate="one_to_one",
    )
    decided = apply_abstention_policy(
        merged,
        DirectEvidencePolicy(),
        score_column="psl_score",
    )
    by_object = {
        row.object: row
        for row in decided.itertuples(index=False)
    }
    assert by_object["api"].decision_state == "CONFIRMED"
    assert by_object["db"].decision_state == "CONFIRMED"
    assert by_object["false-weak"].decision_state == "ABSTAIN"
    assert by_object["api"].psl_direct_score >= 0.9
    assert by_object["db"].psl_direct_score >= 0.9
    assert result.metadata["negative_relation_rule_count"] == 0
    assert result.metadata["uses_a2_prior_for_confirmation"] is False
    assert result.metadata["candidate_count"] == 3
    assert result.metadata["temporary_data_cleaned"] is True


@pytest.mark.skipif(
    not (HAS_PSLPYTHON and HAS_PYARROW),
    reason="pslpython and pyarrow are required",
)
def test_end_to_end_v2_writes_two_state_results() -> None:
    source = synthetic_source()
    with tempfile.TemporaryDirectory(prefix="phase4-v2-e2e-") as parent:
        root = Path(parent)
        source_path = root / "candidate.parquet"
        source.to_parquet(source_path, index=False)
        import hashlib

        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        config = full_config(source, source_sha)
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(config), encoding="utf-8"
        )
        output = root / "output"
        run_phase4_psl_v2(
            candidate_analysis_path=source_path,
            config_path=config_path,
            output=output,
        )
        result = json.loads(
            (
                output
                / "published"
                / "task_a_phase4_psl_v2_results.json"
            ).read_text(encoding="utf-8")
        )
        assert result["status"] == "PASS"
        assert result["mechanism_gate"]["passed"] is True
        assert result["data_eligibility"]["eligible"] is True
        assert result["heldout"]["unsupported_confirmation_count"] == 0
        assert result["decision_semantics"]["states"] == [
            "CONFIRMED",
            "ABSTAIN",
        ]
        assert (
            output
            / "model_output"
            / "psl_v2_candidate_decisions.parquet"
        ).is_file()
