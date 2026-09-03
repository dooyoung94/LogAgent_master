from __future__ import annotations

import json

import pytest

from logagent_benchmark.task_a_phase3_r2 import Phase3R2Error
from logagent_benchmark.task_a_phase3_r2_compat import (
    decode_edge_key,
    evaluator_flags,
)


def test_decode_edge_key_accepts_mapping_and_array_forms():
    expected = ("svc-a", "CALLS", "svc-b")
    assert decode_edge_key(
        {"subject": "svc-a", "predicate": "CALLS", "object": "svc-b"},
        field_name="mapping",
    ) == expected
    assert decode_edge_key(
        ["svc-a", "CALLS", "svc-b"], field_name="array"
    ) == expected
    assert decode_edge_key(
        ("svc-a", "CALLS", "svc-b"), field_name="tuple"
    ) == expected


def test_decode_edge_key_rejects_invalid_shapes():
    with pytest.raises(Phase3R2Error):
        decode_edge_key(["svc-a", "CALLS"], field_name="short")
    with pytest.raises(Phase3R2Error):
        decode_edge_key(
            {"subject": "svc-a", "predicate": "CALLS"},
            field_name="missing-object",
        )
    with pytest.raises(Phase3R2Error):
        decode_edge_key("svc-a|CALLS|svc-b", field_name="string")


def test_evaluator_flags_reads_phase2_array_serialization(tmp_path):
    private = tmp_path / "evaluator_private"
    private.mkdir()
    (private / "mask_manifest.json").write_text(
        json.dumps(
            {
                "target_edges": [
                    ["svc-a", "CALLS", "svc-b"],
                ]
            }
        ),
        encoding="utf-8",
    )
    (private / "evaluation.json").write_text(
        json.dumps(
            {
                "A2": {
                    "silver_precision_lower_bound": {
                        "silver_matched_count": 1,
                        "unverified_edges": [
                            ["svc-a", "CALLS", "svc-c"],
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    candidates = {
        ("svc-a", "CALLS", "svc-b"),
        ("svc-a", "CALLS", "svc-c"),
    }
    flags = evaluator_flags(tmp_path, candidates).set_index("object")
    assert bool(flags.loc["svc-b", "is_masked_target"])
    assert bool(flags.loc["svc-b", "is_silver_matched"])
    assert not bool(flags.loc["svc-c", "is_masked_target"])
    assert not bool(flags.loc["svc-c", "is_silver_matched"])


def test_evaluator_flags_remains_compatible_with_mapping_serialization(tmp_path):
    private = tmp_path / "evaluator_private"
    private.mkdir()
    edge = {"subject": "svc-a", "predicate": "CALLS", "object": "svc-b"}
    (private / "mask_manifest.json").write_text(
        json.dumps({"target_edges": [edge]}), encoding="utf-8"
    )
    (private / "evaluation.json").write_text(
        json.dumps(
            {
                "A2": {
                    "silver_precision_lower_bound": {
                        "silver_matched_count": 1,
                        "unverified_edges": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    flags = evaluator_flags(tmp_path, {("svc-a", "CALLS", "svc-b")})
    assert flags["is_masked_target"].tolist() == [True]
    assert flags["is_silver_matched"].tolist() == [True]
