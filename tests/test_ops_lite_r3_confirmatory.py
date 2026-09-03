from __future__ import annotations

import pandas as pd

from logagent_benchmark.ops_lite_r3_confirmatory import (
    _metric_with_true_targets,
    _paired_bootstrap,
    canonicalize_trace_frame,
)


def _raw_traces() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": "2026-01-01T00:00:00Z",
                "trace_id": "t1",
                "span_id": "s1",
                "parent_span_id": None,
                "span_name": "GET /api/order/123",
                "attr.span_kind": "CLIENT",
                "service_name": "gateway",
                "duration": 2_000_000,
                "attr.http.request.method": "GET",
            },
            {
                "time": "2026-01-01T00:00:00.000500Z",
                "trace_id": "t1",
                "span_id": "s2",
                "parent_span_id": "s1",
                "span_name": "GET /api/order/{id}",
                "attr.span_kind": "SERVER",
                "service_name": "order",
                "duration": 1_000_000,
                "attr.http.request.method": "GET",
            },
        ]
    )


def test_canonicalize_prefixes_phase_and_preserves_direct_trace_attributes():
    frame = canonicalize_trace_frame(
        _raw_traces(),
        phase="normal",
        dataset_id="rcabench-ops-lite",
        system_id="ops-lite-ts",
    )
    assert frame["trace_id"].tolist() == ["normal:t1", "normal:t1"]
    assert frame["span_id"].tolist() == ["normal:s1", "normal:s2"]
    assert pd.isna(frame.loc[0, "parent_span_id"])
    assert frame.loc[1, "parent_span_id"] == "normal:s1"
    assert frame["duration_us"].tolist() == [2000, 1000]
    assert frame["span_kind"].tolist() == ["CLIENT", "SERVER"]
    assert frame["http_method"].tolist() == ["GET", "GET"]
    assert frame.loc[0, "service_id"].endswith(":service:gateway")


def test_paired_bootstrap_is_deterministic_and_preserves_constant_delta():
    result = _paired_bootstrap(
        [0.1, 0.1, 0.1],
        samples=100,
        seed=7,
        confidence_level=0.95,
    )
    assert result["mean"] == 0.1
    assert result["lower"] == 0.1
    assert result["upper"] == 0.1


def test_true_target_metric_counts_a2_misses_as_zero_reciprocal_rank():
    scored = pd.DataFrame(
        [
            {
                "subject": "a",
                "predicate": "CALLS",
                "object": "b",
                "selected": True,
                "is_masked_target": True,
                "is_silver_matched": True,
                "a3_r3_score": 1.0,
            }
        ]
    )
    metric = _metric_with_true_targets(scored, true_target_count=2)
    assert metric["recovered_target_count"] == 1
    assert metric["recall"] == 0.5
    assert metric["mrr"] == 0.5
    assert metric["missing_from_a2_count"] == 1
