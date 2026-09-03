from __future__ import annotations

import pandas as pd

from logagent_benchmark.task_a_phase3_r2 import (
    OperationalPolicy,
    _candidate_feature_rows,
    _canonical_trace_frame,
    _http_method,
    _normalize_route,
    _operation_tokens,
    add_profile_scores,
    apply_policy,
    evaluate_cell,
)


def _trace_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trace_id": "trace-1",
                "span_id": "p1",
                "parent_span_id": None,
                "service_id": "svc-a",
                "operation_name": "HTTP GET http://gateway/api/orders/123",
                "method_name": "GET",
                "start_time_us": 0,
                "end_time_us": 100,
                "duration_us": 100,
            },
            {
                "trace_id": "trace-1",
                "span_id": "c1",
                "parent_span_id": None,
                "service_id": "svc-b",
                "operation_name": "GET /api/orders/{id}",
                "method_name": "GET",
                "start_time_us": 10,
                "end_time_us": 90,
                "duration_us": 80,
            },
            {
                "trace_id": "trace-2",
                "span_id": "p2",
                "parent_span_id": None,
                "service_id": "svc-a",
                "operation_name": "POST /api/payment",
                "method_name": "POST",
                "start_time_us": 0,
                "end_time_us": 100,
                "duration_us": 100,
            },
            {
                "trace_id": "trace-2",
                "span_id": "c2",
                "parent_span_id": None,
                "service_id": "svc-c",
                "operation_name": "GET /health",
                "method_name": "GET",
                "start_time_us": 10,
                "end_time_us": 90,
                "duration_us": 80,
            },
        ]
    )


def _candidates():
    return [
        {
            "subject": "svc-a",
            "predicate": "CALLS",
            "object": "svc-b",
            "a2_score": 0.90,
            "proposal_rank": 1,
            "supporting_traces": 1,
            "boundary_spans": 1,
            "reverse_supporting_traces": 0,
            "reverse_boundary_spans": 0,
            "direct_evidence": False,
            "evidence_ids": ("p1", "c1"),
        },
        {
            "subject": "svc-a",
            "predicate": "CALLS",
            "object": "svc-c",
            "a2_score": 0.89,
            "proposal_rank": 2,
            "supporting_traces": 1,
            "boundary_spans": 1,
            "reverse_supporting_traces": 0,
            "reverse_boundary_spans": 0,
            "direct_evidence": False,
            "evidence_ids": ("p2", "c2"),
        },
    ]


def test_http_and_route_parser_normalize_dynamic_ids():
    assert _http_method("HTTP GET http://host/api/orders/123", "") == "GET"
    assert _normalize_route("HTTP GET http://host/api/orders/123", "") == (
        "/api/orders/{id}"
    )
    assert "orders" in _operation_tokens("GET /api/orders/123", "findOrders")


def test_canonical_trace_frame_reports_missing_direct_otel_fields():
    traces, availability = _canonical_trace_frame(_trace_frame())
    assert len(traces) == 4
    assert availability["operation_name"] is True
    assert availability["method_name"] is True
    assert availability["span_kind"] is False
    assert availability["http_route"] is False
    assert availability["source_workload"] is False


def test_operational_features_distinguish_matching_endpoint_from_health_call():
    traces, _ = _canonical_trace_frame(_trace_frame())
    observed = pd.DataFrame(
        [{"subject": "svc-a", "predicate": "CALLS", "object": "svc-b"}]
    )
    features, diagnostics = _candidate_feature_rows(
        _candidates(), traces, observed
    )
    features = add_profile_scores(features)
    by_object = features.set_index("object")
    assert diagnostics["boundary_alignment_macro"] == 1.0
    assert diagnostics["operation_pair_candidate_coverage"] == 1.0
    assert by_object.loc["svc-b", "method_match_rate"] == 1.0
    assert by_object.loc["svc-b", "route_jaccard_mean"] > (
        by_object.loc["svc-c", "route_jaccard_mean"]
    )
    assert by_object.loc["svc-b", "combined_raw"] > (
        by_object.loc["svc-c", "combined_raw"]
    )


def test_operational_policy_selects_supported_candidate():
    traces, _ = _canonical_trace_frame(_trace_frame())
    features, _ = _candidate_feature_rows(
        _candidates(),
        traces,
        pd.DataFrame(columns=["subject", "predicate", "object"]),
    )
    features = add_profile_scores(features)
    features["is_masked_target"] = [True, False]
    features["is_silver_matched"] = [True, False]
    scored = apply_policy(
        features,
        OperationalPolicy(
            profile="combined",
            retention_fraction=0.50,
            minimum_keep=1,
            evidence_weight=0.70,
        ),
    )
    assert scored.loc[scored["object"].eq("svc-b"), "selected"].item()
    assert not scored.loc[scored["object"].eq("svc-c"), "selected"].item()
    metric = evaluate_cell(scored)
    assert metric["recall"] == 1.0
    assert metric["silver_precision_lower_bound"] == 1.0


def test_direct_evidence_is_never_removed_by_shortlist():
    traces, _ = _canonical_trace_frame(_trace_frame())
    candidates = _candidates()
    candidates[1]["direct_evidence"] = True
    features, _ = _candidate_feature_rows(
        candidates,
        traces,
        pd.DataFrame(columns=["subject", "predicate", "object"]),
    )
    features = add_profile_scores(features)
    scored = apply_policy(
        features,
        OperationalPolicy("combined", 0.50, 1, 1.0),
    )
    assert scored.loc[scored["object"].eq("svc-c"), "selected"].item()
