from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from logagent_benchmark.graph import (  # noqa: E402
    CANONICAL_TRACE_COLUMNS,
    build_heldout_silver_graph,
    deterministic_trace_split,
    edge_key_set,
    extract_exact_parent_calls,
)
from logagent_benchmark.masking import (  # noqa: E402
    ModelMaskBundle,
    StructuralMaskResult,
    assert_mask_is_leakage_free,
    make_component_blackout,
    make_iid_mask,
)


REVISION = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
INCIDENT_ID = "rcaeval_tt_smoke_0001"
INJECT_TIME_US = 5 * 60_000_000
SYSTEM_PREFIX = "rcaeval:train-ticket:service:"


def synthetic_traces(trace_count: int = 100) -> pd.DataFrame:
    """Five stable service edges, each observed before and after injection."""

    rows: list[dict[str, object]] = []
    callees = ("svc-b", "svc-c", "svc-d", "svc-e", "svc-f")
    for index in range(trace_count):
        trace_id = f"trace-{index:04d}"
        minute = index % 10
        root_start = minute * 60_000_000 + index * 100
        root_span = f"span-{index:04d}-root"
        rows.append(
            {
                "time": f"00:{minute:02d}",
                "traceID": trace_id,
                "spanID": root_span,
                "serviceName": "svc-a",
                "methodName": None,
                "operationName": "root",
                "parentSpanID": None,
                "startTimeMillis": root_start // 1_000,
                "startTime": root_start,
                "duration": 50_000_000,
                "statusCode": None,
            }
        )
        for offset, service in enumerate(callees, start=1):
            child_start = root_start + offset * 10_000
            rows.append(
                {
                    "time": f"00:{minute:02d}",
                    "traceID": trace_id,
                    "spanID": f"span-{index:04d}-{offset}",
                    "serviceName": service,
                    "methodName": None,
                    "operationName": f"GET /{service}",
                    "parentSpanID": root_span,
                    "startTimeMillis": child_start // 1_000,
                    "startTime": child_start,
                    "duration": 1_000,
                    "statusCode": None,
                }
            )
    return pd.DataFrame(rows)


def canonical_synthetic_traces(trace_count: int = 100) -> pd.DataFrame:
    traces = synthetic_traces(trace_count).rename(
        columns={
            "traceID": "trace_id",
            "spanID": "span_id",
            "parentSpanID": "parent_span_id",
            "serviceName": "service_id",
            "startTime": "start_time_us",
            "duration": "duration_us",
        }
    )
    traces["service_id"] = traces["service_id"].map(
        lambda value: f"canonical/train-ticket/{value}"
    )
    return traces


class SilverGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.traces = synthetic_traces()
        cls.graph = build_heldout_silver_graph(
            cls.traces,
            revision=REVISION,
            incident_id=INCIDENT_ID,
            inject_time_us=INJECT_TIME_US,
        )

    def test_whole_trace_split_is_deterministic_and_disjoint(self) -> None:
        first = deterministic_trace_split(
            self.traces,
            revision=REVISION,
            incident_id=INCIDENT_ID,
        )
        second = deterministic_trace_split(
            self.traces,
            revision=REVISION,
            incident_id=INCIDENT_ID,
        )

        self.assertEqual(first.reference_trace_ids, second.reference_trace_ids)
        self.assertEqual(first.model_trace_ids, second.model_trace_ids)
        self.assertFalse(first.reference_trace_ids & first.model_trace_ids)
        self.assertEqual(
            first.reference_trace_ids | first.model_trace_ids,
            set(self.traces["traceID"].unique()),
        )

        reference_fraction = len(first.reference_trace_ids) / self.traces["traceID"].nunique()
        self.assertLess(abs(reference_fraction - 0.40), 0.15)
        for trace_id, group in self.traces.groupby("traceID"):
            expected_rows = len(group)
            observed_rows = int((first.reference["traceID"] == trace_id).sum()) + int(
                (first.model["traceID"] == trace_id).sum()
            )
            self.assertEqual(observed_rows, expected_rows)
            self.assertFalse(
                (first.reference["traceID"] == trace_id).any()
                and (first.model["traceID"] == trace_id).any()
            )

    def test_exact_parent_calls_are_aggregated_and_attested(self) -> None:
        graph = self.graph
        self.assertEqual(graph.reference_stats.nonroot_parent_coverage, 1.0)
        self.assertEqual(graph.model_stats.nonroot_parent_coverage, 1.0)
        self.assertEqual(len(graph.reference_edges), 5)
        self.assertEqual(len(graph.observed_edges), 5)
        self.assertEqual(edge_key_set(graph.reference_edges), edge_key_set(graph.observed_edges))
        self.assertEqual(set(graph.reference_edges["attestation"]), {"A"})
        self.assertEqual(set(graph.reference_edges["visibility"]), {"evaluator_only"})
        self.assertEqual(set(graph.observed_edges["visibility"]), {"model_input"})
        self.assertTrue((graph.reference_edges["pre_injection_count"] > 0).all())
        self.assertTrue((graph.reference_edges["post_injection_count"] > 0).all())
        self.assertTrue((graph.reference_edges["confidence"] >= 0.95).all())

        expected = {
            (f"{SYSTEM_PREFIX}svc-a", "CALLS", f"{SYSTEM_PREFIX}{callee}")
            for callee in ("svc-b", "svc-c", "svc-d", "svc-e", "svc-f")
        }
        self.assertEqual(edge_key_set(graph.reference_edges), expected)

    def test_iid_masks_have_exact_counts_and_hide_boundaries(self) -> None:
        expected_counts = {0.20: 1, 0.40: 2, 0.60: 3}
        all_span_ids = set(self.graph.trace_split.model["spanID"].astype(str))

        for fraction, expected_count in expected_counts.items():
            with self.subTest(fraction=fraction):
                result = make_iid_mask(self.graph, fraction=fraction, seed=17)
                manifest = result.evaluator_manifest
                self.assertEqual(manifest.target_count, expected_count)
                self.assertEqual(manifest.visibility, "evaluator_only")
                self.assertGreater(manifest.redacted_boundary_spans, 0)

                targets = set(manifest.target_edges)
                self.assertFalse(targets & edge_key_set(result.model.observed_edges))
                derived = extract_exact_parent_calls(result.model.traces)
                derived_keys = edge_key_set(
                    derived.occurrences[
                        ["subject", "predicate", "object"]
                    ].drop_duplicates()
                )
                self.assertFalse(targets & derived_keys)

                masked_parents = result.model.traces["parentSpanID"].dropna().astype(str)
                opaque = set(masked_parents).difference(all_span_ids)
                self.assertTrue(opaque)
                self.assertFalse(opaque & all_span_ids)
                self.assertTrue(all(re.fullmatch(r"[0-9a-f]{16}", value) for value in opaque))
                self.assertFalse(any("MASK" in value.upper() for value in opaque))

        first = make_iid_mask(self.graph, fraction=0.40, seed=9)
        second = make_iid_mask(self.graph, fraction=0.40, seed=9)
        self.assertEqual(
            first.evaluator_manifest.target_edges,
            second.evaluator_manifest.target_edges,
        )
        self.assertEqual(
            list(first.model.traces["parentSpanID"].astype("string")),
            list(second.model.traces["parentSpanID"].astype("string")),
        )

    def test_component_blackout_removes_every_incident_edge(self) -> None:
        component = f"{SYSTEM_PREFIX}svc-a"
        result = make_component_blackout(
            self.graph,
            component_id=component,
            seed=23,
        )
        manifest = result.evaluator_manifest

        self.assertEqual(manifest.policy, "component_blackout")
        self.assertEqual(manifest.component_id, component)
        self.assertEqual(manifest.target_count, 5)
        self.assertTrue(result.model.observed_edges.empty)
        self.assertEqual(set(result.model.traces["serviceName"]), set(self.traces["serviceName"]))

        derived = extract_exact_parent_calls(result.model.traces)
        self.assertTrue(derived.occurrences.empty)

    def test_leakage_assertion_rejects_reference_trace_in_model_bundle(self) -> None:
        result = make_iid_mask(self.graph, fraction=0.20, seed=5)
        leaked_row = self.graph.trace_split.reference.iloc[[0]].copy()
        tampered_model = replace(
            result.model,
            traces=pd.concat([result.model.traces, leaked_row], ignore_index=True),
        )
        tampered_result = StructuralMaskResult(
            model=tampered_model,
            evaluator_manifest=result.evaluator_manifest,
        )
        with self.assertRaisesRegex(AssertionError, "reference traces leaked"):
            assert_mask_is_leakage_free(
                tampered_result,
                reference_trace_ids=self.graph.trace_split.reference_trace_ids,
            )

    def test_leakage_assertion_rejects_target_in_observed_graph(self) -> None:
        result = make_iid_mask(self.graph, fraction=0.20, seed=6)
        target = result.evaluator_manifest.target_edges[0]
        target_row = self.graph.observed_edges.loc[
            [
                tuple(row) == target
                for row in self.graph.observed_edges[
                    ["subject", "predicate", "object"]
                ].itertuples(index=False, name=None)
            ]
        ].iloc[[0]]
        tampered_edges = pd.concat(
            [result.model.observed_edges, target_row],
            ignore_index=True,
        )
        tampered_result = StructuralMaskResult(
            model=ModelMaskBundle(
                traces=result.model.traces,
                observed_edges=tampered_edges,
            ),
            evaluator_manifest=result.evaluator_manifest,
        )
        with self.assertRaisesRegex(AssertionError, "target edges leaked"):
            assert_mask_is_leakage_free(
                tampered_result,
                reference_trace_ids=self.graph.trace_split.reference_trace_ids,
            )

    def test_duplicate_trace_span_key_is_rejected(self) -> None:
        duplicate = pd.concat([self.traces, self.traces.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            extract_exact_parent_calls(duplicate)

    def test_canonical_trace_schema_preserves_existing_service_ids(self) -> None:
        canonical = canonical_synthetic_traces()
        graph = build_heldout_silver_graph(
            canonical,
            revision=REVISION,
            incident_id=INCIDENT_ID,
            inject_time_us=INJECT_TIME_US,
            columns=CANONICAL_TRACE_COLUMNS,
        )
        expected_subject = "canonical/train-ticket/svc-a"
        expected_objects = {
            f"canonical/train-ticket/{value}"
            for value in ("svc-b", "svc-c", "svc-d", "svc-e", "svc-f")
        }
        self.assertEqual(set(graph.reference_edges["subject"]), {expected_subject})
        self.assertEqual(set(graph.reference_edges["object"]), expected_objects)
        self.assertFalse(
            graph.reference_edges["subject"].str.startswith("rcaeval:").any()
        )

        masked = make_iid_mask(
            graph,
            fraction=0.20,
            seed=31,
        )
        self.assertFalse(
            set(masked.evaluator_manifest.target_edges)
            & edge_key_set(masked.model.observed_edges)
        )
        span_ids = set(masked.model.traces["span_id"].astype(str))
        unmatched = set(
            masked.model.traces["parent_span_id"].dropna().astype(str)
        ).difference(span_ids)
        self.assertTrue(unmatched)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{16}", value) for value in unmatched))


if __name__ == "__main__":
    unittest.main()
