from __future__ import annotations

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
    edge_key_set,
    extract_exact_parent_calls,
)
from logagent_benchmark.masking import (  # noqa: E402
    EVIDENCE_LEVEL_L1,
    EVIDENCE_LEVEL_L2,
    make_iid_mask,
    make_iid_parent_dropped_mask,
)
from logagent_benchmark.recovery import (  # noqa: E402
    InferenceContext,
    temporal_containment_details,
    temporal_containment_support,
)


REVISION = "masking-v2-test-revision"
INCIDENT_ID = "masking-v2-test-incident"
INJECT_TIME_US = 5 * 60_000_000


def _canonical_traces(trace_count: int = 100) -> pd.DataFrame:
    """Five repeatable service edges with evidence on both sides of injection."""

    rows: list[dict[str, object]] = []
    callees = ("svc-b", "svc-c", "svc-d", "svc-e", "svc-f")
    for index in range(trace_count):
        trace_id = f"trace-{index:04d}"
        minute = index % 10
        root_start = minute * 60_000_000 + index * 100
        root_span = f"span-{index:04d}-root"
        rows.append(
            {
                "trace_id": trace_id,
                "span_id": root_span,
                "parent_span_id": None,
                "service_id": "svc-a",
                "start_time_us": root_start,
                "duration_us": 50_000_000,
            }
        )
        for offset, service_id in enumerate(callees, start=1):
            rows.append(
                {
                    "trace_id": trace_id,
                    "span_id": f"span-{index:04d}-{offset}",
                    "parent_span_id": root_span,
                    "service_id": service_id,
                    "start_time_us": root_start + offset * 10_000,
                    "duration_us": 1_000,
                }
            )
    return pd.DataFrame(rows)


def _context(masked: object) -> InferenceContext:
    return InferenceContext(
        incident_id=INCIDENT_ID,
        entities=(),
        observed_edges=masked.model.observed_edges,
        traces=masked.model.traces,
    )


class ParentDroppedMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = build_heldout_silver_graph(
            _canonical_traces(),
            revision=REVISION,
            incident_id=INCIDENT_ID,
            inject_time_us=INJECT_TIME_US,
            columns=CANONICAL_TRACE_COLUMNS,
        )

    def test_l2_target_children_share_the_root_null_parent_surface(self) -> None:
        masked = make_iid_parent_dropped_mask(
            self.graph,
            fraction=0.20,
            seed=17,
        )
        before = self.graph.trace_split.model.reset_index(drop=True)
        after = masked.model.traces
        before_surface = before["parent_span_id"].astype("string").fillna("<NULL>")
        after_surface = after["parent_span_id"].astype("string").fillna("<NULL>")
        changed = before_surface.ne(after_surface)

        self.assertEqual(masked.evaluator_manifest.evidence_level, EVIDENCE_LEVEL_L2)
        self.assertEqual(int(changed.sum()), masked.evaluator_manifest.redacted_boundary_spans)
        self.assertTrue(before.loc[changed, "parent_span_id"].notna().all())
        self.assertTrue(after.loc[changed, "parent_span_id"].isna().all())
        self.assertTrue(before.loc[~changed & before["parent_span_id"].isna()].shape[0] > 0)
        self.assertEqual(
            int(after["parent_span_id"].isna().sum()),
            int(before["parent_span_id"].isna().sum()) + int(changed.sum()),
        )

        target_objects = {edge[2] for edge in masked.evaluator_manifest.target_edges}
        self.assertEqual(set(after.loc[changed, "service_id"].astype(str)), target_objects)

    def test_l2_target_does_not_leak_through_exact_graph_views(self) -> None:
        masked = make_iid_parent_dropped_mask(
            self.graph,
            fraction=0.20,
            seed=17,
        )
        targets = set(masked.evaluator_manifest.target_edges)

        self.assertFalse(targets & edge_key_set(masked.model.observed_edges))
        exact = extract_exact_parent_calls(
            masked.model.traces,
            columns=CANONICAL_TRACE_COLUMNS,
        )
        exact_keys = edge_key_set(
            exact.occurrences[["subject", "predicate", "object"]].drop_duplicates()
        )
        self.assertFalse(targets & exact_keys)

    def test_v1_default_keeps_opaque_parent_and_legacy_containment(self) -> None:
        masked = make_iid_mask(
            self.graph,
            fraction=0.20,
            seed=17,
        )
        target = masked.evaluator_manifest.target_edges[0]
        all_span_ids = set(masked.model.traces["span_id"].astype(str))
        unmatched = set(
            masked.model.traces["parent_span_id"].dropna().astype(str)
        ).difference(all_span_ids)

        self.assertEqual(masked.evaluator_manifest.evidence_level, EVIDENCE_LEVEL_L1)
        self.assertTrue(unmatched)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{16}", value) for value in unmatched))
        self.assertIn(target, temporal_containment_details(_context(masked)))
        self.assertIn(target, temporal_containment_support(_context(masked)))

    def test_v2_opt_in_recovers_containment_but_default_does_not(self) -> None:
        masked = make_iid_parent_dropped_mask(
            self.graph,
            fraction=0.20,
            seed=17,
        )
        target = masked.evaluator_manifest.target_edges[0]
        context = _context(masked)

        self.assertNotIn(target, temporal_containment_details(context))
        self.assertNotIn(target, temporal_containment_support(context))

        details = temporal_containment_details(context, include_null_parent=True)
        self.assertIn(target, details)
        recovered = details[target]
        self.assertEqual(
            recovered.boundary_count,
            masked.evaluator_manifest.redacted_boundary_spans,
        )
        self.assertEqual(recovered.trace_count, recovered.boundary_count)
        self.assertGreater(recovered.score, 0.0)


if __name__ == "__main__":
    unittest.main()
