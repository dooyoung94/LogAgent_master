from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from logagent_benchmark.recovery import Candidate, InferenceContext  # noqa: E402
from logagent_benchmark.runtime_context import build_runtime_pair_contexts  # noqa: E402


SOURCE = "service-source"
TARGET = "service-target"
CANDIDATE = Candidate(SOURCE, "CALLS", TARGET, "Service", "Service")


def _busy_context(*, long_values: bool = False) -> InferenceContext:
    suffix = "-" + ("x" * 5_000) if long_values else ""
    entities = (
        {
            "entity_id": SOURCE,
            "entity_type": "Service",
            "canonical_name": "Source Service" + suffix,
            "type_basis": "trace-observed" + suffix,
        },
        {
            "entity_id": TARGET,
            "entity_type": "Service",
            "canonical_name": "Target Service" + suffix,
            "type_basis": "trace-observed" + suffix,
        },
    )
    traces = tuple(
        {
            "trace_id": f"trace-{service}-{index}",
            "span_id": f"span-{service}-{index}",
            "service_id": service,
            "parent_span_id": None,
            "operation_name": f"operation-{index}{suffix}",
        }
        for service in (SOURCE, TARGET)
        for index in range(8)
    )
    observed_edges = tuple(
        edge
        for index in range(8)
        for edge in (
            {
                "subject": f"source-upstream-{index}{suffix}",
                "predicate": "CALLS",
                "object": SOURCE,
            },
            {
                "subject": SOURCE,
                "predicate": "CALLS",
                "object": f"source-downstream-{index}{suffix}",
            },
            {
                "subject": f"target-upstream-{index}{suffix}",
                "predicate": "CALLS",
                "object": TARGET,
            },
            {
                "subject": TARGET,
                "predicate": "CALLS",
                "object": f"target-downstream-{index}{suffix}",
            },
        )
    )
    return InferenceContext(
        incident_id="runtime-context-budget",
        entities=entities,
        traces=traces,
        observed_edges=observed_edges,
    )


class RuntimePairContextSerializationTests(unittest.TestCase):
    def test_source_and_target_have_balanced_sections_and_bounded_lists(self) -> None:
        result = build_runtime_pair_contexts(
            _busy_context(),
            (CANDIDATE,),
            system_label="benchmark-system",
        )[CANDIDATE.key]
        lines = result.contextual_addendum.splitlines()
        source_lines = [line for line in lines if line.startswith("source.")]
        target_lines = [line for line in lines if line.startswith("target.")]

        def sections(rows: list[str], prefix: str) -> tuple[str, ...]:
            return tuple(row[len(prefix) :].split(":", 1)[0] for row in rows)

        self.assertEqual(len(source_lines), len(target_lines))
        self.assertEqual(
            sections(source_lines, "source."),
            sections(target_lines, "target."),
        )
        self.assertEqual(
            set(sections(source_lines, "source.")),
            {"identity", "role_proxy", "neighbors", "telemetry", "operation_examples"},
        )

        list_pattern = re.compile(r"(?:upstream|downstream)=\[([^]]*)\]")
        operation_pattern = re.compile(r"operation_examples: \[([^]]*)\]")
        serialized_lists = [
            match.group(1)
            for line in lines
            for match in list_pattern.finditer(line)
        ] + [
            match.group(1)
            for line in lines
            if (match := operation_pattern.search(line)) is not None
        ]
        self.assertEqual(len(serialized_lists), 6)
        for value in serialized_lists:
            items = [] if value == "unknown" else value.split(", ")
            self.assertLessEqual(len(items), 3)

    def test_adversarial_values_cannot_expand_context_or_pair_labels_without_bound(self) -> None:
        result = build_runtime_pair_contexts(
            _busy_context(long_values=True),
            (CANDIDATE,),
            system_label="benchmark-system-" + ("s" * 5_000),
        )[CANDIDATE.key]
        lines = result.contextual_addendum.splitlines()

        # These are deliberately conservative character guards, not a proxy
        # for the model's exact token budget.  The ONNX backend enforces that
        # separately.  Their purpose is to prevent one untrusted telemetry
        # value from dominating both sides of the pair serialization.
        self.assertLessEqual(len(result.subject_label), 256)
        self.assertLessEqual(len(result.object_label), 256)
        self.assertLessEqual(len(lines), 16)
        self.assertLessEqual(max(map(len, lines)), 512)
        self.assertLessEqual(len(result.contextual_addendum), 4_096)


if __name__ == "__main__":
    unittest.main()
