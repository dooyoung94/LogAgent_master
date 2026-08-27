from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.psl_backend import PslPythonBackend  # noqa: E402
from logagent_benchmark.recovery import (  # noqa: E402
    Availability,
    Candidate,
    InferenceContext,
    READY,
)


HAS_PSLPYTHON = importlib.util.find_spec("pslpython") is not None


def context() -> InferenceContext:
    return InferenceContext(
        incident_id="psl-integration-smoke",
        entities=(
            {"entity_id": "node-a", "entity_type": "Service"},
            {"entity_id": "node-b", "entity_type": "Service"},
        ),
    )


def calls_candidate() -> Candidate:
    return Candidate("node-a", "CALLS", "node-b", "SERVICE", "SERVICE")


class PslBackendContractTests(unittest.TestCase):
    def test_empty_candidates_do_not_require_or_start_psl(self) -> None:
        result = PslPythonBackend().infer(
            context=context(),
            candidates=(),
            local_scores={},
        )
        self.assertEqual({}, result.scores)
        self.assertEqual(0, result.grounded_rule_count)
        self.assertEqual(0, result.grounded_atom_count)
        self.assertTrue(result.metadata["empty_candidate_set"])
        self.assertTrue(result.metadata["temporary_data_cleaned"])
        self.assertIn("psl_runtime_version", result.metadata)
        self.assertIn("compatibility_override", result.metadata)

    def test_rejects_non_calls_without_a_fallback(self) -> None:
        candidate = Candidate(
            "node-a", "USES_DATASOURCE", "node-b", "SERVICE", "DATASOURCE"
        )
        with self.assertRaisesRegex(ValueError, "CALLS candidates only"):
            PslPythonBackend().infer(
                context=context(),
                candidates=(candidate,),
                local_scores={candidate.key: 0.8},
            )

    def test_rejects_missing_or_invalid_local_truth(self) -> None:
        candidate = calls_candidate()
        with self.assertRaisesRegex(KeyError, "missing local score"):
            PslPythonBackend().infer(
                context=context(), candidates=(candidate,), local_scores={}
            )
        for invalid in (-0.01, 1.01, float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and in"):
                    PslPythonBackend().infer(
                        context=context(),
                        candidates=(candidate,),
                        local_scores={candidate.key: invalid},
                    )


@unittest.skipUnless(HAS_PSLPYTHON, "official pslpython is not installed")
class PslBackendIntegrationTests(unittest.TestCase):
    def test_availability_reports_pinned_artifact_and_override(self) -> None:
        availability = PslPythonBackend().availability()
        self.assertIsInstance(availability, Availability)
        self.assertEqual(READY, availability.status, availability.detail)
        metadata = json.loads(availability.detail)
        self.assertEqual("pslpython", metadata["backend"])
        self.assertEqual("CALLS", metadata["relation"])
        self.assertRegex(metadata["pslpython_version"], r"^\d+\.\d+")
        self.assertRegex(metadata["psl_runtime_version"], r"^\d+\.\d+")
        self.assertRegex(metadata["psl_runtime_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsInstance(metadata["compatibility_override"], bool)

    def test_real_grounding_inference_is_deterministic_and_cleans_temp_data(self) -> None:
        candidate = calls_candidate()
        with tempfile.TemporaryDirectory(prefix="psl-test-parent-") as parent:
            backend = PslPythonBackend(random_seed=7, temporary_parent=parent)
            first = backend.infer(
                context=context(),
                candidates=(candidate,),
                local_scores={candidate.key: 1.0},
            )
            second = backend.infer(
                context=context(),
                candidates=(candidate,),
                local_scores={candidate.key: 1.0},
            )
            self.assertEqual([], list(Path(parent).iterdir()))

        self.assertEqual(2, first.grounded_rule_count)
        self.assertGreaterEqual(first.grounded_atom_count, 2)
        self.assertEqual({candidate.key}, set(first.scores))
        self.assertAlmostEqual(0.90944314, first.scores[candidate.key], places=6)
        self.assertAlmostEqual(
            first.scores[candidate.key], second.scores[candidate.key], places=8
        )
        self.assertTrue(first.metadata["temporary_data_cleaned"])
        self.assertEqual(1, first.metadata["grounded_evidence_rule_count"])
        self.assertEqual(1, first.metadata["grounded_prior_rule_count"])
        self.assertEqual(2, first.metadata["maximum_grounded_rule_count"])
        self.assertEqual("CALLS", first.metadata["relation"])
        self.assertEqual(7, first.metadata["random_seed"])
        self.assertIn("java_runtime_version", first.metadata)

    def test_zero_truth_is_valid_when_psl_prunes_satisfied_evidence_rule(self) -> None:
        candidate = calls_candidate()
        result = PslPythonBackend(random_seed=7).infer(
            context=context(),
            candidates=(candidate,),
            local_scores={candidate.key: 0.0},
        )
        self.assertEqual(1, result.grounded_rule_count)
        self.assertEqual(0, result.metadata["grounded_evidence_rule_count"])
        self.assertEqual(1, result.metadata["grounded_prior_rule_count"])
        self.assertEqual(1, result.metadata["pruned_evidence_rule_count"])
        self.assertLess(result.scores[candidate.key], 0.001)


if __name__ == "__main__":
    unittest.main()
