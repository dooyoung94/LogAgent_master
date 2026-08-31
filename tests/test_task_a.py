import unittest

from logagent_benchmark.recovery import Candidate, TemporalContainmentDetail
from logagent_benchmark.task_a import TaskAConfig, select_abductive_proposals


class TaskACandidateBudgetTests(unittest.TestCase):
    def _candidate(self, subject: str, obj: str) -> Candidate:
        return Candidate(subject, "CALLS", obj, "SERVICE", "SERVICE")

    def _detail(self, score: float, traces: int, boundaries: int):
        return TemporalContainmentDetail(score, (), traces, boundaries)

    def test_global_budget_keeps_strongest_candidates_deterministically(self):
        candidates = [
            self._candidate("s1", "t1"),
            self._candidate("s2", "t2"),
            self._candidate("s3", "t3"),
            self._candidate("s4", "t4"),
        ]
        by_key = {candidate.key: candidate for candidate in candidates}
        details = {
            candidates[0].key: self._detail(0.95, 4, 5),
            candidates[1].key: self._detail(0.90, 3, 7),
            candidates[2].key: self._detail(0.90, 2, 9),
            candidates[3].key: self._detail(0.80, 5, 9),
        }
        result = select_abductive_proposals(
            details=details,
            candidate_by_key=by_key,
            direct_keys=set(),
            config=TaskAConfig(max_abductive_proposals=3),
        )
        self.assertEqual(
            result.selected_keys,
            (candidates[0].key, candidates[1].key, candidates[2].key),
        )
        self.assertEqual(result.diagnostics["selected_after_budget"], 3)
        self.assertEqual(result.diagnostics["dropped_global_cap"], 1)
        self.assertTrue(result.diagnostics["budget_saturated"])

    def test_endpoint_caps_prevent_one_hub_from_consuming_budget(self):
        candidates = [self._candidate("hub", f"t{index}") for index in range(4)]
        candidates += [self._candidate("other", "x")]
        by_key = {candidate.key: candidate for candidate in candidates}
        details = {
            candidate.key: self._detail(0.95 - index * 0.01, 3, 3)
            for index, candidate in enumerate(candidates)
        }
        result = select_abductive_proposals(
            details=details,
            candidate_by_key=by_key,
            direct_keys=set(),
            config=TaskAConfig(
                max_abductive_proposals=4,
                max_per_subject=2,
                max_per_object=4,
            ),
        )
        self.assertLessEqual(
            sum(key[0] == "hub" for key in result.selected_keys),
            2,
        )
        self.assertIn(candidates[-1].key, result.selected_keys)
        self.assertEqual(result.diagnostics["dropped_subject_cap"], 2)

    def test_direct_keys_do_not_consume_abductive_budget(self):
        direct = self._candidate("direct", "target")
        abductive = self._candidate("a", "b")
        result = select_abductive_proposals(
            details={
                direct.key: self._detail(0.99, 10, 10),
                abductive.key: self._detail(0.80, 2, 2),
            },
            candidate_by_key={direct.key: direct, abductive.key: abductive},
            direct_keys={direct.key},
            config=TaskAConfig(max_abductive_proposals=1),
        )
        self.assertEqual(result.selected_keys, (abductive.key,))
        self.assertTrue(
            result.diagnostics["direct_evidence_preserved_outside_budget"]
        )

    def test_budget_values_must_be_positive_integers(self):
        with self.assertRaises(ValueError):
            TaskAConfig(max_abductive_proposals=0)
        with self.assertRaises(ValueError):
            TaskAConfig(max_per_subject=True)


if __name__ == "__main__":
    unittest.main()
