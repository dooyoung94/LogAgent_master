from __future__ import annotations

import unittest

from logagent_benchmark.phase3_contract import Phase3Error
from logagent_benchmark.phase3_policy import select_calibrated_policy


class Phase3DiagnosticFallbackTests(unittest.TestCase):
    @staticmethod
    def _candidate(subject: str, obj: str, rank: int):
        return {
            "subject": subject,
            "predicate": "CALLS",
            "object": obj,
            "a2_score": 1.0 - rank * 0.01,
            "supporting_traces": 10 - rank,
            "boundary_spans": 20 - rank,
            "proposal_rank": rank,
            "direct_evidence": False,
            "nli_evidence_score": 0.0,
            "nli_state": "ambiguous",
        }

    def _cells(self):
        candidates = [
            self._candidate("s", "t1", 1),
            self._candidate("s", "t2", 2),
            self._candidate("s", "u1", 3),
            self._candidate("s", "u2", 4),
        ]
        targets = {("s", "CALLS", "t1"), ("s", "CALLS", "t2")}
        return [
            {
                "case": f"c{index}",
                "fault": "cpu",
                "role": "calibration",
                "seed": index,
                "mask_id": "iid40",
                "mask_ratio": 0.4,
                "candidates": candidates,
                "targets": targets,
                "silver": set(targets),
                "a2_mrr": 1.0,
            }
            for index in range(2)
        ]

    @staticmethod
    def _search():
        return {
            "retention_fractions": [0.5],
            "minimum_keep": [2],
            "nli_weights": [0.1, 0.3],
        }

    @staticmethod
    def _gate():
        return {
            "recall_macro_min": 0.95,
            "recall_each_cell_min": 0.9,
            "mrr_noninferiority_tolerance": 0.01,
            "matched_budget_recall_tolerance": 0.0,
            "matched_budget_p_lb_delta_min": 0.0,
            "matched_budget_mrr_delta_min": 0.0,
            "matched_budget_additive_gain_required": True,
        }

    def test_strict_mode_still_raises_when_no_policy_adds_utility(self):
        with self.assertRaises(Phase3Error):
            select_calibrated_policy(
                self._cells(),
                search=self._search(),
                calibration_gate=self._gate(),
            )

    def test_diagnostic_mode_selects_one_marked_infeasible_policy(self):
        policy, grid = select_calibrated_policy(
            self._cells(),
            search=self._search(),
            calibration_gate=self._gate(),
            allow_diagnostic_fallback=True,
        )
        selected = [row for row in grid if row["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0]["feasible"])
        self.assertEqual(
            selected[0]["selection_status"],
            "DIAGNOSTIC_FALLBACK_NO_FEASIBLE_POLICY",
        )
        self.assertGreater(policy.nli_weight, 0.0)
        self.assertGreaterEqual(selected[0]["violation_count"], 1)


if __name__ == "__main__":
    unittest.main()
