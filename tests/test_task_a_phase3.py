from __future__ import annotations

import unittest

from logagent_benchmark.task_a_phase3 import (
    ShortlistPolicy,
    TriStateThresholds,
    _compact_runtime_context,
    apply_policy,
    classify_tri_state,
    evaluate_shortlist,
    select_calibrated_policy,
    stable_case_split,
)


class Phase3TriStateTests(unittest.TestCase):
    def probs(self, e: float, c: float, n: float):
        return {"entailment": e, "contradiction": c, "neutral": n}

    def test_corroborates_requires_direction_and_label_margin(self):
        evidence = classify_tri_state(
            flat_forward=self.probs(0.85, 0.05, 0.10),
            flat_reverse=self.probs(0.10, 0.80, 0.10),
            context_forward=self.probs(0.80, 0.10, 0.10),
            context_reverse=self.probs(0.15, 0.75, 0.10),
            thresholds=TriStateThresholds(),
        )
        self.assertEqual(evidence.state, "corroborates")
        self.assertGreater(evidence.direction_margin, 0.05)

    def test_reverse_dominance_is_contradictory_evidence_not_deletion(self):
        evidence = classify_tri_state(
            flat_forward=self.probs(0.20, 0.10, 0.70),
            flat_reverse=self.probs(0.80, 0.10, 0.10),
            context_forward=self.probs(0.20, 0.15, 0.65),
            context_reverse=self.probs(0.85, 0.05, 0.10),
            thresholds=TriStateThresholds(),
        )
        self.assertEqual(evidence.state, "contradicts")
        candidate = {
            "subject": "a", "predicate": "CALLS", "object": "b",
            "a2_score": 0.9, "supporting_traces": 2, "boundary_spans": 3,
            "proposal_rank": 1, "direct_evidence": False,
            "nli_evidence_score": evidence.evidence_score,
        }
        selected, scored = apply_policy([candidate], ShortlistPolicy(1.0, 1, 0.3))
        self.assertEqual(len(scored), 1)
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]["selected"])


class Phase3SerializationTests(unittest.TestCase):
    def test_runtime_context_has_explicit_bounded_serialization(self):
        text = _compact_runtime_context("\n".join(["x" * 300] * 20))
        self.assertLessEqual(len(text), 960)
        self.assertLessEqual(len(text.splitlines()), 8)
        self.assertTrue(all(len(line) <= 144 for line in text.splitlines()))


class Phase3SplitTests(unittest.TestCase):
    def test_split_is_stable_and_label_independent(self):
        revision = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
        cases = [
            "re2tt_ts-auth-service_cpu_2",
            "re2tt_ts-travel-service_mem_3",
            "re2tt_ts-order-service_disk_3",
            "re2tt_ts-auth-service_delay_3",
            "re2tt_ts-train-service_loss_3",
            "re2tt_ts-travel-service_socket_3",
        ]
        calibration, heldout, hashes = stable_case_split(
            reversed(cases), revision=revision, calibration_incidents=2
        )
        self.assertEqual(
            calibration,
            ("re2tt_ts-travel-service_mem_3", "re2tt_ts-auth-service_cpu_2"),
        )
        self.assertEqual(len(heldout), 4)
        self.assertEqual(set(calibration) | set(heldout), set(cases))
        self.assertEqual(len(hashes), 6)


class Phase3PolicyTests(unittest.TestCase):
    def candidate(self, subject, obj, rank, nli, *, score=None):
        return {
            "subject": subject,
            "predicate": "CALLS",
            "object": obj,
            "a2_score": float(score if score is not None else 1.0 - rank * 0.01),
            "supporting_traces": 10 - rank,
            "boundary_spans": 20 - rank,
            "proposal_rank": rank,
            "direct_evidence": False,
            "nli_evidence_score": nli,
            "nli_state": "corroborates" if nli > 0 else "ambiguous",
        }

    def test_calibration_selects_smallest_feasible_shortlist(self):
        candidates = [
            self.candidate("s", "t1", 1, 0.9),
            self.candidate("s", "u1", 2, -0.8),
            self.candidate("s", "t2", 3, 0.8),
            self.candidate("s", "u2", 4, -0.9),
        ]
        targets = {("s", "CALLS", "t1"), ("s", "CALLS", "t2")}
        silver = set(targets)
        cells = [
            {
                "case": f"c{index}", "fault": "cpu", "seed": index,
                "mask_id": "iid40", "mask_ratio": 0.4,
                "candidates": candidates, "targets": targets, "silver": silver,
                "a2_mrr": 1.0,
            }
            for index in range(2)
        ]
        policy, grid = select_calibrated_policy(
            cells,
            search={
                "retention_fractions": [0.5, 0.75, 1.0],
                "minimum_keep": [2],
                "nli_weights": [0.1, 0.3],
            },
            calibration_gate={
                "recall_macro_min": 0.95,
                "recall_each_cell_min": 0.9,
                "mrr_noninferiority_tolerance": 0.01,
                "matched_budget_recall_tolerance": 0.0,
                "matched_budget_p_lb_delta_min": 0.0,
                "matched_budget_mrr_delta_min": 0.0,
                "matched_budget_additive_gain_required": True,
            },
        )
        self.assertEqual(policy.retention_fraction, 0.5)
        self.assertEqual(policy.minimum_keep, 2)
        self.assertEqual(policy.nli_weight, 0.3)
        self.assertTrue(any(row["feasible"] for row in grid))

    def test_shortlist_metrics_improve_lower_bound_when_unverified_are_removed(self):
        selected = [
            {**self.candidate("s", "t1", 1, 0.8), "a3_score": 0.9},
            {**self.candidate("s", "t2", 2, 0.7), "a3_score": 0.8},
        ]
        targets = {("s", "CALLS", "t1"), ("s", "CALLS", "t2")}
        metrics = evaluate_shortlist(selected, targets=targets, silver=set(targets))
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["silver_precision_lower_bound"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
