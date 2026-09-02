from __future__ import annotations

import unittest

import pandas as pd

from logagent_benchmark.task_a_phase3_r1 import (
    Phase3R1Error,
    StructuredPolicy,
    add_structured_features,
    apply_structured_policy,
)


class Phase3R1FeatureTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "incident_id": "i1",
                    "seed": 11,
                    "mask_id": "iid20",
                    "mask_ratio": 0.2,
                    "subject": "a",
                    "predicate": "CALLS",
                    "object": "b",
                    "a2_score": 0.9,
                    "proposal_rank": 1,
                    "supporting_traces": 8,
                    "boundary_spans": 20,
                    "reverse_supporting_traces": 0,
                    "reverse_boundary_spans": 0,
                    "direct_evidence": False,
                },
                {
                    "incident_id": "i1",
                    "seed": 11,
                    "mask_id": "iid20",
                    "mask_ratio": 0.2,
                    "subject": "a",
                    "predicate": "CALLS",
                    "object": "c",
                    "a2_score": 0.9,
                    "proposal_rank": 2,
                    "supporting_traces": 8,
                    "boundary_spans": 20,
                    "reverse_supporting_traces": 7,
                    "reverse_boundary_spans": 18,
                    "direct_evidence": False,
                },
            ]
        )

    def test_evaluator_columns_are_rejected_during_feature_scoring(self) -> None:
        frame = self.frame()
        frame["is_masked_target"] = True
        with self.assertRaises(Phase3R1Error):
            add_structured_features(frame)

    def test_directional_asymmetry_changes_equal_a2_order(self) -> None:
        features = add_structured_features(self.frame())
        scored = apply_structured_policy(
            features,
            StructuredPolicy(0.5, 1, 1.0, 1.0, 0.0),
        )
        selected = scored.loc[scored["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["object"], "b")

    def test_zero_structure_weight_reproduces_a2_order(self) -> None:
        frame = self.frame().copy()
        frame.loc[frame["object"] == "b", "proposal_rank"] = 2
        frame.loc[frame["object"] == "c", "proposal_rank"] = 1
        features = add_structured_features(frame)
        scored = apply_structured_policy(
            features,
            StructuredPolicy(0.5, 1, 0.0, 1.0, 0.0),
        )
        selected = scored.loc[scored["selected"]]
        self.assertEqual(selected.iloc[0]["object"], "c")

    def test_direct_evidence_is_preserved(self) -> None:
        frame = self.frame().copy()
        frame.loc[frame["object"] == "c", "direct_evidence"] = True
        features = add_structured_features(frame)
        scored = apply_structured_policy(
            features,
            StructuredPolicy(0.5, 1, 1.0, 1.0, 0.0),
        )
        self.assertTrue(
            bool(scored.loc[scored["object"] == "c", "selected"].iloc[0])
        )


if __name__ == "__main__":
    unittest.main()
