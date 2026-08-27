from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from logagent_benchmark.metrics import (  # noqa: E402
    activation_metrics,
    evaluate_recovery,
    silver_precision_lower_bound,
    zero_flip_gate,
)
from logagent_benchmark.recovery import (  # noqa: E402
    ERROR,
    READY,
    SKIPPED,
    AblationConfig,
    Availability,
    Candidate,
    InferenceContext,
    build_typed_candidates,
    run_ablation_suite,
    run_recovery,
    temporal_containment_support,
)


def fixture_context(*, extra_direct: bool = False) -> InferenceContext:
    direct = [
        {
            "evidence_id": "db-span-1",
            "subject": "payment",
            "predicate": "USES_DATASOURCE",
            "object": "payment-db",
        }
    ]
    if extra_direct:
        direct.append(
            {
                "evidence_id": "db-span-2",
                "subject": "auth",
                "predicate": "USES_DATASOURCE",
                "object": "payment-db",
            }
        )
    return InferenceContext(
        incident_id="re2tt-smoke",
        entities=(
            {"entity_id": "web", "entity_type": "Service"},
            {"entity_id": "payment", "entity_type": "Service"},
            {"entity_id": "auth", "entity_type": "Service"},
            {"entity_id": "payment-db", "entity_type": "DataSource"},
            {"entity_id": "payment-01", "entity_type": "Instance"},
            {"entity_id": "node-a", "entity_type": "Host"},
        ),
        observed_edges=(
            {"subject_id": "web", "predicate": "CALLS", "object_id": "auth"},
        ),
        traces=(
            {
                "trace_id": "trace-1",
                "span_id": "parent-1",
                "service_id": "web",
                "start_time_us": 0,
                "end_time_us": 100,
                "parent_span_id": None,
            },
            {
                "trace_id": "trace-1",
                "span_id": "child-1",
                "service_id": "payment",
                "start_time_us": 10,
                "end_time_us": 80,
                # Same 16-hex surface form as RCAEval, but absent from this trace.
                "parent_span_id": "f0e1d2c3b4a59687",
            },
        ),
        evidence=tuple(direct),
    )


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, *, orient):
        if orient != "records":
            raise AssertionError(orient)
        return list(self.rows)


class TestDebertaBackend:
    research_valid = False

    def __init__(self):
        self.score_batch_calls = 0
        self.last_pairs = []

    def availability(self):
        return Availability(READY, research_valid=False)

    def score_pairs(self, pairs):
        self.score_batch_calls += 1
        self.last_pairs = list(pairs)
        output = []
        for _premise, hypothesis in pairs:
            if "web calls payment" in hypothesis:
                output.append({"entailment": 0.96, "contradiction": 0.01})
            elif "payment uses data source payment-db" in hypothesis:
                output.append({"entailment": 0.85, "contradiction": 0.02})
            else:
                output.append({"entailment": 0.05, "contradiction": 0.75})
        return output


class TestPslBackend:
    research_valid = False
    relation = "CALLS"

    def availability(self):
        return Availability(READY, research_valid=False)

    def infer(self, *, context, candidates, local_scores):
        if any(candidate.predicate != "CALLS" for candidate in candidates):
            raise AssertionError("PSL test backend received a non-CALLS candidate")
        scores = {}
        for candidate in candidates:
            if candidate.key == ("web", "CALLS", "payment"):
                scores[candidate.key] = 0.20
            else:
                scores[candidate.key] = min(0.45, local_scores[candidate.key])
        return {
            "scores": scores,
            "grounded_rule_count": 4,
            "grounded_atom_count": 7,
            "metadata": {"runtime_sha256": "test-only", "random_seed": 7},
        }


class AblationContractTests(unittest.TestCase):
    def test_model_api_has_no_reference_or_mask_manifest(self):
        parameter_names = set(inspect.signature(run_recovery).parameters)
        self.assertNotIn("reference", parameter_names)
        self.assertNotIn("reference_graph", parameter_names)
        self.assertNotIn("mask_manifest", parameter_names)
        context_fields = {item.name for item in fields(InferenceContext)}
        self.assertTrue(
            context_fields.isdisjoint(
                {"reference", "reference_graph", "target_edges", "mask_manifest"}
            )
        )

    def test_model_input_adapter_accepts_dataframe_like_objects(self):
        context = InferenceContext.from_model_input(
            {
                "incident": {
                    "incident_id": "case-1",
                    "root_cause_entity_id": "must-not-enter-context",
                },
                "entities": FakeFrame(
                    [{"entity_id": "svc", "entity_type": "Service"}]
                ),
                "traces": FakeFrame([]),
                "logs": FakeFrame([]),
                "metrics": FakeFrame([]),
            }
        )
        self.assertEqual(context.incident_id, "case-1")
        self.assertEqual(len(context.entities), 1)
        self.assertFalse(hasattr(context, "root_cause_entity_id"))

    def test_typed_candidate_universe_respects_types_and_observed_edges(self):
        candidates = build_typed_candidates(fixture_context())
        keys = {candidate.key for candidate in candidates}
        self.assertIn(("web", "CALLS", "payment"), keys)
        self.assertIn(("payment", "USES_DATASOURCE", "payment-db"), keys)
        self.assertIn(("payment-01", "LOCATED_ON", "node-a"), keys)
        self.assertNotIn(("payment-db", "CALLS", "web"), keys)
        self.assertNotIn(("web", "CALLS", "auth"), keys)  # already observed
        self.assertEqual(len(keys), len(candidates))

    def test_all_variants_receive_the_same_candidate_tuple(self):
        suite = run_ablation_suite(fixture_context())
        expected = tuple(candidate.candidate_id for candidate in suite.candidates)
        for result in suite.results.values():
            self.assertEqual(
                tuple(candidate.candidate_id for candidate in result.candidates), expected
            )

    def test_a0_never_adds_an_edge(self):
        result = run_recovery("A0", fixture_context())
        self.assertEqual(result.status, READY)
        self.assertEqual(result.accepted_edges, ())
        self.assertEqual(
            {edge.key for edge in result.completed_edges}, {("web", "CALLS", "auth")}
        )

    def test_a1_accepts_only_deterministic_direct_evidence(self):
        result = run_recovery("A1", fixture_context())
        self.assertEqual(
            {edge.key for edge in result.accepted_edges},
            {("payment", "USES_DATASOURCE", "payment-db")},
        )
        prediction = next(
            item
            for item in result.predictions
            if item.key == ("payment", "USES_DATASOURCE", "payment-db")
        )
        self.assertEqual(prediction.score, 1.0)
        self.assertEqual(prediction.evidence_ids, ("db-span-1",))

    def test_a2_uses_only_unmatched_parent_temporal_containment(self):
        context = fixture_context()
        support = temporal_containment_support(context)
        self.assertIn(("web", "CALLS", "payment"), support)
        result = run_recovery("A2", context)
        self.assertEqual(
            {edge.key for edge in result.accepted_edges},
            {("web", "CALLS", "payment")},
        )

        natural_child = dict(context.traces[1])
        natural_child["parent_span_id"] = "parent-1"
        natural_context = InferenceContext(
            incident_id=context.incident_id,
            entities=context.entities,
            observed_edges=context.observed_edges,
            traces=(context.traces[0], natural_child),
            evidence=context.evidence,
        )
        self.assertEqual(temporal_containment_support(natural_context), {})

    def test_a2_does_not_require_parent_span_id_ground_truth(self):
        context = fixture_context()
        self.assertNotEqual(context.traces[1]["parent_span_id"], "parent-1")
        # The opaque unmatched ID does not disclose its real parent; interval
        # containment supplies the abductive hypothesis without a mask marker.
        recovered = {edge.key for edge in run_recovery("A2", context).accepted_edges}
        self.assertIn(("web", "CALLS", "payment"), recovered)


class OptionalBackendTests(unittest.TestCase):
    def test_heavy_variants_skip_without_actual_backends(self):
        for variant in ("A3", "A4", "A5"):
            result = run_recovery(variant, fixture_context())
            self.assertEqual(result.status, SKIPPED)
            self.assertEqual(result.reason_code, "DEBERTA_BACKEND_MISSING")
            self.assertEqual(result.accepted_edges, ())

    def test_test_deberta_backend_is_forbidden_by_default(self):
        result = run_recovery(
            "A3", fixture_context(), deberta_backend=TestDebertaBackend()
        )
        self.assertEqual(result.status, SKIPPED)
        self.assertEqual(result.reason_code, "DEBERTA_TEST_BACKEND_FORBIDDEN")

    def test_test_backend_can_exercise_contract_but_is_not_research_valid(self):
        result = run_recovery(
            "A3",
            fixture_context(),
            deberta_backend=TestDebertaBackend(),
            allow_test_backends=True,
        )
        self.assertEqual(result.status, READY)
        self.assertFalse(result.research_valid)
        self.assertGreater(result.activation["nli_pair_count"], 0)
        self.assertGreater(result.activation["score_std"], 0)
        self.assertIn(
            ("web", "CALLS", "payment"),
            {edge.key for edge in result.accepted_edges},
        )

    def test_a5_skips_when_real_psl_backend_is_absent(self):
        result = run_recovery(
            "A5",
            fixture_context(),
            deberta_backend=TestDebertaBackend(),
            allow_test_backends=True,
        )
        self.assertEqual(result.status, SKIPPED)
        self.assertEqual(result.reason_code, "PSL_BACKEND_MISSING")

    def test_a4_and_a5_stage_activation_with_explicit_test_backends(self):
        context = fixture_context()
        a4 = run_recovery(
            "A4",
            context,
            deberta_backend=TestDebertaBackend(),
            allow_test_backends=True,
        )
        a5 = run_recovery(
            "A5",
            context,
            deberta_backend=TestDebertaBackend(),
            psl_backend=TestPslBackend(),
            allow_test_backends=True,
        )
        self.assertEqual(a4.status, READY)
        self.assertEqual(a5.status, READY)
        self.assertFalse(a5.research_valid)
        self.assertEqual(a5.activation["grounded_rule_count"], 4)
        self.assertEqual(a5.activation["grounded_atom_count"], 7)
        self.assertEqual(a5.activation["psl_metadata"]["random_seed"], 7)
        self.assertGreater(a5.activation["psl_candidate_count"], 0)
        self.assertGreater(a5.activation["psl_score_delta_count"], 0)
        activation = activation_metrics(a4, a5)
        self.assertGreater(activation["decision_flip_count"], 0)

    def test_suite_computes_abduction_and_deberta_only_once(self):
        import logagent_benchmark.recovery as recovery_module

        deberta = TestDebertaBackend()
        original = recovery_module.temporal_containment_support
        with patch(
            "logagent_benchmark.recovery.temporal_containment_support",
            wraps=original,
        ) as containment_spy:
            suite = run_ablation_suite(
                fixture_context(),
                deberta_backend=deberta,
                psl_backend=TestPslBackend(),
                allow_test_backends=True,
            )
        self.assertEqual(containment_spy.call_count, 1)
        self.assertEqual(deberta.score_batch_calls, 1)
        self.assertEqual(suite.results["A3"].status, READY)
        self.assertEqual(suite.results["A5"].status, READY)

    def test_a3_does_not_compute_abductive_containment(self):
        with patch(
            "logagent_benchmark.recovery.temporal_containment_support"
        ) as containment_spy:
            result = run_recovery(
                "A3",
                fixture_context(),
                deberta_backend=TestDebertaBackend(),
                allow_test_backends=True,
            )
        self.assertEqual(result.status, READY)
        containment_spy.assert_not_called()

    def test_deberta_prefers_both_endpoint_summary_without_trace_rescan_text(self):
        base = fixture_context()
        context = InferenceContext(
            incident_id=base.incident_id,
            entities=base.entities,
            observed_edges=base.observed_edges,
            traces=base.traces,
            evidence=(
                {
                    "evidence_id": "pair-summary-1",
                    "subject_id": "web",
                    "object_id": "payment",
                    "summary": "PAIR_SUMMARY_ONLY",
                },
            ),
        )
        backend = TestDebertaBackend()
        result = run_recovery(
            "A3",
            context,
            deberta_backend=backend,
            allow_test_backends=True,
        )
        self.assertEqual(result.status, READY)
        premise = next(
            premise
            for premise, hypothesis in backend.last_pairs
            if "web calls payment" in hypothesis
        )
        self.assertIn("PAIR_SUMMARY_ONLY", premise)
        self.assertNotIn("child-1", premise)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.context = fixture_context()
        self.masked = {
            ("web", "CALLS", "payment"),
            ("payment", "USES_DATASOURCE", "payment-db"),
        }
        self.silver = {
            ("web", "CALLS", "auth"),
            *self.masked,
        }

    def test_masked_recall_mrr_hits_and_silver_precision(self):
        result = run_recovery("A1", self.context)
        report = evaluate_recovery(
            result,
            masked_edges=self.masked,
            silver_reference_edges=self.silver,
            all_reference_edges=self.silver,
        )
        self.assertEqual(report["candidate_recall"]["recall"]["value"], 1.0)
        self.assertEqual(report["masked_recall"]["recall"]["value"], 0.5)
        self.assertGreater(report["ranking"]["mrr"]["value"], 0.0)
        self.assertIn("1", report["ranking"]["hits"])
        self.assertIn("3", report["ranking"]["hits"])
        self.assertIn("10", report["ranking"]["hits"])
        self.assertEqual(
            report["silver_precision_lower_bound"]["lower_bound"]["value"], 1.0
        )

    def test_unmatched_silver_prediction_is_unverified_not_declared_false(self):
        result = run_recovery("A1", fixture_context(extra_direct=True))
        report = silver_precision_lower_bound(result, self.silver)
        self.assertEqual(report["accepted_count"], 2)
        self.assertEqual(report["silver_matched_count"], 1)
        self.assertEqual(report["unverified_count"], 1)
        self.assertEqual(report["lower_bound"]["value"], 0.5)
        self.assertNotIn("false_count", report)

    def test_zero_denominator_is_na_not_zero(self):
        result = run_recovery("A0", self.context)
        report = evaluate_recovery(
            result,
            masked_edges=(),
            silver_reference_edges=self.silver,
        )
        self.assertIsNone(report["masked_recall"]["recall"]["value"])
        self.assertEqual(
            report["masked_recall"]["recall"]["reason"], "NO_MASKED_TARGET_EDGES"
        )
        self.assertIsNone(
            report["silver_precision_lower_bound"]["lower_bound"]["value"]
        )

    def test_activation_and_zero_flip_gate(self):
        suite = run_ablation_suite(self.context)
        self.assertGreater(
            suite.activation["A0->A1"]["decision_flip_count"], 0
        )
        self.assertGreater(
            suite.activation["A1->A2"]["decision_flip_count"], 0
        )
        self.assertTrue(suite.gate["passed"])

        required_gate = zero_flip_gate(
            suite.results,
            suite.activation,
            require_variants=("A3", "A4", "A5"),
        )
        self.assertFalse(required_gate["passed"])
        self.assertEqual(
            required_gate["skipped_required_variants"], ["A3", "A4", "A5"]
        )

    def test_zero_flip_is_an_explicit_failure(self):
        empty_context = InferenceContext(
            incident_id="empty",
            entities=(
                {"entity_id": "a", "entity_type": "Service"},
                {"entity_id": "b", "entity_type": "Service"},
            ),
        )
        suite = run_ablation_suite(empty_context, variants=("A0", "A1", "A2"))
        self.assertFalse(suite.gate["passed"])
        self.assertIn("ZERO_DECISION_FLIP", suite.gate["reason_codes"])

    def test_smoke_gate_allows_one_inactive_pair_when_another_is_active(self):
        base = fixture_context()
        context = InferenceContext(
            incident_id=base.incident_id,
            entities=base.entities,
            observed_edges=base.observed_edges,
            traces=base.traces,
            evidence=(),
        )
        suite = run_ablation_suite(context, variants=("A0", "A1", "A2"))
        self.assertEqual(
            suite.activation["A0->A1"]["decision_flip_count"], 0
        )
        self.assertGreater(
            suite.activation["A1->A2"]["decision_flip_count"], 0
        )
        self.assertTrue(suite.gate["passed"])

        paper_suite = run_ablation_suite(
            context,
            variants=("A0", "A1", "A2"),
            require_activation_pairs=("A0->A1", "A1->A2"),
        )
        self.assertFalse(paper_suite.gate["passed"])
        self.assertEqual(
            paper_suite.gate["zero_required_activation_pairs"], ["A0->A1"]
        )


if __name__ == "__main__":
    unittest.main()
