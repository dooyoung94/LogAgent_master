from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from logagent_benchmark.cumulative import (  # noqa: E402
    PairRuntimeContext,
    run_cumulative_suite,
)
from logagent_benchmark.recovery import (  # noqa: E402
    READY,
    Availability,
    InferenceContext,
)


WEB_PAYMENT = ("web", "CALLS", "payment")
PAYMENT_LEDGER = ("payment", "CALLS", "ledger")
OUTSIDE_PROPOSALS = ("outside", "CALLS", "proposal")


def _l2_context() -> InferenceContext:
    """Two L2 parent-loss boundaries and no A1 direct evidence."""

    return InferenceContext(
        incident_id="cumulative-l2-fixture",
        entities=(
            {"entity_id": "web", "entity_type": "Service"},
            {"entity_id": "payment", "entity_type": "Service"},
            {"entity_id": "ledger", "entity_type": "Service"},
        ),
        traces=(
            {
                "trace_id": "trace-web-payment",
                "span_id": "web-parent",
                "service_id": "web",
                "start_time_us": 0,
                "end_time_us": 100,
                "parent_span_id": None,
            },
            {
                "trace_id": "trace-web-payment",
                "span_id": "payment-child",
                "service_id": "payment",
                "start_time_us": 10,
                "end_time_us": 80,
                # L2: the collector lost the parent identifier.  This has the
                # same model-visible surface form as an ordinary root span.
                "parent_span_id": None,
            },
            {
                "trace_id": "trace-payment-ledger",
                "span_id": "payment-parent",
                "service_id": "payment",
                "start_time_us": 200,
                "end_time_us": 300,
                "parent_span_id": None,
            },
            {
                "trace_id": "trace-payment-ledger",
                "span_id": "ledger-child",
                "service_id": "ledger",
                "start_time_us": 210,
                "end_time_us": 280,
                "parent_span_id": None,
            },
        ),
    )


def _pair_contexts() -> dict[tuple[str, str, str], PairRuntimeContext]:
    return {
        WEB_PAYMENT: PairRuntimeContext(
            subject_label="Web Gateway",
            object_label="Payment Service",
            contextual_addendum=(
                "CTX_WEB_PAYMENT: Web Gateway is an ingress role and Payment "
                "Service is its model-visible downstream neighbor."
            ),
            provenance=("runtime_hierarchy", "runtime_neighbor"),
        ),
        PAYMENT_LEDGER: PairRuntimeContext(
            subject_label="Payment Service",
            object_label="Ledger Service",
            contextual_addendum=(
                "CTX_PAYMENT_LEDGER: Payment Service is an orchestrator role and "
                "Ledger Service is a model-visible data-facing role."
            ),
            provenance=("runtime_role", "runtime_hierarchy"),
        ),
    }


class _RecordingDeberta:
    research_valid = False

    def __init__(self) -> None:
        self.availability_calls = 0
        self.batches: list[tuple[tuple[str, str], ...]] = []

    def availability(self) -> Availability:
        self.availability_calls += 1
        return Availability(READY, research_valid=False)

    def score_pairs(self, pairs):
        batch = tuple(pairs)
        self.batches.append(batch)
        output = []
        for _premise, hypothesis in batch:
            if hypothesis == (
                "Within this runtime system, Web Gateway directly invokes "
                "Payment Service."
            ):
                entailment = 0.95
            elif hypothesis == (
                "Within this runtime system, Payment Service directly invokes "
                "Web Gateway."
            ):
                entailment = 0.05
            elif hypothesis == (
                "Within this runtime system, Payment Service directly invokes "
                "Ledger Service."
            ):
                entailment = 0.80
            elif hypothesis == (
                "Within this runtime system, Ledger Service directly invokes "
                "Payment Service."
            ):
                # Deliberately stronger than the proposed direction.  A
                # directional verifier must abstain despite high forward NLI.
                entailment = 0.90
            else:
                # D0 uses the legacy flat hypotheses over the all-pairs
                # universe.  It is a control, not part of A3--A5 assertions.
                entailment = 0.05
            output.append(
                {
                    "entailment": entailment,
                    "contradiction": max(0.0, 0.95 - entailment),
                    "neutral": 0.05,
                }
            )
        return output


class _RecordingPsl:
    research_valid = False
    relation = "CALLS"

    def __init__(self) -> None:
        self.availability_calls = 0
        self.calls: list[
            tuple[
                tuple[tuple[str, str, str], ...],
                dict[tuple[str, str, str], float],
            ]
        ] = []

    def availability(self) -> Availability:
        self.availability_calls += 1
        return Availability(READY, research_valid=False)

    def infer(self, *, context, candidates, local_scores):
        del context
        keys = tuple(candidate.key for candidate in candidates)
        self.calls.append((keys, dict(local_scores)))
        scores = {key: 0.91 for key in keys}
        # A backend may return irrelevant atoms, but the adapter must never
        # turn one into a candidate or accepted edge outside P2.
        scores[OUTSIDE_PROPOSALS] = 1.0
        return {
            "scores": scores,
            "grounded_rule_count": 3,
            "grounded_atom_count": len(scores),
            "metadata": {"fixture": "cumulative-v2"},
        }


class _FixedOutputDeberta(_RecordingDeberta):
    def __init__(self, raw_score) -> None:
        super().__init__()
        self.raw_score = raw_score

    def score_pairs(self, pairs):
        batch = tuple(pairs)
        self.batches.append(batch)
        return [self.raw_score for _pair in batch]


class _AcceptAllDirectionalDeberta(_RecordingDeberta):
    """Accept every proposed forward direction and reject every reverse."""

    def score_pairs(self, pairs):
        batch = tuple(pairs)
        self.batches.append(batch)
        return [
            {
                "entailment": 0.90 if index % 2 == 0 else 0.05,
                "contradiction": 0.05 if index % 2 == 0 else 0.90,
                "neutral": 0.05,
            }
            for index, _pair in enumerate(batch)
        ]


class _MalformedPsl(_RecordingPsl):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def infer(self, *, context, candidates, local_scores):
        del context
        keys = tuple(candidate.key for candidate in candidates)
        self.calls.append((keys, dict(local_scores)))
        if self.mode == "missing":
            scores = {}
        elif self.mode == "nan":
            scores = {key: float("nan") for key in keys}
        else:  # pragma: no cover - test fixture misuse
            raise AssertionError(f"unsupported malformed PSL fixture: {self.mode}")
        return {
            "scores": scores,
            "grounded_rule_count": 1,
            "grounded_atom_count": len(scores),
        }


def _direct_only_context() -> InferenceContext:
    return InferenceContext(
        incident_id="direct-only",
        entities=(
            {"entity_id": "web", "entity_type": "Service"},
            {"entity_id": "payment", "entity_type": "Service"},
        ),
        evidence=(
            {
                "subject": "web",
                "predicate": "CALLS",
                "object": "payment",
                "evidence_id": "declared-web-payment",
            },
        ),
    )


def _mixed_direct_and_abductive_context() -> InferenceContext:
    base = _l2_context()
    return InferenceContext(
        incident_id="mixed-direct-abductive",
        entities=base.entities,
        traces=base.traces[2:],
        evidence=(
            {
                "subject": "web",
                "predicate": "CALLS",
                "object": "payment",
                "evidence_id": "declared-web-payment",
            },
        ),
    )


def _run_fixture():
    deberta = _RecordingDeberta()
    psl = _RecordingPsl()
    suite = run_cumulative_suite(
        _l2_context(),
        pair_contexts=_pair_contexts(),
        deberta_backend=deberta,
        psl_backend=psl,
        allow_test_backends=True,
    )
    return suite, deberta, psl


class CumulativeProposalContractTests(unittest.TestCase):
    def test_a3_a4_a5_candidates_are_exactly_a2_proposals(self) -> None:
        suite, _deberta, _psl = _run_fixture()
        expected_ids = tuple(candidate.candidate_id for candidate in suite.proposals)
        expected_keys = tuple(candidate.key for candidate in suite.proposals)

        self.assertEqual(set(candidate.key for candidate in suite.proposals), {
            WEB_PAYMENT,
            PAYMENT_LEDGER,
        })
        for variant in ("A2", "A3", "A4", "A5"):
            result = suite.results[variant]
            self.assertEqual(result.status, READY)
            self.assertEqual(
                tuple(candidate.candidate_id for candidate in result.candidates),
                expected_ids,
            )
            self.assertEqual(
                tuple(prediction.key for prediction in result.predictions),
                expected_keys,
            )
            self.assertEqual(
                len({prediction.key for prediction in result.predictions}),
                len(expected_keys),
            )
            self.assertEqual(
                tuple(edge.key for edge in result.accepted_edges),
                tuple(
                    prediction.key
                    for prediction in result.predictions
                    if prediction.decision == "accepted"
                ),
            )

    def test_each_deberta_stage_scores_two_directions_for_every_p2_proposal(self) -> None:
        suite, deberta, _psl = _run_fixture()
        expected_pair_count = 2 * len(suite.proposals)

        self.assertEqual(
            suite.results["A3"].activation["nli_pair_count"], expected_pair_count
        )
        self.assertEqual(
            suite.results["A4"].activation["nli_pair_count"], expected_pair_count
        )
        # A3 and A4 are the first two directional calls.  A possible later D0
        # negative-control call is deliberately outside this invariant.
        self.assertGreaterEqual(len(deberta.batches), 2)
        self.assertEqual(len(deberta.batches[0]), expected_pair_count)
        self.assertEqual(len(deberta.batches[1]), expected_pair_count)

        for batch in deberta.batches[:2]:
            hypotheses = [hypothesis for _premise, hypothesis in batch]
            for runtime in _pair_contexts().values():
                forward = (
                    f"Within this runtime system, {runtime.subject_label} directly "
                    f"invokes {runtime.object_label}."
                )
                reverse = (
                    f"Within this runtime system, {runtime.object_label} directly "
                    f"invokes {runtime.subject_label}."
                )
                self.assertEqual(hypotheses.count(forward), 1)
                self.assertEqual(hypotheses.count(reverse), 1)

    def test_reverse_dominance_abstains_and_cannot_reenter_at_a5(self) -> None:
        suite, _deberta, _psl = _run_fixture()

        for variant in ("A3", "A4"):
            prediction = next(
                item
                for item in suite.results[variant].predictions
                if item.key == PAYMENT_LEDGER
            )
            self.assertEqual(prediction.decision, "unresolved")
            self.assertGreater(
                prediction.stage_scores["reverse_entailment"],
                prediction.stage_scores["forward_entailment"],
            )
            self.assertIn("REVERSE_ENTAILMENT_TOO_HIGH", prediction.reason_codes)
            self.assertIn("DIRECTION_MARGIN_INSUFFICIENT", prediction.reason_codes)

        a5_prediction = next(
            item
            for item in suite.results["A5"].predictions
            if item.key == PAYMENT_LEDGER
        )
        self.assertEqual(a5_prediction.decision, "unresolved")
        self.assertIn("A4_GATE_NOT_PASSED", a5_prediction.reason_codes)
        self.assertNotIn(
            PAYMENT_LEDGER,
            {edge.key for edge in suite.results["A5"].accepted_edges},
        )

    def test_a4_runtime_context_is_separate_from_a3_flat_premises(self) -> None:
        suite, deberta, _psl = _run_fixture()
        flat_batch, contextual_batch = deberta.batches[:2]

        self.assertEqual(
            [hypothesis for _premise, hypothesis in flat_batch],
            [hypothesis for _premise, hypothesis in contextual_batch],
        )
        for premise, _hypothesis in flat_batch:
            self.assertNotIn("Runtime context:", premise)
            self.assertNotIn("CTX_WEB_PAYMENT", premise)
            self.assertNotIn("CTX_PAYMENT_LEDGER", premise)
        for premise, _hypothesis in contextual_batch:
            self.assertIn("Runtime context:", premise)

        contextual_premises = [premise for premise, _hypothesis in contextual_batch]
        self.assertEqual(
            sum("CTX_WEB_PAYMENT" in premise for premise in contextual_premises), 2
        )
        self.assertEqual(
            sum("CTX_PAYMENT_LEDGER" in premise for premise in contextual_premises), 2
        )
        self.assertEqual(suite.results["A3"].activation["context_mode"], "flat")
        self.assertEqual(
            suite.results["A4"].activation["context_mode"], "runtime_role"
        )
        self.assertEqual(
            suite.results["A4"].activation["context_available_count"],
            len(suite.proposals),
        )
        self.assertNotEqual(
            suite.results["A3"].activation["premise_sha256"],
            suite.results["A4"].activation["premise_sha256"],
        )

    def test_no_post_a2_stage_can_introduce_an_outside_proposal(self) -> None:
        suite, _deberta, psl = _run_fixture()
        proposal_keys = {candidate.key for candidate in suite.proposals}

        for variant in ("A3", "A4", "A5"):
            result = suite.results[variant]
            self.assertTrue(
                {prediction.key for prediction in result.predictions}.issubset(
                    proposal_keys
                )
            )
            self.assertTrue(
                {edge.key for edge in result.accepted_edges}.issubset(proposal_keys)
            )
            self.assertEqual(
                suite.diagnostics["post_a2_outside_proposal_count"][variant], 0
            )

        self.assertNotIn(
            OUTSIDE_PROPOSALS,
            {edge.key for edge in suite.results["A5"].accepted_edges},
        )
        self.assertEqual(len(psl.calls), 1)
        psl_candidates, local_scores = psl.calls[0]
        self.assertEqual(set(psl_candidates), set(local_scores))
        self.assertEqual(
            set(psl_candidates),
            {edge.key for edge in suite.results["A4"].accepted_edges},
        )
        self.assertTrue(set(psl_candidates).issubset(proposal_keys))

    def test_empty_p2_skips_all_heavy_backend_calls(self) -> None:
        context = InferenceContext(
            incident_id="no-proposal",
            entities=(
                {"entity_id": "service-a", "entity_type": "Service"},
                {"entity_id": "service-b", "entity_type": "Service"},
            ),
        )
        deberta = _RecordingDeberta()
        psl = _RecordingPsl()

        suite = run_cumulative_suite(
            context,
            deberta_backend=deberta,
            psl_backend=psl,
            allow_test_backends=True,
        )

        self.assertGreater(len(suite.evaluation_universe), 0)
        self.assertEqual(suite.proposals, ())
        for variant in ("A2", "A3", "A4", "A5"):
            self.assertEqual(suite.results[variant].status, READY)
            self.assertEqual(suite.results[variant].candidates, ())
            self.assertEqual(suite.results[variant].accepted_edges, ())
        self.assertEqual(suite.results["A3"].activation["nli_pair_count"], 0)
        self.assertEqual(suite.results["A4"].activation["nli_pair_count"], 0)
        self.assertEqual(deberta.availability_calls, 0)
        self.assertEqual(psl.availability_calls, 0)
        self.assertEqual(deberta.batches, [])
        self.assertEqual(psl.calls, [])

    def test_direct_only_p2_does_not_touch_heavy_backends(self) -> None:
        deberta = _RecordingDeberta()
        psl = _RecordingPsl()

        suite = run_cumulative_suite(
            _direct_only_context(),
            deberta_backend=deberta,
            psl_backend=psl,
            allow_test_backends=True,
        )

        self.assertEqual(tuple(candidate.key for candidate in suite.proposals), (WEB_PAYMENT,))
        for variant in ("A2", "A3", "A4", "A5"):
            result = suite.results[variant]
            self.assertEqual(result.status, READY)
            self.assertEqual(tuple(candidate.key for candidate in result.candidates), (WEB_PAYMENT,))
            self.assertEqual(tuple(prediction.key for prediction in result.predictions), (WEB_PAYMENT,))
            self.assertEqual(result.predictions[0].decision, "accepted")
            self.assertEqual(result.predictions[0].evidence_ids, ("declared-web-payment",))
        self.assertEqual(deberta.availability_calls, 0)
        self.assertEqual(psl.availability_calls, 0)
        self.assertEqual(deberta.batches, [])
        self.assertEqual(psl.calls, [])
        self.assertEqual(suite.results["A5"].activation["protected_direct_count"], 1)

    def test_malformed_or_nonfinite_nli_scores_are_errors(self) -> None:
        malformed_scores = {
            "missing_contradiction": {"entailment": 0.80, "neutral": 0.20},
            "nan_entailment": {
                "entailment": float("nan"),
                "contradiction": 0.10,
                "neutral": 0.10,
            },
        }
        for label, raw_score in malformed_scores.items():
            with self.subTest(label=label):
                suite = run_cumulative_suite(
                    _l2_context(),
                    pair_contexts=_pair_contexts(),
                    deberta_backend=_FixedOutputDeberta(raw_score),
                    psl_backend=_RecordingPsl(),
                    allow_test_backends=True,
                )
                for variant in ("A3", "A4"):
                    self.assertEqual(suite.results[variant].status, "ERROR")
                    self.assertEqual(
                        suite.results[variant].reason_code,
                        "DEBERTA_SCORE_INVALID",
                    )

    def test_psl_requires_one_finite_score_per_eligible_candidate(self) -> None:
        for mode in ("missing", "nan"):
            with self.subTest(mode=mode):
                psl = _MalformedPsl(mode)
                suite = run_cumulative_suite(
                    _l2_context(),
                    pair_contexts=_pair_contexts(),
                    deberta_backend=_RecordingDeberta(),
                    psl_backend=psl,
                    allow_test_backends=True,
                )

                self.assertEqual(len(psl.calls), 1)
                self.assertEqual(suite.results["A5"].status, "ERROR")
                self.assertEqual(
                    suite.results["A5"].reason_code,
                    "PSL_SCORE_INVALID",
                )

    def test_direct_calls_are_protected_from_psl_pruning(self) -> None:
        deberta = _AcceptAllDirectionalDeberta()
        psl = _RecordingPsl()
        suite = run_cumulative_suite(
            _mixed_direct_and_abductive_context(),
            pair_contexts=_pair_contexts(),
            deberta_backend=deberta,
            psl_backend=psl,
            allow_test_backends=True,
        )

        self.assertEqual(
            tuple(candidate.key for candidate in suite.proposals),
            (PAYMENT_LEDGER, WEB_PAYMENT),
        )
        self.assertEqual(len(psl.calls), 1)
        psl_candidate_keys, local_scores = psl.calls[0]
        self.assertEqual(psl_candidate_keys, (PAYMENT_LEDGER,))
        self.assertEqual(set(local_scores), {PAYMENT_LEDGER})
        self.assertNotIn(WEB_PAYMENT, psl_candidate_keys)

        direct = next(
            prediction
            for prediction in suite.results["A5"].predictions
            if prediction.key == WEB_PAYMENT
        )
        self.assertEqual(direct.decision, "accepted")
        self.assertEqual(direct.score, 1.0)
        self.assertEqual(direct.evidence_ids, ("declared-web-payment",))
        self.assertIn(
            WEB_PAYMENT,
            {edge.key for edge in suite.results["A5"].accepted_edges},
        )
        self.assertEqual(suite.results["A5"].activation["protected_direct_count"], 1)


if __name__ == "__main__":
    unittest.main()
