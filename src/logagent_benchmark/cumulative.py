"""Leakage-safe cumulative relation-recovery pipeline (v2).

The v1 ablations intentionally scored one shared all-pairs universe so that a
standalone DeBERTa control could be measured.  That is useful as a negative
control but is not the intended operational pipeline.  This module enforces a
different contract:

``A2 proposals -> A3 flat directional NLI -> A4 runtime-context NLI -> A5 PSL``

No A3--A5 stage may score or introduce a relation that A2 did not propose.
The evaluator graph and mask manifest are absent from every public API here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .recovery import (
    ERROR,
    READY,
    SKIPPED,
    AblationConfig,
    Availability,
    Candidate,
    CandidatePrediction,
    DebertaBackend,
    Edge,
    InferenceContext,
    PslBackend,
    RecoveryResult,
    RelationSpec,
    DEFAULT_RELATION_SPECS,
    _backend_availability,
    _field,
    build_typed_candidates,
    run_recovery,
    temporal_containment_details,
)


@dataclass(frozen=True)
class PairRuntimeContext:
    """Allowed model-partition context for one ordered candidate pair."""

    subject_label: str
    object_label: str
    contextual_addendum: str = ""
    provenance: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.contextual_addendum.strip() and self.provenance)


@dataclass(frozen=True)
class CumulativeConfig:
    a2_threshold: float = 0.60
    entailment_threshold: float = 0.67
    reverse_entailment_ceiling: float = 0.33
    direction_margin: float = 0.05
    psl_threshold: float = 0.70
    include_null_parent: bool = True

    def __post_init__(self) -> None:
        probability_fields = (
            self.a2_threshold,
            self.entailment_threshold,
            self.reverse_entailment_ceiling,
            self.psl_threshold,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probability_fields):
            raise ValueError("all cumulative thresholds must be finite probabilities")
        if not math.isfinite(self.direction_margin) or not -1.0 <= self.direction_margin <= 1.0:
            raise ValueError("direction_margin must be finite and in [-1, 1]")


@dataclass(frozen=True)
class CumulativeSuiteResult:
    evaluation_universe: tuple[Candidate, ...]
    proposals: tuple[Candidate, ...]
    results: Mapping[str, RecoveryResult]
    controls: Mapping[str, RecoveryResult]
    activation: Mapping[str, Any]
    gate: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def _display_id(entity_id: str) -> str:
    return str(entity_id).rsplit(":", 1)[-1]


def _edge_from_prediction(prediction: CandidatePrediction, variant: str) -> Edge:
    return Edge(
        prediction.subject,
        prediction.predicate,
        prediction.object,
        prediction.score,
        "inferred",
        prediction.evidence_ids,
        variant,
    )


def _ready_result(
    variant: str,
    context: InferenceContext,
    candidates: Sequence[Candidate],
    predictions: Sequence[CandidatePrediction],
    *,
    activation: Mapping[str, Any],
    research_valid: bool = True,
) -> RecoveryResult:
    observed = run_recovery("A0", context, candidates=()).observed_edges
    candidate_tuple = tuple(candidates)
    prediction_tuple = tuple(predictions)
    candidate_keys = tuple(candidate.key for candidate in candidate_tuple)
    prediction_keys = tuple(prediction.key for prediction in prediction_tuple)
    if len(set(candidate_keys)) != len(candidate_keys):
        raise AssertionError(f"{variant} contains duplicate candidates")
    if prediction_keys != candidate_keys:
        raise AssertionError(
            f"{variant} READY predictions must match candidates 1:1 and in order"
        )
    accepted_prediction_keys = {
        prediction.key
        for prediction in prediction_tuple
        if prediction.decision == "accepted"
    }
    if len(accepted_prediction_keys) != sum(
        prediction.decision == "accepted" for prediction in prediction_tuple
    ):
        raise AssertionError(f"{variant} contains duplicate accepted predictions")
    accepted_edges = tuple(
        _edge_from_prediction(prediction, variant)
        for prediction in prediction_tuple
        if prediction.decision == "accepted"
    )
    if {edge.key for edge in accepted_edges} != accepted_prediction_keys:
        raise AssertionError(f"{variant} accepted edges do not match predictions")
    return RecoveryResult(
        variant=variant,
        status=READY,
        candidates=candidate_tuple,
        predictions=prediction_tuple,
        accepted_edges=accepted_edges,
        observed_edges=observed,
        activation=dict(activation),
        research_valid=research_valid,
    )


def _unavailable_result(
    variant: str,
    context: InferenceContext,
    candidates: Sequence[Candidate],
    availability: Availability,
) -> RecoveryResult:
    observed = run_recovery("A0", context, candidates=()).observed_edges
    return RecoveryResult(
        variant=variant,
        status=availability.status,
        candidates=tuple(candidates),
        observed_edges=observed,
        activation={"candidate_count": len(candidates), "stage_calls": 0},
        reason_code=availability.reason_code,
        detail=availability.detail,
        research_valid=availability.research_valid,
    )


def _nli_probabilities(raw: Any) -> tuple[float, float, float]:
    if isinstance(raw, Mapping):
        entailment_raw = _field(raw, "entailment", "p_entailment")
        contradiction_raw = _field(raw, "contradiction", "p_contradiction")
        if entailment_raw is None or contradiction_raw is None:
            raise ValueError("NLI score must contain entailment and contradiction")
        entailment = float(entailment_raw)
        contradiction = float(contradiction_raw)
        neutral_raw = _field(raw, "neutral", "p_neutral")
        neutral = (
            float(neutral_raw)
            if neutral_raw is not None
            else 1.0 - entailment - contradiction
        )
    else:
        entailment = float(raw)
        contradiction = 0.0
        neutral = max(0.0, 1.0 - entailment)
    values = (entailment, contradiction, neutral)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"NLI probabilities must be finite and in [0,1]: {values}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NLI probabilities must sum to one: {values}")
    return values


def _flat_premise(
    candidate: Candidate,
    detail: Any,
    pair_context: PairRuntimeContext | None,
) -> tuple[str, str, str]:
    subject = pair_context.subject_label if pair_context else _display_id(candidate.subject)
    object_id = pair_context.object_label if pair_context else _display_id(candidate.object)
    premise = (
        "Model-partition telemetry contains a missing-parent child span whose "
        f"interval is immediately and unambiguously contained by a {subject} span. "
        f"The contained child belongs to {object_id}. This pattern occurs at "
        f"{detail.boundary_count} span boundaries across {detail.trace_count} whole "
        "traces. It is candidate evidence, not a confirmed dependency."
    )
    return premise, subject, object_id


def _score_directional_stage(
    variant: str,
    *,
    context: InferenceContext,
    candidates: Sequence[Candidate],
    details: Mapping[tuple[str, str, str], Any],
    direct_support: Mapping[tuple[str, str, str], CandidatePrediction],
    pair_contexts: Mapping[tuple[str, str, str], PairRuntimeContext],
    contextual: bool,
    backend: DebertaBackend,
    availability: Availability,
    config: CumulativeConfig,
) -> RecoveryResult:
    direct_keys = set(direct_support)
    score_candidates = [candidate for candidate in candidates if candidate.key not in direct_keys]
    missing_details = [
        candidate.key for candidate in score_candidates if candidate.key not in details
    ]
    if missing_details:
        raise AssertionError(
            "non-direct A2 proposals lack temporal-containment details: "
            f"{missing_details[:5]}"
        )
    missing_context = [
        candidate.key
        for candidate in score_candidates
        if pair_contexts.get(candidate.key) is None
        or not pair_contexts[candidate.key].available
    ]
    if contextual and missing_context:
        return _unavailable_result(
            variant,
            context,
            candidates,
            Availability(
                SKIPPED,
                "ROLE_CONTEXT_INCOMPLETE",
                f"runtime context is missing for {len(missing_context)} A2 proposals",
                availability.research_valid,
            ),
        )

    pairs: list[tuple[str, str]] = []
    metadata: list[tuple[Candidate, str, tuple[str, ...], str]] = []
    for candidate in score_candidates:
        detail = details[candidate.key]
        runtime = pair_contexts.get(candidate.key)
        premise, subject, object_id = _flat_premise(candidate, detail, runtime)
        provenance: tuple[str, ...] = ("temporal_containment",)
        if contextual and runtime is not None and runtime.available:
            premise = f"{premise}\nRuntime context:\n{runtime.contextual_addendum.strip()}"
            provenance += runtime.provenance
        forward = f"Within this runtime system, {subject} directly invokes {object_id}."
        reverse = f"Within this runtime system, {object_id} directly invokes {subject}."
        pairs.extend(((premise, forward), (premise, reverse)))
        metadata.append((candidate, premise, provenance, forward))

    token_counts: tuple[int, ...] = ()
    token_diagnostics: list[dict[str, Any]] = []
    token_length_method = getattr(backend, "pair_token_lengths", None)
    if callable(token_length_method) and pairs:
        try:
            token_counts = tuple(int(value) for value in token_length_method(tuple(pairs)))
            if len(token_counts) != len(pairs):
                raise ValueError(
                    f"expected {len(pairs)} token lengths, got {len(token_counts)}"
                )
            if any(value <= 0 for value in token_counts):
                raise ValueError("NLI token lengths must be positive")
            maximum = int(getattr(backend, "max_length", 512))
            over_budget = [
                (index, value)
                for index, value in enumerate(token_counts)
                if value > maximum
            ]
            if over_budget:
                raise OverflowError(
                    f"NLI pair token budget exceeded: max={maximum}, items={over_budget}"
                )
            for index, (candidate, _premise, _provenance, _hypothesis) in enumerate(metadata):
                token_diagnostics.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "forward_original_tokens": token_counts[index * 2],
                        "reverse_original_tokens": token_counts[index * 2 + 1],
                        "forward_used_tokens": token_counts[index * 2],
                        "reverse_used_tokens": token_counts[index * 2 + 1],
                        "truncation_occurred": False,
                        "dropped_fields": [],
                    }
                )
        except OverflowError as exc:
            return _unavailable_result(
                variant,
                context,
                candidates,
                Availability(
                    ERROR,
                    "DEBERTA_INPUT_TOO_LONG",
                    str(exc),
                    availability.research_valid,
                ),
            )
        except (TypeError, ValueError) as exc:
            return _unavailable_result(
                variant,
                context,
                candidates,
                Availability(
                    ERROR,
                    "DEBERTA_TOKEN_DIAGNOSTIC_INVALID",
                    str(exc),
                    availability.research_valid,
                ),
            )

    try:
        raw_scores = tuple(backend.score_pairs(tuple(pairs))) if pairs else ()
    except Exception as exc:
        return _unavailable_result(
            variant,
            context,
            candidates,
            Availability(ERROR, "DEBERTA_INFERENCE_ERROR", str(exc), availability.research_valid),
        )
    if len(raw_scores) != len(pairs):
        return _unavailable_result(
            variant,
            context,
            candidates,
            Availability(
                ERROR,
                "DEBERTA_SCORE_COUNT_MISMATCH",
                f"expected {len(pairs)} scores, got {len(raw_scores)}",
                availability.research_valid,
            ),
        )

    scored: dict[tuple[str, str, str], CandidatePrediction] = {}
    margins: list[float] = []
    try:
        for index, (candidate, premise, provenance, _forward_hypothesis) in enumerate(metadata):
            forward_entailment, forward_contradiction, forward_neutral = _nli_probabilities(
                raw_scores[index * 2]
            )
            reverse_entailment, reverse_contradiction, reverse_neutral = _nli_probabilities(
                raw_scores[index * 2 + 1]
            )
            margin = forward_entailment - reverse_entailment
            combined = forward_entailment * (1.0 - reverse_entailment)
            margins.append(margin)
            reasons: list[str] = []
            if forward_entailment < config.entailment_threshold:
                reasons.append("FORWARD_ENTAILMENT_BELOW_THRESHOLD")
            if reverse_entailment > config.reverse_entailment_ceiling:
                reasons.append("REVERSE_ENTAILMENT_TOO_HIGH")
            if margin < config.direction_margin:
                reasons.append("DIRECTION_MARGIN_INSUFFICIENT")
            detail = details[candidate.key]
            scored[candidate.key] = CandidatePrediction(
                candidate.subject,
                candidate.predicate,
                candidate.object,
                # The decision and PSL local truth share raw forward
                # entailment.  Reverse support remains an explicit hard gate
                # and a separately reported directional confidence.
                forward_entailment,
                "accepted" if not reasons else "unresolved",
                detail.evidence_ids,
                {
                    "abduction": detail.score,
                    "forward_entailment": forward_entailment,
                    "forward_contradiction": forward_contradiction,
                    "forward_neutral": forward_neutral,
                    "reverse_entailment": reverse_entailment,
                    "reverse_contradiction": reverse_contradiction,
                    "reverse_neutral": reverse_neutral,
                    "direction_margin": margin,
                    "directional_score": combined,
                },
                tuple(reasons),
            )
    except (TypeError, ValueError, OverflowError) as exc:
        return _unavailable_result(
            variant,
            context,
            candidates,
            Availability(
                ERROR,
                "DEBERTA_SCORE_INVALID",
                str(exc),
                availability.research_valid,
            ),
        )

    predictions: list[CandidatePrediction] = []
    for candidate in candidates:
        if candidate.key in direct_keys:
            direct = direct_support[candidate.key]
            predictions.append(
                CandidatePrediction(
                    candidate.subject,
                    candidate.predicate,
                    candidate.object,
                    1.0,
                    "accepted",
                    direct.evidence_ids,
                    {"direct_passthrough": 1.0},
                    ("DIRECT_EVIDENCE_PASSTHROUGH",),
                )
            )
        else:
            predictions.append(scored[candidate.key])

    premise_fingerprint = hashlib.sha256(
        json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _ready_result(
        variant,
        context,
        candidates,
        predictions,
        activation={
            "candidate_count": len(candidates),
            "scored_candidate_count": len(score_candidates),
            "nli_pair_count": len(pairs),
            "forward_reverse_pairs_per_candidate": 2,
            "context_mode": "runtime_role" if contextual else "flat",
            "context_available_count": sum(
                bool(pair_contexts.get(candidate.key) and pair_contexts[candidate.key].available)
                for candidate in score_candidates
            ),
            "mean_direction_margin": sum(margins) / len(margins) if margins else None,
            "minimum_direction_margin": min(margins) if margins else None,
            "maximum_direction_margin": max(margins) if margins else None,
            "premise_sha256": premise_fingerprint,
            "token_budget_checked": bool(token_counts),
            "maximum_pair_tokens": max(token_counts) if token_counts else None,
            "minimum_pair_tokens": min(token_counts) if token_counts else None,
            "truncation_count": 0 if token_counts else None,
            "pair_token_diagnostics": token_diagnostics,
        },
        research_valid=availability.research_valid,
    )


def _psl_result(
    variant: str,
    *,
    context: InferenceContext,
    proposals: Sequence[Candidate],
    a4: RecoveryResult,
    backend: PslBackend | None,
    availability: Availability | None,
    threshold: float,
    psl_relation: str,
    protected_keys: frozenset[tuple[str, str, str]] = frozenset(),
) -> RecoveryResult:
    a4_predictions = {prediction.key: prediction for prediction in a4.predictions}
    eligible = tuple(
        candidate
        for candidate in proposals
        if a4_predictions[candidate.key].decision == "accepted"
        and candidate.predicate == psl_relation
        and candidate.key not in protected_keys
    )
    local_scores = {
        candidate.key: a4_predictions[candidate.key].score for candidate in eligible
    }
    if not eligible:
        predictions = []
        for candidate in proposals:
            previous = a4_predictions[candidate.key]
            if previous.decision == "accepted" and candidate.key in protected_keys:
                predictions.append(
                    replace(
                        previous,
                        reason_codes=tuple(previous.reason_codes)
                        + ("DIRECT_EVIDENCE_PROTECTED",),
                    )
                )
            elif previous.decision == "accepted" and candidate.predicate != psl_relation:
                predictions.append(
                    replace(
                        previous,
                        reason_codes=tuple(previous.reason_codes)
                        + ("PSL_RELATION_PASSTHROUGH",),
                    )
                )
            else:
                predictions.append(
                    replace(
                        previous,
                        decision="unresolved",
                        reason_codes=tuple(previous.reason_codes)
                        + ("A4_GATE_NOT_PASSED",),
                    )
                )
        return _ready_result(
            variant,
            context,
            proposals,
            predictions,
            activation={
                "candidate_count": len(proposals),
                "psl_relation": psl_relation,
                "psl_candidate_count": 0,
                "protected_direct_count": sum(
                    candidate.key in protected_keys for candidate in proposals
                ),
                "grounded_rule_count": 0,
                "grounded_atom_count": 0,
                "psl_metadata": {},
                "outside_a2_proposal_count": 0,
                "stage_activated": False,
                "no_op_reason": "NO_ELIGIBLE_A4_RELATIONS",
            },
            research_valid=a4.research_valid,
        )

    if backend is None or availability is None:
        raise AssertionError("PSL backend and availability are required for eligible atoms")
    try:
        raw = backend.infer(context=context, candidates=eligible, local_scores=local_scores)
    except Exception as exc:
        return _unavailable_result(
            variant,
            context,
            proposals,
            Availability(ERROR, "PSL_INFERENCE_ERROR", str(exc), availability.research_valid),
        )

    try:
        rule_count = int(_field(raw, "grounded_rule_count", default=0))
        atom_count = int(_field(raw, "grounded_atom_count", default=0))
        metadata = dict(_field(raw, "metadata", default={}) or {})
        values = _field(raw, "scores", "posteriors", default=raw)
        if isinstance(values, Mapping):
            psl_scores: dict[tuple[str, str, str], float] = {}
            for candidate in eligible:
                if candidate.key in values:
                    raw_score = values[candidate.key]
                elif candidate.candidate_id in values:
                    raw_score = values[candidate.candidate_id]
                else:
                    raise ValueError(f"PSL score missing for {candidate.candidate_id}")
                psl_scores[candidate.key] = float(raw_score)
        else:
            sequence = tuple(values)
            if len(sequence) != len(eligible):
                raise ValueError("PSL score count differs from A4-accepted candidates")
            psl_scores = {
                candidate.key: float(score)
                for candidate, score in zip(eligible, sequence)
            }
        invalid = {
            key: score
            for key, score in psl_scores.items()
            if not math.isfinite(score) or not 0.0 <= score <= 1.0
        }
        if invalid:
            raise ValueError(f"PSL scores must be finite probabilities: {invalid}")
    except (TypeError, ValueError, OverflowError) as exc:
        return _unavailable_result(
            variant,
            context,
            proposals,
            Availability(ERROR, "PSL_SCORE_INVALID", str(exc), availability.research_valid),
        )

    predictions: list[CandidatePrediction] = []
    for candidate in proposals:
        previous = a4_predictions[candidate.key]
        if previous.decision == "accepted" and candidate.key in protected_keys:
            predictions.append(
                replace(
                    previous,
                    reason_codes=tuple(previous.reason_codes)
                    + ("DIRECT_EVIDENCE_PROTECTED",),
                )
            )
            continue
        if previous.decision == "accepted" and candidate.predicate != psl_relation:
            predictions.append(
                replace(
                    previous,
                    reason_codes=tuple(previous.reason_codes)
                    + ("PSL_RELATION_PASSTHROUGH",),
                )
            )
            continue
        if previous.decision != "accepted":
            predictions.append(
                replace(
                    previous,
                    decision="unresolved",
                    reason_codes=tuple(previous.reason_codes) + ("A4_GATE_NOT_PASSED",),
                )
            )
            continue
        score = psl_scores[candidate.key]
        accepted = score >= threshold
        predictions.append(
            CandidatePrediction(
                candidate.subject,
                candidate.predicate,
                candidate.object,
                score,
                "accepted" if accepted else "unresolved",
                previous.evidence_ids,
                {**previous.stage_scores, "psl": score},
                tuple(previous.reason_codes)
                + (("PSL_ACCEPTED",) if accepted else ("PSL_BELOW_THRESHOLD",)),
            )
        )
    return _ready_result(
        variant,
        context,
        proposals,
        predictions,
        activation={
            "candidate_count": len(proposals),
            "psl_relation": psl_relation,
            "psl_candidate_count": len(eligible),
            "protected_direct_count": sum(
                candidate.key in protected_keys for candidate in proposals
            ),
            "grounded_rule_count": rule_count,
            "grounded_atom_count": atom_count,
            "psl_metadata": metadata,
            "outside_a2_proposal_count": 0,
            "stage_activated": True,
        },
        research_valid=availability.research_valid and a4.research_valid,
    )


def _relabel_control(result: RecoveryResult, variant: str) -> RecoveryResult:
    return replace(
        result,
        variant=variant,
        accepted_edges=tuple(replace(edge, method=variant) for edge in result.accepted_edges),
        activation={
            **dict(result.activation),
            "control_only": True,
            "comparable_to_directional_stages": False,
            "legacy_score_formula": "(1 + entailment - contradiction) / 2",
        },
    )


def _direct_only_directional_result(
    variant: str,
    *,
    context: InferenceContext,
    proposals: Sequence[Candidate],
    direct_support: Mapping[tuple[str, str, str], CandidatePrediction],
    contextual: bool,
) -> RecoveryResult:
    predictions: list[CandidatePrediction] = []
    for candidate in proposals:
        if candidate.key not in direct_support:
            raise AssertionError("direct-only stage received a non-direct proposal")
        direct = direct_support[candidate.key]
        predictions.append(
            CandidatePrediction(
                candidate.subject,
                candidate.predicate,
                candidate.object,
                1.0,
                "accepted",
                direct.evidence_ids,
                {"direct_passthrough": 1.0},
                ("DIRECT_EVIDENCE_PASSTHROUGH",),
            )
        )
    return _ready_result(
        variant,
        context,
        proposals,
        predictions,
        activation={
            "candidate_count": len(proposals),
            "scored_candidate_count": 0,
            "nli_pair_count": 0,
            "context_mode": "runtime_role" if contextual else "flat",
            "context_available_count": 0,
            "stage_activated": False,
            "no_op_reason": "DIRECT_EVIDENCE_ONLY" if proposals else "NO_A2_PROPOSALS",
        },
    )


def _assert_stage_invariants(
    proposals: Sequence[Candidate],
    results: Mapping[str, RecoveryResult],
) -> None:
    proposal_keys = tuple(candidate.key for candidate in proposals)
    if len(set(proposal_keys)) != len(proposal_keys):
        raise AssertionError("A2 proposal set contains duplicate keys")
    for variant in ("A2", "A3", "A4", "A5"):
        result = results[variant]
        candidate_keys = tuple(candidate.key for candidate in result.candidates)
        if candidate_keys != proposal_keys:
            raise AssertionError(
                f"{variant} candidates differ from ordered A2 proposal tuple"
            )
        if len(set(candidate_keys)) != len(candidate_keys):
            raise AssertionError(f"{variant} contains duplicate candidate keys")
        prediction_keys = tuple(prediction.key for prediction in result.predictions)
        if result.status == READY and prediction_keys != proposal_keys:
            raise AssertionError(
                f"{variant} READY predictions differ from ordered A2 proposals"
            )
        if len(set(prediction_keys)) != len(prediction_keys):
            raise AssertionError(f"{variant} contains duplicate prediction keys")
        if not set(prediction_keys).issubset(set(proposal_keys)):
            raise AssertionError(f"{variant} predicted an edge outside A2 proposals")
        accepted_predictions = {
            prediction.key
            for prediction in result.predictions
            if prediction.decision == "accepted"
        }
        accepted_edges = {edge.key for edge in result.accepted_edges}
        if accepted_edges != accepted_predictions:
            raise AssertionError(
                f"{variant} accepted edges differ from accepted predictions"
            )


def run_cumulative_suite(
    context: InferenceContext,
    *,
    pair_contexts: Mapping[tuple[str, str, str], PairRuntimeContext] | None = None,
    config: CumulativeConfig | None = None,
    relation_specs: Mapping[str, RelationSpec] = DEFAULT_RELATION_SPECS,
    deberta_backend: DebertaBackend | None = None,
    psl_backend: PslBackend | None = None,
    allow_test_backends: bool = False,
    run_d0_control: bool = False,
) -> CumulativeSuiteResult:
    """Run v2 without accepting evaluator-only labels or mask answers."""

    config = config or CumulativeConfig()
    pair_contexts = pair_contexts or {}
    universe = build_typed_candidates(context, relation_specs)
    a0 = run_recovery("A0", context, candidates=universe)
    a1 = run_recovery("A1", context, candidates=universe)
    direct_support = {
        prediction.key: prediction
        for prediction in a1.predictions
        if prediction.decision == "accepted"
    }
    direct_keys = set(direct_support)
    details = temporal_containment_details(
        context, include_null_parent=config.include_null_parent
    )
    candidate_by_key = {candidate.key: candidate for candidate in universe}
    abductive_keys = {
        key
        for key, detail in details.items()
        if key in candidate_by_key and detail.score >= config.a2_threshold
    }
    proposal_keys = direct_keys | abductive_keys
    proposals = tuple(candidate_by_key[key] for key in sorted(proposal_keys))
    non_direct_proposal_keys = proposal_keys - direct_keys
    missing_details = sorted(non_direct_proposal_keys - set(details))
    if missing_details:
        raise AssertionError(
            "non-direct A2 proposals lack temporal-containment details: "
            f"{missing_details[:5]}"
        )
    a2_predictions: list[CandidatePrediction] = []
    for candidate in proposals:
        if candidate.key in direct_keys:
            direct = direct_support[candidate.key]
            a2_predictions.append(
                CandidatePrediction(
                    candidate.subject,
                    candidate.predicate,
                    candidate.object,
                    1.0,
                    "accepted",
                    direct.evidence_ids,
                    {"direct": 1.0},
                    ("DIRECT_EVIDENCE",),
                )
            )
        else:
            detail = details[candidate.key]
            a2_predictions.append(
                CandidatePrediction(
                    candidate.subject,
                    candidate.predicate,
                    candidate.object,
                    detail.score,
                    "accepted",
                    detail.evidence_ids,
                    {
                        "abduction": detail.score,
                        "supporting_traces": float(detail.trace_count),
                        "boundary_spans": float(detail.boundary_count),
                    },
                    ("TEMPORAL_CONTAINMENT_PROPOSAL",),
                )
            )
    a2 = _ready_result(
        "A2",
        context,
        proposals,
        a2_predictions,
        activation={
            "evaluation_universe_count": len(universe),
            "candidate_count": len(proposals),
            "direct_proposal_count": len(direct_keys),
            "abductive_proposal_count": len(abductive_keys),
            "compression_ratio": 1.0 - (len(proposals) / len(universe)) if universe else None,
            "include_null_parent": config.include_null_parent,
            "stage_activated": bool(proposals),
        },
    )

    if not non_direct_proposal_keys:
        # Empty and direct-only proposal sets are deterministic pass-throughs;
        # touching a heavy backend here would make availability an accidental
        # prerequisite for a stage that has no model work to perform.
        a3 = _direct_only_directional_result(
            "A3",
            context=context,
            proposals=proposals,
            direct_support=direct_support,
            contextual=False,
        )
        a4 = _direct_only_directional_result(
            "A4",
            context=context,
            proposals=proposals,
            direct_support=direct_support,
            contextual=True,
        )
    else:
        deberta_availability = _backend_availability(
            deberta_backend, "DEBERTA", allow_test_backends
        )
        if deberta_availability.status != READY:
            a3 = _unavailable_result("A3", context, proposals, deberta_availability)
            a4 = _unavailable_result("A4", context, proposals, deberta_availability)
        else:
            assert deberta_backend is not None
            a3 = _score_directional_stage(
                "A3",
                context=context,
                candidates=proposals,
                details=details,
                direct_support=direct_support,
                pair_contexts=pair_contexts,
                contextual=False,
                backend=deberta_backend,
                availability=deberta_availability,
                config=config,
            )
            a4 = _score_directional_stage(
                "A4",
                context=context,
                candidates=proposals,
                details=details,
                direct_support=direct_support,
                pair_contexts=pair_contexts,
                contextual=True,
                backend=deberta_backend,
                availability=deberta_availability,
                config=config,
            )

    if a4.status != READY:
        a5 = _unavailable_result(
            "A5",
            context,
            proposals,
            Availability(
                SKIPPED,
                "A4_NOT_READY",
                a4.reason_code,
                a4.research_valid,
            ),
        )
    else:
        psl_relation = (
            str(getattr(psl_backend, "relation", "CALLS")).strip().upper()
            or "CALLS"
        )
        a4_prediction_map = {prediction.key: prediction for prediction in a4.predictions}
        psl_eligible = tuple(
            candidate
            for candidate in proposals
            if a4_prediction_map[candidate.key].decision == "accepted"
            and candidate.predicate == psl_relation
            and candidate.key not in direct_keys
        )
        if not psl_eligible:
            a5 = _psl_result(
                "A5",
                context=context,
                proposals=proposals,
                a4=a4,
                backend=None,
                availability=None,
                threshold=config.psl_threshold,
                psl_relation=psl_relation,
                protected_keys=frozenset(direct_keys),
            )
        else:
            psl_availability = _backend_availability(
                psl_backend, "PSL", allow_test_backends
            )
            if psl_availability.status != READY:
                a5 = _unavailable_result("A5", context, proposals, psl_availability)
            else:
                assert psl_backend is not None
                a5 = _psl_result(
                    "A5",
                    context=context,
                    proposals=proposals,
                    a4=a4,
                    backend=psl_backend,
                    availability=psl_availability,
                    threshold=config.psl_threshold,
                    psl_relation=psl_relation,
                    protected_keys=frozenset(direct_keys),
                )

    results = {"A0": a0, "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5}
    controls: dict[str, RecoveryResult] = {}
    if run_d0_control:
        if deberta_backend is None:
            controls["D0_LEGACY"] = _unavailable_result(
                "D0_LEGACY",
                context,
                universe,
                Availability(SKIPPED, "DEBERTA_BACKEND_MISSING"),
            )
        else:
            d0_raw = run_recovery(
                "A3",
                context,
                candidates=universe,
                config=AblationConfig(a3_threshold=config.entailment_threshold),
                relation_specs=relation_specs,
                deberta_backend=deberta_backend,
                allow_test_backends=allow_test_backends,
            )
            controls["D0_LEGACY"] = _relabel_control(d0_raw, "D0_LEGACY")

    _assert_stage_invariants(proposals, results)

    from .metrics import activation_matrix, zero_flip_gate

    activation = activation_matrix(
        results,
        pairs=(("A2", "A3"), ("A3", "A4"), ("A4", "A5")),
    )
    gate = dict(zero_flip_gate(
        results,
        activation,
        require_variants=("A3", "A4", "A5"),
        require_activation_pairs=(("A2", "A3"), ("A3", "A4"), ("A4", "A5")),
    ))
    non_research_valid = [
        variant
        for variant in ("A3", "A4", "A5")
        if results[variant].status == READY and not results[variant].research_valid
    ]
    gate["non_research_valid_variants"] = non_research_valid
    if non_research_valid:
        reasons = list(gate.get("reason_codes", ()))
        if "NON_RESEARCH_VALID_STAGE" not in reasons:
            reasons.append("NON_RESEARCH_VALID_STAGE")
        gate["reason_codes"] = reasons
        gate["passed"] = False
        gate["status"] = "FAIL"
    accepted = {
        variant: {edge.key for edge in result.accepted_edges}
        for variant, result in results.items()
    }
    outside = {
        variant: sorted(accepted[variant] - proposal_keys)
        for variant in ("A3", "A4", "A5")
    }
    if any(outside.values()):
        raise AssertionError(f"post-A2 stage introduced an outside proposal: {outside}")
    diagnostics = {
        "evaluation_universe_count": len(universe),
        "a2_proposal_count": len(proposals),
        "a2_candidate_compression_ratio": (
            1.0 - len(proposals) / len(universe) if universe else None
        ),
        "post_a2_outside_proposal_count": {
            variant: len(values) for variant, values in outside.items()
        },
        "a3_a4_same_candidate_tuple": tuple(a3.candidates) == tuple(a4.candidates),
        "a4_context_available_count": a4.activation.get("context_available_count", 0),
    }
    return CumulativeSuiteResult(
        evaluation_universe=universe,
        proposals=proposals,
        results=results,
        controls=controls,
        activation=activation,
        gate=gate,
        diagnostics=diagnostics,
    )


__all__ = [
    "CumulativeConfig",
    "CumulativeSuiteResult",
    "PairRuntimeContext",
    "run_cumulative_suite",
]
