"""Evaluator-only metrics for relation-recovery ablations.

Unlike :mod:`logagent_benchmark.recovery`, functions here may consume masked
targets and a held-out silver graph.  This boundary is intentional and tested.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

from .recovery import READY, CandidatePrediction, RecoveryResult, edge_key


DEFAULT_ACTIVATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("A0", "A1"),
    ("A1", "A2"),
    ("A0", "A3"),
    ("A2", "A4"),
    ("A3", "A4"),
    ("A4", "A5"),
)


def _edge_set(edges: Iterable[Any] | None) -> set[tuple[str, str, str]]:
    if edges is None:
        return set()
    output: set[tuple[str, str, str]] = set()
    for edge in edges:
        output.add(edge_key(edge))
    return output


def _metric(value: float | None, *, reason: str | None = None) -> Mapping[str, Any]:
    return {"value": value, "reason": reason}


def _safe_ratio(numerator: int | float, denominator: int | float, reason: str) -> Mapping[str, Any]:
    if denominator == 0:
        return _metric(None, reason=reason)
    return _metric(float(numerator) / float(denominator))


def candidate_recall(
    result: RecoveryResult,
    masked_edges: Iterable[Any],
) -> Mapping[str, Any]:
    targets = _edge_set(masked_edges)
    candidate_keys = {candidate.key for candidate in result.candidates}
    matched = targets & candidate_keys
    return {
        "candidate_count": len(candidate_keys),
        "target_count": len(targets),
        "matched_target_count": len(matched),
        "recall": _safe_ratio(len(matched), len(targets), "NO_MASKED_TARGET_EDGES"),
        "missing_target_edges": [list(key) for key in sorted(targets - candidate_keys)],
    }


def masked_recall(
    result: RecoveryResult,
    masked_edges: Iterable[Any],
) -> Mapping[str, Any]:
    targets = _edge_set(masked_edges)
    accepted = {edge.key for edge in result.accepted_edges}
    recovered = targets & accepted
    return {
        "target_count": len(targets),
        "recovered_count": len(recovered),
        "recall": _safe_ratio(len(recovered), len(targets), "NO_MASKED_TARGET_EDGES"),
        "recovered_edges": [list(key) for key in sorted(recovered)],
    }


def relation_masked_recall(
    result: RecoveryResult,
    masked_edges: Iterable[Any],
) -> Mapping[str, Any]:
    targets = _edge_set(masked_edges)
    accepted = {edge.key for edge in result.accepted_edges}
    predicates = sorted({key[1] for key in targets})
    output: dict[str, Any] = {}
    for predicate in predicates:
        relation_targets = {key for key in targets if key[1] == predicate}
        recovered = relation_targets & accepted
        output[predicate] = {
            "target_count": len(relation_targets),
            "recovered_count": len(recovered),
            "recall": _safe_ratio(
                len(recovered), len(relation_targets), "NO_MASKED_TARGET_EDGES"
            ),
        }
    return output


def silver_precision_lower_bound(
    result: RecoveryResult,
    silver_reference_edges: Iterable[Any],
) -> Mapping[str, Any]:
    """Conservative precision under an incomplete silver graph.

    A prediction absent from silver is reported as *unverified*, not definitely
    false.  Consequently ``matched / accepted`` is a lower bound rather than a
    conventional closed-world precision estimate.
    """

    silver = _edge_set(silver_reference_edges)
    accepted = {edge.key for edge in result.accepted_edges}
    matched = accepted & silver
    unverified = accepted - silver
    return {
        "accepted_count": len(accepted),
        "silver_matched_count": len(matched),
        "unverified_count": len(unverified),
        "lower_bound": _safe_ratio(
            len(matched), len(accepted), "NO_ACCEPTED_RECOVERED_EDGES"
        ),
        "unverified_edges": [list(key) for key in sorted(unverified)],
    }


def _prediction_map(result: RecoveryResult) -> Mapping[tuple[str, str, str], CandidatePrediction]:
    return {prediction.key: prediction for prediction in result.predictions}


def masked_ranking_metrics(
    result: RecoveryResult,
    masked_edges: Iterable[Any],
    *,
    all_reference_edges: Iterable[Any] | None = None,
    hits_at: Sequence[int] = (1, 3, 10),
) -> Mapping[str, Any]:
    """Filtered object-ranking metrics for ``(subject, predicate, ?)`` queries.

    Other known-true objects for the same query are filtered.  Score ties are
    ranked pessimistically, avoiding entity-name-dependent optimistic results.
    A target absent from the typed universe receives reciprocal rank zero.
    """

    targets = sorted(_edge_set(masked_edges))
    if not targets:
        return {
            "target_count": 0,
            "mrr": _metric(None, reason="NO_MASKED_TARGET_EDGES"),
            "hits": {str(k): _metric(None, reason="NO_MASKED_TARGET_EDGES") for k in hits_at},
            "missing_candidate_count": 0,
            "tie_policy": "pessimistic",
        }
    if result.status != READY:
        return {
            "target_count": len(targets),
            "mrr": _metric(None, reason=f"VARIANT_{result.status}"),
            "hits": {str(k): _metric(None, reason=f"VARIANT_{result.status}") for k in hits_at},
            "missing_candidate_count": len(targets),
            "tie_policy": "pessimistic",
        }

    predictions = _prediction_map(result)
    reference = _edge_set(all_reference_edges) if all_reference_edges is not None else set(targets)
    by_query: dict[tuple[str, str], list[CandidatePrediction]] = defaultdict(list)
    for prediction in result.predictions:
        by_query[(prediction.subject, prediction.predicate)].append(prediction)

    reciprocal_ranks: list[float] = []
    ranks: list[int | None] = []
    missing = 0
    epsilon = 1e-12
    for target in targets:
        target_prediction = predictions.get(target)
        if target_prediction is None:
            reciprocal_ranks.append(0.0)
            ranks.append(None)
            missing += 1
            continue
        query = target[0], target[1]
        competitors = []
        for candidate in by_query.get(query, []):
            if candidate.key == target:
                continue
            # Filter another independently known true answer for this query.
            if candidate.key in reference:
                continue
            competitors.append(candidate)
        higher = sum(candidate.score > target_prediction.score + epsilon for candidate in competitors)
        tied = sum(abs(candidate.score - target_prediction.score) <= epsilon for candidate in competitors)
        rank = 1 + higher + tied
        reciprocal_ranks.append(1.0 / rank)
        ranks.append(rank)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    hit_values = {
        str(k): _metric(sum(rank is not None and rank <= k for rank in ranks) / len(ranks))
        for k in hits_at
    }
    return {
        "target_count": len(targets),
        "mrr": _metric(mrr),
        "hits": hit_values,
        "missing_candidate_count": missing,
        "ranks": ranks,
        "tie_policy": "pessimistic",
    }


def evaluate_recovery(
    result: RecoveryResult,
    *,
    masked_edges: Iterable[Any],
    silver_reference_edges: Iterable[Any],
    all_reference_edges: Iterable[Any] | None = None,
) -> Mapping[str, Any]:
    """Evaluate one result; evaluator-only labels enter at this boundary."""

    return {
        "variant": result.variant,
        "status": result.status,
        "candidate_recall": candidate_recall(result, masked_edges),
        "masked_recall": masked_recall(result, masked_edges),
        "relation_masked_recall": relation_masked_recall(result, masked_edges),
        "ranking": masked_ranking_metrics(
            result,
            masked_edges,
            all_reference_edges=(
                all_reference_edges
                if all_reference_edges is not None
                else silver_reference_edges
            ),
        ),
        "silver_precision_lower_bound": silver_precision_lower_bound(
            result, silver_reference_edges
        ),
        "activation": dict(result.activation),
    }


def activation_metrics(
    baseline: RecoveryResult,
    enhanced: RecoveryResult,
    *,
    score_epsilon: float = 1e-9,
) -> Mapping[str, Any]:
    """Measure whether an added stage materially changes the same candidates."""

    baseline_candidates = {candidate.key for candidate in baseline.candidates}
    enhanced_candidates = {candidate.key for candidate in enhanced.candidates}
    shared_universe = baseline_candidates == enhanced_candidates
    if baseline.status != READY or enhanced.status != READY:
        return {
            "status": "NOT_COMPARABLE",
            "baseline_status": baseline.status,
            "enhanced_status": enhanced.status,
            "shared_candidate_universe": shared_universe,
            "decision_flip_count": None,
            "score_delta_count": None,
            "accepted_add_count": None,
            "accepted_drop_count": None,
        }

    left = _prediction_map(baseline)
    right = _prediction_map(enhanced)
    common = sorted(set(left) & set(right))
    decision_flips = sum(left[key].decision != right[key].decision for key in common)
    score_deltas = sum(abs(left[key].score - right[key].score) > score_epsilon for key in common)
    accepted_left = {key for key, value in left.items() if value.decision == "accepted"}
    accepted_right = {key for key, value in right.items() if value.decision == "accepted"}
    return {
        "status": "COMPARABLE" if shared_universe else "CANDIDATE_MISMATCH",
        "baseline_status": baseline.status,
        "enhanced_status": enhanced.status,
        "shared_candidate_universe": shared_universe,
        "candidate_count": len(common),
        "decision_flip_count": decision_flips,
        "score_delta_count": score_deltas,
        "accepted_add_count": len(accepted_right - accepted_left),
        "accepted_drop_count": len(accepted_left - accepted_right),
        "accepted_add_edges": [list(key) for key in sorted(accepted_right - accepted_left)],
        "accepted_drop_edges": [list(key) for key in sorted(accepted_left - accepted_right)],
    }


def activation_matrix(
    results: Mapping[str, RecoveryResult],
    *,
    pairs: Sequence[tuple[str, str]] = DEFAULT_ACTIVATION_PAIRS,
) -> Mapping[str, Any]:
    output: dict[str, Any] = {}
    for baseline_id, enhanced_id in pairs:
        if baseline_id not in results or enhanced_id not in results:
            continue
        output[f"{baseline_id}->{enhanced_id}"] = activation_metrics(
            results[baseline_id], results[enhanced_id]
        )
    return output


def zero_flip_gate(
    results: Mapping[str, RecoveryResult],
    activation: Mapping[str, Any] | None = None,
    *,
    require_variants: Sequence[str] = (),
    require_activation_pairs: Sequence[str | tuple[str, str]] = (),
) -> Mapping[str, Any]:
    """D3 gate: active ablations must change at least one final decision.

    In smoke mode at least one comparable pair must have a decision flip.  A
    legitimately inactive A0->A1 does not hide an active A1->A2.  Paper mode
    can require every intended stage explicitly through
    ``require_activation_pairs``.  A score-only change is recorded but does not
    satisfy a required prediction activation.
    """

    activation = activation or activation_matrix(results)
    required = {variant.upper() for variant in require_variants}
    skipped_required = sorted(
        variant
        for variant in required
        if variant not in results or results[variant].status != READY
    )
    candidate_mismatch_pairs: list[str] = []
    zero_flip_pairs: list[str] = []
    nonzero_flip_pairs: list[str] = []
    active_pairs = 0
    for pair, values in activation.items():
        if values.get("status") == "CANDIDATE_MISMATCH":
            candidate_mismatch_pairs.append(pair)
            continue
        if values.get("status") != "COMPARABLE":
            continue
        active_pairs += 1
        if values.get("decision_flip_count") == 0:
            zero_flip_pairs.append(pair)
        else:
            nonzero_flip_pairs.append(pair)

    normalized_required_pairs = {
        "->".join(pair) if isinstance(pair, tuple) else str(pair)
        for pair in require_activation_pairs
    }
    unavailable_required_pairs = sorted(
        pair
        for pair in normalized_required_pairs
        if pair not in activation or activation[pair].get("status") != "COMPARABLE"
    )
    zero_required_pairs = sorted(
        pair
        for pair in normalized_required_pairs
        if pair in activation
        and activation[pair].get("status") == "COMPARABLE"
        and activation[pair].get("decision_flip_count") == 0
    )

    reasons: list[str] = []
    if skipped_required:
        reasons.append("REQUIRED_VARIANT_SKIPPED")
    if candidate_mismatch_pairs:
        reasons.append("CANDIDATE_UNIVERSE_MISMATCH")
    if active_pairs > 0 and not nonzero_flip_pairs:
        reasons.append("ZERO_DECISION_FLIP")
    if unavailable_required_pairs:
        reasons.append("REQUIRED_ACTIVATION_PAIR_UNAVAILABLE")
    if zero_required_pairs:
        reasons.append("REQUIRED_ACTIVATION_PAIR_ZERO_FLIP")
    if active_pairs == 0:
        reasons.append("NO_ACTIVE_COMPARISONS")
    passed = not reasons
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "reason_codes": reasons,
        "active_pair_count": active_pairs,
        "zero_flip_pairs": zero_flip_pairs,
        "nonzero_flip_pairs": nonzero_flip_pairs,
        "candidate_mismatch_pairs": candidate_mismatch_pairs,
        "skipped_required_variants": skipped_required,
        "unavailable_required_activation_pairs": unavailable_required_pairs,
        "zero_required_activation_pairs": zero_required_pairs,
    }
