"""Task A phase 1: bounded abductive relation-candidate recovery.

This module deliberately stops at A2.  It validates whether a small, leakage-safe
abductive proposal set can retain masked runtime relations before any DeBERTa or
PSL tuning is attempted.  A3--A5 are emitted as explicit deferred stages so the
existing RCAEval artifact writer can be reused without making a downstream RCA
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .cumulative import CumulativeConfig, CumulativeSuiteResult, PairRuntimeContext
from .recovery import (
    SKIPPED,
    Candidate,
    CandidatePrediction,
    InferenceContext,
    RecoveryResult,
    RelationSpec,
    DEFAULT_RELATION_SPECS,
    build_typed_candidates,
    run_recovery,
    temporal_containment_details,
)


@dataclass(frozen=True)
class TaskAConfig(CumulativeConfig):
    """Frozen proposal-budget contract for the first Task A experiment."""

    max_abductive_proposals: int = 32
    max_per_subject: int = 8
    max_per_object: int = 8
    min_supporting_traces: int = 1
    min_boundary_count: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        integer_fields = {
            "max_abductive_proposals": self.max_abductive_proposals,
            "max_per_subject": self.max_per_subject,
            "max_per_object": self.max_per_object,
            "min_supporting_traces": self.min_supporting_traces,
            "min_boundary_count": self.min_boundary_count,
        }
        invalid = {
            name: value
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        }
        if invalid:
            raise ValueError(f"Task A candidate-budget values must be positive integers: {invalid}")


@dataclass(frozen=True)
class CandidateBudgetResult:
    """Label-blind candidate selection and aggregate diagnostics."""

    selected_keys: tuple[tuple[str, str, str], ...]
    rank_by_key: Mapping[tuple[str, str, str], int]
    diagnostics: Mapping[str, Any]


def _detail_rank(
    key: tuple[str, str, str],
    detail: Any,
) -> tuple[float, int, int, tuple[str, str, str]]:
    """Order by evidence strength and use the edge key only as a stable tie-break."""

    return (
        -float(detail.score),
        -int(detail.trace_count),
        -int(detail.boundary_count),
        key,
    )


def select_abductive_proposals(
    *,
    details: Mapping[tuple[str, str, str], Any],
    candidate_by_key: Mapping[tuple[str, str, str], Candidate],
    direct_keys: set[tuple[str, str, str]] | frozenset[tuple[str, str, str]],
    config: TaskAConfig,
) -> CandidateBudgetResult:
    """Select a bounded A2 proposal set without evaluator labels.

    Directly observed evidence is handled separately and never consumes this
    budget.  Abductive candidates must pass the frozen support thresholds and
    are then greedily selected by evidence strength under global, source, and
    target caps.  The mask manifest, reference graph, and root label are absent
    from the API by construction.
    """

    eligible: list[tuple[tuple[str, str, str], Any]] = []
    below_threshold = 0
    below_trace_floor = 0
    below_boundary_floor = 0
    outside_universe = 0

    for key, detail in details.items():
        if key in direct_keys:
            continue
        if key not in candidate_by_key:
            outside_universe += 1
            continue
        if float(detail.score) < config.a2_threshold:
            below_threshold += 1
            continue
        if int(detail.trace_count) < config.min_supporting_traces:
            below_trace_floor += 1
            continue
        if int(detail.boundary_count) < config.min_boundary_count:
            below_boundary_floor += 1
            continue
        eligible.append((key, detail))

    eligible.sort(key=lambda item: _detail_rank(item[0], item[1]))
    selected: list[tuple[str, str, str]] = []
    per_subject: dict[tuple[str, str], int] = {}
    per_object: dict[tuple[str, str], int] = {}
    dropped_global = 0
    dropped_subject = 0
    dropped_object = 0

    for key, _detail in eligible:
        subject, predicate, obj = key
        subject_group = (predicate, subject)
        object_group = (predicate, obj)
        if len(selected) >= config.max_abductive_proposals:
            dropped_global += 1
            continue
        if per_subject.get(subject_group, 0) >= config.max_per_subject:
            dropped_subject += 1
            continue
        if per_object.get(object_group, 0) >= config.max_per_object:
            dropped_object += 1
            continue
        selected.append(key)
        per_subject[subject_group] = per_subject.get(subject_group, 0) + 1
        per_object[object_group] = per_object.get(object_group, 0) + 1

    rank_by_key = {key: index for index, key in enumerate(selected, start=1)}
    diagnostics = {
        "policy": "evidence_ranked_global_and_endpoint_caps",
        "ranking": [
            "abduction_score_desc",
            "supporting_trace_count_desc",
            "boundary_count_desc",
            "candidate_key_asc",
        ],
        "direct_evidence_preserved_outside_budget": True,
        "eligible_before_budget": len(eligible),
        "selected_after_budget": len(selected),
        "dropped_by_budget": len(eligible) - len(selected),
        "dropped_global_cap": dropped_global,
        "dropped_subject_cap": dropped_subject,
        "dropped_object_cap": dropped_object,
        "below_a2_threshold": below_threshold,
        "below_trace_floor": below_trace_floor,
        "below_boundary_floor": below_boundary_floor,
        "outside_typed_universe": outside_universe,
        "budget_saturated": len(eligible) > len(selected),
        "max_abductive_proposals": config.max_abductive_proposals,
        "max_per_subject": config.max_per_subject,
        "max_per_object": config.max_per_object,
        "min_supporting_traces": config.min_supporting_traces,
        "min_boundary_count": config.min_boundary_count,
    }
    return CandidateBudgetResult(tuple(selected), rank_by_key, diagnostics)


def _ready_result(
    variant: str,
    context: InferenceContext,
    candidates: Sequence[Candidate],
    predictions: Sequence[CandidatePrediction],
    *,
    activation: Mapping[str, Any],
) -> RecoveryResult:
    observed = run_recovery("A0", context, candidates=()).observed_edges
    candidate_tuple = tuple(candidates)
    prediction_tuple = tuple(predictions)
    if tuple(prediction.key for prediction in prediction_tuple) != tuple(
        candidate.key for candidate in candidate_tuple
    ):
        raise AssertionError(f"{variant} predictions must match candidates in order")
    # Construct inferred edges through the same public class used by RecoveryResult.
    from .recovery import Edge

    accepted = tuple(
        Edge(
            prediction.subject,
            prediction.predicate,
            prediction.object,
            prediction.score,
            "inferred",
            prediction.evidence_ids,
            variant,
        )
        for prediction in prediction_tuple
        if prediction.decision == "accepted"
    )
    return RecoveryResult(
        variant=variant,
        status="READY",
        candidates=candidate_tuple,
        predictions=prediction_tuple,
        accepted_edges=accepted,
        observed_edges=observed,
        activation=dict(activation),
        research_valid=True,
    )


def _deferred_result(
    variant: str,
    context: InferenceContext,
    proposals: Sequence[Candidate],
) -> RecoveryResult:
    observed = run_recovery("A0", context, candidates=()).observed_edges
    return RecoveryResult(
        variant=variant,
        status=SKIPPED,
        candidates=tuple(proposals),
        observed_edges=observed,
        activation={
            "candidate_count": len(proposals),
            "stage_calls": 0,
            "stage_activated": False,
            "phase": "A0_A2_CANDIDATE_READINESS",
        },
        reason_code="TASK_A_PHASE1_DEFERRED",
        detail="A3-A5 start only after the bounded A2 candidate-recall gate passes.",
        research_valid=True,
    )


def run_task_a_candidate_suite(
    context: InferenceContext,
    *,
    pair_contexts: Mapping[tuple[str, str, str], PairRuntimeContext] | None = None,
    config: TaskAConfig | None = None,
    relation_specs: Mapping[str, RelationSpec] = DEFAULT_RELATION_SPECS,
    deberta_backend: Any = None,
    psl_backend: Any = None,
    allow_test_backends: bool = False,
    run_d0_control: bool = False,
) -> CumulativeSuiteResult:
    """Run A0/A1/A2 with bounded proposals; defer A3/A4/A5 explicitly."""

    del pair_contexts, deberta_backend, psl_backend, allow_test_backends, run_d0_control
    config = config or TaskAConfig()
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
        context,
        include_null_parent=config.include_null_parent,
    )
    candidate_by_key = {candidate.key: candidate for candidate in universe}
    budget = select_abductive_proposals(
        details=details,
        candidate_by_key=candidate_by_key,
        direct_keys=direct_keys,
        config=config,
    )
    proposal_keys = direct_keys | set(budget.selected_keys)
    proposals = tuple(candidate_by_key[key] for key in sorted(proposal_keys))

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
                    {"direct": 1.0, "proposal_rank": 0.0},
                    ("DIRECT_EVIDENCE", "BUDGET_EXEMPT"),
                )
            )
            continue
        detail = details[candidate.key]
        a2_predictions.append(
            CandidatePrediction(
                candidate.subject,
                candidate.predicate,
                candidate.object,
                float(detail.score),
                "accepted",
                detail.evidence_ids,
                {
                    "abduction": float(detail.score),
                    "supporting_traces": float(detail.trace_count),
                    "boundary_spans": float(detail.boundary_count),
                    "proposal_rank": float(budget.rank_by_key[candidate.key]),
                },
                ("TEMPORAL_CONTAINMENT_PROPOSAL", "WITHIN_CANDIDATE_BUDGET"),
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
            "abductive_proposal_count": len(budget.selected_keys),
            "compression_ratio": (
                1.0 - len(proposals) / len(universe) if universe else None
            ),
            "include_null_parent": config.include_null_parent,
            "stage_activated": bool(proposals),
            "candidate_budget": dict(budget.diagnostics),
        },
    )
    a3 = _deferred_result("A3", context, proposals)
    a4 = _deferred_result("A4", context, proposals)
    a5 = _deferred_result("A5", context, proposals)
    results = {"A0": a0, "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5}

    diagnostics = {
        "phase": "A0_A2_CANDIDATE_READINESS",
        "evaluation_universe_count": len(universe),
        "a2_proposal_count": len(proposals),
        "a2_abductive_proposal_count": len(budget.selected_keys),
        "a2_candidate_compression_ratio": (
            1.0 - len(proposals) / len(universe) if universe else None
        ),
        "candidate_budget": dict(budget.diagnostics),
        "a3_a5_deferred": True,
    }
    activation = {
        "phase": "A0_A2_CANDIDATE_READINESS",
        "A0": dict(a0.activation),
        "A1": dict(a1.activation),
        "A2": dict(a2.activation),
        "A3_A5": {"stage_activated": False, "reason_code": "TASK_A_PHASE1_DEFERRED"},
    }
    gate = {
        "status": "PENDING_EVALUATOR",
        "passed": None,
        "gate_id": "D2_BOUNDED_CANDIDATE_RECALL",
        "reason_codes": ["CANDIDATE_RECALL_REQUIRES_EVALUATOR_ONLY_MASK_TARGETS"],
        "pass_conditions": {
            "candidate_recall_each_mask_min": 0.90,
            "abductive_proposal_count_max": config.max_abductive_proposals,
            "leakage_checks_all_pass": True,
        },
    }
    return CumulativeSuiteResult(
        evaluation_universe=universe,
        proposals=proposals,
        results=results,
        controls={},
        activation=activation,
        gate=gate,
        diagnostics=diagnostics,
    )


__all__ = [
    "CandidateBudgetResult",
    "TaskAConfig",
    "run_task_a_candidate_suite",
    "select_abductive_proposals",
]
