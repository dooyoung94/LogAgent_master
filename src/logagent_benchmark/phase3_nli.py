"""Tri-state DeBERTa evidence extraction for frozen A2 proposals."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import pandas as pd

from .onnx_deberta import OnnxDebertaNLIBackend
from .phase3_contract import (
    NliEvidence, Phase3Error, RUNTIME_CONTEXT_LINE_CHAR_LIMIT,
    RUNTIME_CONTEXT_MAX_LINES, RUNTIME_CONTEXT_TOTAL_CHAR_LIMIT,
    TriStateThresholds, _edge_key, _json_mapping, _json_sequence, _load_json,
)

def classify_tri_state(
    *,
    flat_forward: Mapping[str, float],
    flat_reverse: Mapping[str, float],
    context_forward: Mapping[str, float],
    context_reverse: Mapping[str, float],
    thresholds: TriStateThresholds,
) -> NliEvidence:
    """Convert two directional NLI views into non-destructive evidence.

    A contradictory state is a ranking signal only.  It never removes a
    candidate by itself.
    """

    forward_entailment = statistics.fmean(
        (float(flat_forward["entailment"]), float(context_forward["entailment"]))
    )
    reverse_entailment = statistics.fmean(
        (float(flat_reverse["entailment"]), float(context_reverse["entailment"]))
    )
    forward_contradiction = max(
        float(flat_forward["contradiction"]),
        float(context_forward["contradiction"]),
    )
    forward_neutral = statistics.fmean(
        (float(flat_forward["neutral"]), float(context_forward["neutral"]))
    )
    direction_margin = forward_entailment - reverse_entailment
    label_margin = forward_entailment - forward_contradiction

    reverse_dominates = (
        reverse_entailment >= thresholds.contradict_probability_min
        and reverse_entailment - forward_entailment >= thresholds.contradiction_margin_min
    )
    contradiction_dominates = (
        forward_contradiction >= thresholds.contradict_probability_min
        and forward_contradiction - forward_entailment
        >= thresholds.contradiction_margin_min
    )
    corroborates = (
        forward_entailment >= thresholds.corroborate_entailment_min
        and direction_margin >= thresholds.evidence_margin_min
        and label_margin >= thresholds.evidence_margin_min
    )
    if reverse_dominates or contradiction_dominates:
        state = "contradicts"
    elif corroborates:
        state = "corroborates"
    else:
        state = "ambiguous"

    raw_score = (
        0.50 * direction_margin
        + 0.35 * label_margin
        + 0.15 * (1.0 - forward_neutral)
    )
    evidence_score = max(-1.0, min(1.0, raw_score))
    return NliEvidence(
        state=state,
        evidence_score=evidence_score,
        forward_entailment=forward_entailment,
        reverse_entailment=reverse_entailment,
        forward_contradiction=forward_contradiction,
        forward_neutral=forward_neutral,
        direction_margin=direction_margin,
        label_margin=label_margin,
    )


def _normal_probabilities(raw: Mapping[str, Any]) -> dict[str, float]:
    output = {
        name: float(raw[name])
        for name in ("entailment", "contradiction", "neutral")
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in output.values()):
        raise Phase3Error(f"invalid NLI probabilities: {output}")
    if not math.isclose(sum(output.values()), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise Phase3Error(f"NLI probabilities do not sum to one: {output}")
    return output


def _display(entity_id: str) -> str:
    return str(entity_id).rsplit(":", 1)[-1]


def _flat_premise(record: Mapping[str, Any]) -> str:
    return (
        "Directional telemetry evidence from a sanitized model partition. "
        f"Outer service={record['subject_label']}; inner service={record['object_label']}. "
        "The outer span begins before the inner span and ends after it while the "
        "explicit parent identifier is unavailable. "
        f"Forward containment occurs at {int(record['boundary_spans'])} boundaries "
        f"across {int(record['supporting_traces'])} whole traces. "
        f"Reverse containment occurs at {int(record['reverse_boundary_spans'])} "
        f"boundaries across {int(record['reverse_supporting_traces'])} whole traces. "
        "This is abductive candidate evidence, not a confirmed dependency."
    )


def _compact_runtime_context(value: Any) -> str:
    """Serialize runtime context with a frozen, explicit character budget.

    The model tokenizer is never allowed to truncate silently.  This compact
    serializer is part of the A3 contract and is applied before tokenization.
    """

    normalized = [
        " ".join(line.split())
        for line in str(value or "").splitlines()
        if line.strip()
    ]
    if not normalized:
        return "No additional runtime role context is available."
    clipped = [
        line[:RUNTIME_CONTEXT_LINE_CHAR_LIMIT]
        for line in normalized[:RUNTIME_CONTEXT_MAX_LINES]
    ]
    text = "\n".join(clipped)
    return text[:RUNTIME_CONTEXT_TOTAL_CHAR_LIMIT].rstrip()


def _runtime_premise(record: Mapping[str, Any]) -> str:
    addendum = _compact_runtime_context(record.get("contextual_addendum", ""))
    return f"{_flat_premise(record)}\nRuntime context:\n{addendum}"


def _hypotheses(record: Mapping[str, Any]) -> tuple[str, str]:
    subject = record["subject_label"]
    object_id = record["object_label"]
    return (
        f"Within this runtime system, {subject} directly invokes {object_id}.",
        f"Within this runtime system, {object_id} directly invokes {subject}.",
    )


def _cell_root(phase2_root: Path, row: Mapping[str, Any]) -> Path:
    summary = phase2_root / str(row["run_summary"])
    return summary.parent / "masks" / str(row["mask_id"])


def _load_cell_candidates(phase2_root: Path, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    cell_root = _cell_root(phase2_root, row)
    a2_path = cell_root / "predictions" / "A2.parquet"
    context_path = cell_root / "model_input" / "runtime_pair_context.parquet"
    if not a2_path.is_file() or not context_path.is_file():
        raise Phase3Error(f"Phase-2 cell lacks A2/context artifacts: {cell_root}")
    a2 = pd.read_parquet(a2_path)
    runtime = pd.read_parquet(context_path)
    required = {
        "subject", "predicate", "object", "score", "decision",
        "stage_scores_json", "reason_codes_json",
    }
    missing = sorted(required.difference(a2.columns))
    if missing:
        raise Phase3Error(f"A2 parquet missing columns {missing}: {a2_path}")
    runtime_required = {
        "subject", "predicate", "object", "subject_label", "object_label",
        "contextual_addendum", "provenance",
    }
    runtime_missing = sorted(runtime_required.difference(runtime.columns))
    if runtime_missing:
        raise Phase3Error(f"runtime context missing columns {runtime_missing}: {context_path}")

    contexts = {
        (str(item.subject), str(item.predicate), str(item.object)): item
        for item in runtime.itertuples(index=False)
    }
    raw_records: list[dict[str, Any]] = []
    for item in a2.itertuples(index=False):
        key = (str(item.subject), str(item.predicate), str(item.object))
        if str(item.decision) != "accepted":
            raise Phase3Error(f"A2 Phase-2 proposal is not accepted: {key}")
        stage = _json_mapping(item.stage_scores_json, field_name="stage_scores_json")
        reasons = _json_sequence(item.reason_codes_json, field_name="reason_codes_json")
        runtime_row = contexts.get(key)
        if runtime_row is None:
            raise Phase3Error(f"runtime context missing for A2 proposal: {key}")
        raw_records.append(
            {
                "subject": key[0],
                "predicate": key[1],
                "object": key[2],
                "subject_label": str(runtime_row.subject_label or _display(key[0])),
                "object_label": str(runtime_row.object_label or _display(key[2])),
                "contextual_addendum": str(runtime_row.contextual_addendum or ""),
                "context_provenance": runtime_row.provenance,
                "a2_score": float(item.score),
                "supporting_traces": int(float(stage.get("supporting_traces", 0))),
                "boundary_spans": int(float(stage.get("boundary_spans", 0))),
                "proposal_rank": int(float(stage.get("proposal_rank", 0))),
                "direct_evidence": "DIRECT_EVIDENCE" in {str(value) for value in reasons},
            }
        )
    by_key = {(record["subject"], record["predicate"], record["object"]): record for record in raw_records}
    for record in raw_records:
        reverse = by_key.get((record["object"], record["predicate"], record["subject"]))
        record["reverse_supporting_traces"] = int(reverse["supporting_traces"]) if reverse else 0
        record["reverse_boundary_spans"] = int(reverse["boundary_spans"]) if reverse else 0
    if len(raw_records) != int(row["a2_proposal_count"]):
        raise Phase3Error(
            f"A2 proposal count mismatch for {row['case']} {row['mask_id']}: "
            f"csv={row['a2_proposal_count']} parquet={len(raw_records)}"
        )
    return raw_records


def _load_evaluator_sets(
    phase2_root: Path,
    row: Mapping[str, Any],
    candidate_keys: set[tuple[str, str, str]],
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    private = _cell_root(phase2_root, row) / "evaluator_private"
    manifest = _load_json(private / "mask_manifest.json")
    evaluation = _load_json(private / "evaluation.json")
    targets = {_edge_key(item) for item in manifest["target_edges"]}
    a2_eval = evaluation["A2"]
    unverified = {
        _edge_key(item)
        for item in a2_eval["silver_precision_lower_bound"].get("unverified_edges", ())
    }
    silver = candidate_keys - unverified
    expected_matched = int(a2_eval["silver_precision_lower_bound"]["silver_matched_count"])
    if len(silver) != expected_matched:
        raise Phase3Error(
            f"unable to reconstruct A2 silver set: expected {expected_matched}, got {len(silver)}"
        )
    if not targets.issubset(candidate_keys):
        raise Phase3Error("Phase-2 A2 candidate set no longer contains every masked target")
    return targets, silver


def _score_all_candidates(
    cells: Sequence[dict[str, Any]],
    *,
    backend: OnnxDebertaNLIBackend,
    thresholds: TriStateThresholds,
) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    locations: list[tuple[dict[str, Any], str, str]] = []
    for cell in cells:
        for record in cell["candidates"]:
            forward, reverse = _hypotheses(record)
            for view, premise in (("flat", _flat_premise(record)), ("context", _runtime_premise(record))):
                pairs.extend(((premise, forward), (premise, reverse)))
                locations.extend(((record, view, "forward"), (record, view, "reverse")))

    token_lengths = tuple(backend.pair_token_lengths(tuple(pairs))) if pairs else ()
    if token_lengths and max(token_lengths) > backend.max_length:
        raise Phase3Error(
            f"A3 pair exceeds token budget: max={max(token_lengths)} budget={backend.max_length}"
        )
    raw_scores = tuple(backend.score_pairs(tuple(pairs))) if pairs else ()
    if len(raw_scores) != len(locations):
        raise Phase3Error(f"expected {len(locations)} NLI rows, got {len(raw_scores)}")
    for (record, view, direction), raw in zip(locations, raw_scores):
        record[f"{view}_{direction}"] = _normal_probabilities(raw)
    for cell in cells:
        for record in cell["candidates"]:
            evidence = classify_tri_state(
                flat_forward=record["flat_forward"],
                flat_reverse=record["flat_reverse"],
                context_forward=record["context_forward"],
                context_reverse=record["context_reverse"],
                thresholds=thresholds,
            )
            record.update({f"nli_{key}": value for key, value in asdict(evidence).items()})
    return {
        "candidate_count": sum(len(cell["candidates"]) for cell in cells),
        "nli_pair_count": len(pairs),
        "pairs_per_candidate": 4,
        "minimum_pair_tokens": min(token_lengths) if token_lengths else None,
        "maximum_pair_tokens": max(token_lengths) if token_lengths else None,
        "truncation_count": 0,
        "context_serialization": {
            "max_lines": RUNTIME_CONTEXT_MAX_LINES,
            "line_char_limit": RUNTIME_CONTEXT_LINE_CHAR_LIMIT,
            "total_char_limit": RUNTIME_CONTEXT_TOTAL_CHAR_LIMIT,
            "silent_tokenizer_truncation": False,
        },
    }


__all__ = [
    "_compact_runtime_context", "_load_cell_candidates",
    "_load_evaluator_sets", "_score_all_candidates", "classify_tri_state",
]
