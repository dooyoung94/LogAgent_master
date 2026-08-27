"""Structural blind-spot masks for held-out service-call benchmarks.

The public/model side receives only :class:`ModelMaskBundle`.  Target triples
live in :class:`EvaluatorMaskManifest` and must remain evaluator-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
from typing import Iterable

import pandas as pd

from .graph import (
    EDGE_KEY_COLUMNS,
    RCAEVAL_TRACE_COLUMNS,
    SilverGraph,
    TraceColumns,
    edge_key_set,
    extract_exact_parent_calls,
)


TargetEdge = tuple[str, str, str]
EVIDENCE_LEVEL_L1 = "L1_BOUNDARY_HIDDEN"
IID_FRACTIONS = (0.20, 0.40, 0.60)


@dataclass(frozen=True)
class ModelMaskBundle:
    """The only structural-mask object that a recovery model may receive."""

    traces: pd.DataFrame
    observed_edges: pd.DataFrame


@dataclass(frozen=True)
class EvaluatorMaskManifest:
    """Private mask answers used only after model inference."""

    policy: str
    seed: int
    evidence_level: str
    target_edges: tuple[TargetEdge, ...]
    target_count: int
    redacted_boundary_spans: int
    fraction: float | None = None
    component_id: str | None = None
    visibility: str = "evaluator_only"

    def to_private_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target_edges"] = [
            {"subject": subject, "predicate": predicate, "object": object_id}
            for subject, predicate, object_id in self.target_edges
        ]
        return payload


@dataclass(frozen=True)
class StructuralMaskResult:
    model: ModelMaskBundle
    evaluator_manifest: EvaluatorMaskManifest


def _attested_keys(edges: pd.DataFrame) -> set[TargetEdge]:
    selected = edges
    if "attestation" in edges.columns:
        selected = edges.loc[edges["attestation"].eq("A")]
    return edge_key_set(selected)


def _eligible_keys(
    reference_edges: pd.DataFrame,
    observed_edges: pd.DataFrame | None,
) -> set[TargetEdge]:
    keys = _attested_keys(reference_edges)
    if observed_edges is not None:
        keys &= edge_key_set(observed_edges)
    return keys


def select_iid_targets(
    reference_edges: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
    observed_edges: pd.DataFrame | None = None,
) -> tuple[TargetEdge, ...]:
    """Select an exact 20/40/60% seeded sample of attestation-A edges."""

    allowed = next(
        (candidate for candidate in IID_FRACTIONS if math.isclose(fraction, candidate)),
        None,
    )
    if allowed is None:
        raise ValueError(f"IID fraction must be one of {IID_FRACTIONS}")

    eligible = sorted(_eligible_keys(reference_edges, observed_edges))
    if not eligible:
        raise ValueError("no attestation-A reference edges are available for masking")

    target_count = int(math.floor(len(eligible) * allowed + 0.5))
    target_count = max(1, min(len(eligible), target_count))
    targets = random.Random(seed).sample(eligible, target_count)
    return tuple(sorted(targets))


def select_component_targets(
    reference_edges: pd.DataFrame,
    *,
    component_id: str,
    observed_edges: pd.DataFrame | None = None,
) -> tuple[TargetEdge, ...]:
    """Select every attested incoming/outgoing edge of one component."""

    eligible = _eligible_keys(reference_edges, observed_edges)
    targets = sorted(
        edge
        for edge in eligible
        if edge[0] == component_id or edge[2] == component_id
    )
    if not targets:
        raise ValueError(f"component has no maskable attestation-A edge: {component_id}")
    return tuple(targets)


def _drop_target_edges(
    observed_edges: pd.DataFrame,
    target_edges: set[TargetEdge],
) -> pd.DataFrame:
    missing = [column for column in EDGE_KEY_COLUMNS if column not in observed_edges.columns]
    if missing:
        raise ValueError(f"observed edge table is missing key columns: {missing}")
    keep = [
        tuple(map(str, key)) not in target_edges
        for key in observed_edges[list(EDGE_KEY_COLUMNS)].itertuples(index=False, name=None)
    ]
    return observed_edges.loc[keep].copy().reset_index(drop=True)


def _opaque_parent_id(
    *,
    trace_id: object,
    span_id: object,
    seed: int,
    policy: str,
) -> str:
    material = f"{policy}|{seed}|{trace_id}|{span_id}".encode("utf-8")
    # RCAEval span IDs are 16 hexadecimal characters.  Preserve that surface
    # shape so the model observes only an ordinary unmatched parent rather
    # than a mask-specific marker.
    return hashlib.sha256(material).hexdigest()[:16]


def apply_l1_boundary_hidden(
    model_traces: pd.DataFrame,
    observed_edges: pd.DataFrame,
    *,
    target_edges: Iterable[TargetEdge],
    seed: int,
    policy: str,
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
    columns: TraceColumns = RCAEVAL_TRACE_COLUMNS,
    service_ids_are_canonical: bool | None = None,
) -> tuple[ModelMaskBundle, int]:
    """Break every exact parent link that directly attests a target edge.

    The child span is retained, but its ``parentSpanID`` becomes a deterministic
    unmatched opaque value.  This preserves a realistic missing-parent signal
    without revealing which service was the original parent.
    """

    targets = {tuple(map(str, edge)) for edge in target_edges}
    if not targets:
        raise ValueError("target_edges must not be empty")

    traces = model_traces.reset_index(drop=True).copy()
    before = extract_exact_parent_calls(
        traces,
        dataset_id=dataset_id,
        system_id=system_id,
        columns=columns,
        service_ids_are_canonical=service_ids_are_canonical,
    )
    before_keys = edge_key_set(
        before.occurrences[["subject", "predicate", "object"]].drop_duplicates()
    )
    missing_targets = targets.difference(before_keys)
    if missing_targets:
        raise ValueError(
            "synthetic mask targets must be observable in the model trace partition: "
            f"{sorted(missing_targets)[:3]}"
        )

    boundary_rows = before.occurrences.loc[
        [
            (row.subject, row.predicate, row.object) in targets
            for row in before.occurrences.itertuples(index=False)
        ]
    ]
    child_row_ids = sorted(set(boundary_rows["child_row_id"].astype(int)))
    existing_span_ids = set(traces[columns.span_id].astype(str))
    generated_tokens: set[str] = set()

    for row_id in child_row_ids:
        token = _opaque_parent_id(
            trace_id=traces.at[row_id, columns.trace_id],
            span_id=traces.at[row_id, columns.span_id],
            seed=seed,
            policy=policy,
        )
        if token in existing_span_ids or token in generated_tokens:
            raise AssertionError("opaque masked parent ID collision detected")
        generated_tokens.add(token)
        traces.at[row_id, columns.parent_span_id] = token

    masked_edges = _drop_target_edges(observed_edges, targets)
    model_bundle = ModelMaskBundle(traces=traces, observed_edges=masked_edges)

    # Re-derive the graph rather than trusting the row update.
    after = extract_exact_parent_calls(
        traces,
        dataset_id=dataset_id,
        system_id=system_id,
        columns=columns,
        service_ids_are_canonical=service_ids_are_canonical,
    )
    after_keys = edge_key_set(
        after.occurrences[["subject", "predicate", "object"]].drop_duplicates()
    )
    leaked = targets.intersection(after_keys)
    if leaked:
        raise AssertionError(f"masked CALLS remain derivable from exact parents: {sorted(leaked)}")

    return model_bundle, len(child_row_ids)


def assert_mask_is_leakage_free(
    result: StructuralMaskResult,
    *,
    reference_trace_ids: Iterable[str],
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
    columns: TraceColumns = RCAEVAL_TRACE_COLUMNS,
    service_ids_are_canonical: bool | None = None,
) -> None:
    manifest = result.evaluator_manifest
    if manifest.visibility != "evaluator_only":
        raise AssertionError("mask answer manifest must be evaluator_only")
    if manifest.target_count != len(manifest.target_edges):
        raise AssertionError("mask target_count does not match target_edges")

    targets = set(manifest.target_edges)
    observed_leak = targets.intersection(edge_key_set(result.model.observed_edges))
    if observed_leak:
        raise AssertionError(f"target edges leaked into observed graph: {sorted(observed_leak)}")

    derived = extract_exact_parent_calls(
        result.model.traces,
        dataset_id=dataset_id,
        system_id=system_id,
        columns=columns,
        service_ids_are_canonical=service_ids_are_canonical,
    )
    derived_keys = edge_key_set(
        derived.occurrences[["subject", "predicate", "object"]].drop_duplicates()
    )
    trace_leak = targets.intersection(derived_keys)
    if trace_leak:
        raise AssertionError(f"target edges leaked through exact trace joins: {sorted(trace_leak)}")

    model_trace_ids = set(result.model.traces[columns.trace_id].astype(str).unique())
    split_leak = set(map(str, reference_trace_ids)).intersection(model_trace_ids)
    if split_leak:
        raise AssertionError(
            f"evaluator reference traces leaked into model input: {sorted(split_leak)[:3]}"
        )


def make_iid_mask(
    graph: SilverGraph,
    *,
    fraction: float,
    seed: int,
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
    columns: TraceColumns | None = None,
    service_ids_are_canonical: bool | None = None,
) -> StructuralMaskResult:
    active_columns = columns or graph.trace_columns
    canonical_mode = (
        graph.service_ids_are_canonical
        if service_ids_are_canonical is None
        else service_ids_are_canonical
    )
    targets = select_iid_targets(
        graph.reference_edges,
        fraction=fraction,
        seed=seed,
        observed_edges=graph.observed_edges,
    )
    model, redacted = apply_l1_boundary_hidden(
        graph.trace_split.model,
        graph.observed_edges,
        target_edges=targets,
        seed=seed,
        policy="iid",
        dataset_id=dataset_id,
        system_id=system_id,
        columns=active_columns,
        service_ids_are_canonical=canonical_mode,
    )
    manifest = EvaluatorMaskManifest(
        policy="iid",
        seed=seed,
        evidence_level=EVIDENCE_LEVEL_L1,
        target_edges=targets,
        target_count=len(targets),
        redacted_boundary_spans=redacted,
        fraction=fraction,
    )
    result = StructuralMaskResult(model=model, evaluator_manifest=manifest)
    assert_mask_is_leakage_free(
        result,
        reference_trace_ids=graph.trace_split.reference_trace_ids,
        dataset_id=dataset_id,
        system_id=system_id,
        columns=active_columns,
        service_ids_are_canonical=canonical_mode,
    )
    return result


def make_component_blackout(
    graph: SilverGraph,
    *,
    component_id: str,
    seed: int = 0,
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
    columns: TraceColumns | None = None,
    service_ids_are_canonical: bool | None = None,
) -> StructuralMaskResult:
    active_columns = columns or graph.trace_columns
    canonical_mode = (
        graph.service_ids_are_canonical
        if service_ids_are_canonical is None
        else service_ids_are_canonical
    )
    targets = select_component_targets(
        graph.reference_edges,
        component_id=component_id,
        observed_edges=graph.observed_edges,
    )
    model, redacted = apply_l1_boundary_hidden(
        graph.trace_split.model,
        graph.observed_edges,
        target_edges=targets,
        seed=seed,
        policy="component_blackout",
        dataset_id=dataset_id,
        system_id=system_id,
        columns=active_columns,
        service_ids_are_canonical=canonical_mode,
    )
    manifest = EvaluatorMaskManifest(
        policy="component_blackout",
        seed=seed,
        evidence_level=EVIDENCE_LEVEL_L1,
        target_edges=targets,
        target_count=len(targets),
        redacted_boundary_spans=redacted,
        component_id=component_id,
    )
    result = StructuralMaskResult(model=model, evaluator_manifest=manifest)
    assert_mask_is_leakage_free(
        result,
        reference_trace_ids=graph.trace_split.reference_trace_ids,
        dataset_id=dataset_id,
        system_id=system_id,
        columns=active_columns,
        service_ids_are_canonical=canonical_mode,
    )
    return result
