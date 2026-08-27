"""Held-out silver service-call graph construction for RCAEval traces.

The reference graph is deliberately narrower than a general topology graph:
``A CALLS B`` means that a span executed by service A is the exact parent of a
span executed by service B in a held-out distributed trace.  It does not mean
that A causally caused an incident in B.

Only pandas and the Python standard library are used so that the smoke
benchmark remains easy to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata
from typing import Iterable

import pandas as pd


EDGE_KEY_COLUMNS = ("subject", "predicate", "object")


@dataclass(frozen=True)
class TraceColumns:
    """Column mapping for a distributed-trace table."""

    trace_id: str = "traceID"
    span_id: str = "spanID"
    parent_span_id: str = "parentSpanID"
    service_name: str = "serviceName"
    start_us: str = "startTime"
    duration_us: str = "duration"

    @property
    def required(self) -> tuple[str, ...]:
        return (
            self.trace_id,
            self.span_id,
            self.parent_span_id,
            self.service_name,
            self.start_us,
            self.duration_us,
        )


RCAEVAL_TRACE_COLUMNS = TraceColumns()
CANONICAL_TRACE_COLUMNS = TraceColumns(
    trace_id="trace_id",
    span_id="span_id",
    parent_span_id="parent_span_id",
    service_name="service_id",
    start_us="start_time_us",
    duration_us="duration_us",
)


@dataclass(frozen=True)
class TraceSplit:
    """Whole-trace, deterministic evaluator/model partition."""

    reference: pd.DataFrame
    model: pd.DataFrame
    reference_trace_ids: frozenset[str]
    model_trace_ids: frozenset[str]
    reference_ratio: float


@dataclass(frozen=True)
class CallExtractionStats:
    total_spans: int
    unique_traces: int
    root_spans: int
    nonroot_spans: int
    matched_parent_spans: int
    orphan_parent_spans: int
    nonroot_parent_coverage: float
    services: int
    cross_service_occurrences: int


@dataclass(frozen=True)
class CallExtraction:
    occurrences: pd.DataFrame
    stats: CallExtractionStats


@dataclass(frozen=True)
class SilverGraph:
    """Artifacts needed by the model runner and the private evaluator."""

    trace_split: TraceSplit
    reference_occurrences: pd.DataFrame
    model_occurrences: pd.DataFrame
    reference_edges: pd.DataFrame
    observed_edges: pd.DataFrame
    reference_stats: CallExtractionStats
    model_stats: CallExtractionStats
    trace_columns: TraceColumns
    service_ids_are_canonical: bool


def _validate_trace_frame(traces: pd.DataFrame, columns: TraceColumns) -> None:
    missing = [column for column in columns.required if column not in traces.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")
    if traces.empty:
        raise ValueError("trace table must not be empty")

    non_nullable = (
        columns.trace_id,
        columns.span_id,
        columns.service_name,
        columns.start_us,
        columns.duration_us,
    )
    null_columns = [column for column in non_nullable if traces[column].isna().any()]
    if null_columns:
        raise ValueError(f"trace table contains null values in {null_columns}")

    duplicate_keys = traces.duplicated([columns.trace_id, columns.span_id], keep=False)
    if duplicate_keys.any():
        raise ValueError("(traceID, spanID) must be unique")

    starts = pd.to_numeric(traces[columns.start_us], errors="coerce")
    durations = pd.to_numeric(traces[columns.duration_us], errors="coerce")
    if starts.isna().any() or durations.isna().any():
        raise ValueError("start and duration columns must be numeric")
    if (durations < 0).any():
        raise ValueError("span duration must be non-negative")


def _stable_bucket(value: str, *, revision: str, incident_id: str) -> int:
    material = f"{revision}|{incident_id}|{value}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16) % 100


def deterministic_trace_split(
    traces: pd.DataFrame,
    *,
    revision: str,
    incident_id: str,
    reference_ratio: float = 0.40,
    columns: TraceColumns = RCAEVAL_TRACE_COLUMNS,
) -> TraceSplit:
    """Split complete traces into evaluator-only and model partitions.

    A trace is assigned by hashing ``revision|incident_id|traceID``.  Every
    span belonging to the same trace therefore stays in exactly one partition.
    The split is frozen independently of structural-mask seeds.
    """

    _validate_trace_frame(traces, columns)
    if not 0.0 < reference_ratio < 1.0:
        raise ValueError("reference_ratio must be between 0 and 1")

    threshold = int(round(reference_ratio * 100))
    trace_values = traces[columns.trace_id].astype(str)
    unique_trace_ids = sorted(trace_values.unique())
    assignments = {
        trace_id: _stable_bucket(
            trace_id,
            revision=revision,
            incident_id=incident_id,
        )
        < threshold
        for trace_id in unique_trace_ids
    }
    reference_mask = trace_values.map(assignments).astype(bool)

    reference = traces.loc[reference_mask].copy().reset_index(drop=True)
    model = traces.loc[~reference_mask].copy().reset_index(drop=True)
    reference_ids = frozenset(reference[columns.trace_id].astype(str).unique())
    model_ids = frozenset(model[columns.trace_id].astype(str).unique())

    if not reference_ids or not model_ids:
        raise ValueError(
            "deterministic trace split produced an empty partition; "
            "use more traces or a different incident identifier"
        )

    split = TraceSplit(
        reference=reference,
        model=model,
        reference_trace_ids=reference_ids,
        model_trace_ids=model_ids,
        reference_ratio=reference_ratio,
    )
    assert_trace_disjoint(split)
    return split


_K8S_DNS_SUFFIXES = (
    ".default.svc.cluster.local",
    ".svc.cluster.local",
    ".cluster.local",
)
_POD_SUFFIX_RE = re.compile(r"-[a-f0-9]{8,10}-[a-z0-9]{5}$")


def normalize_service_name(raw_name: object) -> str:
    """Apply conservative, deterministic service-name normalization."""

    if raw_name is None or pd.isna(raw_name):
        raise ValueError("service name must not be null")
    value = unicodedata.normalize("NFKC", str(raw_name)).strip().lower()
    if not value:
        raise ValueError("service name must not be empty")

    # Strip a numeric port, but retain semantic suffixes such as ``-service``.
    value = re.sub(r":\d+$", "", value)
    for suffix in _K8S_DNS_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if _POD_SUFFIX_RE.search(value):
        value = _POD_SUFFIX_RE.sub("", value)
    return value


def canonical_service_id(
    raw_name: object,
    *,
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
) -> str:
    return f"{dataset_id}:{system_id}:service:{normalize_service_name(raw_name)}"


def _service_value(
    raw_name: object,
    *,
    dataset_id: str,
    system_id: str,
    service_ids_are_canonical: bool,
) -> str:
    if service_ids_are_canonical:
        if raw_name is None or pd.isna(raw_name):
            raise ValueError("canonical service_id must not be null")
        value = str(raw_name).strip()
        if not value:
            raise ValueError("canonical service_id must not be empty")
        return value
    return canonical_service_id(
        raw_name,
        dataset_id=dataset_id,
        system_id=system_id,
    )


def _canonical_service_mode(
    columns: TraceColumns,
    service_ids_are_canonical: bool | None,
) -> bool:
    if service_ids_are_canonical is not None:
        return service_ids_are_canonical
    return columns.service_name == CANONICAL_TRACE_COLUMNS.service_name


def _root_mask(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().fillna("").eq("")


def extract_exact_parent_calls(
    traces: pd.DataFrame,
    *,
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
    columns: TraceColumns = RCAEVAL_TRACE_COLUMNS,
    containment_tolerance_us: int = 10_000,
    service_ids_are_canonical: bool | None = None,
) -> CallExtraction:
    """Extract directed cross-service CALLS from exact parent-span joins."""

    _validate_trace_frame(traces, columns)
    if containment_tolerance_us < 0:
        raise ValueError("containment_tolerance_us must be non-negative")

    work = traces.reset_index(drop=True).copy()
    work["_row_id"] = range(len(work))
    roots = _root_mask(work[columns.parent_span_id])

    parents = work[
        [
            columns.trace_id,
            columns.span_id,
            columns.service_name,
            columns.start_us,
            columns.duration_us,
            "_row_id",
        ]
    ].rename(
        columns={
            columns.span_id: columns.parent_span_id,
            columns.service_name: "_parent_service",
            columns.start_us: "_parent_start_us",
            columns.duration_us: "_parent_duration_us",
            "_row_id": "parent_row_id",
        }
    )

    children = work.loc[
        ~roots,
        [
            columns.trace_id,
            columns.span_id,
            columns.parent_span_id,
            columns.service_name,
            columns.start_us,
            columns.duration_us,
            "_row_id",
        ],
    ].rename(columns={"_row_id": "child_row_id"})

    joined = children.merge(
        parents,
        on=[columns.trace_id, columns.parent_span_id],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    matched = joined.loc[joined["_merge"].eq("both")].copy()
    matched.drop(columns=["_merge"], inplace=True)

    canonical_mode = _canonical_service_mode(columns, service_ids_are_canonical)
    if canonical_mode:
        parent_values = matched["_parent_service"].astype(str).str.strip()
        child_values = matched[columns.service_name].astype(str).str.strip()
    else:
        parent_values = matched["_parent_service"].map(normalize_service_name)
        child_values = matched[columns.service_name].map(normalize_service_name)
    cross_service = parent_values.ne(child_values)
    calls = matched.loc[cross_service].copy()
    parent_values = parent_values.loc[cross_service]
    child_values = child_values.loc[cross_service]

    parent_start = pd.to_numeric(calls["_parent_start_us"])
    parent_duration = pd.to_numeric(calls["_parent_duration_us"])
    child_start = pd.to_numeric(calls[columns.start_us])
    child_duration = pd.to_numeric(calls[columns.duration_us])

    occurrences = pd.DataFrame(
        {
            "subject": [
                _service_value(
                    name,
                    dataset_id=dataset_id,
                    system_id=system_id,
                    service_ids_are_canonical=canonical_mode,
                )
                for name in parent_values
            ],
            "predicate": "CALLS",
            "object": [
                _service_value(
                    name,
                    dataset_id=dataset_id,
                    system_id=system_id,
                    service_ids_are_canonical=canonical_mode,
                )
                for name in child_values
            ],
            "subject_id": [
                _service_value(
                    name,
                    dataset_id=dataset_id,
                    system_id=system_id,
                    service_ids_are_canonical=canonical_mode,
                )
                for name in parent_values
            ],
            "object_id": [
                _service_value(
                    name,
                    dataset_id=dataset_id,
                    system_id=system_id,
                    service_ids_are_canonical=canonical_mode,
                )
                for name in child_values
            ],
            "trace_id": calls[columns.trace_id].astype(str).to_numpy(),
            "parent_span_id": calls[columns.parent_span_id].astype(str).to_numpy(),
            "child_span_id": calls[columns.span_id].astype(str).to_numpy(),
            "parent_row_id": calls["parent_row_id"].astype(int).to_numpy(),
            "child_row_id": calls["child_row_id"].astype(int).to_numpy(),
            "start_us": child_start.astype("int64").to_numpy(),
            "duration_us": child_duration.astype("int64").to_numpy(),
            "temporal_contained": (
                child_start.ge(parent_start - containment_tolerance_us)
                & (child_start + child_duration).le(
                    parent_start + parent_duration + containment_tolerance_us
                )
            ).to_numpy(),
            "join_method": "EXACT_TRACE_AND_PARENT_SPAN_ID",
            "normalization_confidence": 1.0,
        }
    )
    occurrences["evidence_id"] = (
        occurrences["trace_id"]
        + "|"
        + occurrences["parent_span_id"]
        + "|"
        + occurrences["child_span_id"]
    )
    occurrences.sort_values(
        ["subject", "object", "trace_id", "start_us", "child_span_id"],
        inplace=True,
        ignore_index=True,
    )

    nonroot_count = int((~roots).sum())
    matched_count = len(matched)
    stats = CallExtractionStats(
        total_spans=len(work),
        unique_traces=work[columns.trace_id].nunique(),
        root_spans=int(roots.sum()),
        nonroot_spans=nonroot_count,
        matched_parent_spans=matched_count,
        orphan_parent_spans=nonroot_count - matched_count,
        nonroot_parent_coverage=(matched_count / nonroot_count if nonroot_count else 1.0),
        services=(
            work[columns.service_name].astype(str).str.strip().nunique()
            if canonical_mode
            else work[columns.service_name].map(normalize_service_name).nunique()
        ),
        cross_service_occurrences=len(occurrences),
    )
    return CallExtraction(occurrences=occurrences, stats=stats)


REFERENCE_EDGE_COLUMNS = (
    "edge_id",
    "dataset_id",
    "system_id",
    "incident_id",
    "subject",
    "predicate",
    "object",
    "subject_id",
    "object_id",
    "subject_type",
    "object_type",
    "directed",
    "relation_layer",
    "status",
    "source",
    "join_method",
    "occurrence_count",
    "unique_trace_count",
    "unique_minute_count",
    "pre_injection_count",
    "post_injection_count",
    "temporal_containment_rate",
    "normalization_confidence",
    "attestation",
    "confidence",
    "confidence_semantics",
    "first_seen_us",
    "last_seen_us",
    "visibility",
    "extractor_version",
    "normalizer_version",
    "source_revision",
    "evidence_digest",
)


def _evidence_digest(evidence_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for evidence_id in sorted(evidence_ids):
        digest.update(evidence_id.encode("utf-8"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def aggregate_call_edges(
    occurrences: pd.DataFrame,
    *,
    incident_id: str,
    source_revision: str,
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
    inject_time_us: int | None = None,
    status: str = "reference_confirmed",
    visibility: str = "evaluator_only",
    source: str = "heldout_trace_parent_child",
) -> pd.DataFrame:
    """Aggregate occurrence-level calls and assign transparent attestation."""

    required = {
        "subject",
        "predicate",
        "object",
        "trace_id",
        "start_us",
        "temporal_contained",
        "normalization_confidence",
        "join_method",
        "evidence_id",
    }
    missing = sorted(required.difference(occurrences.columns))
    if missing:
        raise ValueError(f"call occurrences are missing columns: {missing}")
    if occurrences.empty:
        return pd.DataFrame(columns=REFERENCE_EDGE_COLUMNS)

    rows: list[dict[str, object]] = []
    grouped = occurrences.groupby(list(EDGE_KEY_COLUMNS), sort=True, dropna=False)
    for (subject, predicate, object_id), group in grouped:
        starts = pd.to_numeric(group["start_us"])
        occurrence_count = len(group)
        unique_trace_count = group["trace_id"].nunique()
        unique_minute_count = (starts // 60_000_000).nunique()
        containment_rate = float(group["temporal_contained"].mean())
        normalization_quality = float(group["normalization_confidence"].min())

        if inject_time_us is None:
            pre_count: int | None = None
            post_count: int | None = None
            stable_across_injection = True
        else:
            pre_count = int(starts.lt(inject_time_us).sum())
            post_count = int(starts.ge(inject_time_us).sum())
            stable_across_injection = pre_count > 0 and post_count > 0

        repeatability = min(1.0, unique_trace_count / 5.0) * min(
            1.0, unique_minute_count / 3.0
        )
        quality_terms = (1.0, containment_rate, normalization_quality, repeatability)
        confidence = math.prod(max(0.0, term) for term in quality_terms) ** 0.25

        if (
            unique_trace_count >= 5
            and unique_minute_count >= 3
            and containment_rate >= 0.95
            and normalization_quality >= 0.95
            and stable_across_injection
        ):
            attestation = "A"
        elif (
            unique_trace_count >= 2
            and containment_rate >= 0.80
            and normalization_quality >= 0.80
        ):
            attestation = "B"
        else:
            attestation = "C"

        edge_id = f"{system_id}|{subject}|{predicate}|{object_id}"
        rows.append(
            {
                "edge_id": edge_id,
                "dataset_id": dataset_id,
                "system_id": system_id,
                "incident_id": incident_id,
                "subject": subject,
                "predicate": predicate,
                "object": object_id,
                "subject_id": subject,
                "object_id": object_id,
                "subject_type": "Service",
                "object_type": "Service",
                "directed": True,
                "relation_layer": "runtime",
                "status": status,
                "source": source,
                "join_method": "EXACT_TRACE_AND_PARENT_SPAN_ID",
                "occurrence_count": occurrence_count,
                "unique_trace_count": unique_trace_count,
                "unique_minute_count": unique_minute_count,
                "pre_injection_count": pre_count,
                "post_injection_count": post_count,
                "temporal_containment_rate": containment_rate,
                "normalization_confidence": normalization_quality,
                "attestation": attestation,
                "confidence": confidence,
                "confidence_semantics": "deterministic_extraction_quality_v1",
                "first_seen_us": int(starts.min()),
                "last_seen_us": int(starts.max()),
                "visibility": visibility,
                "extractor_version": "trace_calls_v1",
                "normalizer_version": "rcaeval_service_v1",
                "source_revision": source_revision,
                "evidence_digest": _evidence_digest(group["evidence_id"].astype(str)),
            }
        )

    return pd.DataFrame(rows, columns=REFERENCE_EDGE_COLUMNS).sort_values(
        list(EDGE_KEY_COLUMNS), ignore_index=True
    )


def edge_key_set(edges: pd.DataFrame) -> set[tuple[str, str, str]]:
    missing = [column for column in EDGE_KEY_COLUMNS if column not in edges.columns]
    if missing:
        raise ValueError(f"edge table is missing key columns: {missing}")
    return {
        (str(row.subject), str(row.predicate), str(row.object))
        for row in edges[list(EDGE_KEY_COLUMNS)].itertuples(index=False)
    }


def assert_trace_disjoint(split: TraceSplit) -> None:
    overlap = split.reference_trace_ids.intersection(split.model_trace_ids)
    if overlap:
        sample = sorted(overlap)[:3]
        raise AssertionError(f"reference/model trace leakage detected: {sample}")


def build_heldout_silver_graph(
    traces: pd.DataFrame,
    *,
    revision: str,
    incident_id: str,
    inject_time_us: int | None,
    dataset_id: str = "rcaeval",
    system_id: str = "train-ticket",
    reference_ratio: float = 0.40,
    columns: TraceColumns = RCAEVAL_TRACE_COLUMNS,
    service_ids_are_canonical: bool | None = None,
) -> SilverGraph:
    """Build disjoint reference and model-side service-call graphs."""

    canonical_mode = _canonical_service_mode(columns, service_ids_are_canonical)
    split = deterministic_trace_split(
        traces,
        revision=revision,
        incident_id=incident_id,
        reference_ratio=reference_ratio,
        columns=columns,
    )
    reference_extraction = extract_exact_parent_calls(
        split.reference,
        dataset_id=dataset_id,
        system_id=system_id,
        columns=columns,
        service_ids_are_canonical=canonical_mode,
    )
    model_extraction = extract_exact_parent_calls(
        split.model,
        dataset_id=dataset_id,
        system_id=system_id,
        columns=columns,
        service_ids_are_canonical=canonical_mode,
    )
    reference_edges = aggregate_call_edges(
        reference_extraction.occurrences,
        incident_id=incident_id,
        source_revision=revision,
        dataset_id=dataset_id,
        system_id=system_id,
        inject_time_us=inject_time_us,
        status="reference_confirmed",
        visibility="evaluator_only",
        source="heldout_trace_parent_child",
    )
    observed_edges = aggregate_call_edges(
        model_extraction.occurrences,
        incident_id=incident_id,
        source_revision=revision,
        dataset_id=dataset_id,
        system_id=system_id,
        inject_time_us=inject_time_us,
        status="observed",
        visibility="model_input",
        source="model_trace_parent_child",
    )
    return SilverGraph(
        trace_split=split,
        reference_occurrences=reference_extraction.occurrences,
        model_occurrences=model_extraction.occurrences,
        reference_edges=reference_edges,
        observed_edges=observed_edges,
        reference_stats=reference_extraction.stats,
        model_stats=model_extraction.stats,
        trace_columns=columns,
        service_ids_are_canonical=canonical_mode,
    )
