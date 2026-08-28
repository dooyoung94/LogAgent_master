"""Model-visible runtime hierarchy and role-proxy features for v2 NLI.

RCAEval does not provide Application, Instance, Host, or deployment ownership.
This module therefore emits only the partial hierarchy actually supported by
the sanitized model partition: System -> Service -> observed Operation.  Graph
roles are continuous telemetry-derived proxies, never asserted CMDB labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Mapping, Sequence

from .cumulative import PairRuntimeContext
from .recovery import Candidate, InferenceContext, _field, _iter_records, edge_key


_HTTP_OPERATION = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+\S+", re.I)
_DATA_TOKENS = ("repository", "database", "datastore", "mongo", "mysql", "redis", "sql")


def _display(entity_id: str) -> str:
    return str(entity_id).rsplit(":", 1)[-1]


def _compact(value: object, limit: int = 128) -> str:
    text = " ".join(str(value).split())
    return text[:limit]


def _bounded(
    values: Sequence[str],
    limit: int = 3,
    *,
    item_char_limit: int = 72,
) -> str:
    selected = [
        _compact(value, item_char_limit) for value in sorted(set(values))[:limit]
    ]
    return ", ".join(selected) if selected else "unknown"


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def build_runtime_pair_contexts(
    context: InferenceContext,
    candidates: Sequence[Candidate],
    *,
    system_label: str,
) -> Mapping[tuple[str, str, str], PairRuntimeContext]:
    """Build deterministic ordered-pair context from model-side data only.

    The signature deliberately accepts no reference graph, target edge, mask
    manifest, root-cause label, fault label, or injection time.
    """

    candidate_entities = {
        endpoint
        for candidate in candidates
        for endpoint in (candidate.subject, candidate.object)
    }
    labels: dict[str, str] = {
        entity_id: _compact(_display(entity_id)) for entity_id in candidate_entities
    }
    entity_basis: dict[str, str] = {}
    for record in _iter_records(context.entities):
        entity_id = _field(record, "entity_id", "id", "service_id")
        if entity_id is None or str(entity_id) not in candidate_entities:
            continue
        canonical = _field(record, "canonical_name", "name")
        if canonical:
            labels[str(entity_id)] = _compact(canonical)
        basis = _field(record, "type_basis", "provenance", "source")
        if basis:
            entity_basis[str(entity_id)] = _compact(basis, 96)

    span_count: Counter[str] = Counter()
    operation_count: dict[str, Counter[str]] = defaultdict(Counter)
    http_count: Counter[str] = Counter()
    data_count: Counter[str] = Counter()
    for record in _iter_records(context.traces):
        service_id = _field(record, "service_id", "service_name", "service")
        if service_id is None or str(service_id) not in candidate_entities:
            continue
        service = str(service_id)
        span_count[service] += 1
        operation = _field(record, "operation_name", "operationName", "operation")
        if operation is None:
            continue
        operation_text = _compact(operation, 160)
        if not operation_text:
            continue
        operation_count[service][operation_text] += 1
        if _HTTP_OPERATION.match(operation_text):
            http_count[service] += 1
        lowered = operation_text.lower()
        if any(token in lowered for token in _DATA_TOKENS):
            data_count[service] += 1

    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)
    for record in _iter_records(context.observed_edges):
        try:
            subject, predicate, object_id = edge_key(record)
        except ValueError:
            continue
        if predicate != "CALLS":
            continue
        outbound[subject].add(object_id)
        inbound[object_id].add(subject)

    def service_lines(entity_id: str, prefix: str) -> list[str]:
        total = span_count[entity_id]
        in_degree = len(inbound[entity_id])
        out_degree = len(outbound[entity_id])
        degree_total = in_degree + out_degree
        operations = [name for name, _count in operation_count[entity_id].most_common(3)]
        upstream = _bounded(
            [labels.get(value, _display(value)) for value in inbound[entity_id]]
        )
        downstream = _bounded(
            [labels.get(value, _display(value)) for value in outbound[entity_id]]
        )
        return [
            (
                f"{prefix}.identity: label={labels[entity_id]}; type=Service; "
                f"basis={entity_basis.get(entity_id, 'model-trace-observed')}"
            ),
            (
                f"{prefix}.role_proxy: in_degree={in_degree}; out_degree={out_degree}; "
                f"orchestrator={_ratio(out_degree, degree_total):.4f}; "
                f"provider={_ratio(in_degree, degree_total):.4f}"
            ),
            f"{prefix}.neighbors: upstream=[{upstream}]; downstream=[{downstream}]",
            (
                f"{prefix}.telemetry: http={_ratio(http_count[entity_id], total):.6f}; "
                f"data_access={_ratio(data_count[entity_id], total):.6f}"
            ),
            f"{prefix}.operation_examples: [{_bounded(operations)}]",
        ]

    output: dict[tuple[str, str, str], PairRuntimeContext] = {}
    compact_system_label = _compact(system_label)
    for candidate in candidates:
        lines = [
            "Balanced runtime-role context from the sanitized model partition.",
            f"partial_hierarchy: System={compact_system_label} -> Service; Application, Instance, Host, and Deployment are unknown.",
            *service_lines(candidate.subject, "source"),
            *service_lines(candidate.object, "target"),
        ]
        provenance = ["model_masked_observed_graph", "model_trace_service_statistics"]
        if operation_count[candidate.subject] or operation_count[candidate.object]:
            provenance.append("model_trace_operations")
        if compact_system_label:
            provenance.append("incident_system_membership")
        output[candidate.key] = PairRuntimeContext(
            subject_label=labels[candidate.subject],
            object_label=labels[candidate.object],
            contextual_addendum="\n".join(lines),
            provenance=tuple(provenance + ["balanced_pair_serialization_v1"]),
        )
    return output


__all__ = ["build_runtime_pair_contexts"]
