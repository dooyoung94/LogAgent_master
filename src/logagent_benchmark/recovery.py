"""Minimal, leakage-safe A0--A5 relation-recovery ablations.

The model-facing API in this module deliberately has no reference graph or mask
manifest argument.  A reference graph belongs to :mod:`metrics`, never to a
recovery method.  Inputs may be plain mappings/sequences or pandas DataFrames;
the latter support is duck-typed so pandas is not a core dependency.

The heavy stages are intentionally strict:

* A3/A4 are ``SKIPPED`` unless an actual DeBERTa NLI backend is supplied.
* A5 is ``SKIPPED`` unless both DeBERTa and PSL backends are supplied.
* There is no lexical or rule-based fallback carrying a heavy-stage label.

Test doubles are accepted only with ``allow_test_backends=True`` and results are
then marked ``research_valid=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


READY = "READY"
SKIPPED = "SKIPPED"
ERROR = "ERROR"


def _field(record: Any, *names: str, default: Any = None) -> Any:
    """Read the first non-null field from a mapping, dataclass, or object."""

    if record is None:
        return default
    for name in names:
        if isinstance(record, Mapping) and name in record:
            value = record[name]
        else:
            value = getattr(record, name, None)
        if value is not None:
            # pandas uses NaN/NA for missing scalar values.  Avoid importing it.
            try:
                if isinstance(value, float) and math.isnan(value):
                    continue
            except (TypeError, ValueError):
                pass
            return value
    return default


def _records(value: Any) -> tuple[Any, ...]:
    """Convert DataFrames, mappings, and iterables to an immutable record list."""

    if value is None:
        return ()
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        try:
            return tuple(value.to_dict(orient="records"))
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _model_source(value: Any) -> Any:
    """Preserve a DataFrame for lazy row iteration; normalize small sources."""

    if value is not None and hasattr(value, "itertuples") and hasattr(value, "columns"):
        return value
    return _records(value)


def _iter_records(value: Any) -> Iterable[Any]:
    """Iterate DataFrame rows without materializing hundreds of thousands of dicts."""

    if value is None:
        return
    if hasattr(value, "itertuples") and hasattr(value, "columns"):
        columns = [str(column) for column in value.columns]
        for row in value.itertuples(index=False, name=None):
            yield dict(zip(columns, row))
        return
    if isinstance(value, Mapping):
        yield value
        return
    if isinstance(value, (str, bytes)):
        yield value
        return
    try:
        yield from value
    except TypeError:
        yield value


def _normal_type(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _normal_predicate(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _entity_id(record: Any) -> str | None:
    value = _field(
        record,
        "entity_id",
        "id",
        "service_id",
        "component_id",
        "canonical_name",
        "name",
    )
    return None if value is None else str(value)


def edge_key(edge: Any) -> tuple[str, str, str]:
    """Return a canonical edge tuple, accepting both public edge aliases."""

    if isinstance(edge, tuple) and len(edge) >= 3:
        return str(edge[0]), _normal_predicate(edge[1]), str(edge[2])
    subject = _field(edge, "subject", "subject_id", "source", "source_id")
    predicate = _field(edge, "predicate", "relation", "relation_type", "edge_type")
    obj = _field(edge, "object", "object_id", "target", "target_id")
    if subject is None or predicate is None or obj is None:
        raise ValueError(f"edge is missing subject/predicate/object: {edge!r}")
    return str(subject), _normal_predicate(predicate), str(obj)


@dataclass(frozen=True, order=True)
class Edge:
    subject: str
    predicate: str
    object: str
    confidence: float = field(default=1.0, compare=False)
    status: str = field(default="observed", compare=False)
    evidence_ids: tuple[str, ...] = field(default=(), compare=False)
    method: str = field(default="", compare=False)

    @property
    def subject_id(self) -> str:
        return self.subject

    @property
    def object_id(self) -> str:
        return self.object

    @property
    def key(self) -> tuple[str, str, str]:
        return self.subject, _normal_predicate(self.predicate), self.object


@dataclass(frozen=True)
class RelationSpec:
    domain_types: frozenset[str]
    range_types: frozenset[str]
    allow_self: bool = False
    functional: bool = False

    def allows(self, subject_type: str, object_type: str) -> bool:
        return (
            _normal_type(subject_type) in self.domain_types
            and _normal_type(object_type) in self.range_types
        )


DEFAULT_RELATION_SPECS: Mapping[str, RelationSpec] = {
    "CALLS": RelationSpec(
        frozenset({"SERVICE", "APPLICATION", "INSTANCE"}),
        frozenset({"SERVICE", "APPLICATION", "INSTANCE"}),
    ),
    "INSTANCE_OF": RelationSpec(
        frozenset({"INSTANCE"}),
        frozenset({"APPLICATION", "SERVICE"}),
        functional=True,
    ),
    "EXPOSES": RelationSpec(
        frozenset({"APPLICATION", "SERVICE"}),
        frozenset({"ENDPOINT", "APIENDPOINT"}),
    ),
    "ROUTES_TO": RelationSpec(
        frozenset({"WEBPAGE", "URL"}),
        frozenset({"ENDPOINT", "APIENDPOINT"}),
    ),
    "USES_DATASOURCE": RelationSpec(
        frozenset({"SERVICE", "APPLICATION"}),
        frozenset({"DATASOURCE", "DATABASE"}),
    ),
    "EXECUTES": RelationSpec(
        frozenset({"TRANSACTION"}),
        frozenset({"SQLPATTERN"}),
    ),
    "LOCATED_ON": RelationSpec(
        frozenset({"INSTANCE"}),
        frozenset({"HOST"}),
        functional=True,
    ),
}


@dataclass(frozen=True, order=True)
class Candidate:
    subject: str
    predicate: str
    object: str
    subject_type: str = field(compare=False)
    object_type: str = field(compare=False)
    origins: tuple[str, ...] = field(default=("typed",), compare=False)

    @property
    def subject_id(self) -> str:
        return self.subject

    @property
    def object_id(self) -> str:
        return self.object

    @property
    def key(self) -> tuple[str, str, str]:
        return self.subject, self.predicate, self.object

    @property
    def candidate_id(self) -> str:
        # Human-readable and stable; no UUID/random identity is introduced.
        return "|".join(self.key)


@dataclass(frozen=True)
class InferenceContext:
    """Model-visible data only.

    There is intentionally no ``reference_graph``, ``target_edges``, root label,
    or mask-manifest field here.
    """

    incident_id: str
    entities: Any
    observed_edges: Any = ()
    traces: Any = ()
    evidence: Any = ()
    logs: Any = ()
    metrics: Any = ()
    decision_time: Any = None

    @classmethod
    def from_model_input(cls, model_input: Mapping[str, Any]) -> "InferenceContext":
        """Build a context from ``IncidentBundle.model_input()`` output."""

        incident = model_input.get("incident", {})
        incident_id = _field(incident, "incident_id", "case_id", "id", default="unknown")
        return cls(
            incident_id=str(incident_id),
            entities=_model_source(model_input.get("entities")),
            observed_edges=_model_source(model_input.get("observed_edges")),
            traces=_model_source(model_input.get("traces")),
            evidence=_model_source(model_input.get("evidence")),
            logs=_model_source(model_input.get("logs")),
            metrics=_model_source(model_input.get("metrics")),
            decision_time=_field(incident, "decision_time", "end_time", "inject_time"),
        )


@dataclass(frozen=True)
class Availability:
    status: str
    reason_code: str = ""
    detail: str = ""
    research_valid: bool = True


@dataclass(frozen=True)
class CandidatePrediction:
    subject: str
    predicate: str
    object: str
    score: float
    decision: str
    evidence_ids: tuple[str, ...] = ()
    stage_scores: Mapping[str, float] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str, str]:
        return self.subject, self.predicate, self.object


@dataclass(frozen=True)
class RecoveryResult:
    variant: str
    status: str
    candidates: tuple[Candidate, ...]
    predictions: tuple[CandidatePrediction, ...] = ()
    accepted_edges: tuple[Edge, ...] = ()
    observed_edges: tuple[Edge, ...] = ()
    activation: Mapping[str, Any] = field(default_factory=dict)
    reason_code: str = ""
    detail: str = ""
    research_valid: bool = True

    @property
    def completed_edges(self) -> tuple[Edge, ...]:
        by_key = {edge.key: edge for edge in self.observed_edges}
        by_key.update({edge.key: edge for edge in self.accepted_edges})
        return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True)
class AblationConfig:
    a2_threshold: float = 0.60
    a3_threshold: float = 0.67
    a4_threshold: float = 0.67
    a5_threshold: float = 0.70
    a5_temperature: float = 1.0
    functional_margin: float = 0.10
    relation_thresholds: Mapping[str, float] = field(default_factory=dict)
    max_evidence_records: int = 40
    max_fallback_scan_records: int = 50_000

    def threshold(self, variant: str, predicate: str) -> float:
        key = f"{variant}:{predicate}"
        if key in self.relation_thresholds:
            return float(self.relation_thresholds[key])
        return float(getattr(self, f"{variant.lower()}_threshold"))


class DebertaBackend(Protocol):
    research_valid: bool

    def availability(self) -> Availability: ...

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> Sequence[Any]: ...


class PslBackend(Protocol):
    research_valid: bool

    def availability(self) -> Availability: ...

    def infer(
        self,
        *,
        context: InferenceContext,
        candidates: Sequence[Candidate],
        local_scores: Mapping[tuple[str, str, str], float],
    ) -> Any: ...


@dataclass
class RecoveryFeatureCache:
    """Per-context feature cache shared only across an ablation suite.

    The cache contains model-visible derived features, never evaluator labels.
    It binds to the Python context identity and exact candidate IDs to prevent
    accidental reuse across incidents or mask partitions.
    """

    context_identity: int | None = None
    candidate_ids: tuple[str, ...] = ()
    abduction_support: Mapping[
        tuple[str, str, str], tuple[float, tuple[str, ...]]
    ] | None = None
    deberta_backend_identity: int | None = None
    deberta_availability: Availability | None = None
    deberta_scores: Mapping[tuple[str, str, str], float] | None = None
    deberta_pair_count: int = 0
    deberta_error: Availability | None = None
    premises: Mapping[tuple[str, str, str], str] | None = None

    def bind(self, context: InferenceContext, candidates: Sequence[Candidate]) -> None:
        identity = id(context)
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if self.context_identity == identity and self.candidate_ids == candidate_ids:
            return
        self.context_identity = identity
        self.candidate_ids = candidate_ids
        self.abduction_support = None
        self.deberta_backend_identity = None
        self.deberta_availability = None
        self.deberta_scores = None
        self.deberta_pair_count = 0
        self.deberta_error = None
        self.premises = None

    def bind_deberta(self, backend: Any) -> None:
        identity = id(backend)
        if self.deberta_backend_identity == identity:
            return
        self.deberta_backend_identity = identity
        self.deberta_availability = None
        self.deberta_scores = None
        self.deberta_pair_count = 0
        self.deberta_error = None


class HuggingFaceDebertaNLIBackend:
    """Lazy, actual Hugging Face DeBERTa NLI cross-encoder backend.

    A plain DeBERTa masked-language model is rejected: its label map must expose
    both entailment and contradiction labels.  ``local_files_only=True`` avoids
    an implicit network download during a supposedly frozen experiment.
    """

    research_valid = True

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None,
        local_files_only: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._entailment_index: int | None = None
        self._contradiction_index: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        kwargs = {
            "revision": self.revision,
            "local_files_only": self.local_files_only,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, **kwargs
        )
        self._torch = torch
        labels = {
            int(index): str(label).lower()
            for index, label in self._model.config.id2label.items()
        }
        self._entailment_index = next(
            (index for index, label in labels.items() if "entail" in label), None
        )
        self._contradiction_index = next(
            (index for index, label in labels.items() if "contrad" in label), None
        )
        if self._entailment_index is None or self._contradiction_index is None:
            raise ValueError("configured DeBERTa artifact is not an NLI classifier")
        if self.device:
            self._model.to(self.device)
        self._model.eval()

    def availability(self) -> Availability:
        try:
            self._load()
        except (ImportError, ModuleNotFoundError) as exc:
            return Availability(SKIPPED, "DEBERTA_DEPENDENCY_MISSING", str(exc))
        except OSError as exc:
            return Availability(SKIPPED, "DEBERTA_MODEL_ARTIFACT_MISSING", str(exc))
        except Exception as exc:  # a present-but-invalid model is an error, not a skip
            return Availability(ERROR, "DEBERTA_MODEL_INVALID", str(exc))
        return Availability(READY)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> Sequence[Mapping[str, float]]:
        self._load()
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._entailment_index is not None
        assert self._contradiction_index is not None
        outputs: list[Mapping[str, float]] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset : offset + self.batch_size]
            premises = [item[0] for item in batch]
            hypotheses = [item[1] for item in batch]
            encoded = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device:
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1).detach().cpu().tolist()
            for row in probabilities:
                outputs.append(
                    {
                        "entailment": float(row[self._entailment_index]),
                        "contradiction": float(row[self._contradiction_index]),
                    }
                )
        return outputs


def _observed_edges(context: InferenceContext) -> tuple[Edge, ...]:
    output: list[Edge] = []
    for raw in _iter_records(context.observed_edges):
        try:
            subject, predicate, obj = edge_key(raw)
        except ValueError:
            continue
        output.append(
            Edge(
                subject,
                predicate,
                obj,
                confidence=float(_field(raw, "confidence", default=1.0)),
                status=str(_field(raw, "status", default="observed")),
                evidence_ids=tuple(_field(raw, "evidence_ids", default=()) or ()),
                method="observed",
            )
        )
    by_key = {edge.key: edge for edge in output}
    return tuple(by_key[key] for key in sorted(by_key))


def build_typed_candidates(
    context: InferenceContext,
    relation_specs: Mapping[str, RelationSpec] = DEFAULT_RELATION_SPECS,
) -> tuple[Candidate, ...]:
    """Create the one shared, label-blind candidate universe for A0--A5."""

    entities: dict[str, str] = {}
    for raw in _iter_records(context.entities):
        entity_id = _entity_id(raw)
        entity_type = _field(raw, "entity_type", "type", "node_type", "class_name")
        if entity_id is not None and entity_type is not None:
            entities[entity_id] = _normal_type(entity_type)

    observed = {edge.key for edge in _observed_edges(context)}
    candidates: list[Candidate] = []
    for predicate in sorted(relation_specs):
        spec = relation_specs[predicate]
        for subject in sorted(entities):
            for obj in sorted(entities):
                if subject == obj and not spec.allow_self:
                    continue
                if not spec.allows(entities[subject], entities[obj]):
                    continue
                key = subject, predicate, obj
                if key in observed:
                    continue
                candidates.append(
                    Candidate(
                        subject,
                        predicate,
                        obj,
                        entities[subject],
                        entities[obj],
                    )
                )
    return tuple(candidates)


def _evidence_id(record: Any, fallback: str) -> str:
    value = _field(record, "evidence_id", "event_id", "span_id", "log_id", "metric_id")
    return str(value) if value is not None else fallback


def _direct_relation(record: Any) -> tuple[str, str, str] | None:
    # Explicit edge-shaped evidence is deterministic by construction.
    try:
        return edge_key(record)
    except ValueError:
        pass

    caller = _field(record, "caller_service_id", "source_service_id", "parent_service_id")
    callee = _field(record, "callee_service_id", "peer_service_id", "target_service_id")
    if caller is not None and callee is not None:
        return str(caller), "CALLS", str(callee)

    service = _field(record, "service_id", "application_id")
    data_source = _field(record, "data_source_id", "datasource_id", "database_id", "db_id")
    if service is not None and data_source is not None:
        return str(service), "USES_DATASOURCE", str(data_source)

    transaction = _field(record, "transaction_id", "tx_id", "tx_uuid")
    sql_pattern = _field(record, "sql_pattern_id", "sql_id")
    if transaction is not None and sql_pattern is not None:
        return str(transaction), "EXECUTES", str(sql_pattern)

    instance = _field(record, "instance_id")
    host = _field(record, "host_id")
    if instance is not None and host is not None:
        return str(instance), "LOCATED_ON", str(host)
    return None


def _direct_support(context: InferenceContext) -> Mapping[tuple[str, str, str], tuple[str, ...]]:
    support: dict[tuple[str, str, str], set[str]] = {}
    # A1 consumes only an explicit, model-visible evidence table.  Treating a
    # raw log/metric/trace row as a declared edge would silently turn A1 into
    # another extractor and requires scanning the full 839k-span case.
    for index, record in enumerate(_iter_records(context.evidence)):
        relation = _direct_relation(record)
        if relation is not None:
            key = relation[0], _normal_predicate(relation[1]), relation[2]
            support.setdefault(key, set()).add(
                _evidence_id(record, f"direct-{index}")
            )
    return {key: tuple(sorted(ids)) for key, ids in support.items()}


def _time_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class _Span:
    trace_id: str
    span_id: str
    service_id: str
    start: float
    end: float
    parent_span_id: str | None

    @property
    def width(self) -> float:
        return self.end - self.start


def _span(record: Any, index: int) -> _Span | None:
    trace_id = _field(record, "trace_id", "traceid", "traceId")
    service_id = _field(
        record,
        "service_id",
        "service_name",
        "service",
        "application_id",
        "instance_id",
    )
    start = _time_value(
        _field(
            record,
            "start_time",
            "start_timestamp",
            "startTime",
            "start_time_unix_nano",
            "start_time_us",
            "timestamp",
        )
    )
    end = _time_value(
        _field(
            record,
            "end_time",
            "end_timestamp",
            "endTime",
            "end_time_unix_nano",
            "end_time_us",
        )
    )
    if end is None and start is not None:
        duration = _time_value(
            _field(record, "duration", "duration_ns", "duration_us", "elapsed")
        )
        if duration is not None and duration >= 0:
            end = start + duration
    if trace_id is None or service_id is None or start is None or end is None or end < start:
        return None
    span_id = _field(record, "span_id", "spanid", "spanId", default=f"span-{index}")
    parent_span_id = _field(record, "parent_span_id", "parentspanid", "parentSpanId")
    parent_text = None if parent_span_id is None else str(parent_span_id).strip()
    if parent_text is not None and parent_text.lower() in {
        "",
        "<na>",
        "nan",
        "nat",
        "none",
        "null",
    }:
        parent_text = None
    return _Span(str(trace_id), str(span_id), str(service_id), start, end, parent_text)


def temporal_containment_support(
    context: InferenceContext,
) -> Mapping[tuple[str, str, str], tuple[float, tuple[str, ...]]]:
    """Abduce CALLS edges from immediate, unambiguous interval containment.

    The masked model partition retains an ordinary-looking but unmatched parent
    ID.  A child is therefore selected when its non-null ``parent_span_id`` is
    absent from the span IDs of the same trace; no mask prefix or evaluator
    manifest is needed.  For each selected child, the narrowest strictly
    containing span is selected. Equal-width parents from different services
    are treated as ambiguous.
    """

    spans = [
        span
        for index, raw in enumerate(_iter_records(context.traces))
        if (span := _span(raw, index))
    ]
    by_trace: dict[str, list[_Span]] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)

    evidence: dict[tuple[str, str, str], set[str]] = {}
    traces: dict[tuple[str, str, str], set[str]] = {}
    for trace_id, trace_spans in by_trace.items():
        span_ids = {span.span_id for span in trace_spans}
        # Wider spans with the same start are visited first and can therefore be
        # active containers.  The active set is normally bounded by trace depth,
        # avoiding an all-pairs scan over large RCAEval traces.
        trace_spans.sort(key=lambda item: (item.start, -item.end, item.span_id))
        active: list[_Span] = []
        for child in trace_spans:
            active = [parent for parent in active if parent.end >= child.start]
            parent_unmatched = (
                child.parent_span_id is not None
                and child.parent_span_id not in span_ids
            )
            if not parent_unmatched:
                active.append(child)
                continue
            containers = [
                parent
                for parent in active
                if parent.span_id != child.span_id
                and parent.service_id != child.service_id
                and parent.end >= child.end
                and (parent.start < child.start or parent.end > child.end)
            ]
            if not containers:
                active.append(child)
                continue
            containers.sort(key=lambda item: (item.width, item.start, item.span_id))
            minimum_width = containers[0].width
            tied_services = {
                item.service_id
                for item in containers
                if math.isclose(item.width, minimum_width, rel_tol=0.0, abs_tol=1e-12)
            }
            if len(tied_services) != 1:
                active.append(child)
                continue
            parent = containers[0]
            key = parent.service_id, "CALLS", child.service_id
            evidence.setdefault(key, set()).update({parent.span_id, child.span_id})
            traces.setdefault(key, set()).add(trace_id)
            active.append(child)

    output: dict[tuple[str, str, str], tuple[float, tuple[str, ...]]] = {}
    for key, evidence_ids in evidence.items():
        unique_trace_count = len(traces[key])
        score = 1.0 - math.exp(-float(unique_trace_count))
        output[key] = score, tuple(sorted(evidence_ids))
    return output


def _safe_record_text(record: Any) -> str:
    if isinstance(record, Mapping):
        blocked = ("root_cause", "ground_truth", "reference", "mask", "injection", "fault_type")
        items = [
            (str(key), value)
            for key, value in record.items()
            if value is not None and not any(token in str(key).lower() for token in blocked)
        ]
        items.sort(key=lambda item: item[0])
        return "; ".join(f"{key}={value}" for key, value in items)
    return str(record)


_ENTITY_MENTION_FIELDS = (
    "subject",
    "subject_id",
    "object",
    "object_id",
    "source",
    "source_id",
    "target",
    "target_id",
    "entity_id",
    "service_id",
    "application_id",
    "instance_id",
    "caller_service_id",
    "callee_service_id",
    "source_service_id",
    "target_service_id",
    "peer_service_id",
    "data_source_id",
    "database_id",
    "db_id",
    "endpoint_id",
    "host_id",
)


def _record_mentions(
    record: Any,
    entity_ids: set[str],
    *,
    scan_text: bool,
) -> tuple[set[str], str | None]:
    mentions: set[str] = set()
    for name in _ENTITY_MENTION_FIELDS:
        value = _field(record, name)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        for item in values:
            text_value = str(item)
            if text_value in entity_ids:
                mentions.add(text_value)
    text: str | None = None
    if scan_text:
        text = _safe_record_text(record)
        # Text scanning is restricted to the small pair-summary source.  Large
        # trace/log fallback uses structured entity IDs and is scanned once.
        mentions.update(entity_id for entity_id in entity_ids if entity_id in text)
    return mentions, text


def _build_premises(
    context: InferenceContext,
    candidates: Sequence[Candidate],
    *,
    per_candidate_limit: int,
    fallback_scan_limit: int,
) -> Mapping[tuple[str, str, str], str]:
    """Build candidate premises in one bounded pass over telemetry.

    A both-endpoint summary from ``context.evidence`` wins immediately and is
    used alone.  Only candidates lacking such a summary receive bounded
    subject/object fallback snippets.  This avoids candidates independently
    rescanning an RCAEval trace table with hundreds of thousands of rows.
    """

    entity_ids = {
        endpoint
        for candidate in candidates
        for endpoint in (candidate.subject, candidate.object)
    }
    pair_summaries: dict[frozenset[str], str] = {}
    for record in _iter_records(context.evidence):
        mentions, text = _record_mentions(record, entity_ids, scan_text=True)
        if text is None or len(mentions) < 2:
            continue
        ordered = sorted(mentions)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                pair_summaries.setdefault(frozenset((left, right)), text)

    per_entity_limit = max(1, (per_candidate_limit + 1) // 2)
    snippets: dict[str, list[str]] = {entity_id: [] for entity_id in entity_ids}
    remaining = set(entity_ids)
    scanned = 0
    for source in (context.evidence, context.traces, context.logs, context.metrics):
        for record in _iter_records(source):
            if scanned >= fallback_scan_limit or not remaining:
                break
            scanned += 1
            mentions, text = _record_mentions(
                record,
                remaining,
                scan_text=(source is context.evidence),
            )
            relevant = mentions & remaining
            if not relevant:
                continue
            if text is None:
                text = _safe_record_text(record)
            for entity_id in relevant:
                if text not in snippets[entity_id]:
                    snippets[entity_id].append(text)
                if len(snippets[entity_id]) >= per_entity_limit:
                    remaining.discard(entity_id)
        if scanned >= fallback_scan_limit or not remaining:
            break

    output: dict[tuple[str, str, str], str] = {}
    for candidate in candidates:
        pair_summary = pair_summaries.get(
            frozenset((candidate.subject, candidate.object))
        )
        if pair_summary is not None:
            output[candidate.key] = pair_summary
            continue
        selected = (
            snippets.get(candidate.subject, [])
            + snippets.get(candidate.object, [])
        )[:per_candidate_limit]
        output[candidate.key] = (
            "\n".join(selected)
            if selected
            else "No incident-local evidence mentions both operational entities."
        )
    return output


def _hypothesis(candidate: Candidate) -> str:
    templates = {
        "CALLS": "{s} calls {o}.",
        "INSTANCE_OF": "{s} is an instance of {o}.",
        "EXPOSES": "{s} exposes endpoint {o}.",
        "ROUTES_TO": "{s} routes to endpoint {o}.",
        "USES_DATASOURCE": "{s} uses data source {o}.",
        "EXECUTES": "{s} executes SQL pattern {o}.",
        "LOCATED_ON": "{s} is located on host {o}.",
    }
    return templates.get(candidate.predicate, "{s} {p} {o}.").format(
        s=candidate.subject,
        p=candidate.predicate,
        o=candidate.object,
    )


def _backend_availability(backend: Any, kind: str, allow_test_backends: bool) -> Availability:
    if backend is None:
        return Availability(SKIPPED, f"{kind}_BACKEND_MISSING")
    research_valid = bool(getattr(backend, "research_valid", False))
    if not research_valid and not allow_test_backends:
        return Availability(SKIPPED, f"{kind}_TEST_BACKEND_FORBIDDEN", research_valid=False)
    try:
        availability = backend.availability()
    except (ImportError, ModuleNotFoundError) as exc:
        return Availability(SKIPPED, f"{kind}_DEPENDENCY_MISSING", str(exc), research_valid)
    except OSError as exc:
        return Availability(SKIPPED, f"{kind}_ARTIFACT_MISSING", str(exc), research_valid)
    except Exception as exc:
        return Availability(ERROR, f"{kind}_AVAILABILITY_ERROR", str(exc), research_valid)
    if isinstance(availability, Availability):
        return Availability(
            availability.status,
            availability.reason_code,
            availability.detail,
            research_valid and availability.research_valid,
        )
    if availability is True:
        return Availability(READY, research_valid=research_valid)
    if availability is False:
        return Availability(SKIPPED, f"{kind}_UNAVAILABLE", research_valid=research_valid)
    raise TypeError(f"{kind} backend availability() returned an unsupported value")


def _nli_value(raw: Any) -> float:
    if isinstance(raw, Mapping):
        entailment = float(_field(raw, "entailment", "p_entailment", default=0.0))
        contradiction = float(_field(raw, "contradiction", "p_contradiction", default=0.0))
        return min(1.0, max(0.0, (1.0 + entailment - contradiction) / 2.0))
    return min(1.0, max(0.0, float(raw)))


def _deberta_scores(
    backend: DebertaBackend,
    candidates: Sequence[Candidate],
    premises: Mapping[tuple[str, str, str], str],
) -> tuple[Mapping[tuple[str, str, str], float], int]:
    pairs = [
        (premises[candidate.key], _hypothesis(candidate))
        for candidate in candidates
    ]
    raw_scores = backend.score_pairs(pairs)
    if len(raw_scores) != len(candidates):
        raise ValueError("DeBERTa backend returned a score count different from candidates")
    return {
        candidate.key: _nli_value(score) for candidate, score in zip(candidates, raw_scores)
    }, len(pairs)


def _fuse(abduction: float, deberta: float) -> float:
    # Pre-registered no-dependency fusion for smoke tests.  Learned/frozen
    # relation-specific weights can replace this function in full experiments.
    epsilon = 1e-6
    a = min(1.0 - epsilon, max(epsilon, abduction))
    d = min(1.0 - epsilon, max(epsilon, deberta))
    logit = 0.5 * math.log(a / (1.0 - a)) + 0.5 * math.log(d / (1.0 - d))
    return 1.0 / (1.0 + math.exp(-logit))


def _temperature_scale(probability: float, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    epsilon = 1e-9
    p = min(1.0 - epsilon, max(epsilon, probability))
    logit = math.log(p / (1.0 - p)) / temperature
    return 1.0 / (1.0 + math.exp(-logit))


def _score_stats(values: Sequence[float]) -> tuple[float, int]:
    if not values:
        return 0.0, 0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    unique = len({round(value, 12) for value in values})
    return math.sqrt(variance), unique


def _predictions(
    *,
    variant: str,
    candidates: Sequence[Candidate],
    scores: Mapping[tuple[str, str, str], float],
    threshold: Any,
    evidence: Mapping[tuple[str, str, str], tuple[str, ...]] | None = None,
    stage_scores: Mapping[tuple[str, str, str], Mapping[str, float]] | None = None,
    forced_unresolved: set[tuple[str, str, str]] | None = None,
) -> tuple[CandidatePrediction, ...]:
    evidence = evidence or {}
    stage_scores = stage_scores or {}
    forced_unresolved = forced_unresolved or set()
    output: list[CandidatePrediction] = []
    for candidate in candidates:
        score = float(scores.get(candidate.key, 0.0))
        candidate_threshold = float(threshold(candidate.predicate))
        accepted = score >= candidate_threshold and candidate.key not in forced_unresolved
        reasons = []
        if candidate.key in forced_unresolved:
            reasons.append("AMBIGUOUS_MARGIN")
        elif not accepted:
            reasons.append("BELOW_THRESHOLD")
        output.append(
            CandidatePrediction(
                candidate.subject,
                candidate.predicate,
                candidate.object,
                score,
                "accepted" if accepted else "unresolved",
                evidence.get(candidate.key, ()),
                stage_scores.get(candidate.key, {variant.lower(): score}),
                tuple(reasons),
            )
        )
    return tuple(output)


def _result(
    variant: str,
    context: InferenceContext,
    candidates: tuple[Candidate, ...],
    predictions: tuple[CandidatePrediction, ...],
    *,
    activation: Mapping[str, Any],
    research_valid: bool = True,
) -> RecoveryResult:
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
        for prediction in predictions
        if prediction.decision == "accepted"
    )
    return RecoveryResult(
        variant,
        READY,
        candidates,
        predictions,
        accepted,
        _observed_edges(context),
        activation,
        research_valid=research_valid,
    )


def _unavailable_result(
    variant: str,
    context: InferenceContext,
    candidates: tuple[Candidate, ...],
    availability: Availability,
) -> RecoveryResult:
    return RecoveryResult(
        variant,
        availability.status,
        candidates,
        observed_edges=_observed_edges(context),
        activation={"stage_calls": 0, "candidate_count": len(candidates)},
        reason_code=availability.reason_code,
        detail=availability.detail,
        research_valid=availability.research_valid,
    )


def run_recovery(
    variant: str,
    context: InferenceContext,
    *,
    candidates: Sequence[Candidate] | None = None,
    config: AblationConfig | None = None,
    relation_specs: Mapping[str, RelationSpec] = DEFAULT_RELATION_SPECS,
    deberta_backend: DebertaBackend | None = None,
    psl_backend: PslBackend | None = None,
    allow_test_backends: bool = False,
    feature_cache: RecoveryFeatureCache | None = None,
) -> RecoveryResult:
    """Run one ablation without accepting evaluator-only inputs."""

    variant = variant.upper()
    if variant not in {"A0", "A1", "A2", "A3", "A4", "A5"}:
        raise ValueError(f"unsupported ablation: {variant}")
    config = config or AblationConfig()
    shared = tuple(candidates) if candidates is not None else build_typed_candidates(context, relation_specs)
    cache = feature_cache or RecoveryFeatureCache()
    cache.bind(context, shared)

    if variant == "A0":
        scores = {candidate.key: 0.0 for candidate in shared}
        predictions = _predictions(
            variant=variant,
            candidates=shared,
            scores=scores,
            threshold=lambda _predicate: 1.1,
        )
        return _result(
            variant,
            context,
            shared,
            predictions,
            activation={
                "candidate_count": len(shared),
                "stage_calls": 0,
                "score_std": 0.0,
                "unique_score_count": 1 if shared else 0,
            },
        )

    if variant == "A1":
        support = _direct_support(context)
        scores = {candidate.key: 1.0 if candidate.key in support else 0.0 for candidate in shared}
        predictions = _predictions(
            variant=variant,
            candidates=shared,
            scores=scores,
            threshold=lambda _predicate: 1.0,
            evidence=support,
        )
        score_std, unique = _score_stats(list(scores.values()))
        return _result(
            variant,
            context,
            shared,
            predictions,
            activation={
                "candidate_count": len(shared),
                "stage_calls": len(support),
                "direct_supported_count": len(support),
                "score_std": score_std,
                "unique_score_count": unique,
            },
        )

    # A3 is intentionally DeBERTa-only and must not pay for or consume an
    # abductive feature.  A2/A4/A5 share one containment pass in suite mode.
    if variant in {"A2", "A4", "A5"}:
        if cache.abduction_support is None:
            cache.abduction_support = temporal_containment_support(context)
        containment = cache.abduction_support
    else:
        containment = {}
    abduction_scores = {candidate.key: containment.get(candidate.key, (0.0, ()))[0] for candidate in shared}
    abduction_evidence = {key: value[1] for key, value in containment.items()}

    if variant == "A2":
        predictions = _predictions(
            variant=variant,
            candidates=shared,
            scores=abduction_scores,
            threshold=lambda predicate: config.threshold("A2", predicate),
            evidence=abduction_evidence,
        )
        score_std, unique = _score_stats(list(abduction_scores.values()))
        return _result(
            variant,
            context,
            shared,
            predictions,
            activation={
                "candidate_count": len(shared),
                "stage_calls": len(context.traces),
                "abductive_supported_count": sum(score > 0 for score in abduction_scores.values()),
                "score_std": score_std,
                "unique_score_count": unique,
            },
        )

    cache.bind_deberta(deberta_backend)
    if cache.deberta_availability is None:
        cache.deberta_availability = _backend_availability(
            deberta_backend, "DEBERTA", allow_test_backends
        )
    deberta_availability = cache.deberta_availability
    if deberta_availability.status != READY:
        return _unavailable_result(variant, context, shared, deberta_availability)
    assert deberta_backend is not None
    if cache.deberta_error is not None:
        return _unavailable_result(variant, context, shared, cache.deberta_error)
    if cache.deberta_scores is None:
        try:
            if cache.premises is None:
                cache.premises = _build_premises(
                    context,
                    shared,
                    per_candidate_limit=config.max_evidence_records,
                    fallback_scan_limit=config.max_fallback_scan_records,
                )
            cache.deberta_scores, cache.deberta_pair_count = _deberta_scores(
                deberta_backend, shared, cache.premises
            )
        except Exception as exc:
            cache.deberta_error = Availability(
                ERROR,
                "DEBERTA_INFERENCE_ERROR",
                str(exc),
                deberta_availability.research_valid,
            )
            return _unavailable_result(variant, context, shared, cache.deberta_error)
    deberta_scores = cache.deberta_scores
    nli_calls = cache.deberta_pair_count

    if variant == "A3":
        predictions = _predictions(
            variant=variant,
            candidates=shared,
            scores=deberta_scores,
            threshold=lambda predicate: config.threshold("A3", predicate),
            stage_scores={key: {"deberta": score} for key, score in deberta_scores.items()},
        )
        score_std, unique = _score_stats(list(deberta_scores.values()))
        return _result(
            variant,
            context,
            shared,
            predictions,
            activation={
                "candidate_count": len(shared),
                "stage_calls": nli_calls,
                "nli_pair_count": nli_calls,
                "score_std": score_std,
                "unique_score_count": unique,
            },
            research_valid=deberta_availability.research_valid,
        )

    fused_scores = {
        candidate.key: _fuse(abduction_scores[candidate.key], deberta_scores[candidate.key])
        for candidate in shared
    }
    fused_stages = {
        candidate.key: {
            "abduction": abduction_scores[candidate.key],
            "deberta": deberta_scores[candidate.key],
            "fusion": fused_scores[candidate.key],
        }
        for candidate in shared
    }

    if variant == "A4":
        predictions = _predictions(
            variant=variant,
            candidates=shared,
            scores=fused_scores,
            threshold=lambda predicate: config.threshold("A4", predicate),
            evidence=abduction_evidence,
            stage_scores=fused_stages,
        )
        score_std, unique = _score_stats(list(fused_scores.values()))
        fusion_delta = sum(
            abs(fused_scores[key] - deberta_scores[key]) > 1e-9 for key in fused_scores
        )
        return _result(
            variant,
            context,
            shared,
            predictions,
            activation={
                "candidate_count": len(shared),
                "stage_calls": nli_calls + len(context.traces),
                "nli_pair_count": nli_calls,
                "fusion_score_delta_count": fusion_delta,
                "score_std": score_std,
                "unique_score_count": unique,
            },
            research_valid=deberta_availability.research_valid,
        )

    psl_availability = _backend_availability(psl_backend, "PSL", allow_test_backends)
    if psl_availability.status != READY:
        return _unavailable_result(variant, context, shared, psl_availability)
    assert psl_backend is not None
    psl_relation = _normal_predicate(getattr(psl_backend, "relation", "CALLS"))
    psl_candidates = tuple(
        candidate for candidate in shared if candidate.predicate == psl_relation
    )
    psl_local_scores = {
        candidate.key: fused_scores[candidate.key] for candidate in psl_candidates
    }
    try:
        raw_psl = psl_backend.infer(
            context=context,
            candidates=psl_candidates,
            local_scores=psl_local_scores,
        )
        grounded_rule_count = int(_field(raw_psl, "grounded_rule_count", default=0))
        grounded_atom_count = int(_field(raw_psl, "grounded_atom_count", default=0))
        psl_metadata = dict(_field(raw_psl, "metadata", default={}) or {})
        raw_values = _field(raw_psl, "scores", "posteriors", default=raw_psl)
        if isinstance(raw_values, Mapping):
            inferred_psl_scores = {
                candidate.key: float(
                    raw_values.get(candidate.key, raw_values.get(candidate.candidate_id, 0.0))
                )
                for candidate in psl_candidates
            }
        else:
            values = list(raw_values)
            if len(values) != len(psl_candidates):
                raise ValueError("PSL backend returned a score count different from candidates")
            inferred_psl_scores = {
                candidate.key: float(value)
                for candidate, value in zip(psl_candidates, values)
            }
        # The official PSL rule set currently models CALLS only.  Other typed
        # relations keep their A4 local score; they are not silently fed into a
        # relation-incompatible PSL program.
        psl_scores = dict(fused_scores)
        psl_scores.update(inferred_psl_scores)
    except Exception as exc:
        return _unavailable_result(
            variant,
            context,
            shared,
            Availability(
                ERROR,
                "PSL_INFERENCE_ERROR",
                str(exc),
                deberta_availability.research_valid and psl_availability.research_valid,
            ),
        )

    calibrated = {
        key: _temperature_scale(min(1.0, max(0.0, score)), config.a5_temperature)
        for key, score in psl_scores.items()
    }
    forced_unresolved: set[tuple[str, str, str]] = set()
    for predicate, spec in relation_specs.items():
        if not spec.functional:
            continue
        groups: dict[str, list[tuple[tuple[str, str, str], float]]] = {}
        for key, score in calibrated.items():
            if key[1] == predicate:
                groups.setdefault(key[0], []).append((key, score))
        for values in groups.values():
            values.sort(key=lambda item: (-item[1], item[0][2]))
            if len(values) >= 2 and values[0][1] - values[1][1] < config.functional_margin:
                # A functional relation with no clear winner must abstain for
                # the whole subject group; accepting the runner-up would invert
                # the intended ambiguity rule.
                forced_unresolved.update(key for key, _score in values)

    stages = {
        candidate.key: {
            **fused_stages[candidate.key],
            "psl": psl_scores[candidate.key],
            "calibrated": calibrated[candidate.key],
        }
        for candidate in shared
    }
    predictions = _predictions(
        variant=variant,
        candidates=shared,
        scores=calibrated,
        threshold=lambda predicate: config.threshold("A5", predicate),
        evidence=abduction_evidence,
        stage_scores=stages,
        forced_unresolved=forced_unresolved,
    )
    score_std, unique = _score_stats(list(calibrated.values()))
    return _result(
        variant,
        context,
        shared,
        predictions,
        activation={
            "candidate_count": len(shared),
            "stage_calls": nli_calls + grounded_rule_count,
            "nli_pair_count": nli_calls,
            "psl_relation": psl_relation,
            "psl_candidate_count": len(psl_candidates),
            "grounded_rule_count": grounded_rule_count,
            "grounded_atom_count": grounded_atom_count,
            "psl_metadata": psl_metadata,
            "psl_score_delta_count": sum(
                abs(psl_scores[key] - fused_scores[key]) > 1e-9 for key in psl_scores
            ),
            "calibration_score_delta_count": sum(
                abs(calibrated[key] - psl_scores[key]) > 1e-9 for key in calibrated
            ),
            "abstention_count": len(forced_unresolved),
            "score_std": score_std,
            "unique_score_count": unique,
        },
        research_valid=(
            deberta_availability.research_valid and psl_availability.research_valid
        ),
    )


@dataclass(frozen=True)
class AblationSuiteResult:
    candidates: tuple[Candidate, ...]
    results: Mapping[str, RecoveryResult]
    activation: Mapping[str, Any]
    gate: Mapping[str, Any]


def run_ablation_suite(
    context: InferenceContext,
    *,
    variants: Sequence[str] = ("A0", "A1", "A2", "A3", "A4", "A5"),
    config: AblationConfig | None = None,
    relation_specs: Mapping[str, RelationSpec] = DEFAULT_RELATION_SPECS,
    deberta_backend: DebertaBackend | None = None,
    psl_backend: PslBackend | None = None,
    allow_test_backends: bool = False,
    require_variants: Sequence[str] = (),
    require_activation_pairs: Sequence[str | tuple[str, str]] = (),
) -> AblationSuiteResult:
    """Run every requested variant over exactly the same candidate tuple."""

    shared = build_typed_candidates(context, relation_specs)
    feature_cache = RecoveryFeatureCache()
    results = {
        variant.upper(): run_recovery(
            variant,
            context,
            candidates=shared,
            config=config,
            relation_specs=relation_specs,
            deberta_backend=deberta_backend,
            psl_backend=psl_backend,
            allow_test_backends=allow_test_backends,
            feature_cache=feature_cache,
        )
        for variant in variants
    }
    # Local import avoids making the core method depend on evaluator code.
    from .metrics import activation_matrix, zero_flip_gate

    activation = activation_matrix(results)
    gate = zero_flip_gate(
        results,
        activation,
        require_variants=require_variants,
        require_activation_pairs=require_activation_pairs,
    )
    return AblationSuiteResult(shared, results, activation, gate)


def experiment_fingerprint(
    context: InferenceContext,
    candidates: Sequence[Candidate],
    config: AblationConfig,
) -> str:
    """Stable, non-sensitive fingerprint for reproducibility manifests."""

    payload = {
        "incident_id": context.incident_id,
        "observed_edges": sorted(
            edge_key(edge) for edge in _iter_records(context.observed_edges)
        ),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "config": {
            "a2_threshold": config.a2_threshold,
            "a3_threshold": config.a3_threshold,
            "a4_threshold": config.a4_threshold,
            "a5_threshold": config.a5_threshold,
            "a5_temperature": config.a5_temperature,
            "functional_margin": config.functional_margin,
            "relation_thresholds": dict(config.relation_thresholds),
            "max_evidence_records": config.max_evidence_records,
            "max_fallback_scan_records": config.max_fallback_scan_records,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
