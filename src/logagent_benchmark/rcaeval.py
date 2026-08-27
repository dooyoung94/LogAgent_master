"""Convert an RCAEval case into the LogAgent incident contract.

The source case directory and RCAEval case index contain evaluator answers in
their names and columns.  This module deliberately keeps those values out of
``model_input()``.  Parent-span links are also retained only in
``canonical_traces`` so that a reference graph can be built without handing
the exact edge answer to a relation-recovery model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = "1.0"
_REQUIRED_METRIC_COLUMNS = {"time"}
_REQUIRED_LOG_COLUMNS = {"timestamp", "container_name", "message"}
_REQUIRED_TRACE_COLUMNS = {
    "time",
    "traceID",
    "spanID",
    "serviceName",
    "methodName",
    "operationName",
    "parentSpanID",
    "startTimeMillis",
    "startTime",
    "duration",
    "statusCode",
}
_LABEL_KEYS = {"case", "root_cause_service", "fault", "fault_description"}


class RCAEvalSchemaError(ValueError):
    """Raised when an RCAEval case violates the expected source contract."""


@dataclass(frozen=True)
class IncidentBundle:
    """Normalized case with model input and evaluator truth kept apart.

    ``canonical_traces`` contains ``parent_span_id`` and is therefore
    restricted to reference-graph construction.  The ``traces`` property and
    ``model_input()`` omit that field and the derived root-span flag.
    """

    incident: Mapping[str, Any]
    entities: pd.DataFrame
    metrics: pd.DataFrame
    logs: pd.DataFrame
    canonical_traces: pd.DataFrame
    evaluator_labels: Mapping[str, Any]
    restricted_provenance: Mapping[str, Any]

    @property
    def traces(self) -> pd.DataFrame:
        """Return the model-safe trace view without exact parent answers."""

        return self.canonical_traces.drop(
            columns=["parent_span_id", "is_root_span"], errors="ignore"
        )

    def model_input(self) -> dict[str, Any]:
        """Return only fields that an RCA or relation model may consume."""

        incident = dict(self.incident)
        leaked = _LABEL_KEYS.intersection(incident)
        if leaked:
            raise RCAEvalSchemaError(
                f"model incident metadata contains evaluator labels: {sorted(leaked)}"
            )
        return {
            "incident": incident,
            "entities": self.entities.copy(deep=False),
            "metrics": self.metrics.copy(deep=False),
            "logs": self.logs.copy(deep=False),
            "traces": self.traces,
        }

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        """Persist the bundle using physically separate model/evaluator paths."""

        return write_incident_bundle(self, output_dir)


_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_KV_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|authorization|access[_-]?token|secret)"
    r"(\s*[:=]\s*|\s+)([^\s,;]+)"
)
_LOGIN_USER_RE = re.compile(
    r"(?i)(LOGIN\s+USER\s*:\s*)(\S+)(\s+__\s+)(\S+)"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_IPV4_RE = re.compile(
    r"(?<![0-9])(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})"
    r"(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}(?![0-9])"
)
_SEVERITY_RE = re.compile(r"\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b")


def _redact_message(value: Any) -> tuple[Any, int]:
    if pd.isna(value):
        return pd.NA, 0
    text = str(value)
    count = 0
    substitutions = (
        (_JWT_RE, "<JWT>"),
        (_BEARER_RE, "Bearer <CREDENTIAL>"),
        (_LOGIN_USER_RE, r"\1<USER>\3<CREDENTIAL>"),
        (_SECRET_KV_RE, r"\1=<CREDENTIAL>"),
        (_EMAIL_RE, "<EMAIL>"),
        (_UUID_RE, "<UUID>"),
        (_IPV4_RE, "<IPV4>"),
    )
    for pattern, replacement in substitutions:
        text, changed = pattern.subn(replacement, text)
        count += changed
    return text, count


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RCAEvalSchemaError(f"{source} is missing columns: {missing}")


def _read_optional_parquet(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame({name: pd.Series(dtype="object") for name in sorted(required)})
    frame = pd.read_parquet(path)
    _require_columns(frame, required, path.name)
    return frame


def _as_integer(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise RCAEvalSchemaError(f"{label} contains null or non-numeric values")
    fractional = values.astype("float64") % 1
    if fractional.ne(0).any():
        raise RCAEvalSchemaError(f"{label} contains non-integral values")
    return values.astype("int64")


def _entity_type(alias: str, observed_traces: bool) -> tuple[str, str, float]:
    lowered = alias.lower()
    if lowered.endswith(("-mongo", "-mysql")):
        return "DataSource", "heuristic_suffix", 0.95
    if lowered.endswith("-dashboard"):
        return "WebApplication", "heuristic_suffix", 0.90
    if observed_traces:
        return "Service", "trace.serviceName", 0.99
    if lowered.endswith("service"):
        return "Service", "heuristic_suffix", 0.90
    return "Component", "unresolved", 0.50


def _entity_id(system_id: str, entity_type: str, alias: str) -> str:
    type_token = entity_type.lower().replace(" ", "-")
    return f"rcaeval:{system_id}:{type_token}:{alias}"


def _parse_metric_columns(columns: list[str]) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    for column in columns:
        if column == "time":
            continue
        entity, separator, metric_name = column.rpartition("_")
        if not separator or not entity or not metric_name:
            raise RCAEvalSchemaError(
                f"metric column {column!r} does not match <entity>_<metric>"
            )
        parsed[column] = (entity, metric_name)
    if not parsed:
        raise RCAEvalSchemaError("metrics.parquet contains no metric series")
    return parsed


def _opaque_incident_id(case_name: str, revision: str) -> tuple[str, str]:
    source_key = f"RCAEval\x1f{revision}\x1f{case_name}"
    digest = sha256(source_key.encode("utf-8")).hexdigest()
    return f"inc_rcaeval_{digest[:24]}", digest


def _canonical_system_id(source_code: str, system_name: str) -> str:
    known = {
        "tt": "train-ticket",
        "ob": "online-boutique",
        "ss": "sock-shop",
    }
    if source_code.lower() in known:
        return known[source_code.lower()]
    slug = re.sub(r"[^a-z0-9]+", "-", system_name.strip().lower()).strip("-")
    if not slug:
        raise RCAEvalSchemaError("unable to derive a canonical system_id")
    return slug


def _validated_incident_id(
    requested: str | None,
    generated: str,
    *,
    case_name: str,
    root_alias: str,
    fault: str,
) -> tuple[str, str]:
    if requested is None:
        return generated, "sha256-derived"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", requested):
        raise RCAEvalSchemaError(
            "incident_id override must be an 8-128 character opaque identifier"
        )
    lowered = requested.lower()
    forbidden = (case_name.lower(), root_alias.lower(), fault.lower())
    if any(value and value in lowered for value in forbidden):
        raise RCAEvalSchemaError(
            "incident_id override must not encode the source case, root label, or fault"
        )
    return requested, "caller-supplied-opaque"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_index_row(index_path: Path, case_name: str) -> pd.Series:
    if not index_path.is_file():
        raise RCAEvalSchemaError(f"RCAEval case index not found: {index_path}")
    index = pd.read_parquet(index_path)
    _require_columns(
        index,
        {
            "case",
            "dataset",
            "suite",
            "system",
            "system_name",
            "root_cause_service",
            "fault",
            "inject_time",
            "time_start",
            "time_end",
        },
        index_path.name,
    )
    matches = index[index["case"].astype("string").eq(case_name)]
    if len(matches) != 1:
        raise RCAEvalSchemaError(
            f"expected one index row for the source case, found {len(matches)}"
        )
    return matches.iloc[0]


def _build_entities(
    incident_id: str,
    system_id: str,
    metric_aliases: set[str],
    log_aliases: set[str],
    trace_aliases: set[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    records: list[dict[str, Any]] = []
    entity_ids: dict[str, str] = {}
    for alias in sorted(metric_aliases | log_aliases | trace_aliases):
        entity_type, basis, confidence = _entity_type(alias, alias in trace_aliases)
        canonical_id = _entity_id(system_id, entity_type, alias)
        entity_ids[alias] = canonical_id
        records.append(
            {
                "incident_id": incident_id,
                "entity_id": canonical_id,
                "canonical_name": alias,
                "entity_type": entity_type,
                "source_alias": alias,
                "observed_metrics": alias in metric_aliases,
                "observed_logs": alias in log_aliases,
                "observed_traces": alias in trace_aliases,
                "type_basis": basis,
                "type_confidence": confidence,
            }
        )
    entities = pd.DataFrame.from_records(
        records,
        columns=[
            "incident_id",
            "entity_id",
            "canonical_name",
            "entity_type",
            "source_alias",
            "observed_metrics",
            "observed_logs",
            "observed_traces",
            "type_basis",
            "type_confidence",
        ],
    )
    return entities, entity_ids


def _normalize_metrics(
    raw: pd.DataFrame,
    incident_id: str,
    parsed_columns: Mapping[str, tuple[str, str]],
    entity_ids: Mapping[str, str],
) -> pd.DataFrame:
    raw = raw.copy()
    raw["time"] = _as_integer(raw["time"], "metrics.time")
    if raw["time"].duplicated().any() or not raw["time"].is_monotonic_increasing:
        raise RCAEvalSchemaError("metrics.time must be unique and monotonically increasing")

    long = raw.melt(
        id_vars=["time"],
        value_vars=list(parsed_columns),
        var_name="source_metric",
        value_name="value",
    )
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    long["entity_id"] = long["source_metric"].map(
        lambda name: entity_ids[parsed_columns[str(name)][0]]
    )
    long["metric_name"] = long["source_metric"].map(
        lambda name: parsed_columns[str(name)][1]
    )
    long["event_time_us"] = long["time"].astype("int64") * 1_000_000
    long["value"] = pd.to_numeric(long["value"], errors="raise").astype("float64")
    long.insert(0, "incident_id", incident_id)
    long["unit"] = "unknown"
    return long[
        [
            "incident_id",
            "event_time_us",
            "entity_id",
            "metric_name",
            "value",
            "unit",
        ]
    ]


def _normalize_logs(
    raw: pd.DataFrame,
    incident_id: str,
    entity_ids: Mapping[str, str],
) -> pd.DataFrame:
    columns = [
        "incident_id",
        "event_id",
        "event_time_us",
        "entity_id",
        "body",
        "severity",
        "redaction_count",
        "quality_flag",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    timestamps = _as_integer(raw["timestamp"], "logs.timestamp")
    aliases = raw["container_name"].astype("string")
    if aliases.isna().any() or not set(aliases.astype(str)).issubset(entity_ids):
        raise RCAEvalSchemaError("logs.container_name contains an unresolved entity")

    bodies: list[Any] = []
    counts: list[int] = []
    severities: list[Any] = []
    quality: list[str] = []
    for value in raw["message"]:
        redacted, count = _redact_message(value)
        bodies.append(redacted)
        counts.append(count)
        if pd.isna(value):
            severities.append(pd.NA)
            quality.append("missing_body")
        else:
            match = _SEVERITY_RE.search(str(value))
            if match:
                severity = match.group(1)
                severities.append("WARN" if severity == "WARNING" else severity)
            else:
                severities.append(pd.NA)
            quality.append("observed")

    frame = pd.DataFrame(
        {
            "incident_id": incident_id,
            "event_id": [
                f"{incident_id}:log:{position:09d}" for position in range(len(raw))
            ],
            "event_time_us": timestamps * 1_000_000,
            "entity_id": aliases.astype(str).map(entity_ids),
            "body": pd.array(bodies, dtype="string"),
            "severity": pd.array(severities, dtype="string"),
            "redaction_count": pd.Series(counts, dtype="int64"),
            "quality_flag": pd.array(quality, dtype="string"),
        }
    )
    return frame[columns]


def _normalize_traces(
    raw: pd.DataFrame,
    incident_id: str,
    entity_ids: Mapping[str, str],
) -> pd.DataFrame:
    columns = [
        "incident_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "service_id",
        "operation_name",
        "method_name",
        "source_time_label",
        "start_time_us",
        "end_time_us",
        "duration_us",
        "status_code",
        "is_root_span",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    start_us = _as_integer(raw["startTime"], "traces.startTime")
    start_ms = _as_integer(raw["startTimeMillis"], "traces.startTimeMillis")
    precision_delta = start_us - start_ms * 1_000
    if precision_delta.lt(0).any() or precision_delta.gt(999).any():
        raise RCAEvalSchemaError(
            "traces.startTime is not a microsecond refinement of startTimeMillis"
        )
    duration_us = _as_integer(raw["duration"], "traces.duration")
    if duration_us.lt(0).any():
        raise RCAEvalSchemaError("traces.duration contains a negative value")

    trace_ids = raw["traceID"].astype("string")
    span_ids = raw["spanID"].astype("string")
    if trace_ids.isna().any() or span_ids.isna().any():
        raise RCAEvalSchemaError("traceID and spanID must be populated")
    if pd.DataFrame({"trace": trace_ids, "span": span_ids}).duplicated().any():
        raise RCAEvalSchemaError("(traceID, spanID) must be unique")

    aliases = raw["serviceName"].astype("string")
    if aliases.isna().any() or not set(aliases.astype(str)).issubset(entity_ids):
        raise RCAEvalSchemaError("traces.serviceName contains an unresolved entity")

    parent_ids = raw["parentSpanID"].astype("string").replace("", pd.NA)
    status = pd.to_numeric(raw["statusCode"], errors="coerce").astype("Int64")
    frame = pd.DataFrame(
        {
            "incident_id": incident_id,
            "trace_id": trace_ids,
            "span_id": span_ids,
            "parent_span_id": parent_ids,
            "service_id": aliases.astype(str).map(entity_ids),
            "operation_name": raw["operationName"].astype("string"),
            "method_name": raw["methodName"].astype("string"),
            "source_time_label": raw["time"].astype("string"),
            "start_time_us": start_us,
            "end_time_us": start_us + duration_us,
            "duration_us": duration_us,
            "status_code": status,
            "is_root_span": parent_ids.isna(),
        }
    )
    return frame[columns]


def _derived_root_indicator(
    root_alias: str, fault: str, parsed_columns: Mapping[str, tuple[str, str]]
) -> tuple[str | None, str]:
    candidates = {
        "cpu": ("cpu",),
        "mem": ("mem",),
        "disk": ("diskio",),
        "socket": ("socket",),
        "delay": ("latency-90",),
        "loss": ("error",),
    }.get(fault.lower(), ())
    available = set(parsed_columns.values())
    for metric_name in candidates:
        if (root_alias, metric_name) in available:
            return (
                f"{root_alias}_{metric_name}",
                "derived_from_root_service_and_fault;not_source_ground_truth",
            )
    return None, "unavailable"


def convert_rcaeval_case(
    case_dir: str | Path,
    *,
    cases_index_path: str | Path | None = None,
    dataset_revision: str = "unversioned",
    incident_id: str | None = None,
) -> IncidentBundle:
    """Normalize one RCAEval Parquet case.

    Parameters
    ----------
    case_dir:
        Directory containing ``metrics.parquet`` and optional log/trace files.
        Its label-bearing basename is used internally only to locate the index
        row and derive an opaque ID.
    cases_index_path:
        ``cases.parquet`` path.  Defaults to the parent of ``case_dir``.
    dataset_revision:
        Immutable Hugging Face/Git revision used as part of the opaque ID and
        restricted provenance.
    incident_id:
        Optional caller-controlled opaque ID for cross-stage reproducibility.
        Label-bearing IDs are rejected.  The default is SHA-256-derived.
    """

    source_dir = Path(case_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise RCAEvalSchemaError(f"case directory not found: {source_dir}")
    index_path = (
        Path(cases_index_path).expanduser().resolve()
        if cases_index_path is not None
        else source_dir.parent / "cases.parquet"
    )
    label_row = _case_index_row(index_path, source_dir.name)
    generated_incident_id, source_case_digest = _opaque_incident_id(
        source_dir.name, dataset_revision
    )
    root_alias = str(label_row["root_cause_service"])
    fault = str(label_row["fault"])
    incident_id, incident_id_strategy = _validated_incident_id(
        incident_id,
        generated_incident_id,
        case_name=source_dir.name,
        root_alias=root_alias,
        fault=fault,
    )

    metrics_path = source_dir / "metrics.parquet"
    if not metrics_path.is_file():
        raise RCAEvalSchemaError(f"metrics.parquet not found in {source_dir}")
    raw_metrics = pd.read_parquet(metrics_path)
    _require_columns(raw_metrics, _REQUIRED_METRIC_COLUMNS, metrics_path.name)
    parsed_metrics = _parse_metric_columns(list(raw_metrics.columns))
    raw_logs = _read_optional_parquet(source_dir / "logs.parquet", _REQUIRED_LOG_COLUMNS)
    raw_traces = _read_optional_parquet(
        source_dir / "traces.parquet", _REQUIRED_TRACE_COLUMNS
    )

    metric_aliases = {entity for entity, _ in parsed_metrics.values()}
    log_aliases = (
        set(raw_logs["container_name"].dropna().astype(str))
        if not raw_logs.empty
        else set()
    )
    trace_aliases = (
        set(raw_traces["serviceName"].dropna().astype(str))
        if not raw_traces.empty
        else set()
    )
    source_system_code = str(label_row["system"])
    system_id = _canonical_system_id(
        source_system_code, str(label_row["system_name"])
    )
    entities, entity_ids = _build_entities(
        incident_id,
        system_id,
        metric_aliases,
        log_aliases,
        trace_aliases,
    )

    metrics = _normalize_metrics(
        raw_metrics, incident_id, parsed_metrics, entity_ids
    )
    logs = _normalize_logs(raw_logs, incident_id, entity_ids)
    canonical_traces = _normalize_traces(raw_traces, incident_id, entity_ids)

    inject_file = source_dir / "inject_time.txt"
    if not inject_file.is_file():
        raise RCAEvalSchemaError(f"inject_time.txt not found in {source_dir}")
    try:
        inject_time = int(inject_file.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise RCAEvalSchemaError("inject_time.txt is not an integer") from exc
    indexed_inject_time = int(label_row["inject_time"])
    if inject_time != indexed_inject_time:
        raise RCAEvalSchemaError(
            "inject_time.txt does not match the evaluator index"
        )

    window_start = int(label_row["time_start"])
    window_end = int(label_row["time_end"])
    if not window_start <= inject_time <= window_end:
        raise RCAEvalSchemaError("injection timestamp falls outside the incident window")

    incident = {
        "schema_version": SCHEMA_VERSION,
        "incident_id": incident_id,
        "dataset_id": "rcaeval",
        "benchmark_dataset": str(label_row["dataset"]),
        "suite": str(label_row["suite"]),
        "system_id": system_id,
        "system_name": str(label_row["system_name"]),
        "window_start_us": window_start * 1_000_000,
        "anomaly_time_us": inject_time * 1_000_000,
        "window_end_us": window_end * 1_000_000,
        "entity_count": int(len(entities)),
        "metric_observation_count": int(len(metrics)),
        "log_event_count": int(len(logs)),
        "span_count": int(len(canonical_traces)),
    }

    if root_alias not in entity_ids:
        raise RCAEvalSchemaError("evaluator root service is absent from the entity registry")
    root_indicator, indicator_provenance = _derived_root_indicator(
        root_alias, fault, parsed_metrics
    )
    evaluator_labels = {
        "schema_version": SCHEMA_VERSION,
        "incident_id": incident_id,
        "root_cause_entity_id": entity_ids[root_alias],
        "root_cause_service": root_alias,
        "fault_type": fault,
        "root_cause_indicator": root_indicator,
        "root_cause_indicator_provenance": indicator_provenance,
        "gold_cause_path": None,
        "gold_impact_path": None,
        "label_source": "cases.parquet",
    }

    source_files: dict[str, dict[str, Any]] = {}
    for name in ("metrics.parquet", "logs.parquet", "traces.parquet", "inject_time.txt"):
        path = source_dir / name
        if path.is_file():
            source_files[name] = {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
    restricted_provenance = {
        "schema_version": SCHEMA_VERSION,
        "incident_id": incident_id,
        "dataset_revision": dataset_revision,
        "incident_id_strategy": incident_id_strategy,
        "source_system_code": source_system_code,
        "source_case_key_sha256": source_case_digest,
        "case_index_sha256": _file_sha256(index_path),
        "source_files": source_files,
        "raw_case_name_stored": False,
    }

    return IncidentBundle(
        incident=incident,
        entities=entities,
        metrics=metrics,
        logs=logs,
        canonical_traces=canonical_traces,
        evaluator_labels=evaluator_labels,
        restricted_provenance=restricted_provenance,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_incident_bundle(
    bundle: IncidentBundle, output_dir: str | Path
) -> dict[str, Path]:
    """Write normalized artifacts without mixing evaluator truth into input."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise RCAEvalSchemaError(f"refusing to overwrite non-empty output: {target}")
    telemetry_dir = target / "telemetry"
    evaluator_dir = target / "evaluator"
    restricted_dir = target / "restricted"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    restricted_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "incident": target / "incident.json",
        "entities": target / "entities.parquet",
        "metrics": telemetry_dir / "metrics.parquet",
        "logs": telemetry_dir / "logs.parquet",
        "traces": telemetry_dir / "traces.parquet",
        "canonical_traces": restricted_dir / "canonical_traces.parquet",
        "evaluator_labels": evaluator_dir / "labels.json",
        "provenance": restricted_dir / "provenance.json",
    }
    _write_json(paths["incident"], bundle.incident)
    bundle.entities.to_parquet(paths["entities"], index=False)
    bundle.metrics.to_parquet(paths["metrics"], index=False)
    bundle.logs.to_parquet(paths["logs"], index=False)
    bundle.traces.to_parquet(paths["traces"], index=False)
    bundle.canonical_traces.to_parquet(paths["canonical_traces"], index=False)
    _write_json(paths["evaluator_labels"], bundle.evaluator_labels)
    _write_json(paths["provenance"], bundle.restricted_provenance)
    return paths
