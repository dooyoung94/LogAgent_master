"""Task A Phase 3-R2: operational evidence reranking before NLI.

The stage keeps the bounded A2 proposal set immutable, derives candidate-level
CALLS evidence from sanitized model traces, and asks whether operation/HTTP/
role evidence improves ranking over an equal-size A2-only control.

RCAEval exposes operationName and methodName but not OpenTelemetry span.kind,
http.route, source_workload, or destination_workload.  The extractor therefore
uses direct fields when present and emits explicit availability diagnostics;
otherwise it falls back to transparent proxies learned from unmasked spans.
Evaluator labels are attached only after all model-visible features are frozen.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .phase3_contract import _json_mapping, _json_sequence, stable_case_split


CELL_KEY = ("case", "seed", "mask_id")
CANDIDATE_KEY = ("subject", "predicate", "object")
MODEL_COLUMNS = {
    "subject",
    "predicate",
    "object",
    "a2_score",
    "proposal_rank",
    "supporting_traces",
    "boundary_spans",
    "reverse_supporting_traces",
    "reverse_boundary_spans",
    "direct_evidence",
}
EVALUATOR_COLUMNS = {"is_masked_target", "is_silver_matched", "fault", "role"}
HTTP_VERBS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
HTTP_VERB_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b", re.I)
URL_RE = re.compile(r"https?://[^/\s]+(?P<path>/[^\s?#]*)", re.I)
PATH_RE = re.compile(r"(?P<path>/[A-Za-z0-9._~!$&'()*+,;=:@%/{}-]+)")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27,}$", re.I)
HEX_RE = re.compile(r"^[0-9a-f]{12,}$", re.I)
NUMBER_RE = re.compile(r"^\d+$")
TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]+", re.I)
TOKEN_STOP = frozenset(
    {
        "http",
        "https",
        "service",
        "server",
        "client",
        "request",
        "response",
        "call",
        "calls",
        "invoke",
        "invokes",
        "unknown",
        *{verb.lower() for verb in HTTP_VERBS},
    }
)


class Phase3R2Error(RuntimeError):
    """Raised when the frozen R2 experiment contract is violated."""


@dataclass(frozen=True)
class OperationalPolicy:
    profile: str
    retention_fraction: float
    minimum_keep: int
    evidence_weight: float

    def __post_init__(self) -> None:
        if self.profile not in {
            "direction_role",
            "operation_endpoint",
            "method_route",
            "combined",
        }:
            raise ValueError(f"unsupported operational profile: {self.profile}")
        if not 0.0 < self.retention_fraction <= 1.0:
            raise ValueError("retention_fraction must be in (0,1]")
        if isinstance(self.minimum_keep, bool) or self.minimum_keep <= 0:
            raise ValueError("minimum_keep must be a positive integer")
        if not 0.0 <= self.evidence_weight <= 1.0:
            raise ValueError("evidence_weight must be in [0,1]")


@dataclass(frozen=True)
class GateConfig:
    recall_macro_min: float = 0.95
    recall_pooled_min: float = 0.95
    recall_each_cell_min: float = 0.90
    selected_count_ratio_max: float = 0.95
    p_lb_delta_vs_full_min: float = 0.0
    mrr_delta_vs_full_min: float = 0.0
    matched_budget_recall_tolerance: float = 0.0
    matched_budget_p_lb_delta_min: float = 0.0
    matched_budget_mrr_delta_min: float = 0.0
    additive_gain_min: float = 1e-12
    boundary_alignment_macro_min: float = 0.95
    operation_pair_coverage_min: float = 0.20


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split()).strip()


def _raw_operation_key(operation: Any, method_name: Any) -> str:
    return f"{_safe_text(operation)}\x1f{_safe_text(method_name)}"


def _http_method(operation: Any, method_name: Any, explicit: Any = None) -> str:
    for value in (explicit, method_name, operation):
        text = _safe_text(value).upper()
        if text in HTTP_VERBS:
            return text
        match = HTTP_VERB_RE.search(text)
        if match:
            return match.group(1).upper()
    return ""


def _normalize_route(operation: Any, method_name: Any, explicit: Any = None) -> str:
    for value in (explicit, operation, method_name):
        text = _safe_text(value)
        if not text:
            continue
        match = URL_RE.search(text) or PATH_RE.search(text)
        if not match:
            continue
        route = match.group("path").split("?", 1)[0].rstrip("/") or "/"
        normalized = []
        for segment in route.split("/"):
            lowered = segment.lower()
            if not lowered:
                normalized.append("")
            elif NUMBER_RE.match(lowered) or UUID_RE.match(lowered) or HEX_RE.match(lowered):
                normalized.append("{id}")
            else:
                normalized.append(lowered)
        return "/".join(normalized)
    return ""


def _operation_tokens(operation: Any, method_name: Any) -> frozenset[str]:
    text = f"{_safe_text(operation)} {_safe_text(method_name)}".lower()
    return frozenset(
        token
        for token in TOKEN_RE.findall(text)
        if token not in TOKEN_STOP and len(token) > 1
    )


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float | None:
    a, b = set(left), set(right)
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def _mean(values: Iterable[float | None], default: float = 0.0) -> float:
    selected = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.fmean(selected) if selected else default


def _rank_normalize(frame: pd.DataFrame, score_column: str) -> pd.Series:
    order = frame.sort_values(
        [score_column, "a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).index
    denominator = max(1, len(frame) - 1)
    output = pd.Series(0.0, index=frame.index, dtype=float)
    for rank, index in enumerate(order):
        output.loc[index] = 1.0 - rank / denominator
    return output


def _parse_a2(a2_path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(a2_path)
    required = {
        "subject",
        "predicate",
        "object",
        "score",
        "decision",
        "evidence_ids_json",
        "stage_scores_json",
        "reason_codes_json",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Phase3R2Error(f"A2 parquet missing columns {missing}: {a2_path}")
    records: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        if str(item.decision) != "accepted":
            raise Phase3R2Error("R2 expects the frozen accepted A2 proposal set")
        stage = _json_mapping(
            item.stage_scores_json, field_name="stage_scores_json"
        )
        reasons = {
            str(value)
            for value in _json_sequence(
                item.reason_codes_json, field_name="reason_codes_json"
            )
        }
        evidence_ids = tuple(
            str(value)
            for value in _json_sequence(
                item.evidence_ids_json, field_name="evidence_ids_json"
            )
        )
        records.append(
            {
                "subject": str(item.subject),
                "predicate": str(item.predicate),
                "object": str(item.object),
                "a2_score": float(item.score),
                "proposal_rank": int(float(stage.get("proposal_rank", 0))),
                "supporting_traces": int(
                    float(stage.get("supporting_traces", 0))
                ),
                "boundary_spans": int(float(stage.get("boundary_spans", 0))),
                "direct_evidence": "DIRECT_EVIDENCE" in reasons,
                "evidence_ids": evidence_ids,
            }
        )
    by_key = {
        (row["subject"], row["predicate"], row["object"]): row
        for row in records
    }
    for row in records:
        reverse = by_key.get((row["object"], row["predicate"], row["subject"]))
        row["reverse_supporting_traces"] = (
            int(reverse["supporting_traces"]) if reverse else 0
        )
        row["reverse_boundary_spans"] = (
            int(reverse["boundary_spans"]) if reverse else 0
        )
    return records


def _trace_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    aliases = {
        "trace_id": ("trace_id", "traceID"),
        "span_id": ("span_id", "spanID"),
        "parent_span_id": ("parent_span_id", "parentSpanID"),
        "service_id": ("service_id", "serviceName"),
        "operation_name": ("operation_name", "operationName"),
        "method_name": ("method_name", "methodName"),
        "start_time_us": ("start_time_us", "startTime"),
        "end_time_us": ("end_time_us",),
        "duration_us": ("duration_us", "duration"),
        "span_kind": ("span_kind", "spanKind", "attr.span_kind"),
        "http_method": ("http_method", "attr.http.request.method"),
        "http_route": ("http_route", "attr.http.route", "attr.url.path"),
        "source_workload": ("source_workload", "attr.source_workload"),
        "destination_workload": (
            "destination_workload",
            "attr.destination_workload",
        ),
    }
    resolved: dict[str, str | None] = {}
    for canonical, names in aliases.items():
        resolved[canonical] = next(
            (name for name in names if name in frame.columns), None
        )
    required = (
        "trace_id",
        "span_id",
        "parent_span_id",
        "service_id",
        "start_time_us",
    )
    missing = [name for name in required if resolved[name] is None]
    if missing:
        raise Phase3R2Error(f"trace artifact missing required fields: {missing}")
    if resolved["end_time_us"] is None and resolved["duration_us"] is None:
        raise Phase3R2Error("trace artifact requires end_time_us or duration_us")
    return resolved


def _canonical_trace_frame(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    columns = _trace_columns(raw)
    output = pd.DataFrame(
        {
            "trace_id": raw[columns["trace_id"]].astype(str),
            "span_id": raw[columns["span_id"]].astype(str),
            "parent_span_id": raw[columns["parent_span_id"]].astype("string"),
            "service_id": raw[columns["service_id"]].astype(str),
            "operation_name": (
                raw[columns["operation_name"]].astype("string")
                if columns["operation_name"]
                else pd.Series("", index=raw.index, dtype="string")
            ),
            "method_name": (
                raw[columns["method_name"]].astype("string")
                if columns["method_name"]
                else pd.Series("", index=raw.index, dtype="string")
            ),
            "start_time_us": pd.to_numeric(
                raw[columns["start_time_us"]], errors="raise"
            ),
        }
    )
    if columns["end_time_us"]:
        output["end_time_us"] = pd.to_numeric(
            raw[columns["end_time_us"]], errors="raise"
        )
    else:
        output["end_time_us"] = output["start_time_us"] + pd.to_numeric(
            raw[columns["duration_us"]], errors="raise"
        )
    output["span_kind"] = (
        raw[columns["span_kind"]].astype("string")
        if columns["span_kind"]
        else pd.Series("", index=raw.index, dtype="string")
    )
    output["http_method"] = (
        raw[columns["http_method"]].astype("string")
        if columns["http_method"]
        else pd.Series("", index=raw.index, dtype="string")
    )
    output["http_route"] = (
        raw[columns["http_route"]].astype("string")
        if columns["http_route"]
        else pd.Series("", index=raw.index, dtype="string")
    )
    output["source_workload"] = (
        raw[columns["source_workload"]].astype("string")
        if columns["source_workload"]
        else pd.Series("", index=raw.index, dtype="string")
    )
    output["destination_workload"] = (
        raw[columns["destination_workload"]].astype("string")
        if columns["destination_workload"]
        else pd.Series("", index=raw.index, dtype="string")
    )
    parent = output["parent_span_id"].str.strip().str.lower()
    output.loc[
        parent.isin({"", "<na>", "nan", "none", "null"}), "parent_span_id"
    ] = pd.NA
    availability = {
        "operation_name": columns["operation_name"] is not None,
        "method_name": columns["method_name"] is not None,
        "span_kind": columns["span_kind"] is not None,
        "http_method": columns["http_method"] is not None,
        "http_route": columns["http_route"] is not None,
        "source_workload": columns["source_workload"] is not None,
        "destination_workload": columns["destination_workload"] is not None,
    }
    return output, availability


def _degree_features(
    observed_edges: pd.DataFrame, services: set[str]
) -> dict[str, dict[str, float]]:
    if observed_edges.empty:
        return {
            service: {
                "in_degree": 0.0,
                "out_degree": 0.0,
                "caller_ratio": 0.5,
                "callee_ratio": 0.5,
            }
            for service in services
        }
    required = {"subject", "predicate", "object"}
    missing = required.difference(observed_edges.columns)
    if missing:
        raise Phase3R2Error(f"observed edge table missing {sorted(missing)}")
    calls = observed_edges.loc[
        observed_edges["predicate"].astype(str).str.upper().eq("CALLS")
    ]
    outbound = (
        calls.groupby(calls["subject"].astype(str))["object"].nunique().to_dict()
    )
    inbound = (
        calls.groupby(calls["object"].astype(str))["subject"].nunique().to_dict()
    )
    result: dict[str, dict[str, float]] = {}
    for service in services:
        out_degree = float(outbound.get(service, 0))
        in_degree = float(inbound.get(service, 0))
        total = in_degree + out_degree
        result[service] = {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "caller_ratio": (out_degree + 1.0) / (total + 2.0),
            "callee_ratio": (in_degree + 1.0) / (total + 2.0),
        }
    return result


def _selected_span_frame(
    traces: pd.DataFrame, records: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    evidence_ids = {
        evidence_id
        for record in records
        for evidence_id in record.get("evidence_ids", ())
    }
    if not evidence_ids:
        return traces.iloc[0:0].copy().set_index("span_id", drop=False)
    selected = traces.loc[traces["span_id"].isin(evidence_ids)].copy()
    return selected.set_index("span_id", drop=False)


def _boundary_pairs(
    selected_spans: pd.DataFrame,
    *,
    evidence_ids: set[str],
    subject: str,
    object_id: str,
) -> list[tuple[Any, Any]]:
    available = [
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id in selected_spans.index
    ]
    if not available:
        return []
    subset = selected_spans.loc[available]
    if isinstance(subset, pd.Series):
        subset = subset.to_frame().T
    subset = subset.loc[subset["service_id"].isin({subject, object_id})]
    pairs: list[tuple[Any, Any]] = []
    for _trace_id, group in subset.groupby("trace_id", sort=False):
        sources = list(
            group.loc[group["service_id"].eq(subject)].itertuples(index=False)
        )
        if not sources:
            continue
        for child in group.loc[
            group["service_id"].eq(object_id)
        ].itertuples(index=False):
            containers = [
                parent
                for parent in sources
                if float(parent.start_time_us) <= float(child.start_time_us)
                and float(parent.end_time_us) >= float(child.end_time_us)
                and (
                    float(parent.start_time_us) < float(child.start_time_us)
                    or float(parent.end_time_us) > float(child.end_time_us)
                )
            ]
            if not containers:
                continue
            containers.sort(
                key=lambda parent: (
                    float(parent.end_time_us) - float(parent.start_time_us),
                    float(parent.start_time_us),
                    str(parent.span_id),
                )
            )
            pairs.append((containers[0], child))
    return pairs


def _operation_role_priors(
    traces: pd.DataFrame, relevant_raw_keys: set[str]
) -> dict[str, dict[str, float]]:
    if not relevant_raw_keys:
        return {}
    raw_keys = (
        traces["operation_name"].fillna("").astype(str)
        + "\x1f"
        + traces["method_name"].fillna("").astype(str)
    )
    mask = raw_keys.isin(relevant_raw_keys)
    selected = traces.loc[
        mask, ["trace_id", "span_id", "parent_span_id"]
    ].copy()
    if selected.empty:
        return {}
    selected["raw_operation_key"] = raw_keys.loc[mask].to_numpy()
    selected["has_parent"] = selected["parent_span_id"].notna().astype(float)

    parent_refs = traces.loc[
        traces["parent_span_id"].notna(), ["trace_id", "parent_span_id"]
    ].drop_duplicates()
    parent_refs.columns = ["trace_id", "span_id"]
    parent_index = pd.MultiIndex.from_frame(parent_refs)
    selected_index = pd.MultiIndex.from_frame(
        selected[["trace_id", "span_id"]]
    )
    selected["has_child"] = selected_index.isin(parent_index).astype(float)

    grouped = selected.groupby("raw_operation_key", sort=False).agg(
        occurrence_count=("span_id", "size"),
        parent_role_rate=("has_child", "mean"),
        child_role_rate=("has_parent", "mean"),
    )
    return {
        str(index): {
            "occurrence_count": float(row.occurrence_count),
            "parent_role_rate": float(row.parent_role_rate),
            "child_role_rate": float(row.child_role_rate),
        }
        for index, row in grouped.iterrows()
    }


def _pair_metrics(
    parent: Any, child: Any, subject: str, object_id: str
) -> dict[str, Any]:
    parent_method = _http_method(
        parent.operation_name, parent.method_name, parent.http_method
    )
    child_method = _http_method(
        child.operation_name, child.method_name, child.http_method
    )
    parent_route = _normalize_route(
        parent.operation_name, parent.method_name, parent.http_route
    )
    child_route = _normalize_route(
        child.operation_name, child.method_name, child.http_route
    )
    operation_jaccard = _jaccard(
        _operation_tokens(parent.operation_name, parent.method_name),
        _operation_tokens(child.operation_name, child.method_name),
    )
    route_jaccard = _jaccard(
        [segment for segment in parent_route.split("/") if segment],
        [segment for segment in child_route.split("/") if segment],
    )
    source_kind = _safe_text(parent.span_kind).upper()
    target_kind = _safe_text(child.span_kind).upper()
    kind_known = bool(source_kind and target_kind)
    kind_compatible = (
        source_kind in {"CLIENT", "PRODUCER"}
        and target_kind in {"SERVER", "CONSUMER"}
        if kind_known
        else None
    )
    source_workload = _safe_text(parent.source_workload)
    destination_workload = _safe_text(parent.destination_workload)
    workload_known = bool(source_workload or destination_workload)
    workload_match = None
    if workload_known:
        workload_match = float(
            (not source_workload or source_workload in subject)
            and (not destination_workload or destination_workload in object_id)
        )
    return {
        "source_raw_operation_key": _raw_operation_key(
            parent.operation_name, parent.method_name
        ),
        "target_raw_operation_key": _raw_operation_key(
            child.operation_name, child.method_name
        ),
        "operation_pair_key": (
            _raw_operation_key(parent.operation_name, parent.method_name),
            _raw_operation_key(child.operation_name, child.method_name),
        ),
        "method_known": bool(parent_method and child_method),
        "method_match": (
            float(parent_method == child_method)
            if parent_method and child_method
            else None
        ),
        "route_known": bool(parent_route and child_route),
        "route_exact": (
            float(parent_route == child_route)
            if parent_route and child_route
            else None
        ),
        "route_jaccard": route_jaccard,
        "operation_jaccard": operation_jaccard,
        "span_kind_known": kind_known,
        "span_kind_compatible": (
            float(kind_compatible) if kind_compatible is not None else None
        ),
        "workload_known": workload_known,
        "workload_match": workload_match,
    }


def _candidate_feature_rows(
    records: Sequence[dict[str, Any]],
    traces: pd.DataFrame,
    observed_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    services = {
        str(record[endpoint])
        for record in records
        for endpoint in ("subject", "object")
    }
    degree = _degree_features(observed_edges, services)
    selected_spans = _selected_span_frame(traces, records)
    candidate_pairs: dict[tuple[str, str, str], list[tuple[Any, Any]]] = {}
    relevant_raw_keys: set[str] = set()
    for record in records:
        key = (record["subject"], record["predicate"], record["object"])
        pairs = _boundary_pairs(
            selected_spans,
            evidence_ids=set(record["evidence_ids"]),
            subject=record["subject"],
            object_id=record["object"],
        )
        candidate_pairs[key] = pairs
        for parent, child in pairs:
            relevant_raw_keys.add(
                _raw_operation_key(parent.operation_name, parent.method_name)
            )
            relevant_raw_keys.add(
                _raw_operation_key(child.operation_name, child.method_name)
            )
    role_priors = _operation_role_priors(traces, relevant_raw_keys)

    rows: list[dict[str, Any]] = []
    pair_count_total = 0
    expected_boundary_total = 0
    method_known_total = 0
    route_known_total = 0
    span_kind_known_total = 0
    workload_known_total = 0
    for record in records:
        key = (record["subject"], record["predicate"], record["object"])
        pairs = candidate_pairs[key]
        metrics = [
            _pair_metrics(parent, child, record["subject"], record["object"])
            for parent, child in pairs
        ]
        pair_count_total += len(metrics)
        expected_boundary_total += int(record["boundary_spans"])
        method_known_total += sum(bool(item["method_known"]) for item in metrics)
        route_known_total += sum(bool(item["route_known"]) for item in metrics)
        span_kind_known_total += sum(
            bool(item["span_kind_known"]) for item in metrics
        )
        workload_known_total += sum(
            bool(item["workload_known"]) for item in metrics
        )

        pair_frequencies: dict[tuple[str, str], int] = {}
        for item in metrics:
            pair_key = item["operation_pair_key"]
            pair_frequencies[pair_key] = pair_frequencies.get(pair_key, 0) + 1
        pair_concentration = (
            max(pair_frequencies.values()) / len(metrics) if metrics else 0.0
        )

        source_parent_prior = _mean(
            (
                role_priors.get(item["source_raw_operation_key"], {}).get(
                    "parent_role_rate"
                )
                for item in metrics
            ),
            default=0.5,
        )
        source_child_prior = _mean(
            (
                role_priors.get(item["source_raw_operation_key"], {}).get(
                    "child_role_rate"
                )
                for item in metrics
            ),
            default=0.5,
        )
        target_parent_prior = _mean(
            (
                role_priors.get(item["target_raw_operation_key"], {}).get(
                    "parent_role_rate"
                )
                for item in metrics
            ),
            default=0.5,
        )
        target_child_prior = _mean(
            (
                role_priors.get(item["target_raw_operation_key"], {}).get(
                    "child_role_rate"
                )
                for item in metrics
            ),
            default=0.5,
        )
        operation_role_score = _mean(
            [
                source_parent_prior,
                target_child_prior,
                1.0 - target_parent_prior,
                1.0 - source_child_prior,
            ],
            default=0.5,
        )

        source_role = degree[record["subject"]]
        target_role = degree[record["object"]]
        graph_role_score = _mean(
            [
                source_role["caller_ratio"],
                target_role["callee_ratio"],
                1.0 - target_role["caller_ratio"],
                1.0 - source_role["callee_ratio"],
            ],
            default=0.5,
        )

        f_trace = float(record["supporting_traces"])
        r_trace = float(record["reverse_supporting_traces"])
        f_boundary = float(record["boundary_spans"])
        r_boundary = float(record["reverse_boundary_spans"])
        trace_margin = (f_trace - r_trace) / (f_trace + r_trace + 1.0)
        boundary_margin = (f_boundary - r_boundary) / (
            f_boundary + r_boundary + 1.0
        )
        direction_score = 0.55 * ((trace_margin + 1.0) / 2.0) + 0.45 * (
            (boundary_margin + 1.0) / 2.0
        )

        method_match = _mean(
            (item["method_match"] for item in metrics), default=0.5
        )
        route_exact = _mean(
            (item["route_exact"] for item in metrics), default=0.0
        )
        route_jaccard = _mean(
            (item["route_jaccard"] for item in metrics), default=0.0
        )
        operation_jaccard = _mean(
            (item["operation_jaccard"] for item in metrics), default=0.0
        )
        span_kind_score = _mean(
            (item["span_kind_compatible"] for item in metrics), default=0.5
        )
        workload_score = _mean(
            (item["workload_match"] for item in metrics), default=0.5
        )
        method_coverage = (
            sum(bool(item["method_known"]) for item in metrics) / len(metrics)
            if metrics
            else 0.0
        )
        route_coverage = (
            sum(bool(item["route_known"]) for item in metrics) / len(metrics)
            if metrics
            else 0.0
        )
        span_kind_coverage = (
            sum(bool(item["span_kind_known"]) for item in metrics) / len(metrics)
            if metrics
            else 0.0
        )
        workload_coverage = (
            sum(bool(item["workload_known"]) for item in metrics) / len(metrics)
            if metrics
            else 0.0
        )
        endpoint_components: list[tuple[float, float]] = [
            (operation_jaccard, 0.35),
            (pair_concentration, 0.15),
        ]
        if method_coverage > 0:
            endpoint_components.append((method_match, 0.20))
        if route_coverage > 0:
            endpoint_components.extend(
                ((route_jaccard, 0.20), (route_exact, 0.10))
            )
        weight_total = sum(weight for _value, weight in endpoint_components)
        endpoint_score = (
            sum(value * weight for value, weight in endpoint_components)
            / weight_total
        )

        alignment = (
            min(1.0, len(metrics) / int(record["boundary_spans"]))
            if int(record["boundary_spans"]) > 0
            else 1.0
        )
        row = {
            **{name: record[name] for name in MODEL_COLUMNS},
            "evidence_id_count": len(record["evidence_ids"]),
            "reconstructed_boundary_pairs": len(metrics),
            "boundary_alignment": alignment,
            "trace_direction_margin": trace_margin,
            "boundary_direction_margin": boundary_margin,
            "direction_score": direction_score,
            "source_out_degree": source_role["out_degree"],
            "source_in_degree": source_role["in_degree"],
            "target_out_degree": target_role["out_degree"],
            "target_in_degree": target_role["in_degree"],
            "graph_role_score": graph_role_score,
            "source_operation_parent_prior": source_parent_prior,
            "source_operation_child_prior": source_child_prior,
            "target_operation_parent_prior": target_parent_prior,
            "target_operation_child_prior": target_child_prior,
            "operation_role_score": operation_role_score,
            "operation_pair_concentration": pair_concentration,
            "method_coverage": method_coverage,
            "method_match_rate": method_match,
            "route_coverage": route_coverage,
            "route_exact_rate": route_exact,
            "route_jaccard_mean": route_jaccard,
            "operation_jaccard_mean": operation_jaccard,
            "endpoint_compatibility_score": endpoint_score,
            "span_kind_coverage": span_kind_coverage,
            "span_kind_compatibility_score": span_kind_score,
            "workload_coverage": workload_coverage,
            "workload_match_score": workload_score,
        }
        rows.append(row)

    diagnostics = {
        "candidate_count": len(records),
        "selected_span_count": len(selected_spans),
        "reconstructed_boundary_pairs": pair_count_total,
        "expected_boundary_spans": expected_boundary_total,
        "boundary_alignment_macro": (
            statistics.fmean(float(row["boundary_alignment"]) for row in rows)
            if rows
            else None
        ),
        "operation_pair_candidate_coverage": (
            sum(int(row["reconstructed_boundary_pairs"]) > 0 for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "method_pair_coverage": (
            method_known_total / pair_count_total if pair_count_total else 0.0
        ),
        "route_pair_coverage": (
            route_known_total / pair_count_total if pair_count_total else 0.0
        ),
        "span_kind_pair_coverage": (
            span_kind_known_total / pair_count_total if pair_count_total else 0.0
        ),
        "workload_pair_coverage": (
            workload_known_total / pair_count_total if pair_count_total else 0.0
        ),
    }
    return pd.DataFrame.from_records(rows), diagnostics


def add_profile_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["direction_role_raw"] = (
        0.60 * output["direction_score"]
        + 0.25 * output["graph_role_score"]
        + 0.15 * output["operation_role_score"]
    )
    output["operation_endpoint_raw"] = (
        0.55 * output["endpoint_compatibility_score"]
        + 0.30 * output["operation_role_score"]
        + 0.15 * output["operation_pair_concentration"]
    )
    output["method_route_raw"] = (
        0.30 * output["method_match_rate"]
        + 0.25 * output["route_jaccard_mean"]
        + 0.15 * output["route_exact_rate"]
        + 0.20 * output["operation_jaccard_mean"]
        + 0.10 * output["operation_pair_concentration"]
    )
    direct_bonus = (
        0.05
        * output["span_kind_coverage"]
        * output["span_kind_compatibility_score"]
        + 0.05
        * output["workload_coverage"]
        * output["workload_match_score"]
    )
    output["combined_raw"] = (
        0.30 * output["direction_score"]
        + 0.15 * output["graph_role_score"]
        + 0.20 * output["operation_role_score"]
        + 0.30 * output["endpoint_compatibility_score"]
        + direct_bonus
    )
    output["a2_rank_normalized"] = _rank_normalize(output, "a2_score")
    for profile in (
        "direction_role",
        "operation_endpoint",
        "method_route",
        "combined",
    ):
        output[f"{profile}_rank_normalized"] = _rank_normalize(
            output, f"{profile}_raw"
        )
    return output


def apply_policy(frame: pd.DataFrame, policy: OperationalPolicy) -> pd.DataFrame:
    scored = frame.copy()
    evidence_rank = scored[f"{policy.profile}_rank_normalized"].astype(float)
    scored["operational_evidence_rank"] = evidence_rank
    scored["a3_r2_score"] = (
        (1.0 - policy.evidence_weight)
        * scored["a2_rank_normalized"].astype(float)
        + policy.evidence_weight * evidence_rank
    )
    keep = min(
        len(scored),
        max(
            policy.minimum_keep,
            int(math.ceil(policy.retention_fraction * len(scored))),
        ),
    )
    scored["selected"] = False
    direct = set(scored.index[scored["direct_evidence"].map(_truthy)])
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        ["a3_r2_score", "a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    selected = set(direct)
    for index in ranked.index:
        if len(selected) >= keep:
            break
        selected.add(index)
    scored.loc[list(selected), "selected"] = True
    return scored


def _a2_control(frame: pd.DataFrame, selected_count: int) -> pd.DataFrame:
    scored = frame.copy()
    scored["a3_r2_score"] = scored["a2_rank_normalized"].astype(float)
    scored["selected"] = False
    direct = set(scored.index[scored["direct_evidence"].map(_truthy)])
    ranked = scored.loc[~scored.index.isin(direct)].sort_values(
        ["a2_score", "proposal_rank", "subject", "object"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    selected = set(direct)
    for index in ranked.index:
        if len(selected) >= selected_count:
            break
        selected.add(index)
    scored.loc[list(selected), "selected"] = True
    return scored


def evaluate_cell(scored: pd.DataFrame) -> dict[str, Any]:
    selected = scored.loc[scored["selected"].map(_truthy)]
    target_keys = {
        tuple(map(str, row))
        for row in scored.loc[
            scored["is_masked_target"].map(_truthy), list(CANDIDATE_KEY)
        ].itertuples(index=False, name=None)
    }
    silver_keys = {
        tuple(map(str, row))
        for row in scored.loc[
            scored["is_silver_matched"].map(_truthy), list(CANDIDATE_KEY)
        ].itertuples(index=False, name=None)
    }
    selected_keys = {
        tuple(map(str, row))
        for row in selected[list(CANDIDATE_KEY)].itertuples(
            index=False, name=None
        )
    }
    recovered = selected_keys & target_keys
    by_query: dict[tuple[str, str], list[Any]] = {}
    by_key: dict[tuple[str, str, str], Any] = {}
    for row in selected.itertuples(index=False):
        key = (str(row.subject), str(row.predicate), str(row.object))
        by_key[key] = row
        by_query.setdefault((key[0], key[1]), []).append(row)
    reciprocal: list[float] = []
    ranks: list[int | None] = []
    epsilon = 1e-12
    for target in sorted(target_keys):
        target_item = by_key.get(target)
        if target_item is None:
            reciprocal.append(0.0)
            ranks.append(None)
            continue
        target_score = float(target_item.a3_r2_score)
        competitors = []
        for item in by_query.get((target[0], target[1]), ()):
            key = (str(item.subject), str(item.predicate), str(item.object))
            if key == target or key in silver_keys:
                continue
            competitors.append(item)
        higher = sum(
            float(item.a3_r2_score) > target_score + epsilon
            for item in competitors
        )
        tied = sum(
            abs(float(item.a3_r2_score) - target_score) <= epsilon
            for item in competitors
        )
        rank = 1 + higher + tied
        reciprocal.append(1.0 / rank)
        ranks.append(rank)
    return {
        "selected_count": len(selected_keys),
        "target_count": len(target_keys),
        "recovered_target_count": len(recovered),
        "recall": len(recovered) / len(target_keys) if target_keys else None,
        "silver_matched_count": len(selected_keys & silver_keys),
        "silver_precision_lower_bound": (
            len(selected_keys & silver_keys) / len(selected_keys)
            if selected_keys
            else None
        ),
        "mrr": statistics.fmean(reciprocal) if reciprocal else None,
        "ranks": json.dumps(ranks, separators=(",", ":")),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    recalls = [
        float(row["recall"]) for row in rows if row["recall"] is not None
    ]
    p_lbs = [
        float(row["silver_precision_lower_bound"])
        for row in rows
        if row["silver_precision_lower_bound"] is not None
    ]
    mrrs = [float(row["mrr"]) for row in rows if row["mrr"] is not None]
    counts = [int(row["selected_count"]) for row in rows]
    targets = sum(int(row["target_count"]) for row in rows)
    recovered = sum(int(row["recovered_target_count"]) for row in rows)
    return {
        "cell_count": len(rows),
        "recall_macro": statistics.fmean(recalls),
        "recall_min": min(recalls),
        "recall_pooled": recovered / targets if targets else None,
        "selected_count_mean": statistics.fmean(counts),
        "selected_count_median": statistics.median(counts),
        "selected_count_max": max(counts),
        "silver_precision_lower_bound_macro": statistics.fmean(p_lbs),
        "silver_precision_lower_bound_min": min(p_lbs),
        "mrr_macro": statistics.fmean(mrrs),
        "mrr_min": min(mrrs),
    }


def _delta(
    enhanced: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "recall_macro": float(enhanced["recall_macro"])
        - float(baseline["recall_macro"]),
        "selected_count_mean": float(enhanced["selected_count_mean"])
        - float(baseline["selected_count_mean"]),
        "silver_precision_lower_bound_macro": float(
            enhanced["silver_precision_lower_bound_macro"]
        )
        - float(baseline["silver_precision_lower_bound_macro"]),
        "mrr_macro": float(enhanced["mrr_macro"])
        - float(baseline["mrr_macro"]),
    }


def _evaluate_policy(
    cells: Mapping[tuple[str, int, str], pd.DataFrame],
    policy: OperationalPolicy,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for key, group in sorted(cells.items()):
        scored = apply_policy(group.reset_index(drop=True), policy)
        metric = evaluate_cell(scored)
        control = _a2_control(
            group.reset_index(drop=True), int(metric["selected_count"])
        )
        control_metric = evaluate_cell(control)
        first = group.iloc[0]
        common = {
            "case": key[0],
            "fault": str(first["fault"]),
            "role": str(first["role"]),
            "seed": int(key[1]),
            "mask_id": str(key[2]),
            "mask_ratio": float(first["mask_ratio"]),
        }
        rows.append({**common, **metric})
        control_rows.append({**common, **control_metric})
        scored = scored.copy()
        for name, value in common.items():
            scored[name] = value
        scored_frames.append(scored)
    return (
        pd.DataFrame.from_records(rows),
        _aggregate(rows),
        pd.DataFrame.from_records(control_rows),
        (
            pd.concat(scored_frames, ignore_index=True)
            if scored_frames
            else pd.DataFrame()
        ),
    )


def _baseline(
    cells: Mapping[tuple[str, int, str], pd.DataFrame]
) -> dict[str, Any]:
    rows = []
    for group in cells.values():
        pseudo = group.copy()
        pseudo["selected"] = True
        pseudo["a3_r2_score"] = pseudo["a2_rank_normalized"].astype(float)
        rows.append(evaluate_cell(pseudo))
    return _aggregate(rows)


def _policy_grid(config: Mapping[str, Any]) -> list[OperationalPolicy]:
    search = config["policy_search"]
    return [
        OperationalPolicy(
            str(profile), float(retention), int(minimum_keep), float(weight)
        )
        for profile in search["profiles"]
        for retention in search["retention_fractions"]
        for minimum_keep in search["minimum_keep"]
        for weight in search["evidence_weights"]
    ]


def _select_policy(
    cells: Mapping[tuple[str, int, str], pd.DataFrame],
    config: Mapping[str, Any],
    gate: GateConfig,
) -> tuple[OperationalPolicy, pd.DataFrame, bool]:
    baseline = _baseline(cells)
    rows: list[dict[str, Any]] = []
    for policy in _policy_grid(config):
        _cell_rows, aggregate, control_rows, _ = _evaluate_policy(cells, policy)
        control = _aggregate(control_rows.to_dict(orient="records"))
        delta_control = _delta(aggregate, control)
        selected_ratio = (
            aggregate["selected_count_mean"] / baseline["selected_count_mean"]
        )
        additive_gain = max(
            delta_control["silver_precision_lower_bound_macro"],
            delta_control["mrr_macro"],
        )
        conditions = {
            "recall_macro": aggregate["recall_macro"] >= gate.recall_macro_min,
            "recall_each_cell": aggregate["recall_min"]
            >= gate.recall_each_cell_min,
            "candidate_count_reduced": selected_ratio
            <= gate.selected_count_ratio_max,
            "mrr_noninferior_to_full_a2": aggregate["mrr_macro"]
            >= baseline["mrr_macro"] - 0.01,
            "matched_budget_recall": delta_control["recall_macro"]
            >= -gate.matched_budget_recall_tolerance,
            "matched_budget_p_lb": delta_control[
                "silver_precision_lower_bound_macro"
            ]
            >= gate.matched_budget_p_lb_delta_min,
            "matched_budget_mrr": delta_control["mrr_macro"]
            >= gate.matched_budget_mrr_delta_min,
            "matched_budget_additive_gain": additive_gain
            >= gate.additive_gain_min,
        }
        rows.append(
            {
                **asdict(policy),
                **aggregate,
                "baseline_mrr_macro": baseline["mrr_macro"],
                "selected_count_ratio": selected_ratio,
                "control_recall_macro": control["recall_macro"],
                "control_p_lb_macro": control[
                    "silver_precision_lower_bound_macro"
                ],
                "control_mrr_macro": control["mrr_macro"],
                "matched_budget_recall_delta": delta_control["recall_macro"],
                "matched_budget_p_lb_delta": delta_control[
                    "silver_precision_lower_bound_macro"
                ],
                "matched_budget_mrr_delta": delta_control["mrr_macro"],
                "matched_budget_additive_gain": additive_gain,
                "feasible": all(conditions.values()),
                "violation_count": sum(
                    not value for value in conditions.values()
                ),
            }
        )
    grid = pd.DataFrame.from_records(rows)
    feasible = grid.loc[grid["feasible"].map(_truthy)]
    pool = feasible if not feasible.empty else grid
    chosen = pool.sort_values(
        [
            "violation_count" if feasible.empty else "selected_count_mean",
            "matched_budget_additive_gain",
            "silver_precision_lower_bound_macro",
            "mrr_macro",
            "evidence_weight",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    policy = OperationalPolicy(
        str(chosen.profile),
        float(chosen.retention_fraction),
        int(chosen.minimum_keep),
        float(chosen.evidence_weight),
    )
    grid["selected"] = (
        grid["profile"].astype(str).eq(policy.profile)
        & grid["retention_fraction"].astype(float).eq(
            policy.retention_fraction
        )
        & grid["minimum_keep"].astype(int).eq(policy.minimum_keep)
        & grid["evidence_weight"].astype(float).eq(policy.evidence_weight)
    )
    return policy, grid, not feasible.empty


def _cell_root(phase2_root: Path, row: Mapping[str, Any]) -> Path:
    summary = phase2_root / str(row["run_summary"])
    return summary.parent / "masks" / str(row["mask_id"])


def _evaluator_flags(
    cell_root: Path, candidate_keys: set[tuple[str, str, str]]
) -> pd.DataFrame:
    manifest = json.loads(
        (cell_root / "evaluator_private" / "mask_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    evaluation = json.loads(
        (cell_root / "evaluator_private" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        (str(item["subject"]), str(item["predicate"]), str(item["object"]))
        for item in manifest["target_edges"]
    }
    unverified = {
        (str(item["subject"]), str(item["predicate"]), str(item["object"]))
        for item in evaluation["A2"]["silver_precision_lower_bound"].get(
            "unverified_edges", ()
        )
    }
    silver = candidate_keys - unverified
    return pd.DataFrame.from_records(
        [
            {
                "subject": key[0],
                "predicate": key[1],
                "object": key[2],
                "is_masked_target": key in targets,
                "is_silver_matched": key in silver,
            }
            for key in sorted(candidate_keys)
        ]
    )


def _process_cell(
    phase2_root: Path, row: Mapping[str, Any], role: str
) -> tuple[tuple[str, int, str], pd.DataFrame, dict[str, Any]]:
    cell_root = _cell_root(phase2_root, row)
    records = _parse_a2(cell_root / "predictions" / "A2.parquet")
    trace_path = cell_root / "model_input" / "traces.parquet"
    edge_path = cell_root / "model_input" / "observed_edges.parquet"
    if not trace_path.is_file() or not edge_path.is_file():
        raise Phase3R2Error(
            f"cell lacks heavy trace/edge artifacts: {cell_root}"
        )
    traces, availability = _canonical_trace_frame(pd.read_parquet(trace_path))
    observed_edges = pd.read_parquet(edge_path)
    model_features, diagnostics = _candidate_feature_rows(
        records, traces, observed_edges
    )
    model_features = add_profile_scores(model_features)

    # Evaluator-only answers are loaded after model features are frozen.
    candidate_keys = {
        (str(item.subject), str(item.predicate), str(item.object))
        for item in model_features.itertuples(index=False)
    }
    flags = _evaluator_flags(cell_root, candidate_keys)
    frame = model_features.merge(
        flags, on=list(CANDIDATE_KEY), validate="one_to_one"
    )
    frame["case"] = str(row["case"])
    frame["fault"] = str(row["fault"])
    frame["role"] = role
    frame["seed"] = int(row["seed"])
    frame["mask_id"] = str(row["mask_id"])
    frame["mask_ratio"] = float(row["mask_ratio"])
    diagnostics.update(
        {
            "case": str(row["case"]),
            "seed": int(row["seed"]),
            "mask_id": str(row["mask_id"]),
            "mask_ratio": float(row["mask_ratio"]),
            "trace_rows": len(traces),
            "field_availability": availability,
        }
    )
    return (
        (str(row["case"]), int(row["seed"]), str(row["mask_id"])),
        frame,
        diagnostics,
    )


def _render_report(summary: Mapping[str, Any]) -> str:
    heldout = summary["heldout"]
    base = heldout["baseline_a2_full"]
    proposed = heldout["proposed_a3_r2"]
    delta_full = heldout["delta_vs_full_a2"]
    delta_control = heldout["delta_vs_equal_size_a2"]
    reasons = ", ".join(summary["gate"]["reason_codes"]) or "없음"
    return f"""# Task A Phase 3-R2 결과 — Operation·HTTP·Role Evidence

- 최종 Gate: **{summary['status']}**
- Calibration feasible 정책: **{summary['calibration']['feasible_policy_count']} / {summary['calibration']['searched_policy_count']}**
- 선택 정책: `{summary['selected_policy']}`
- 미통과 조건: `{reasons}`

## Held-out 40 Cell

| 지표 | A2 전체 | A3-R2 | 변화 | 동일 크기 A2 대비 |
|---|---:|---:|---:|---:|
| Recall Macro | {base['recall_macro']:.4f} | {proposed['recall_macro']:.4f} | {delta_full['recall_macro']:+.4f} | {delta_control['recall_macro']:+.4f} |
| Recall Minimum | {base['recall_min']:.4f} | {proposed['recall_min']:.4f} | - | - |
| 후보 수 평균 | {base['selected_count_mean']:.3f} | {proposed['selected_count_mean']:.3f} | {delta_full['selected_count_mean']:+.3f} | {delta_control['selected_count_mean']:+.3f} |
| P-LB Macro | {base['silver_precision_lower_bound_macro']:.4f} | {proposed['silver_precision_lower_bound_macro']:.4f} | {delta_full['silver_precision_lower_bound_macro']:+.4f} | {delta_control['silver_precision_lower_bound_macro']:+.4f} |
| MRR Macro | {base['mrr_macro']:.4f} | {proposed['mrr_macro']:.4f} | {delta_full['mrr_macro']:+.4f} | {delta_control['mrr_macro']:+.4f} |

## 데이터 가용성

```json
{json.dumps(summary['feature_diagnostics'], ensure_ascii=False, indent=2, sort_keys=True)}
```

## 해석

- 이 단계는 일반 NLI 문장보다 먼저, 실제 Trace의 Operation·Method·Route·Role Evidence가 후보별 판별력을 만드는지 검증한다.
- `span.kind`, `http.route`, workload가 원본에 없으면 Proxy 사용 사실과 Coverage를 결과에 명시한다.
- PASS는 동일 후보 수의 A2-only보다 P-LB 또는 MRR이 실제로 개선됐다는 뜻이다.
- `CALLS`는 runtime 구조 관계이며 causal `CAUSES`를 의미하지 않는다.
"""


def run_phase3_r2(
    *,
    phase2_root: Path,
    output: Path,
    config_path: Path,
    max_workers: int = 2,
) -> Path:
    phase2_root = phase2_root.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if output.exists():
        raise Phase3R2Error(f"refusing to overwrite output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != (
        "rcaeval-task-a-phase3-r2-operational-evidence"
    ):
        raise Phase3R2Error("unexpected experiment_id")
    phase2_summary = json.loads(
        (phase2_root / "summary.json").read_text(encoding="utf-8")
    )
    if not phase2_summary.get("gate", {}).get("passed"):
        raise Phase3R2Error("Phase-2 D3 gate must pass before R2")
    cells_frame = pd.read_csv(phase2_root / "cells.csv")
    source_contract = config["source_contract"]
    if len(cells_frame) != int(source_contract["required_cells"]):
        raise Phase3R2Error("unexpected Phase-2 cell count")
    if cells_frame["case"].nunique() != int(
        source_contract["required_incidents"]
    ):
        raise Phase3R2Error("unexpected Phase-2 incident count")
    expected_candidates = int(source_contract["required_candidate_rows"])
    observed_candidates = int(
        pd.to_numeric(
            cells_frame["a2_proposal_count"], errors="raise"
        ).sum()
    )
    if observed_candidates != expected_candidates:
        raise Phase3R2Error(
            f"unexpected A2 candidate count: expected={expected_candidates} "
            f"observed={observed_candidates}"
        )

    calibration_cases, heldout_cases, case_hashes = stable_case_split(
        cells_frame["case"].astype(str),
        revision=str(config["dataset_revision"]),
        calibration_incidents=int(
            config["split_contract"]["calibration_incidents"]
        ),
    )
    role_by_case = {case: "calibration" for case in calibration_cases}
    role_by_case.update({case: "heldout" for case in heldout_cases})

    cells: dict[tuple[str, int, str], pd.DataFrame] = {}
    diagnostics: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_cell,
                phase2_root,
                row,
                role_by_case[str(row["case"])],
            ): row
            for row in cells_frame.to_dict(orient="records")
        }
        for index, future in enumerate(as_completed(futures), start=1):
            key, frame, diagnostic = future.result()
            cells[key] = frame
            diagnostics.append(diagnostic)
            print(
                f"[{index}/{len(futures)}] {key[0]} seed={key[1]} "
                f"mask={key[2]} operational-evidence=READY",
                flush=True,
            )

    calibration = {
        key: frame
        for key, frame in cells.items()
        if str(frame.iloc[0]["role"]) == "calibration"
    }
    heldout = {
        key: frame
        for key, frame in cells.items()
        if str(frame.iloc[0]["role"]) == "heldout"
    }
    split = config["split_contract"]
    if len(calibration) != int(split["calibration_cells"]) or len(
        heldout
    ) != int(split["heldout_cells"]):
        raise Phase3R2Error("calibration/heldout cell contract mismatch")

    gate = GateConfig(**dict(config["gate"]))
    policy, grid, calibration_feasible = _select_policy(
        calibration, config, gate
    )
    (
        calibration_rows,
        calibration_aggregate,
        calibration_control_rows,
        _,
    ) = _evaluate_policy(calibration, policy)
    (
        heldout_rows,
        heldout_aggregate,
        heldout_control_rows,
        heldout_scored,
    ) = _evaluate_policy(heldout, policy)
    baseline_calibration = _baseline(calibration)
    baseline_heldout = _baseline(heldout)
    control_heldout = _aggregate(
        heldout_control_rows.to_dict(orient="records")
    )
    delta_full = _delta(heldout_aggregate, baseline_heldout)
    delta_control = _delta(heldout_aggregate, control_heldout)
    selected_ratio = (
        heldout_aggregate["selected_count_mean"]
        / baseline_heldout["selected_count_mean"]
    )
    additive_gain = max(
        delta_control["silver_precision_lower_bound_macro"],
        delta_control["mrr_macro"],
    )

    diagnostic_frame = pd.DataFrame.from_records(diagnostics)
    feature_diagnostics = {
        "boundary_alignment_macro": float(
            diagnostic_frame["boundary_alignment_macro"].mean()
        ),
        "boundary_alignment_min": float(
            diagnostic_frame["boundary_alignment_macro"].min()
        ),
        "operation_pair_candidate_coverage_macro": float(
            diagnostic_frame["operation_pair_candidate_coverage"].mean()
        ),
        "method_pair_coverage_macro": float(
            diagnostic_frame["method_pair_coverage"].mean()
        ),
        "route_pair_coverage_macro": float(
            diagnostic_frame["route_pair_coverage"].mean()
        ),
        "span_kind_pair_coverage_macro": float(
            diagnostic_frame["span_kind_pair_coverage"].mean()
        ),
        "workload_pair_coverage_macro": float(
            diagnostic_frame["workload_pair_coverage"].mean()
        ),
        "source_schema_availability": {
            key: bool(
                any(
                    item["field_availability"].get(key, False)
                    for item in diagnostics
                )
            )
            for key in (
                "operation_name",
                "method_name",
                "span_kind",
                "http_method",
                "http_route",
                "source_workload",
                "destination_workload",
            )
        },
    }
    conditions = {
        "calibration_policy_feasible": calibration_feasible,
        "heldout_complete": len(heldout_rows) == int(split["heldout_cells"]),
        "recall_macro": heldout_aggregate["recall_macro"]
        >= gate.recall_macro_min,
        "recall_pooled": heldout_aggregate["recall_pooled"]
        >= gate.recall_pooled_min,
        "recall_each_cell": heldout_aggregate["recall_min"]
        >= gate.recall_each_cell_min,
        "candidate_count_reduced": selected_ratio
        <= gate.selected_count_ratio_max,
        "p_lb_improved_vs_full_a2": delta_full[
            "silver_precision_lower_bound_macro"
        ]
        >= gate.p_lb_delta_vs_full_min,
        "mrr_improved_vs_full_a2": delta_full["mrr_macro"]
        >= gate.mrr_delta_vs_full_min,
        "matched_budget_recall_noninferior": delta_control["recall_macro"]
        >= -gate.matched_budget_recall_tolerance,
        "matched_budget_p_lb_noninferior": delta_control[
            "silver_precision_lower_bound_macro"
        ]
        >= gate.matched_budget_p_lb_delta_min,
        "matched_budget_mrr_noninferior": delta_control["mrr_macro"]
        >= gate.matched_budget_mrr_delta_min,
        "matched_budget_additive_gain": additive_gain
        >= gate.additive_gain_min,
        "operational_weight_active": policy.evidence_weight > 0.0,
        "boundary_alignment": feature_diagnostics["boundary_alignment_macro"]
        >= gate.boundary_alignment_macro_min,
        "operation_pair_coverage": feature_diagnostics[
            "operation_pair_candidate_coverage_macro"
        ]
        >= gate.operation_pair_coverage_min,
        "a2_candidates_preserved": sum(len(frame) for frame in cells.values())
        == int(cells_frame["a2_proposal_count"].sum()),
    }
    passed = all(conditions.values())

    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "model_output"
    evaluator_dir = output / "evaluator_private"
    published = output / "published"
    model_dir.mkdir()
    evaluator_dir.mkdir()
    published.mkdir()
    model_frames = []
    for frame in cells.values():
        model_frames.append(
            frame.drop(
                columns=[
                    column
                    for column in EVALUATOR_COLUMNS | {"case"}
                    if column in frame.columns
                ]
            )
        )
    pd.concat(model_frames, ignore_index=True).to_parquet(
        model_dir / "a3_r2_operational_features.parquet", index=False
    )
    heldout_scored.to_parquet(
        evaluator_dir / "heldout_candidate_analysis.parquet", index=False
    )
    diagnostic_frame.to_json(
        evaluator_dir / "feature_diagnostics.json",
        orient="records",
        indent=2,
    )
    heldout_control_rows.to_csv(
        evaluator_dir / "a2_equal_size_control_cells.csv", index=False
    )
    calibration_rows.to_csv(
        published / "task_a_phase3_r2_calibration_cells.csv", index=False
    )
    heldout_rows.to_csv(
        published / "task_a_phase3_r2_heldout_cells.csv", index=False
    )
    grid.to_csv(
        published / "task_a_phase3_r2_policy_grid.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "PASS" if passed else "FAIL",
        "gate_id": "D4B_A3_R2_OPERATIONAL_EVIDENCE_UTILITY",
        "source": {
            "phase2_root": str(phase2_root),
            "phase2_cells_sha256": hashlib.sha256(
                (phase2_root / "cells.csv").read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "candidate_rows": sum(len(frame) for frame in cells.values()),
            "candidate_cells": len(cells),
        },
        "split": {
            "calibration_cases": list(calibration_cases),
            "heldout_cases": list(heldout_cases),
            "case_hashes": case_hashes,
            "calibration_cells": len(calibration),
            "heldout_cells": len(heldout),
        },
        "protocol_status": dict(config.get("protocol_status", {})),
        "leakage_boundary": {
            "feature_inputs": (
                "sanitized model traces, observed graph, frozen A2 evidence only"
            ),
            "evaluator_labels_loaded_after_feature_freeze": True,
            "fault_and_root_labels_used_for_scoring": False,
            "policy_selection": "calibration incidents only",
            "heldout_labels_used_for_selection": False,
        },
        "selected_policy": asdict(policy),
        "feature_diagnostics": feature_diagnostics,
        "evidence_contract": dict(config.get("evidence_contract", {})),
        "calibration": {
            "feasible": calibration_feasible,
            "searched_policy_count": len(grid),
            "feasible_policy_count": int(
                grid["feasible"].map(_truthy).sum()
            ),
            "baseline_a2_full": baseline_calibration,
            "proposed_a3_r2": calibration_aggregate,
            "equal_size_a2_control": _aggregate(
                calibration_control_rows.to_dict(orient="records")
            ),
        },
        "heldout": {
            "baseline_a2_full": baseline_heldout,
            "proposed_a3_r2": heldout_aggregate,
            "equal_size_a2_control": control_heldout,
            "delta_vs_full_a2": delta_full,
            "delta_vs_equal_size_a2": delta_control,
            "selected_count_ratio": selected_ratio,
            "additive_gain": additive_gain,
        },
        "gate": {
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "conditions": conditions,
            "reason_codes": [
                name.upper() for name, value in conditions.items() if not value
            ],
            "required": asdict(gate),
        },
        "claim_limit": (
            "Operation/HTTP/role-assisted CALLS candidate shortlisting on six "
            "RCAEval TrainTicket incidents; no causal-edge, RCA, LLM, DeBERTa, "
            "or production-generalization claim."
        ),
    }
    (published / "task_a_phase3_r2_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (published / "task_a_phase3_r2_results.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    (published / "task_a_phase3_r2_status.txt").write_text(
        summary["status"] + "\n", encoding="utf-8"
    )
    return output


__all__ = [
    "GateConfig",
    "OperationalPolicy",
    "Phase3R2Error",
    "add_profile_scores",
    "apply_policy",
    "evaluate_cell",
    "run_phase3_r2",
    "_candidate_feature_rows",
    "_canonical_trace_frame",
    "_http_method",
    "_normalize_route",
    "_operation_tokens",
]
