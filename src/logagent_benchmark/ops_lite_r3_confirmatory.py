"""Independent RCABench OPS-Lite confirmation of a frozen Task A R3 policy.

The experiment applies the checksum-locked RCAEval R3 policy without tuning to
preselected cases from the official RCABench OPS-Lite test split. Model-side
features are computed only from sanitized trace partitions, observed CALLS
edges, and bounded A2 proposals. Mask targets and silver labels are joined
only after all operational and DeBERTa scores are frozen.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

import pandas as pd

from .graph import (
    CANONICAL_TRACE_COLUMNS,
    build_heldout_silver_graph,
    canonical_service_id,
    edge_key_set,
)
from .masking import make_iid_parent_dropped_mask
from .onnx_deberta import OnnxDebertaNLIBackend
from .recovery import DEFAULT_RELATION_SPECS, InferenceContext
from .task_a import TaskAConfig, run_task_a_candidate_suite
from . import task_a_phase3_r2 as r2
from . import task_a_phase3_r3_channel_v2 as r3


CANDIDATE_KEY = ("subject", "predicate", "object")
CELL_KEY = ("incident_token", "seed", "mask_id")


class OpsLiteConfirmatoryError(RuntimeError):
    """Raised when the frozen external-confirmation contract is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OpsLiteConfirmatoryError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_incident_token(revision: str, case_id: str) -> str:
    return hashlib.sha256(
        f"{revision}|ops-lite-confirmatory|{case_id}".encode("utf-8")
    ).hexdigest()[:20]


def _prefix_nullable_id(series: pd.Series, prefix: str) -> pd.Series:
    values = series.astype("string")
    normalized = values.str.strip()
    missing = values.isna() | normalized.fillna("").str.lower().isin(
        {"", "<na>", "nan", "none", "null"}
    )
    output = (prefix + normalized.fillna("")).astype("string")
    output.loc[missing] = pd.NA
    return output


def canonicalize_trace_frame(
    raw: pd.DataFrame,
    *,
    phase: str,
    dataset_id: str,
    system_id: str,
) -> pd.DataFrame:
    """Map one native OPS-Lite trace table to the frozen canonical schema."""

    required = {
        "time",
        "trace_id",
        "span_id",
        "parent_span_id",
        "span_name",
        "service_name",
        "duration",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise OpsLiteConfirmatoryError(
            f"OPS-Lite trace table is missing columns: {missing}"
        )
    prefix = f"{phase}:"
    timestamps = pd.to_datetime(raw["time"], utc=True, errors="raise")
    start_us = timestamps.astype("int64") // 1_000
    duration_ns = pd.to_numeric(raw["duration"], errors="raise")
    if duration_ns.isna().any() or (duration_ns < 0).any():
        raise OpsLiteConfirmatoryError("trace duration must be non-negative")
    duration_us = (duration_ns // 1_000).astype("int64")

    output = pd.DataFrame(
        {
            "trace_id": prefix + raw["trace_id"].astype(str),
            "span_id": prefix + raw["span_id"].astype(str),
            "parent_span_id": _prefix_nullable_id(
                raw["parent_span_id"], prefix
            ),
            "service_id": [
                canonical_service_id(
                    value,
                    dataset_id=dataset_id,
                    system_id=system_id,
                )
                for value in raw["service_name"]
            ],
            "operation_name": raw["span_name"].astype("string").fillna(""),
            "method_name": (
                raw["attr.http.request.method"].astype("string").fillna("")
                if "attr.http.request.method" in raw.columns
                else pd.Series("", index=raw.index, dtype="string")
            ),
            "start_time_us": start_us.astype("int64"),
            "duration_us": duration_us,
        }
    )
    output["end_time_us"] = (
        output["start_time_us"] + output["duration_us"]
    )
    output["span_kind"] = (
        raw["attr.span_kind"].astype("string").fillna("")
        if "attr.span_kind" in raw.columns
        else pd.Series("", index=raw.index, dtype="string")
    )
    output["http_method"] = output["method_name"].astype("string")
    output["http_route"] = pd.Series("", index=raw.index, dtype="string")
    output["source_workload"] = pd.Series(
        "", index=raw.index, dtype="string"
    )
    output["destination_workload"] = pd.Series(
        "", index=raw.index, dtype="string"
    )
    if output.duplicated(["trace_id", "span_id"]).any():
        raise OpsLiteConfirmatoryError(
            f"duplicate trace/span identifiers after {phase} canonicalization"
        )
    return output


def load_case_traces(
    data_root: Path,
    *,
    case_id: str,
    dataset_id: str,
    system_id: str,
) -> pd.DataFrame:
    case_root = data_root / "cases" / case_id
    frames = []
    for phase, filename in (
        ("normal", "normal_traces.parquet"),
        ("abnormal", "abnormal_traces.parquet"),
    ):
        path = case_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing OPS-Lite trace source: {path}")
        frames.append(
            canonicalize_trace_frame(
                pd.read_parquet(path),
                phase=phase,
                dataset_id=dataset_id,
                system_id=system_id,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["trace_id", "span_id"]).any():
        raise OpsLiteConfirmatoryError(
            f"combined trace identifiers repeat for case {case_id}"
        )
    return combined


def _model_entities(traces: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "entity_id": service_id,
            "canonical_name": service_id.rsplit(":", 1)[-1],
            "entity_type": "Service",
            "type_basis": "sanitized_ops_lite_trace_partition",
            "type_confidence": 1.0,
        }
        for service_id in sorted(traces["service_id"].astype(str).unique())
    )


def _a2_records(result: Any) -> list[dict[str, Any]]:
    records = []
    for prediction in result.predictions:
        if prediction.decision != "accepted":
            continue
        stage = dict(prediction.stage_scores)
        reasons = {str(value) for value in prediction.reason_codes}
        records.append(
            {
                "subject": str(prediction.subject),
                "predicate": str(prediction.predicate),
                "object": str(prediction.object),
                "a2_score": float(prediction.score),
                "proposal_rank": int(float(stage.get("proposal_rank", 0))),
                "supporting_traces": int(
                    float(stage.get("supporting_traces", 0))
                ),
                "boundary_spans": int(float(stage.get("boundary_spans", 0))),
                "direct_evidence": "DIRECT_EVIDENCE" in reasons,
                "evidence_ids": tuple(
                    str(value) for value in prediction.evidence_ids
                ),
            }
        )
    by_key = {
        (row["subject"], row["predicate"], row["object"]): row
        for row in records
    }
    for row in records:
        reverse = by_key.get(
            (row["object"], row["predicate"], row["subject"])
        )
        row["reverse_supporting_traces"] = (
            int(reverse["supporting_traces"]) if reverse else 0
        )
        row["reverse_boundary_spans"] = (
            int(reverse["boundary_spans"]) if reverse else 0
        )
    return records


def _a2_config(contract: Mapping[str, Any]) -> TaskAConfig:
    return TaskAConfig(
        a2_threshold=float(contract["a2_threshold"]),
        include_null_parent=bool(contract["include_null_parent"]),
        max_abductive_proposals=int(contract["max_abductive_proposals"]),
        max_per_subject=int(contract["max_per_subject"]),
        max_per_object=int(contract["max_per_object"]),
        min_supporting_traces=int(contract["min_supporting_traces"]),
        min_boundary_count=int(contract["min_boundary_count"]),
    )


def build_case_cells(
    *,
    traces: pd.DataFrame,
    case_id: str,
    system_label: str,
    revision: str,
    config: Mapping[str, Any],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    """Build model-only A2/R2 features and private evaluator labels per cell."""

    trace_contract = config["trace_contract"]
    dataset_id = str(trace_contract["dataset_id"])
    system_id = f"ops-lite-{system_label}"
    incident_token = _opaque_incident_token(revision, case_id)
    graph = build_heldout_silver_graph(
        traces,
        revision=revision,
        incident_id=incident_token,
        inject_time_us=None,
        dataset_id=dataset_id,
        system_id=system_id,
        reference_ratio=float(trace_contract["reference_ratio"]),
        columns=CANONICAL_TRACE_COLUMNS,
        service_ids_are_canonical=True,
    )
    reference_keys = edge_key_set(graph.reference_edges)
    observed_keys = edge_key_set(graph.observed_edges)
    attested = graph.reference_edges
    if "attestation" in attested.columns:
        attested = attested.loc[attested["attestation"].eq("A")]
    eligible = edge_key_set(attested).intersection(observed_keys)

    model_frames: list[pd.DataFrame] = []
    cell_records: list[dict[str, Any]] = []
    mask_contract = config["mask_contract"]
    relation_specs = {"CALLS": DEFAULT_RELATION_SPECS["CALLS"]}
    a2_config = _a2_config(config["a2_contract"])

    for seed in mask_contract["seeds"]:
        for ratio in mask_contract["ratios"]:
            mask_id = f"iid{int(round(float(ratio) * 100))}_l2_s{int(seed)}"
            common = {
                "case_id": case_id,
                "incident_token": incident_token,
                "system": system_label,
                "seed": int(seed),
                "mask_id": mask_id,
                "mask_ratio": float(ratio),
            }
            try:
                mask = make_iid_parent_dropped_mask(
                    graph,
                    fraction=float(ratio),
                    seed=int(seed),
                    dataset_id=dataset_id,
                    system_id=system_id,
                    columns=CANONICAL_TRACE_COLUMNS,
                    service_ids_are_canonical=True,
                )
                context = InferenceContext(
                    incident_id=incident_token,
                    entities=_model_entities(mask.model.traces),
                    observed_edges=mask.model.observed_edges,
                    traces=mask.model.traces,
                )
                suite = run_task_a_candidate_suite(
                    context,
                    config=a2_config,
                    relation_specs=relation_specs,
                )
                a2 = suite.results["A2"]
                if a2.status != "READY":
                    raise OpsLiteConfirmatoryError(
                        f"A2 stage is not READY: {a2.status} {a2.reason_code}"
                    )
                records = _a2_records(a2)
                candidate_keys = {
                    (row["subject"], row["predicate"], row["object"])
                    for row in records
                }
                targets = set(mask.evaluator_manifest.target_edges)
                recovered_targets = targets.intersection(candidate_keys)

                canonical_traces, availability = r2._canonical_trace_frame(
                    mask.model.traces
                )
                operational, diagnostics = r2._candidate_feature_rows(
                    records,
                    canonical_traces,
                    mask.model.observed_edges,
                )
                operational = r2.add_profile_scores(operational)
                operational["incident_token"] = incident_token
                operational["seed"] = int(seed)
                operational["mask_id"] = mask_id
                operational["mask_ratio"] = float(ratio)
                operational["subject_label"] = operational["subject"].map(
                    lambda value: str(value).rsplit(":", 1)[-1]
                )
                operational["object_label"] = operational["object"].map(
                    lambda value: str(value).rsplit(":", 1)[-1]
                )
                operational["is_masked_target"] = [
                    tuple(map(str, values)) in targets
                    for values in operational[list(CANDIDATE_KEY)].itertuples(
                        index=False, name=None
                    )
                ]
                operational["is_silver_matched"] = [
                    tuple(map(str, values)) in reference_keys
                    for values in operational[list(CANDIDATE_KEY)].itertuples(
                        index=False, name=None
                    )
                ]
                operational["case"] = case_id
                operational["fault"] = "evaluator_metadata_withheld_until_scoring"
                operational["role"] = "confirmatory"
                model_frames.append(operational)

                diagnostics = dict(diagnostics)
                diagnostics.update(
                    {
                        **common,
                        "status": "READY",
                        "target_count": len(targets),
                        "a2_recovered_target_count": len(recovered_targets),
                        "a2_candidate_recall": (
                            len(recovered_targets) / len(targets)
                            if targets
                            else None
                        ),
                        "candidate_count": len(candidate_keys),
                        "typed_universe_count": len(
                            suite.evaluation_universe
                        ),
                        "reference_edge_count": len(reference_keys),
                        "eligible_attestation_a_edge_count": len(eligible),
                        "trace_field_availability": availability,
                    }
                )
                cell_records.append(diagnostics)
            except Exception as exc:
                cell_records.append(
                    {
                        **common,
                        "status": "NOT_READY",
                        "reason_code": type(exc).__name__,
                        "detail": str(exc),
                        "eligible_attestation_a_edge_count": len(eligible),
                    }
                )

    case_diagnostic = {
        "case_id": case_id,
        "incident_token": incident_token,
        "system": system_label,
        "trace_rows": len(traces),
        "trace_count": traces["trace_id"].nunique(),
        "service_count": traces["service_id"].nunique(),
        "reference_edge_count": len(reference_keys),
        "observed_edge_count": len(observed_keys),
        "eligible_attestation_a_edge_count": len(eligible),
        "reference_parent_coverage": graph.reference_stats.nonroot_parent_coverage,
        "model_parent_coverage": graph.model_stats.nonroot_parent_coverage,
    }
    return model_frames, cell_records, case_diagnostic


def _metric_with_true_targets(
    scored: pd.DataFrame,
    *,
    true_target_count: int,
) -> dict[str, Any]:
    metric = r3.evaluate_cell(scored)
    represented = int(metric["target_count"])
    if true_target_count < represented:
        raise OpsLiteConfirmatoryError(
            "represented masked targets exceed evaluator target count"
        )
    if true_target_count > 0:
        metric["recall"] = (
            int(metric["recovered_target_count"]) / true_target_count
        )
        metric["mrr"] = (
            float(metric["mrr"]) * represented / true_target_count
            if metric["mrr"] is not None
            else 0.0
        )
    metric["target_count"] = int(true_target_count)
    metric["missing_from_a2_count"] = true_target_count - represented
    return metric


def _evaluate_frozen_policy(
    cells: Mapping[tuple[str, int, str], pd.DataFrame],
    target_counts: Mapping[tuple[str, int, str], int],
    policy: r3.R3Policy,
):
    proposed_rows = []
    a2_rows = []
    r2_rows = []
    full_rows = []
    scored_frames = []
    for key, group in sorted(cells.items()):
        group = group.reset_index(drop=True)
        target_count = int(target_counts[key])
        proposed = r3.apply_policy(group, policy)
        proposed_metric = _metric_with_true_targets(
            proposed, true_target_count=target_count
        )
        a2_control = r3._control(
            group,
            int(proposed_metric["selected_count"]),
            kind="a2",
            policy=policy,
        )
        r2_control = r3._control(
            group,
            int(proposed_metric["selected_count"]),
            kind="r2",
            policy=policy,
        )
        full = group.copy()
        full["selected"] = True
        full["a3_r3_score"] = full["a2_rank_normalized"].astype(float)

        proposed_result = _metric_with_true_targets(
            proposed, true_target_count=target_count
        )
        a2_result = _metric_with_true_targets(
            a2_control, true_target_count=target_count
        )
        r2_result = _metric_with_true_targets(
            r2_control, true_target_count=target_count
        )
        full_result = _metric_with_true_targets(
            full, true_target_count=target_count
        )
        first = group.iloc[0]
        common = {
            "incident_token": key[0],
            "case": str(first["case"]),
            "fault": str(first["fault"]),
            "role": "confirmatory",
            "system": str(first["system"]),
            "seed": int(key[1]),
            "mask_id": str(key[2]),
            "mask_ratio": float(first["mask_ratio"]),
        }
        proposed_rows.append({**common, **proposed_result})
        a2_rows.append({**common, **a2_result})
        r2_rows.append({**common, **r2_result})
        full_rows.append({**common, **full_result})
        for name, value in common.items():
            proposed[name] = value
        scored_frames.append(proposed)

    proposed_frame = pd.DataFrame.from_records(proposed_rows)
    a2_frame = pd.DataFrame.from_records(a2_rows)
    r2_frame = pd.DataFrame.from_records(r2_rows)
    full_frame = pd.DataFrame.from_records(full_rows)
    scored = pd.concat(scored_frames, ignore_index=True)
    return (
        proposed_frame,
        r3._aggregate(proposed_rows),
        a2_frame,
        r3._aggregate(a2_rows),
        r2_frame,
        r3._aggregate(r2_rows),
        full_frame,
        r3._aggregate(full_rows),
        scored,
    )


def _paired_bootstrap(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    if samples <= 0:
        raise ValueError("paired bootstrap samples must be positive")
    rng = random.Random(seed)
    observed = [float(value) for value in values]
    means = []
    for _ in range(samples):
        draw = [observed[rng.randrange(len(observed))] for _ in observed]
        means.append(statistics.fmean(draw))
    means.sort()
    alpha = 1.0 - confidence_level
    lower_index = max(0, int(math.floor((alpha / 2.0) * (samples - 1))))
    upper_index = min(
        samples - 1,
        int(math.ceil((1.0 - alpha / 2.0) * (samples - 1))),
    )
    return {
        "n": len(observed),
        "mean": statistics.fmean(observed),
        "lower": means[lower_index],
        "upper": means[upper_index],
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence_level,
    }


def _delta(enhanced: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
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


def _render_report(summary: Mapping[str, Any]) -> str:
    proposed = summary.get("confirmatory", {}).get("proposed_a3_r3", {})
    full = summary.get("confirmatory", {}).get("baseline_a2_full", {})
    a2_delta = summary.get("confirmatory", {}).get("delta_vs_equal_size_a2", {})
    r2_delta = summary.get("confirmatory", {}).get("delta_vs_equal_size_r2", {})
    reasons = ", ".join(summary["gate"]["reason_codes"]) or "없음"
    if not proposed:
        table = "실행 가능한 확인 Cell이 없어 성능표를 생성하지 못했습니다."
    else:
        table = f"""| 지표 | A2 전체 | Frozen R3 | 동일 크기 A2 대비 | 동일 크기 R2 대비 |
|---|---:|---:|---:|---:|
| Recall Macro | {full['recall_macro']:.4f} | {proposed['recall_macro']:.4f} | {a2_delta['recall_macro']:+.4f} | {r2_delta['recall_macro']:+.4f} |
| Recall Minimum | {full['recall_min']:.4f} | {proposed['recall_min']:.4f} | - | - |
| 후보 수 평균 | {full['selected_count_mean']:.3f} | {proposed['selected_count_mean']:.3f} | {a2_delta['selected_count_mean']:+.3f} | {r2_delta['selected_count_mean']:+.3f} |
| P-LB Macro | {full['silver_precision_lower_bound_macro']:.4f} | {proposed['silver_precision_lower_bound_macro']:.4f} | {a2_delta['silver_precision_lower_bound_macro']:+.4f} | {r2_delta['silver_precision_lower_bound_macro']:+.4f} |
| MRR Macro | {full['mrr_macro']:.4f} | {proposed['mrr_macro']:.4f} | {a2_delta['mrr_macro']:+.4f} | {r2_delta['mrr_macro']:+.4f} |"""
    return f"""# RCABench OPS-Lite 독립 확인시험 — Frozen A3-R3

- 최종 과학적 Gate: **{summary['status']}**
- 선택 Incident: **{summary['execution']['selected_incident_count']}**
- 기대/완료 Cell: **{summary['execution']['expected_cell_count']} / {summary['execution']['ready_cell_count']}**
- 시스템: `{', '.join(summary['execution']['systems'])}`
- 정책 재학습·Threshold 재조정: **없음**
- 미통과 조건: `{reasons}`

## 결과

{table}

## 통계적 확인

```json
{json.dumps(summary.get('paired_bootstrap', {}), ensure_ascii=False, indent=2, sort_keys=True)}
```

## 해석 한계

- 이 시험은 공식 RCABench OPS-Lite Test Split에서 사전 고정한 Incident에 RCAEval R3 정책을 그대로 적용한다.
- 대상은 runtime `CALLS` 관계 후보 복원·재랭킹이며 causal `CAUSES`, RCA 경로 또는 LLM 성능을 검증하지 않는다.
- 실패 Cell과 구조적으로 Mask 불가능한 Incident는 교체하지 않고 결과에 남긴다.
"""


def run_ops_lite_confirmatory(
    *,
    data_root: Path,
    output: Path,
    config_path: Path,
    frozen_policy_path: Path,
    model_dir: Path,
) -> Path:
    data_root = data_root.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    frozen_policy_path = frozen_policy_path.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    config = _load_json(config_path)
    frozen = _load_json(frozen_policy_path)
    if config.get("experiment_id") != (
        "rcabench-ops-lite-task-a-r3-independent-confirmatory"
    ):
        raise OpsLiteConfirmatoryError("unexpected confirmatory experiment_id")
    if frozen.get("scientific_status") != "PASS":
        raise OpsLiteConfirmatoryError(
            "R3 policy cannot be frozen from a non-PASS development result"
        )
    if frozen.get("policy_tuning_allowed") is not False:
        raise OpsLiteConfirmatoryError("frozen policy must disable tuning")

    source_manifest = _load_json(data_root / "source_manifest.json")
    dataset = config["dataset"]
    if source_manifest["dataset"]["revision"] != dataset["revision"]:
        raise OpsLiteConfirmatoryError("source revision differs from config")
    if source_manifest["dataset"]["manifest_sha256"] != dataset["manifest_sha256"]:
        raise OpsLiteConfirmatoryError("source manifest digest differs from config")
    if source_manifest["dataset"]["test_split_sha256"] != dataset["test_split_sha256"]:
        raise OpsLiteConfirmatoryError("source test split digest differs from config")

    policy = r3.R3Policy(**dict(frozen["selected_policy"]))
    tri_state = r3.TriStateConfig(**dict(frozen["tri_state"]))
    channel_weights = {
        str(key): float(value)
        for key, value in frozen["channel_weights"].items()
    }
    backend_contract = frozen["backend"]
    backend = OnnxDebertaNLIBackend(
        model_dir,
        onnx_filename=str(backend_contract["onnx_file"]),
        expected_sha256=str(backend_contract["onnx_sha256"]),
        revision=str(backend_contract["revision"]),
        batch_size=int(backend_contract["batch_size"]),
        performance_mode=False,
        max_length=int(backend_contract["max_length"]),
    )

    metadata = source_manifest["selection"]["metadata"]
    all_model_frames = []
    cell_records = []
    case_diagnostics = []
    revision = str(dataset["revision"])
    dataset_id = str(config["trace_contract"]["dataset_id"])
    for case_spec in config["selection_contract"]["selected_cases"]:
        case_id = str(case_spec["case_id"])
        system = str(case_spec["system"])
        traces = load_case_traces(
            data_root,
            case_id=case_id,
            dataset_id=dataset_id,
            system_id=f"ops-lite-{system}",
        )
        frames, records, diagnostic = build_case_cells(
            traces=traces,
            case_id=case_id,
            system_label=system,
            revision=revision,
            config=config,
        )
        primary_kind = str(metadata.get(case_id, {}).get("primary_kind", "unknown"))
        for frame in frames:
            frame["fault"] = primary_kind
            frame["system"] = system
        for record in records:
            record["fault"] = primary_kind
        all_model_frames.extend(frames)
        cell_records.extend(records)
        case_diagnostics.append(diagnostic)
        print(
            f"OPS_LITE_CASE_READY case={case_id} system={system} "
            f"cells={sum(record['status'] == 'READY' for record in records)}",
            flush=True,
        )

    expected_cells = int(config["mask_contract"]["expected_cells"])
    ready_records = [row for row in cell_records if row.get("status") == "READY"]
    ready_keys = {
        (
            str(row["incident_token"]),
            int(row["seed"]),
            str(row["mask_id"]),
        ): row
        for row in ready_records
    }
    if len(ready_keys) != len(ready_records):
        raise OpsLiteConfirmatoryError("ready confirmatory cell keys repeat")

    if all_model_frames:
        labelled = pd.concat(all_model_frames, ignore_index=True)
        evaluator_columns = {
            "case",
            "fault",
            "role",
            "system",
            "is_masked_target",
            "is_silver_matched",
        }
        evaluator = labelled[
            [*CELL_KEY, *CANDIDATE_KEY, *sorted(evaluator_columns)]
        ].copy()
        model = labelled.drop(columns=sorted(evaluator_columns))
        scored_model, nli_diagnostics = r3.score_channel_nli(
            model,
            backend=backend,
            tri_state=tri_state,
            channel_weights=channel_weights,
        )
        scored = scored_model.merge(
            evaluator,
            on=[*CELL_KEY, *CANDIDATE_KEY],
            how="inner",
            validate="one_to_one",
        )
        if len(scored) != len(labelled):
            raise OpsLiteConfirmatoryError(
                "model/evaluator rejoin changed candidate count"
            )
        cells = {
            tuple(key): group.copy()
            for key, group in scored.groupby(
                list(CELL_KEY), sort=True, dropna=False
            )
        }
        target_counts = {
            key: int(ready_keys[key]["target_count"])
            for key in cells
        }
        (
            proposed_rows,
            proposed,
            a2_rows,
            a2_control,
            r2_rows,
            r2_control,
            full_rows,
            baseline,
            scored_candidates,
        ) = _evaluate_frozen_policy(cells, target_counts, policy)

        delta_full = _delta(proposed, baseline)
        delta_a2 = _delta(proposed, a2_control)
        delta_r2 = _delta(proposed, r2_control)
        selected_ratio = (
            float(proposed["selected_count_mean"])
            / float(baseline["selected_count_mean"])
        )
        merged_r2 = proposed_rows.merge(
            r2_rows,
            on=[
                "incident_token",
                "case",
                "fault",
                "role",
                "system",
                "seed",
                "mask_id",
                "mask_ratio",
            ],
            suffixes=("_r3", "_r2"),
            validate="one_to_one",
        )
        statistics_contract = config["statistics"]
        bootstrap_p_lb = _paired_bootstrap(
            (
                merged_r2["silver_precision_lower_bound_r3"]
                - merged_r2["silver_precision_lower_bound_r2"]
            ).tolist(),
            samples=int(statistics_contract["paired_bootstrap_samples"]),
            seed=int(statistics_contract["paired_bootstrap_seed"]),
            confidence_level=float(statistics_contract["confidence_level"]),
        )
        bootstrap_mrr = _paired_bootstrap(
            (merged_r2["mrr_r3"] - merged_r2["mrr_r2"]).tolist(),
            samples=int(statistics_contract["paired_bootstrap_samples"]),
            seed=int(statistics_contract["paired_bootstrap_seed"]) + 1,
            confidence_level=float(statistics_contract["confidence_level"]),
        )

        a2_recalls = [
            float(row["a2_candidate_recall"])
            for row in ready_records
            if row.get("a2_candidate_recall") is not None
        ]
        gate = config["gate"]
        additive_gain = max(
            delta_r2["silver_precision_lower_bound_macro"],
            delta_r2["mrr_macro"],
        )
        bootstrap_gain = bool(
            (
                bootstrap_p_lb["mean"] >= float(gate["nli_additive_gain_min"])
                and bootstrap_p_lb["lower"]
                >= float(gate["paired_bootstrap_lower_bound_min"])
            )
            or (
                bootstrap_mrr["mean"] >= float(gate["nli_additive_gain_min"])
                and bootstrap_mrr["lower"]
                >= float(gate["paired_bootstrap_lower_bound_min"])
            )
        )
        systems = sorted({str(item["system"]) for item in case_diagnostics})
        conditions = {
            "all_selected_incidents_ready": len(case_diagnostics)
            == len(config["selection_contract"]["selected_cases"]),
            "all_expected_cells_ready": len(ready_records) == expected_cells,
            "required_systems_present": set(gate["required_systems"]).issubset(
                systems
            ),
            "a2_candidate_recall_macro": bool(a2_recalls)
            and statistics.fmean(a2_recalls)
            >= float(gate["a2_candidate_recall_macro_min"]),
            "a2_candidate_recall_each_cell": bool(a2_recalls)
            and min(a2_recalls)
            >= float(gate["a2_candidate_recall_each_cell_min"]),
            "r3_recall_macro": float(proposed["recall_macro"])
            >= float(gate["r3_recall_macro_min"]),
            "r3_recall_pooled": float(proposed["recall_pooled"])
            >= float(gate["r3_recall_pooled_min"]),
            "r3_recall_each_cell": float(proposed["recall_min"])
            >= float(gate["r3_recall_each_cell_min"]),
            "candidate_count_reduced": selected_ratio
            <= float(gate["selected_count_ratio_max"]),
            "p_lb_noninferior_to_full_a2": delta_full[
                "silver_precision_lower_bound_macro"
            ]
            >= float(gate["p_lb_delta_vs_full_a2_min"]),
            "mrr_noninferior_to_full_a2": delta_full["mrr_macro"]
            >= float(gate["mrr_delta_vs_full_a2_min"]),
            "matched_a2_recall_noninferior": delta_a2["recall_macro"]
            >= -float(gate["matched_a2_recall_tolerance"]),
            "matched_a2_p_lb_noninferior": delta_a2[
                "silver_precision_lower_bound_macro"
            ]
            >= float(gate["matched_a2_p_lb_delta_min"]),
            "matched_a2_mrr_noninferior": delta_a2["mrr_macro"]
            >= float(gate["matched_a2_mrr_delta_min"]),
            "matched_r2_recall_noninferior": delta_r2["recall_macro"]
            >= -float(gate["matched_r2_recall_tolerance"]),
            "matched_r2_p_lb_noninferior": delta_r2[
                "silver_precision_lower_bound_macro"
            ]
            >= float(gate["matched_r2_p_lb_delta_min"]),
            "matched_r2_mrr_noninferior": delta_r2["mrr_macro"]
            >= float(gate["matched_r2_mrr_delta_min"]),
            "nli_additive_gain": additive_gain
            >= float(gate["nli_additive_gain_min"]),
            "paired_bootstrap_gain": bootstrap_gain,
            "nli_candidate_coverage": float(
                nli_diagnostics["candidate_coverage"]
            )
            >= float(gate["nli_candidate_coverage_min"]),
            "nli_score_variance": float(nli_diagnostics["nli_score_std"])
            >= float(gate["nli_score_std_min"]),
            "frozen_policy_applied_without_search": True,
        }
        passed = all(conditions.values())
        summary = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "PASS" if passed else "FAIL",
            "gate_id": "D5_EXTERNAL_R3_CONFIRMATORY",
            "frozen_policy": {
                "path": str(frozen_policy_path),
                "sha256": _sha256(frozen_policy_path),
                "source": frozen.get("source"),
                "selected_policy": asdict(policy),
                "tri_state": asdict(tri_state),
                "channel_weights": channel_weights,
                "backend": backend.metadata(),
                "policy_search_performed": False,
            },
            "source": {
                "data_root": str(data_root),
                "source_manifest_sha256": _sha256(
                    data_root / "source_manifest.json"
                ),
                "config_sha256": _sha256(config_path),
                "dataset": source_manifest["dataset"],
                "selection": source_manifest["selection"],
            },
            "execution": {
                "selected_incident_count": len(case_diagnostics),
                "expected_cell_count": expected_cells,
                "ready_cell_count": len(ready_records),
                "not_ready_cell_count": len(cell_records) - len(ready_records),
                "systems": systems,
                "candidate_rows": len(scored),
            },
            "a2_candidate_recovery": {
                "recall_macro": statistics.fmean(a2_recalls)
                if a2_recalls
                else None,
                "recall_min": min(a2_recalls) if a2_recalls else None,
            },
            "nli_diagnostics": nli_diagnostics,
            "confirmatory": {
                "baseline_a2_full": baseline,
                "proposed_a3_r3": proposed,
                "equal_size_a2_control": a2_control,
                "equal_size_r2_control": r2_control,
                "delta_vs_full_a2": delta_full,
                "delta_vs_equal_size_a2": delta_a2,
                "delta_vs_equal_size_r2": delta_r2,
                "selected_count_ratio": selected_ratio,
                "nli_additive_gain": additive_gain,
            },
            "paired_bootstrap": {
                "p_lb_delta_vs_equal_size_r2": bootstrap_p_lb,
                "mrr_delta_vs_equal_size_r2": bootstrap_mrr,
            },
            "case_diagnostics": case_diagnostics,
            "gate": {
                "status": "PASS" if passed else "FAIL",
                "passed": passed,
                "conditions": conditions,
                "reason_codes": [
                    name.upper()
                    for name, value in conditions.items()
                    if not value
                ],
                "required": gate,
            },
            "claim_limit": config["claim_limit"],
        }
    else:
        conditions = {
            "all_expected_cells_ready": False,
            "at_least_one_model_cell": False,
        }
        summary = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "FAIL",
            "gate_id": "D5_EXTERNAL_R3_CONFIRMATORY",
            "execution": {
                "selected_incident_count": len(case_diagnostics),
                "expected_cell_count": expected_cells,
                "ready_cell_count": 0,
                "not_ready_cell_count": len(cell_records),
                "systems": sorted(
                    {str(item["system"]) for item in case_diagnostics}
                ),
            },
            "case_diagnostics": case_diagnostics,
            "gate": {
                "status": "FAIL",
                "passed": False,
                "conditions": conditions,
                "reason_codes": [name.upper() for name in conditions],
                "required": config["gate"],
            },
            "claim_limit": config["claim_limit"],
        }

    output.mkdir(parents=True, exist_ok=False)
    published = output / "published"
    evaluator = output / "evaluator_private"
    model_output = output / "model_output"
    published.mkdir()
    evaluator.mkdir()
    model_output.mkdir()
    pd.DataFrame.from_records(cell_records).to_csv(
        published / "ops_lite_confirmatory_cell_readiness.csv", index=False
    )
    pd.DataFrame.from_records(case_diagnostics).to_csv(
        published / "ops_lite_confirmatory_case_diagnostics.csv", index=False
    )
    if all_model_frames:
        proposed_rows.to_csv(
            published / "ops_lite_confirmatory_cells.csv", index=False
        )
        a2_rows.to_csv(
            evaluator / "equal_size_a2_control_cells.csv", index=False
        )
        r2_rows.to_csv(
            evaluator / "equal_size_r2_control_cells.csv", index=False
        )
        full_rows.to_csv(
            evaluator / "full_a2_baseline_cells.csv", index=False
        )
        scored_candidates.to_parquet(
            evaluator / "confirmatory_candidate_analysis.parquet", index=False
        )
        model.to_parquet(
            model_output / "confirmatory_model_features.parquet", index=False
        )
    result_path = published / "ops_lite_r3_confirmatory_results.json"
    result_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (published / "ops_lite_r3_confirmatory_results.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    (published / "ops_lite_r3_confirmatory_status.txt").write_text(
        summary["status"] + "\n", encoding="utf-8"
    )
    return output


__all__ = [
    "OpsLiteConfirmatoryError",
    "build_case_cells",
    "canonicalize_trace_frame",
    "load_case_traces",
    "run_ops_lite_confirmatory",
]
