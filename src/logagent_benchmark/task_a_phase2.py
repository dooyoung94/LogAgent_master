"""Task A Phase 2: multi-incident, multi-seed bounded relation recovery.

This module orchestrates the already frozen Phase-1 runner.  It deliberately
keeps the incident identifier stable across masking seeds so that the held-out
reference/model trace partition is unchanged within an incident.  Only the IID
masking seed varies.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE2_CONFIG = PROJECT_ROOT / "configs" / "experiment_task_a_rcaeval_phase2.json"


class Phase2Error(RuntimeError):
    """Raised when the Phase-2 preregistered contract cannot be satisfied."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase2Error(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase2Error(f"invalid JSON in {path}: {exc}") from exc


def validate_phase2_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise Phase2Error("Phase-2 config schema_version must be 1")
    dataset = config.get("dataset", {})
    if dataset.get("dataset_id") != "rcaeval":
        raise Phase2Error("Phase-2 currently accepts only RCAEval")
    selection = dataset.get("case_selection", {})
    faults = tuple(str(value).lower() for value in selection.get("fault_order", ()))
    if faults != ("cpu", "mem", "disk", "delay", "loss", "socket"):
        raise Phase2Error("fault_order must be cpu, mem, disk, delay, loss, socket")
    if int(selection.get("incident_count", 0)) != len(faults):
        raise Phase2Error("incident_count must equal the number of stratified faults")

    seeds = tuple(config.get("seeds", ()))
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise Phase2Error("Phase-2 requires at least five unique seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise Phase2Error("all masking seeds must be integers")

    masks = tuple(config.get("masks", ()))
    ratios = sorted(float(mask.get("ratio", -1.0)) for mask in masks)
    if len(masks) != 2 or ratios != [0.2, 0.4]:
        raise Phase2Error("Phase-2 requires exactly IID20 and IID40")
    if any(mask.get("kind") != "iid" for mask in masks):
        raise Phase2Error("Phase-2 accepts IID masks only")
    if any(mask.get("level") != "L2_PARENT_DROPPED" for mask in masks):
        raise Phase2Error("Phase-2 requires L2_PARENT_DROPPED masks")

    budget = config.get("candidate_budget", {})
    positive_integer_fields = (
        "max_abductive_proposals",
        "max_per_subject",
        "max_per_object",
        "min_supporting_traces",
        "min_boundary_count",
    )
    for name in positive_integer_fields:
        value = budget.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Phase2Error(f"candidate_budget.{name} must be a positive integer")


def _truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _selection_hash(revision: str, case_id: str) -> str:
    material = f"{revision}|task-a-phase2|{case_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def select_phase2_cases(index: pd.DataFrame, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select one eligible RE2-TT incident per fault without using recovery labels.

    The CPU case is a fixed continuity anchor from Phase 1.  Every other fault
    uses the minimum stable SHA-256 key at the frozen dataset revision.  The
    selection never observes a recovered edge, mask target, or relation score.
    """

    validate_phase2_config(config)
    required = {
        "case",
        "dataset",
        "root_cause_service",
        "fault",
        "repetition",
        "has_logs",
        "n_logs",
        "has_traces",
        "n_traces",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise Phase2Error(f"cases.parquet is missing columns: {missing}")

    dataset = config["dataset"]
    selection = dataset["case_selection"]
    suite = str(selection["suite"]).strip().upper().replace("_", "-")
    revision = str(dataset["source_revision"])

    normalized_suite = index["dataset"].astype(str).str.strip().str.upper().str.replace("_", "-", regex=False)
    eligible = index.loc[normalized_suite.eq(suite)].copy()
    if bool(selection.get("require_traces", True)):
        eligible = eligible.loc[
            _truthy_series(eligible["has_traces"])
            & pd.to_numeric(eligible["n_traces"], errors="coerce").fillna(0).gt(0)
        ]
    if bool(selection.get("require_logs", True)):
        eligible = eligible.loc[
            _truthy_series(eligible["has_logs"])
            & pd.to_numeric(eligible["n_logs"], errors="coerce").fillna(0).gt(0)
        ]
    if eligible.empty:
        raise Phase2Error(f"no eligible cases remain for suite {suite}")

    eligible["case"] = eligible["case"].astype(str)
    eligible["fault"] = eligible["fault"].astype(str).str.lower()
    eligible["selection_sha256"] = eligible["case"].map(
        lambda case_id: _selection_hash(revision, case_id)
    )

    anchor_case = str(selection["anchor_case"])
    anchor_rows = eligible.loc[eligible["case"].eq(anchor_case)]
    if len(anchor_rows) != 1:
        raise Phase2Error(
            f"anchor case must exist exactly once among eligible rows: {anchor_case}"
        )
    anchor = anchor_rows.iloc[0]
    if str(anchor["fault"]).lower() != "cpu":
        raise Phase2Error("the fixed continuity anchor must be a CPU case")

    chosen: list[pd.Series] = [anchor]
    for fault in tuple(str(value).lower() for value in selection["fault_order"]):
        if fault == "cpu":
            continue
        candidates = eligible.loc[eligible["fault"].eq(fault)].copy()
        if candidates.empty:
            raise Phase2Error(f"no eligible {fault} case is available")
        candidates = candidates.sort_values(
            ["selection_sha256", "case"], kind="mergesort"
        )
        chosen.append(candidates.iloc[0])

    records: list[dict[str, Any]] = []
    for order, row in enumerate(chosen, start=1):
        records.append(
            {
                "selection_order": order,
                "case": str(row["case"]),
                "dataset": str(row["dataset"]),
                "fault": str(row["fault"]).lower(),
                "root_cause_service": str(row["root_cause_service"]),
                "repetition": int(row["repetition"]),
                "n_logs": int(row["n_logs"]),
                "n_traces": int(row["n_traces"]),
                "selection_sha256": str(row["selection_sha256"]),
                "selection_role": "phase1_anchor" if order == 1 else "fault_stratified_hash_min",
            }
        )

    expected_count = int(selection["incident_count"])
    if len(records) != expected_count:
        raise AssertionError(f"expected {expected_count} selected cases, got {len(records)}")
    selected_faults = tuple(record["fault"] for record in records)
    expected_faults = tuple(str(value).lower() for value in selection["fault_order"])
    if selected_faults != expected_faults:
        raise AssertionError(
            f"selected fault order differs from preregistration: {selected_faults}"
        )
    if len({record["case"] for record in records}) != len(records):
        raise AssertionError("case selection contains duplicates")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_source_files(selected_cases: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names = ["cases.parquet"]
    for record in selected_cases:
        case_id = str(record["case"])
        names.extend(
            [
                f"{case_id}/inject_time.txt",
                f"{case_id}/metrics.parquet",
                f"{case_id}/logs.parquet",
                f"{case_id}/traces.parquet",
            ]
        )
    return tuple(names)


def write_verified_provenance(
    raw_root: Path,
    *,
    config: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Hash every selected source file and emit the Phase-1-compatible manifest."""

    raw_root = raw_root.resolve()
    verified: dict[str, Any] = {}
    total_bytes = 0
    for relative_name in required_source_files(selected_cases):
        path = (raw_root / relative_name).resolve()
        try:
            path.relative_to(raw_root)
        except ValueError as exc:
            raise Phase2Error(f"unsafe source path: {relative_name}") from exc
        if not path.is_file():
            raise Phase2Error(f"required Phase-2 source file is missing: {relative_name}")
        byte_count = path.stat().st_size
        total_bytes += byte_count
        verified[relative_name] = {
            "bytes": byte_count,
            "checksum": f"sha256:{sha256_file(path)}",
        }

    dataset = config["dataset"]
    selection_payload = json.dumps(
        list(selected_cases), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = {
        "dataset": "rcaeval",
        "profile": "task_a_phase2_dynamic_subset",
        "adapter": "huggingface",
        "repo_id": dataset["repo_id"],
        "revision": dataset["source_revision"],
        "selection_policy": dataset["case_selection"]["policy"],
        "selected_case_count": len(selected_cases),
        "selected_cases_sha256": hashlib.sha256(selection_payload).hexdigest(),
        "total_verified_bytes": total_bytes,
        "verified": verified,
    }
    path = raw_root / ".logagent-source.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_cell_config(
    base_config: Mapping[str, Any],
    phase2_config: Mapping[str, Any],
    case_record: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Create one two-mask config while keeping the case split stable across seeds."""

    config = deepcopy(dict(base_config))
    case_id = str(case_record["case"])
    dataset = config["dataset"]
    dataset["profile"] = "task_a_phase2_dynamic_subset"
    dataset["source_revision"] = phase2_config["dataset"]["source_revision"]
    dataset["raw_case"] = case_id
    dataset["incident_id"] = f"rcaeval_{case_id}"
    dataset["system_id"] = phase2_config["dataset"]["system_id"]

    config["experiment_id"] = f"task-a-phase2-{case_id}-seed-{seed}"
    masks = []
    for specification in phase2_config["masks"]:
        masks.append(
            {
                "id": f"{specification['name']}_l2_s{seed}",
                "kind": "iid",
                "ratio": float(specification["ratio"]),
                "seed": int(seed),
                "level": "L2_PARENT_DROPPED",
            }
        )
    config["masks"] = masks

    pipeline = config["pipeline"]
    pipeline["candidate_budget"] = deepcopy(phase2_config["candidate_budget"])
    execution_policy = config.setdefault("execution_policy", {})
    execution_policy.update(
        {
            "profile": "task_a_phase2_cell",
            "task_b_enabled": False,
            "included_masks": [mask["id"] for mask in masks],
            "excluded_masks": ["iid60_l2", "component_l2"],
            "reason": "multi-incident multi-seed Task A generalization validation",
            "claim_limit": phase2_config["claim_limit"],
        }
    )
    config["notes"] = [
        "Phase-2 cell generated from the frozen Phase-1 contract.",
        "The incident_id is independent of the masking seed, fixing the whole-trace split.",
        f"Selected case fault stratum: {case_record['fault']}.",
    ]
    return config


def _metric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, Mapping):
        return None
    for key in ("value", "recall", "lower_bound"):
        if key in value:
            parsed = _metric_value(value[key])
            if parsed is not None:
                return parsed
    return None


def _prune_heavy_run_artifacts(run_root: Path) -> list[str]:
    removed: list[str] = []
    for path in run_root.glob("masks/*/model_input/traces.parquet"):
        if path.is_file():
            removed.append(str(path.relative_to(run_root)))
            path.unlink()
    for cache_dir in run_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
    return removed


@dataclass(frozen=True)
class RunTask:
    case_record: Mapping[str, Any]
    seed: int
    config_path: Path
    output_path: Path
    timeout_seconds: int
    prune_heavy: bool


def _extract_cells(
    summary: Mapping[str, Any],
    *,
    task: RunTask,
    generated_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ratios = {
        str(mask["id"]): float(mask["ratio"])
        for mask in generated_config["masks"]
    }
    cells: list[dict[str, Any]] = []
    for mask_summary in summary.get("masks", ()):
        mask_id = str(mask_summary["mask_id"])
        evaluation = mask_summary.get("evaluation", {}).get("A2", {})
        diagnostics = mask_summary.get("diagnostics", {})
        budget = diagnostics.get("candidate_budget", {})
        gate = mask_summary.get("stage_gate", {})
        cells.append(
            {
                "case": str(task.case_record["case"]),
                "fault": str(task.case_record["fault"]),
                "root_cause_service": str(task.case_record["root_cause_service"]),
                "seed": int(task.seed),
                "mask_id": mask_id,
                "mask_ratio": ratios[mask_id],
                "target_count": int(mask_summary.get("target_count", 0)),
                "redacted_boundary_spans": int(mask_summary.get("redacted_boundary_spans", 0)),
                "typed_universe_count": int(mask_summary.get("evaluation_universe_count", 0)),
                "a2_proposal_count": int(diagnostics.get("a2_abductive_proposal_count", mask_summary.get("a2_proposal_count", 0))),
                "candidate_recall": _metric_value(evaluation.get("candidate_recall")),
                "masked_recall": _metric_value(evaluation.get("masked_recall")),
                "mrr_within_a2": _metric_value(evaluation.get("mrr")),
                "silver_precision_lower_bound": _metric_value(evaluation.get("silver_precision_lower_bound")),
                "compression_ratio": float(diagnostics.get("a2_candidate_compression_ratio", 0.0)),
                "budget_saturated": bool(budget.get("budget_saturated", False)),
                "dropped_by_budget": int(budget.get("dropped_by_budget", 0)),
                "leakage_checks_all_pass": all(
                    bool(check.get("passed"))
                    for check in mask_summary.get("leakage_checks", ())
                ),
                "cell_gate_status": str(gate.get("status", "UNKNOWN")),
                "cell_gate_passed": gate.get("passed"),
                "run_summary": str((task.output_path / "summary.json").relative_to(task.output_path.parents[2])),
            }
        )
    return cells


def _execute_task(task: RunTask, base_config: Mapping[str, Any], phase2_config: Mapping[str, Any]) -> dict[str, Any]:
    generated = build_cell_config(base_config, phase2_config, task.case_record, task.seed)
    task.config_path.parent.mkdir(parents=True, exist_ok=True)
    task.config_path.write_text(
        json.dumps(generated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_task_a.py"),
        "--config",
        str(task.config_path),
        "--raw-root",
        str(task.output_path.parents[3] / "_raw_placeholder"),
        "--output",
        str(task.output_path),
    ]
    # The raw root is injected by run_phase2 immediately before execution.
    command[command.index("--raw-root") + 1] = os.environ["LOGAGENT_PHASE2_RAW_ROOT"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=task.timeout_seconds,
        check=False,
    )
    log_path = task.output_path.parent / f"seed-{task.seed}.log"
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout
        + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {
            "status": "ERROR",
            "case": task.case_record["case"],
            "fault": task.case_record["fault"],
            "seed": task.seed,
            "returncode": completed.returncode,
            "log_path": str(log_path),
            "detail": completed.stderr[-4000:],
            "cells": [],
        }

    summary_path = task.output_path / "summary.json"
    if not summary_path.is_file():
        return {
            "status": "ERROR",
            "case": task.case_record["case"],
            "fault": task.case_record["fault"],
            "seed": task.seed,
            "returncode": 0,
            "log_path": str(log_path),
            "detail": "summary.json was not created",
            "cells": [],
        }
    summary = _load_json(summary_path)
    removed = _prune_heavy_run_artifacts(task.output_path) if task.prune_heavy else []
    return {
        "status": "READY",
        "case": task.case_record["case"],
        "fault": task.case_record["fault"],
        "seed": task.seed,
        "returncode": 0,
        "log_path": str(log_path),
        "removed_heavy_artifacts": removed,
        "decision_gate": summary.get("decision_gate", {}),
        "cells": _extract_cells(summary, task=task, generated_config=generated),
    }


def _group_summary(cells: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(str(cell[key]), []).append(cell)
    output: dict[str, Any] = {}
    for group, values in sorted(grouped.items()):
        recalls = [float(item["candidate_recall"]) for item in values if item.get("candidate_recall") is not None]
        proposals = [int(item["a2_proposal_count"]) for item in values]
        p_lbs = [float(item["silver_precision_lower_bound"]) for item in values if item.get("silver_precision_lower_bound") is not None]
        output[group] = {
            "cell_count": len(values),
            "candidate_recall_macro": statistics.fmean(recalls) if recalls else None,
            "candidate_recall_min": min(recalls) if recalls else None,
            "proposal_count_mean": statistics.fmean(proposals) if proposals else None,
            "proposal_count_max": max(proposals) if proposals else None,
            "silver_precision_lower_bound_macro": statistics.fmean(p_lbs) if p_lbs else None,
            "budget_saturation_count": sum(bool(item["budget_saturated"]) for item in values),
            "budget_drop_total": sum(int(item["dropped_by_budget"]) for item in values),
        }
    return output


def aggregate_phase2(
    *,
    cells: Sequence[Mapping[str, Any]],
    run_records: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = tuple(int(value) for value in config["seeds"])
    masks = tuple(config["masks"])
    expected_cells = len(selected_cases) * len(seeds) * len(masks)
    expected_runs = len(selected_cases) * len(seeds)
    evaluation = config["evaluation"]

    ready_runs = [record for record in run_records if record.get("status") == "READY"]
    recalls = [float(cell["candidate_recall"]) for cell in cells if cell.get("candidate_recall") is not None]
    proposals = [int(cell["a2_proposal_count"]) for cell in cells]
    p_lbs = [float(cell["silver_precision_lower_bound"]) for cell in cells if cell.get("silver_precision_lower_bound") is not None]
    compressions = [float(cell["compression_ratio"]) for cell in cells]
    saturation_count = sum(bool(cell["budget_saturated"]) for cell in cells)
    saturation_rate = saturation_count / len(cells) if cells else None
    budget_drop_total = sum(int(cell["dropped_by_budget"]) for cell in cells)

    complete_grid = len(cells) == expected_cells and len(ready_runs) == expected_runs
    all_recall_measurable = len(recalls) == expected_cells
    cell_recall_floor = float(evaluation["candidate_recall_each_cell_min"])
    macro_recall_floor = float(evaluation["candidate_recall_macro_min"])
    proposal_cap = int(evaluation["abductive_proposal_count_max"])
    saturation_cap = float(evaluation["budget_saturation_rate_max"])

    conditions = {
        "complete_grid": complete_grid,
        "all_candidate_recall_measurable": all_recall_measurable,
        "candidate_recall_each_cell": bool(recalls) and min(recalls) >= cell_recall_floor,
        "candidate_recall_macro": bool(recalls) and statistics.fmean(recalls) >= macro_recall_floor,
        "proposal_count_cap": bool(proposals) and max(proposals) <= proposal_cap,
        "budget_saturation_rate": saturation_rate is not None and saturation_rate <= saturation_cap,
        "leakage_checks_all_pass": len(cells) == expected_cells and all(bool(cell["leakage_checks_all_pass"]) for cell in cells),
    }
    passed = all(conditions.values())
    reason_codes = [name.upper() for name, value in conditions.items() if not value]

    return {
        "gate_id": evaluation["gate_id"],
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "reason_codes": reason_codes,
        "conditions": conditions,
        "required": {
            "expected_incidents": int(evaluation["expected_incidents"]),
            "expected_seeds": int(evaluation["expected_seeds"]),
            "expected_cells": expected_cells,
            "candidate_recall_each_cell_min": cell_recall_floor,
            "candidate_recall_macro_min": macro_recall_floor,
            "proposal_count_max": proposal_cap,
            "budget_saturation_rate_max": saturation_cap,
        },
        "observed": {
            "incident_count": len(selected_cases),
            "seed_count": len(seeds),
            "run_count": len(run_records),
            "ready_run_count": len(ready_runs),
            "cell_count": len(cells),
            "expected_cell_count": expected_cells,
            "candidate_recall_macro": statistics.fmean(recalls) if recalls else None,
            "candidate_recall_min": min(recalls) if recalls else None,
            "candidate_recall_max": max(recalls) if recalls else None,
            "proposal_count_mean": statistics.fmean(proposals) if proposals else None,
            "proposal_count_median": statistics.median(proposals) if proposals else None,
            "proposal_count_max": max(proposals) if proposals else None,
            "silver_precision_lower_bound_macro": statistics.fmean(p_lbs) if p_lbs else None,
            "silver_precision_lower_bound_min": min(p_lbs) if p_lbs else None,
            "compression_ratio_macro": statistics.fmean(compressions) if compressions else None,
            "compression_ratio_min": min(compressions) if compressions else None,
            "budget_saturation_count": saturation_count,
            "budget_saturation_rate": saturation_rate,
            "budget_drop_total": budget_drop_total,
        },
        "by_mask_ratio": _group_summary(cells, "mask_ratio"),
        "by_case": _group_summary(cells, "case"),
        "by_fault": _group_summary(cells, "fault"),
    }


def run_phase2(
    *,
    config_path: Path,
    raw_root: Path,
    output: Path,
    max_workers: int | None = None,
    keep_heavy_artifacts: bool = False,
) -> Path:
    config_path = config_path.expanduser().resolve()
    raw_root = raw_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise Phase2Error(f"refusing to overwrite existing output: {output}")
    config = _load_json(config_path)
    validate_phase2_config(config)
    base_config_path = (PROJECT_ROOT / str(config["base_experiment_config"])).resolve()
    base_config = _load_json(base_config_path)

    cases_index = raw_root / "cases.parquet"
    if not cases_index.is_file():
        raise Phase2Error(f"cases.parquet is missing under {raw_root}")
    selected_cases = select_phase2_cases(pd.read_parquet(cases_index), config)
    provenance_path = raw_root / ".logagent-source.json"
    if not provenance_path.is_file():
        raise Phase2Error("Phase-2 provenance is missing; run prepare_task_a_phase2_data.py")
    provenance = _load_json(provenance_path)
    if provenance.get("revision") != config["dataset"]["source_revision"]:
        raise Phase2Error("Phase-2 raw data revision differs from the frozen config")
    if provenance.get("selected_cases_sha256") != hashlib.sha256(
        json.dumps(selected_cases, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest():
        raise Phase2Error("selected-case manifest differs from the prepared raw data")

    output.mkdir(parents=True, exist_ok=False)
    (output / "selected_cases.json").write_text(
        json.dumps(selected_cases, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(config_path, output / "phase2_config.json")

    execution = config["execution"]
    worker_count = int(max_workers or execution["max_workers"])
    if worker_count <= 0:
        raise Phase2Error("max_workers must be positive")
    timeout_seconds = int(execution["per_run_timeout_seconds"])
    prune_heavy = bool(execution.get("prune_model_trace_artifacts", True)) and not keep_heavy_artifacts

    tasks: list[RunTask] = []
    for record in selected_cases:
        case_id = str(record["case"])
        for seed in config["seeds"]:
            tasks.append(
                RunTask(
                    case_record=record,
                    seed=int(seed),
                    config_path=output / "cell_configs" / case_id / f"seed-{seed}.json",
                    output_path=output / "runs" / case_id / f"seed-{seed}",
                    timeout_seconds=timeout_seconds,
                    prune_heavy=prune_heavy,
                )
            )

    os.environ["LOGAGENT_PHASE2_RAW_ROOT"] = str(raw_root)
    run_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_execute_task, task, base_config, config): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except subprocess.TimeoutExpired as exc:
                record = {
                    "status": "ERROR",
                    "case": task.case_record["case"],
                    "fault": task.case_record["fault"],
                    "seed": task.seed,
                    "returncode": None,
                    "detail": f"run timed out after {exc.timeout} seconds",
                    "cells": [],
                }
            except Exception as exc:  # Keep a machine-readable partial result.
                record = {
                    "status": "ERROR",
                    "case": task.case_record["case"],
                    "fault": task.case_record["fault"],
                    "seed": task.seed,
                    "returncode": None,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "cells": [],
                }
            run_records.append(record)
            print(
                f"[{len(run_records)}/{len(tasks)}] {record['case']} seed={record['seed']} "
                f"status={record['status']}",
                flush=True,
            )

    run_records.sort(key=lambda item: (str(item["case"]), int(item["seed"])))
    cells = [cell for record in run_records for cell in record.get("cells", ())]
    cells.sort(key=lambda item: (str(item["case"]), int(item["seed"]), float(item["mask_ratio"])))
    gate = aggregate_phase2(
        cells=cells,
        run_records=run_records,
        selected_cases=selected_cases,
        config=config,
    )

    data_manifest_sha256 = sha256_file(provenance_path)
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "research_task": "A_relation_recovery",
        "execution_phase": "multicase_multiseed_candidate_generalization",
        "dataset_revision": config["dataset"]["source_revision"],
        "selection_policy": config["dataset"]["case_selection"]["policy"],
        "selected_cases": selected_cases,
        "seeds": list(config["seeds"]),
        "masks": list(config["masks"]),
        "candidate_budget": dict(config["candidate_budget"]),
        "run_records": [
            {key: value for key, value in record.items() if key != "cells"}
            for record in run_records
        ],
        "gate": gate,
        "provenance": {
            "manifest_sha256": data_manifest_sha256,
            "selected_cases_sha256": provenance["selected_cases_sha256"],
            "verified_file_count": len(provenance["verified"]),
            "total_verified_bytes": provenance["total_verified_bytes"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "max_workers": worker_count,
            "heavy_model_traces_pruned": prune_heavy,
        },
        "scope_warning": (
            "Phase 2 remains limited to RCAEval RE2-TT runtime CALLS candidate recovery. "
            "Cells share incidents and nested mask ratios, so they are not independent "
            "samples for publication-level confidence intervals."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame.from_records(cells).to_csv(output / "cells.csv", index=False)
    pd.DataFrame.from_records(
        [{key: value for key, value in record.items() if key != "cells"} for record in run_records]
    ).to_json(output / "runs.json", orient="records", indent=2, force_ascii=False)
    print(f"Phase-2 gate: {gate['status']} -> {output}", flush=True)
    return output


__all__ = [
    "DEFAULT_PHASE2_CONFIG",
    "Phase2Error",
    "aggregate_phase2",
    "build_cell_config",
    "required_source_files",
    "run_phase2",
    "select_phase2_cases",
    "sha256_file",
    "validate_phase2_config",
    "write_verified_provenance",
]
