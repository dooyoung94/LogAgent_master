"""Task A Phase 3: tri-state DeBERTa evidence over frozen A2 proposals.

A2 proposals remain immutable. DeBERTa contributes non-destructive
corroborates/ambiguous/contradicts evidence and a calibrated shortlist.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .onnx_deberta import OnnxDebertaNLIBackend
from .phase3_contract import (
    DEFAULT_PHASE3_CONFIG,
    NliEvidence,
    Phase3Error,
    REQUIRED_CELL_COLUMNS,
    ShortlistPolicy,
    TriStateThresholds,
    _load_json,
    stable_case_split,
    validate_phase3_config,
)
from .phase3_nli import (
    _compact_runtime_context,
    _load_cell_candidates,
    _load_evaluator_sets,
    _score_all_candidates,
    classify_tri_state,
)
from .phase3_policy import (
    _baseline_aggregate,
    _delta,
    _evaluate_policy_on_cells,
    apply_policy,
    evaluate_shortlist,
    select_calibrated_policy,
)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _render_phase3_report(summary: Mapping[str, Any]) -> str:
    gate = summary["gate"]
    calibration = summary["calibration_selection"]
    policy = summary["selected_policy"]
    baseline = summary["baseline"]["heldout"]
    proposed = summary["proposed_a3"]["heldout"]
    delta = summary["proposed_a3"]["heldout_delta_vs_a2"]
    matched = summary["matched_budget_a2_control"][
        "heldout_delta_a3_minus_control"
    ]
    state = summary["state_distribution"]
    reasons = ", ".join(gate["reason_codes"]) or "없음"
    lines = [
        "# Task A Phase 3 결과 — Tri-state DeBERTa 가설 검증",
        "",
        f"- 최종 Gate: **{gate['status']}**",
        f"- Calibration 선택 상태: **{calibration['status']}**",
        f"- Calibration feasible 정책 수: **{calibration['feasible_policy_count']}**",
        f"- Held-out Cell: **{summary['split']['heldout_cells']}**",
        f"- 미통과 조건: `{reasons}`",
        "",
        "## 선택 정책",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| retention_fraction | {_fmt(policy['retention_fraction'])} |",
        f"| minimum_keep | {policy['minimum_keep']} |",
        f"| nli_weight | {_fmt(policy['nli_weight'])} |",
        f"| calibration_feasible | {calibration['selected_policy_feasible']} |",
        "",
        "## Held-out 결과",
        "",
        "| 지표 | A2 전체 후보 | A3 Shortlist | 변화 |",
        "|---|---:|---:|---:|",
        f"| Candidate Recall Macro | {_fmt(baseline['recall_macro'])} | {_fmt(proposed['recall_macro'])} | {_fmt(delta['recall_macro'])} |",
        f"| Candidate Recall Minimum | {_fmt(baseline['recall_min'])} | {_fmt(proposed['recall_min'])} | - |",
        f"| 후보 수 평균 | {_fmt(baseline['selected_count_mean'], 2)} | {_fmt(proposed['selected_count_mean'], 2)} | {_fmt(delta['selected_count_mean'], 2)} |",
        f"| P-LB Macro | {_fmt(baseline['silver_precision_lower_bound_macro'])} | {_fmt(proposed['silver_precision_lower_bound_macro'])} | {_fmt(delta['silver_precision_lower_bound_macro'])} |",
        f"| MRR Macro | {_fmt(baseline['mrr_macro'])} | {_fmt(proposed['mrr_macro'])} | {_fmt(delta['mrr_macro'])} |",
        "",
        "## 동일 후보 수 A2-only 대조군 대비",
        "",
        "| 지표 | A3 - A2 matched-budget |",
        "|---|---:|",
        f"| Recall Macro | {_fmt(matched['recall_macro'])} |",
        f"| P-LB Macro | {_fmt(matched['silver_precision_lower_bound_macro'])} |",
        f"| MRR Macro | {_fmt(matched['mrr_macro'])} |",
        "",
        "## NLI 상태 분포",
        "",
        f"- corroborates: {state['corroborates']}",
        f"- ambiguous: {state['ambiguous']}",
        f"- contradicts: {state['contradicts']}",
        "",
        "## 해석",
        "",
    ]
    if gate["passed"]:
        lines.extend(
            [
                "- 사전 정의된 Calibration 조건을 통과한 정책이 Held-out Gate도 통과했다.",
                "- A3는 A2 후보를 보존한 상태에서 DeBERTa를 hard veto가 아닌 재랭킹 증거로 사용했다.",
            ]
        )
    else:
        lines.extend(
            [
                "- 사전 정의된 과학적 Gate를 통과하지 못했으므로 A3 개선 주장을 하지 않는다.",
                "- 결과 파일은 실패 원인 분석과 다음 설계 변경을 위한 진단 산출물이다.",
                "- Calibration에서 feasible 정책이 없더라도, Calibration-only 기준으로 고정한 진단용 정책을 Held-out에 적용하여 선택 편향 없이 실패 양상을 기록했다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 범위 제한",
            "",
            f"- {summary['claim_limit']}",
            "- `CALLS` 복원은 runtime 구조 관계이며 causal `CAUSES` 복원을 의미하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_phase3(
    *,
    phase2_root: Path,
    model_dir: Path,
    output: Path,
    config_path: Path = DEFAULT_PHASE3_CONFIG,
) -> Path:
    phase2_root = phase2_root.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if output.exists():
        raise Phase3Error(f"refusing to overwrite existing output: {output}")
    config = _load_json(config_path)
    validate_phase3_config(config)

    cells_path = phase2_root / "cells.csv"
    if not cells_path.is_file():
        raise Phase3Error(f"Phase-2 cells.csv is missing: {cells_path}")
    frame = pd.read_csv(cells_path)
    missing = sorted(REQUIRED_CELL_COLUMNS.difference(frame.columns))
    if missing:
        raise Phase3Error(f"Phase-2 cells.csv is missing columns: {missing}")
    if frame.duplicated(["case", "seed", "mask_id"]).any():
        raise Phase3Error("Phase-2 cells.csv contains duplicate evaluation cells")

    revision = str(config["dataset_revision"])
    calibration_cases, heldout_cases, split_hashes = stable_case_split(
        frame["case"].astype(str),
        revision=revision,
        calibration_incidents=int(
            config["calibration_split"]["calibration_incidents"]
        ),
    )
    role_by_case = {case: "calibration" for case in calibration_cases}
    role_by_case.update({case: "heldout" for case in heldout_cases})

    phase2_summary = _load_json(phase2_root / "summary.json")
    phase2_contract = config["phase2_contract"]
    if not phase2_summary.get("gate", {}).get("passed"):
        raise Phase3Error("Phase-2 D3 gate must pass before A3")
    if len(frame) != int(phase2_contract["required_cells"]):
        raise Phase3Error(
            f"Phase-2 cell count must be {phase2_contract['required_cells']}"
        )
    if frame["case"].nunique() != int(
        phase2_contract["required_incidents"]
    ):
        raise Phase3Error(
            "Phase-2 incident count differs from the A3 contract"
        )
    if set(frame["seed"].astype(int)) != {
        int(value) for value in phase2_contract["required_seeds"]
    }:
        raise Phase3Error("Phase-2 seed grid differs from the A3 contract")
    if set(frame["mask_ratio"].astype(float)) != {
        float(value) for value in phase2_contract["required_mask_ratios"]
    }:
        raise Phase3Error(
            "Phase-2 mask ratios differ from the A3 contract"
        )
    if int(frame["a2_proposal_count"].max()) > int(
        phase2_contract["a2_max_proposals"]
    ):
        raise Phase3Error(
            "Phase-2 proposal count exceeds the frozen A2 budget"
        )
    truthy = {"true", "1", "yes", "y"}
    if not frame["leakage_checks_all_pass"].astype(str).str.lower().isin(
        truthy
    ).all():
        raise Phase3Error("Phase-2 leakage checks did not all pass")
    if not frame["cell_gate_passed"].astype(str).str.lower().isin(
        truthy
    ).all():
        raise Phase3Error("Phase-2 cell gates did not all pass")

    cells: list[dict[str, Any]] = []
    source_rows: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        candidates = _load_cell_candidates(phase2_root, row)
        key = (
            str(row["case"]),
            int(row["seed"]),
            str(row["mask_id"]),
        )
        source_rows[key] = row
        cells.append(
            {
                "case": key[0],
                "fault": str(row["fault"]),
                "seed": key[1],
                "mask_id": key[2],
                "mask_ratio": float(row["mask_ratio"]),
                "role": role_by_case[key[0]],
                "a2_count": int(row["a2_proposal_count"]),
                "a2_recall": float(row["candidate_recall"]),
                "a2_mrr": float(row["mrr_within_a2"]),
                "a2_p_lb": float(row["silver_precision_lower_bound"]),
                "candidates": candidates,
            }
        )

    backend_cfg = config["backend"]
    backend = OnnxDebertaNLIBackend(
        model_dir,
        onnx_filename=str(backend_cfg["onnx_filename"]),
        expected_sha256=str(backend_cfg["onnx_sha256"]),
        revision=str(backend_cfg["revision"]),
        batch_size=1,
        performance_mode=False,
    )
    availability = backend.availability()
    if availability.status != "READY" or not availability.research_valid:
        raise Phase3Error(
            f"frozen DeBERTa backend is unavailable: {availability.status} "
            f"{availability.reason_code} {availability.detail}"
        )
    thresholds = TriStateThresholds(**dict(config["tri_state"]))
    scoring_diagnostics = _score_all_candidates(
        cells,
        backend=backend,
        thresholds=thresholds,
    )

    calibration_cells = [
        cell for cell in cells if cell["role"] == "calibration"
    ]
    heldout_cells = [cell for cell in cells if cell["role"] == "heldout"]

    for cell in calibration_cells:
        row = source_rows[
            (cell["case"], cell["seed"], cell["mask_id"])
        ]
        keys = {
            (item["subject"], item["predicate"], item["object"])
            for item in cell["candidates"]
        }
        cell["targets"], cell["silver"] = _load_evaluator_sets(
            phase2_root,
            row,
            keys,
        )
    policy, calibration_grid = select_calibrated_policy(
        calibration_cells,
        search=config["policy_search"],
        calibration_gate=config["calibration_gate"],
        allow_diagnostic_fallback=True,
    )
    selected_calibration_rows = [
        row for row in calibration_grid if bool(row.get("selected"))
    ]
    if len(selected_calibration_rows) != 1:
        raise Phase3Error(
            "calibration grid must identify exactly one selected policy"
        )
    selected_calibration_row = selected_calibration_rows[0]
    calibration_policy_feasible = bool(
        selected_calibration_row.get("feasible")
    )
    calibration_selection_status = str(
        selected_calibration_row.get("selection_status", "UNKNOWN")
    )
    control_policy = ShortlistPolicy(
        policy.retention_fraction,
        policy.minimum_keep,
        0.0,
    )

    for cell in heldout_cells:
        row = source_rows[
            (cell["case"], cell["seed"], cell["mask_id"])
        ]
        keys = {
            (item["subject"], item["predicate"], item["object"])
            for item in cell["candidates"]
        }
        cell["targets"], cell["silver"] = _load_evaluator_sets(
            phase2_root,
            row,
            keys,
        )

    proposed_rows, proposed_all = _evaluate_policy_on_cells(cells, policy)
    control_rows, control_all = _evaluate_policy_on_cells(
        cells,
        control_policy,
    )
    proposed_cal_rows, proposed_cal = _evaluate_policy_on_cells(
        calibration_cells,
        policy,
    )
    proposed_held_rows, proposed_held = _evaluate_policy_on_cells(
        heldout_cells,
        policy,
    )
    _control_held_rows, control_held = _evaluate_policy_on_cells(
        heldout_cells,
        control_policy,
    )
    baseline_all = _baseline_aggregate(cells)
    baseline_cal = _baseline_aggregate(calibration_cells)
    baseline_held = _baseline_aggregate(heldout_cells)

    gate_cfg = config["heldout_gate"]
    count_ratio = (
        proposed_held["selected_count_mean"]
        / baseline_held["selected_count_mean"]
    )
    matched_budget_delta = _delta(proposed_held, control_held)
    matched_budget_additive_gain = (
        matched_budget_delta["silver_precision_lower_bound_macro"] > 1e-12
        or matched_budget_delta["mrr_macro"] > 1e-12
    )
    conditions = {
        "calibration_policy_feasible": calibration_policy_feasible,
        "heldout_complete": len(proposed_held_rows) == len(heldout_cells),
        "recall_macro": proposed_held["recall_macro"]
        >= float(gate_cfg["recall_macro_min"]),
        "recall_pooled": proposed_held["recall_pooled"]
        >= float(gate_cfg["recall_pooled_min"]),
        "recall_each_cell": proposed_held["recall_min"]
        >= float(gate_cfg["recall_each_cell_min"]),
        "candidate_count_reduced": count_ratio
        <= float(gate_cfg["selected_count_ratio_max"]),
        "p_lb_improved": (
            proposed_held["silver_precision_lower_bound_macro"]
            >= baseline_held["silver_precision_lower_bound_macro"]
            + float(gate_cfg["p_lb_macro_delta_min"])
        ),
        "mrr_improved": (
            proposed_held["mrr_macro"]
            >= baseline_held["mrr_macro"]
            + float(gate_cfg["mrr_macro_delta_min"])
        ),
        "nli_weight_active": policy.nli_weight > 0.0,
        "matched_budget_recall_noninferior": (
            matched_budget_delta["recall_macro"]
            >= -float(gate_cfg["matched_budget_recall_tolerance"])
        ),
        "matched_budget_p_lb_noninferior": (
            matched_budget_delta["silver_precision_lower_bound_macro"]
            >= float(gate_cfg["matched_budget_p_lb_delta_min"])
        ),
        "matched_budget_mrr_noninferior": (
            matched_budget_delta["mrr_macro"]
            >= float(gate_cfg["matched_budget_mrr_delta_min"])
        ),
        "matched_budget_additive_gain": (
            matched_budget_additive_gain
            if bool(
                gate_cfg.get(
                    "matched_budget_additive_gain_required",
                    True,
                )
            )
            else True
        ),
        "a2_evidence_preserved": sum(
            len(cell["candidates"]) for cell in cells
        )
        == int(frame["a2_proposal_count"].sum()),
    }
    passed = all(conditions.values())

    detailed_scores: list[dict[str, Any]] = []
    policy_by_cell = {
        (row["case"], int(row["seed"]), row["mask_id"]): row
        for row in proposed_rows
    }
    for cell in cells:
        selected, scored = apply_policy(cell["candidates"], policy)
        del selected
        key = (cell["case"], cell["seed"], cell["mask_id"])
        for record in scored:
            detailed_scores.append(
                {
                    "incident_id": hashlib.sha256(
                        (
                            f"{revision}|task-a-phase3-output|"
                            f"{cell['case']}"
                        ).encode("utf-8")
                    ).hexdigest()[:24],
                    "case": cell["case"],
                    "fault": cell["fault"],
                    "role": cell["role"],
                    "seed": cell["seed"],
                    "mask_id": cell["mask_id"],
                    "mask_ratio": cell["mask_ratio"],
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"],
                    "a2_score": record["a2_score"],
                    "proposal_rank": record["proposal_rank"],
                    "supporting_traces": record["supporting_traces"],
                    "boundary_spans": record["boundary_spans"],
                    "reverse_supporting_traces": record[
                        "reverse_supporting_traces"
                    ],
                    "reverse_boundary_spans": record[
                        "reverse_boundary_spans"
                    ],
                    "nli_state": record["nli_state"],
                    "nli_evidence_score": record["nli_evidence_score"],
                    "nli_forward_entailment": record[
                        "nli_forward_entailment"
                    ],
                    "nli_reverse_entailment": record[
                        "nli_reverse_entailment"
                    ],
                    "nli_forward_contradiction": record[
                        "nli_forward_contradiction"
                    ],
                    "nli_forward_neutral": record[
                        "nli_forward_neutral"
                    ],
                    "nli_direction_margin": record[
                        "nli_direction_margin"
                    ],
                    "nli_label_margin": record["nli_label_margin"],
                    "a3_score": record["a3_score"],
                    "selected": record["selected"],
                    "is_masked_target": (
                        record["subject"],
                        record["predicate"],
                        record["object"],
                    )
                    in cell["targets"],
                    "is_silver_matched": (
                        record["subject"],
                        record["predicate"],
                        record["object"],
                    )
                    in cell["silver"],
                    "cell_selected_count": policy_by_cell[key][
                        "selected_count"
                    ],
                }
            )

    output.mkdir(parents=True, exist_ok=False)
    model_dir_out = output / "model_output"
    evaluator_dir_out = output / "evaluator_private"
    published_dir = output / "published"
    model_dir_out.mkdir(parents=True, exist_ok=False)
    evaluator_dir_out.mkdir(parents=True, exist_ok=False)
    published_dir.mkdir(parents=True, exist_ok=False)
    detailed_frame = pd.DataFrame.from_records(detailed_scores)
    evaluator_only_columns = {
        "case",
        "fault",
        "role",
        "is_masked_target",
        "is_silver_matched",
    }
    model_columns = [
        column
        for column in detailed_frame.columns
        if column not in evaluator_only_columns
    ]
    detailed_frame[model_columns].to_parquet(
        model_dir_out / "a3_candidate_evidence.parquet",
        index=False,
    )
    detailed_frame.to_parquet(
        evaluator_dir_out / "a3_candidate_analysis.parquet",
        index=False,
    )
    proposed_frame = pd.DataFrame.from_records(proposed_rows)
    control_frame = pd.DataFrame.from_records(control_rows)
    calibration_frame = pd.DataFrame.from_records(calibration_grid)
    proposed_cal_frame = pd.DataFrame.from_records(proposed_cal_rows)
    proposed_held_frame = pd.DataFrame.from_records(proposed_held_rows)
    proposed_frame.to_csv(evaluator_dir_out / "a3_cells.csv", index=False)
    control_frame.to_csv(
        evaluator_dir_out / "a2_budget_control_cells.csv",
        index=False,
    )
    calibration_frame.to_csv(
        evaluator_dir_out / "calibration_grid.csv",
        index=False,
    )
    proposed_cal_frame.to_csv(
        evaluator_dir_out / "calibration_cells.csv",
        index=False,
    )
    proposed_held_frame.to_csv(
        evaluator_dir_out / "heldout_cells.csv",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "PASS" if passed else "FAIL",
        "gate_id": gate_cfg["gate_id"],
        "config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "phase2_cells_sha256": hashlib.sha256(
            cells_path.read_bytes()
        ).hexdigest(),
        "split": {
            "method": config["calibration_split"]["method"],
            "calibration_cases": list(calibration_cases),
            "heldout_cases": list(heldout_cases),
            "case_hashes": split_hashes,
            "calibration_cells": len(calibration_cells),
            "heldout_cells": len(heldout_cells),
        },
        "backend": {
            **backend.metadata(),
            "availability": asdict(availability),
            "cache": asdict(backend.cache_info()),
            "scoring": scoring_diagnostics,
        },
        "tri_state": asdict(thresholds),
        "calibration_selection": {
            "status": calibration_selection_status,
            "selected_policy_feasible": calibration_policy_feasible,
            "feasible_policy_count": sum(
                bool(row.get("feasible")) for row in calibration_grid
            ),
            "searched_policy_count": len(calibration_grid),
            "selected_policy_row": selected_calibration_row,
            "labels_used": "calibration incidents only",
            "heldout_labels_used_for_selection": False,
        },
        "selected_policy": asdict(policy),
        "matched_budget_control_policy": asdict(control_policy),
        "baseline": {
            "all": baseline_all,
            "calibration": baseline_cal,
            "heldout": baseline_held,
        },
        "proposed_a3": {
            "all": proposed_all,
            "calibration": proposed_cal,
            "heldout": proposed_held,
            "heldout_delta_vs_a2": _delta(
                proposed_held,
                baseline_held,
            ),
        },
        "matched_budget_a2_control": {
            "all": control_all,
            "heldout": control_held,
            "heldout_delta_a3_minus_control": matched_budget_delta,
        },
        "gate": {
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "conditions": conditions,
            "reason_codes": [
                name.upper() for name, value in conditions.items() if not value
            ],
            "required": dict(gate_cfg),
            "observed_selected_count_ratio": count_ratio,
        },
        "state_distribution": {
            state: sum(
                record["nli_state"] == state for record in detailed_scores
            )
            for state in ("corroborates", "ambiguous", "contradicts")
        },
        "leakage_boundary": {
            "model_scoring_before_any_evaluator_labels": True,
            "heldout_labels_loaded_after_policy_freeze": True,
            "model_output_separated_from_evaluator_private": True,
        },
        "claim_limit": (
            "A3 evaluates tri-state NLI-assisted shortlisting of frozen A2 "
            "CALLS proposals on six RCAEval TrainTicket incidents. It does "
            "not establish causal-edge recovery or RCA/LLM improvement."
        ),
    }
    summary_text = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    (output / "summary.json").write_text(
        summary_text,
        encoding="utf-8",
    )

    (published_dir / "task_a_phase3_results.json").write_text(
        summary_text,
        encoding="utf-8",
    )
    (published_dir / "task_a_phase3_results.md").write_text(
        _render_phase3_report(summary),
        encoding="utf-8",
    )
    proposed_held_frame.to_csv(
        published_dir / "task_a_phase3_heldout_cells.csv",
        index=False,
    )
    proposed_cal_frame.to_csv(
        published_dir / "task_a_phase3_calibration_cells.csv",
        index=False,
    )
    calibration_frame.to_csv(
        published_dir / "task_a_phase3_policy_grid.csv",
        index=False,
    )
    (published_dir / "task_a_phase3_status.txt").write_text(
        f"{summary['status']}\n"
        f"gate={summary['gate_id']}\n"
        f"calibration={calibration_selection_status}\n"
        f"reason_codes={','.join(summary['gate']['reason_codes'])}\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFAULT_PHASE3_CONFIG",
    "NliEvidence",
    "Phase3Error",
    "ShortlistPolicy",
    "TriStateThresholds",
    "_compact_runtime_context",
    "apply_policy",
    "classify_tri_state",
    "evaluate_shortlist",
    "run_phase3",
    "select_calibrated_policy",
    "stable_case_split",
    "validate_phase3_config",
]
