"""RCAEval Task A phase-1 runner: IID20/IID40 bounded abduction only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import cli_v2
from .task_a import TaskAConfig, run_task_a_candidate_suite


DEFAULT_TASK_A_CONFIG_PATH = Path("configs/experiment_task_a_rcaeval.json")
_BASE_LOAD_CONFIG = cli_v2._load_config
_RUNTIME_CONTEXT_ENV = "LOGAGENT_TASK_A_MATERIALIZE_RUNTIME_CONTEXT"


def _materialize_runtime_context_requested() -> bool:
    """Return whether model-visible A3 context should be persisted.

    The default remains the lightweight A0-A2 contract.  Phase 3 may opt in so
    that ``cli_v2`` computes and writes pair contexts while A3-A5 themselves
    stay deferred.  This changes only model artifacts, not candidate selection.
    """

    return os.getenv(_RUNTIME_CONTEXT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _load_task_a_config(path: Path) -> dict[str, Any]:
    config = _BASE_LOAD_CONFIG(path)
    pipeline = config.get("pipeline", {})
    if pipeline.get("task") != "A_relation_recovery":
        raise ValueError("pipeline.task must be A_relation_recovery")
    if pipeline.get("phase") != "A0_A2_candidate_readiness":
        raise ValueError("pipeline.phase must be A0_A2_candidate_readiness")
    masks = tuple(config.get("masks", ()))
    ratios = sorted(float(mask.get("ratio", -1.0)) for mask in masks)
    if len(masks) != 2 or ratios != [0.2, 0.4]:
        raise ValueError("Task A phase 1 requires exactly IID20 and IID40 masks")
    if any(mask.get("kind") != "iid" for mask in masks):
        raise ValueError("Task A phase 1 accepts IID masks only")
    if "candidate_budget" not in pipeline:
        raise ValueError("pipeline.candidate_budget is required")
    return config


def _task_a_config(config: Mapping[str, Any]) -> TaskAConfig:
    pipeline = config["pipeline"]
    thresholds = pipeline["thresholds"]
    budget = pipeline["candidate_budget"]
    return TaskAConfig(
        a2_threshold=float(thresholds["a2"]),
        entailment_threshold=float(thresholds["forward_entailment"]),
        reverse_entailment_ceiling=float(thresholds["reverse_entailment_ceiling"]),
        direction_margin=float(thresholds["direction_margin"]),
        psl_threshold=float(thresholds["psl"]),
        include_null_parent=bool(pipeline.get("include_null_parent", True)),
        max_abductive_proposals=int(budget["max_abductive_proposals"]),
        max_per_subject=int(budget["max_per_subject"]),
        max_per_object=int(budget["max_per_object"]),
        min_supporting_traces=int(budget["min_supporting_traces"]),
        min_boundary_count=int(budget["min_boundary_count"]),
    )


def _task_a_implementation_fingerprint(config_path: Path) -> str:
    package_dir = Path(__file__).resolve().parent
    paths = (
        package_dir / "masking.py",
        package_dir / "recovery.py",
        package_dir / "graph.py",
        package_dir / "rcaeval.py",
        package_dir / "cli_v2.py",
        package_dir / "task_a.py",
        package_dir / "cli_task_a.py",
        config_path,
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _metric_value(metric: Any) -> float | None:
    """Read a scalar from the structured metric objects emitted by metrics.py."""

    if metric is None or isinstance(metric, bool):
        return None
    if isinstance(metric, (int, float)):
        return float(metric)
    if not isinstance(metric, Mapping):
        return None
    for key in ("value", "recall", "lower_bound"):
        if key not in metric:
            continue
        value = _metric_value(metric[key])
        if value is not None:
            return value
    return None


def _task_a_gate(mask_summary: Mapping[str, Any], max_proposals: int) -> dict[str, Any]:
    a2_evaluation = mask_summary.get("evaluation", {}).get("A2", {})
    recall = _metric_value(a2_evaluation.get("candidate_recall"))
    diagnostics = mask_summary.get("diagnostics", {})
    abductive_count = diagnostics.get(
        "a2_abductive_proposal_count",
        mask_summary.get("a2_proposal_count"),
    )
    leakage_checks = tuple(mask_summary.get("leakage_checks", ()))
    leakage_passed = bool(leakage_checks) and all(
        bool(check.get("passed")) for check in leakage_checks
    )
    measurable = recall is not None and abductive_count is not None
    passed = bool(
        measurable
        and recall >= 0.90
        and int(abductive_count) <= max_proposals
        and leakage_passed
    )
    return {
        "gate_id": "D2_BOUNDED_CANDIDATE_RECALL",
        "status": "PASS" if passed else ("FAIL" if measurable else "PENDING"),
        "passed": passed if measurable else None,
        "observed": {
            "candidate_recall": recall,
            "abductive_proposal_count": abductive_count,
            "leakage_checks_all_pass": leakage_passed,
        },
        "required": {
            "candidate_recall_min": 0.90,
            "abductive_proposal_count_max": max_proposals,
            "leakage_checks_all_pass": True,
        },
    }


def _rewrite_summary(output: Path, config_path: Path) -> None:
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = _load_task_a_config(config_path)
    summary["research_task"] = "A_relation_recovery"
    summary["execution_phase"] = "A0_A2_candidate_readiness"
    summary["pipeline_contract"] = {
        "relation_scope": ["CALLS"],
        "active_stages": ["A0", "A1", "A2"],
        "deferred_stages": ["A3", "A4", "A5"],
        "reason_for_deferral": (
            "Freeze candidate recall and candidate volume before NLI/PSL tuning."
        ),
        "model_visible_inputs": [
            "sanitized model traces",
            "observed CALLS graph",
            "typed Service entities",
        ],
        "runtime_pair_context_materialized": _materialize_runtime_context_requested(),
        "evaluator_only": [
            "masked target edges",
            "reference graph",
            "mask manifest",
            "fault/root labels",
        ],
    }
    summary["candidate_budget_contract"] = dict(
        config["pipeline"]["candidate_budget"]
    )
    summary["threshold_contract"] = {
        "a2": config["pipeline"]["thresholds"]["a2"],
        "a2_formula": "1 - exp(-supporting_trace_count)",
        "ranking": config["pipeline"]["candidate_budget"]["ranking"],
        "threshold_origin": "FROZEN_TASK_A_PHASE1_CONFIG",
    }
    max_proposals = int(
        config["pipeline"]["candidate_budget"]["max_abductive_proposals"]
    )
    mask_gates = {}
    for mask_summary in summary.get("masks", ()):
        gate = _task_a_gate(mask_summary, max_proposals)
        mask_summary["stage_gate"] = gate
        mask_gates[str(mask_summary["mask_id"])] = gate
        gate_path = (
            output
            / "masks"
            / str(mask_summary["mask_id"])
            / "evaluator_private"
            / "stage_gate.json"
        )
        gate_path.write_text(
            json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    measurable = all(gate["passed"] is not None for gate in mask_gates.values())
    overall_passed = bool(mask_gates) and measurable and all(
        bool(gate["passed"]) for gate in mask_gates.values()
    )
    summary["decision_gate"] = {
        "gate_id": "D2_BOUNDED_CANDIDATE_RECALL",
        "status": (
            "PASS" if overall_passed else ("FAIL" if measurable else "PENDING")
        ),
        "passed": overall_passed if measurable else None,
        "masks": mask_gates,
    }
    summary["scope_warning"] = (
        "Task A phase 1 evaluates bounded relation-candidate recovery on one "
        "RCAEval incident and one nested seed at IID20/IID40. It does not test "
        "DeBERTa, PSL, causal paths, Task B, or LLM RCA improvement."
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Path:
    heavy_requested = bool(
        args.require_heavy
        or args.deberta_model_dir is not None
        or args.enable_psl
        or args.psl_compatibility_override
    )
    if heavy_requested:
        raise ValueError(
            "Task A phase 1 intentionally defers A3-A5; remove DeBERTa/PSL options"
        )
    # Heavy artifacts are deliberately not loaded during candidate-readiness runs.
    args.deberta_model_dir = None
    args.enable_psl = False
    args.require_heavy = False

    original = (
        cli_v2._load_config,
        cli_v2._cumulative_config,
        cli_v2.run_cumulative_suite,
        cli_v2._implementation_fingerprint,
        cli_v2.build_runtime_pair_contexts,
    )
    cli_v2._load_config = _load_task_a_config
    cli_v2._cumulative_config = _task_a_config
    cli_v2.run_cumulative_suite = run_task_a_candidate_suite
    cli_v2._implementation_fingerprint = _task_a_implementation_fingerprint
    if not _materialize_runtime_context_requested():
        cli_v2.build_runtime_pair_contexts = lambda *_args, **_kwargs: {}
    try:
        output = cli_v2.run(args)
    finally:
        (
            cli_v2._load_config,
            cli_v2._cumulative_config,
            cli_v2.run_cumulative_suite,
            cli_v2._implementation_fingerprint,
            cli_v2.build_runtime_pair_contexts,
        ) = original
    _rewrite_summary(output, args.config.expanduser().resolve())
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = cli_v2.build_parser()
    parser.description = __doc__
    parser.set_defaults(config=DEFAULT_TASK_A_CONFIG_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
