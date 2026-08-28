"""Executable RCAEval cumulative relation-recovery smoke (v2)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

from .cli import (
    _build_bundle,
    _build_graph,
    _highest_degree_component,
    _load_backends,
    _prediction_frame,
    _verify_source_provenance,
    _write_json,
)
from .cumulative import CumulativeConfig, run_cumulative_suite
from .graph import CANONICAL_TRACE_COLUMNS, SilverGraph
from .masking import (
    EVIDENCE_LEVEL_L2,
    StructuralMaskResult,
    make_component_parent_dropped_mask,
    make_iid_parent_dropped_mask,
)
from .metrics import evaluate_recovery
from .onnx_deberta import (
    NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
    NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
    NLI_DEBERTA_V3_SMALL_REPO_ID,
    NLI_DEBERTA_V3_SMALL_REVISION,
)
from .rcaeval import IncidentBundle
from .recovery import (
    DEFAULT_RELATION_SPECS,
    InferenceContext,
    RelationSpec,
    build_typed_candidates,
)
from .runtime_context import build_runtime_pair_contexts


_STAGE_ROLES = {
    "A0": "observed_graph_baseline",
    "A1": "direct_evidence",
    "A2": "abductive_proposal_layer",
    "A3": "flat_directional_control",
    "A4": "runtime_context_directional_verifier",
    "A5": "psl_pruning_baseline",
}

DEFAULT_BUDGET_CONFIG_PATH = Path(
    "configs/experiment_rcaeval_smoke_v2_budget.json"
)


def _fingerprint(values: Iterable[str]) -> str:
    material = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _implementation_fingerprint(config_path: Path) -> str:
    package_dir = Path(__file__).resolve().parent
    paths = (
        package_dir / "masking.py",
        package_dir / "recovery.py",
        package_dir / "onnx_deberta.py",
        package_dir / "cumulative.py",
        package_dir / "runtime_context.py",
        package_dir / "cli_v2.py",
        config_path,
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _bind_backend_config(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    backend = config["optional_backends"]
    frozen = {
        "deberta_model": NLI_DEBERTA_V3_SMALL_REPO_ID,
        "deberta_revision": NLI_DEBERTA_V3_SMALL_REVISION,
        "deberta_artifact": NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
        "deberta_artifact_sha256": NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
    }
    mismatches = {
        key: (backend.get(key), expected)
        for key, expected in frozen.items()
        if backend.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"v2 backend config differs from frozen implementation: {mismatches}")

    configured_sha = str(backend["deberta_artifact_sha256"])
    if args.deberta_model_sha256 is None:
        args.deberta_model_sha256 = configured_sha
    elif str(args.deberta_model_sha256) != configured_sha:
        raise ValueError("--deberta-model-sha256 differs from the v2 config")

    configured_seed = int(backend["psl_seed"])
    if args.psl_seed is None:
        args.psl_seed = configured_seed
    elif int(args.psl_seed) != configured_seed:
        raise ValueError("--psl-seed differs from the v2 config")


def _unavailable_evaluation(result: Any) -> dict[str, Any]:
    return {
        "variant": result.variant,
        "status": result.status,
        "reason_code": result.reason_code,
        "candidate_recall": None,
        "masked_recall": None,
        "relation_masked_recall": None,
        "ranking": {"mrr": None, "hits": {}},
        "silver_precision_lower_bound": None,
        "activation": dict(result.activation),
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("v2 runner requires experiment config schema_version=2")
    if config.get("dataset", {}).get("dataset_id") != "rcaeval":
        raise ValueError("v2 runner accepts only the RCAEval contract")
    if config.get("pipeline", {}).get("mode") != "cumulative_a2_gated":
        raise ValueError("pipeline.mode must be cumulative_a2_gated")
    for mask in config.get("masks", ()):
        if mask.get("level") != EVIDENCE_LEVEL_L2:
            raise ValueError("v2 smoke currently requires L2_PARENT_DROPPED masks")
    return config


def _call_relation_specs() -> dict[str, RelationSpec]:
    return {"CALLS": DEFAULT_RELATION_SPECS["CALLS"]}


def _make_mask(
    graph: SilverGraph,
    specification: Mapping[str, Any],
) -> StructuralMaskResult:
    common = {
        "seed": int(specification.get("seed", 0)),
        "dataset_id": "rcaeval",
        "system_id": "train-ticket",
        "columns": CANONICAL_TRACE_COLUMNS,
    }
    if specification["kind"] == "iid":
        return make_iid_parent_dropped_mask(
            graph,
            fraction=float(specification["ratio"]),
            **common,
        )
    if specification["kind"] == "component_blackout":
        component_id = (
            _highest_degree_component(graph)
            if specification.get("selector") == "highest_total_degree"
            else str(specification["component_id"])
        )
        return make_component_parent_dropped_mask(
            graph,
            component_id=component_id,
            **common,
        )
    raise ValueError(f"unsupported v2 mask kind: {specification['kind']}")


def _model_entities(mask: StructuralMaskResult) -> tuple[dict[str, Any], ...]:
    """Create an inductive vocabulary from sanitized model traces only."""

    service_ids = sorted(mask.model.traces["service_id"].astype(str).unique())
    return tuple(
        {
            "entity_id": service_id,
            "canonical_name": service_id.rsplit(":", 1)[-1],
            "entity_type": "Service",
            "type_basis": "model_trace_partition",
            "type_confidence": 1.0,
        }
        for service_id in service_ids
    )


def _model_context(mask: StructuralMaskResult, incident_id: str) -> InferenceContext:
    allowed_columns = [
        "trace_id",
        "span_id",
        "parent_span_id",
        "service_id",
        "operation_name",
        "start_time_us",
        "end_time_us",
        "duration_us",
    ]
    available_columns = [
        column for column in allowed_columns if column in mask.model.traces.columns
    ]
    traces = mask.model.traces[available_columns]
    service_ids = {
        record["entity_id"] for record in _model_entities(mask)
    }
    observed = mask.model.observed_edges.loc[
        mask.model.observed_edges["subject"].astype(str).isin(service_ids)
        & mask.model.observed_edges["object"].astype(str).isin(service_ids)
    ]
    return InferenceContext(
        incident_id=incident_id,
        entities=_model_entities(mask),
        observed_edges=tuple(observed.to_dict(orient="records")),
        traces=tuple(traces.itertuples(index=False, name="TraceRecordV2")),
        evidence=(),
        decision_time=None,
    )


def _pair_context_frame(pair_contexts: Mapping[Any, Any]) -> pd.DataFrame:
    records = []
    for key in sorted(pair_contexts):
        context = pair_contexts[key]
        records.append(
            {
                "subject": key[0],
                "predicate": key[1],
                "object": key[2],
                "subject_label": context.subject_label,
                "object_label": context.object_label,
                "contextual_addendum": context.contextual_addendum,
                "provenance": list(context.provenance),
            }
        )
    return pd.DataFrame.from_records(records)


def _cumulative_config(config: Mapping[str, Any]) -> CumulativeConfig:
    thresholds = config["pipeline"]["thresholds"]
    return CumulativeConfig(
        a2_threshold=float(thresholds["a2"]),
        entailment_threshold=float(thresholds["forward_entailment"]),
        reverse_entailment_ceiling=float(thresholds["reverse_entailment_ceiling"]),
        direction_margin=float(thresholds["direction_margin"]),
        psl_threshold=float(thresholds["psl"]),
        include_null_parent=True,
    )


def _run_mask(
    *,
    output: Path,
    specification: Mapping[str, Any],
    graph: SilverGraph,
    bundle: IncidentBundle,
    config: Mapping[str, Any],
    deberta_backend: Any,
    psl_backend: Any,
    require_heavy: bool,
) -> dict[str, Any]:
    mask_id = str(specification["id"])
    mask = _make_mask(graph, specification)
    mask_root = output / "masks" / mask_id
    model_dir = mask_root / "model_input"
    private_dir = mask_root / "evaluator_private"
    prediction_dir = mask_root / "predictions"
    control_dir = mask_root / "diagnostic_controls"
    for path in (model_dir, private_dir, prediction_dir, control_dir):
        path.mkdir(parents=True, exist_ok=True)

    context = _model_context(mask, str(bundle.incident["incident_id"]))
    universe = build_typed_candidates(context, _call_relation_specs())
    pair_contexts = build_runtime_pair_contexts(
        context,
        universe,
        system_label=str(bundle.incident.get("system_name", "Train Ticket")),
    )
    suite = run_cumulative_suite(
        context,
        pair_contexts=pair_contexts,
        config=_cumulative_config(config),
        relation_specs=_call_relation_specs(),
        deberta_backend=deberta_backend,
        psl_backend=psl_backend,
        run_d0_control=False,
    )

    if require_heavy:
        unavailable = [
            variant
            for variant in ("A3", "A4", "A5")
            if suite.results[variant].status != "READY"
        ]
        if unavailable:
            raise RuntimeError(f"required v2 stages are unavailable: {unavailable}")

    # Public model artifacts contain only the sanitized mask partition.  v2
    # never writes the pre-mask model traces or graph to the run tree.
    mask.model.traces.to_parquet(model_dir / "traces.parquet", index=False)
    pd.DataFrame.from_records(context.entities).to_parquet(
        model_dir / "entities.parquet", index=False
    )
    pd.DataFrame.from_records(context.observed_edges).to_parquet(
        model_dir / "observed_edges.parquet", index=False
    )
    _pair_context_frame(pair_contexts).to_parquet(
        model_dir / "runtime_pair_context.parquet", index=False
    )
    observation_end = int(mask.model.traces["end_time_us"].max())
    model_incident = {
        "incident_id": bundle.incident["incident_id"],
        "dataset_id": bundle.incident["dataset_id"],
        "system_id": bundle.incident["system_id"],
        "observation_end_us": observation_end,
        "analysis_mode": "offline_topology_recovery",
    }
    _write_json(model_dir / "incident.json", model_incident)

    # Evaluator-only material is written after every model stage has finished.
    _write_json(private_dir / "mask_manifest.json", mask.evaluator_manifest.to_private_dict())
    evaluation: dict[str, Any] = {}
    statuses: dict[str, Any] = {}
    silver_records = graph.reference_edges.to_dict(orient="records")
    for variant, result in suite.results.items():
        ranking_scope = "typed_universe_U" if variant in {"A0", "A1"} else "a2_proposals_P2"
        statuses[variant] = {
            "status": result.status,
            "reason_code": result.reason_code,
            "research_valid": result.research_valid,
            "stage_role": _STAGE_ROLES[variant],
            "candidate_scope": ranking_scope,
            "ranking_scope": ranking_scope,
            "candidate_count": len(result.candidates),
            "prediction_count": len(result.predictions),
            "accepted_edge_count": len(result.accepted_edges),
            "unresolved_count": sum(
                prediction.decision != "accepted" for prediction in result.predictions
            ),
            "model_scored_count": result.activation.get("scored_candidate_count"),
            "nli_pair_count": result.activation.get("nli_pair_count"),
            "maximum_pair_tokens": result.activation.get("maximum_pair_tokens"),
            "truncation_count": result.activation.get("truncation_count"),
        }
        if result.predictions:
            _prediction_frame(result).to_parquet(
                prediction_dir / f"{variant}.parquet", index=False
            )
        evaluation[variant] = (
            evaluate_recovery(
                result,
                masked_edges=mask.evaluator_manifest.target_edges,
                silver_reference_edges=silver_records,
                all_reference_edges=silver_records,
            )
            if result.status == "READY"
            else _unavailable_evaluation(result)
        )
    for control, result in suite.controls.items():
        if result.predictions:
            _prediction_frame(result).to_parquet(
                control_dir / f"{control}.parquet", index=False
            )

    _write_json(private_dir / "evaluation.json", evaluation)
    _write_json(private_dir / "activation.json", suite.activation)
    _write_json(
        private_dir / "stage_activation.json",
        {variant: dict(result.activation) for variant, result in suite.results.items()},
    )
    _write_json(private_dir / "stage_gate.json", suite.gate)
    model_trace_services = set(mask.model.traces["service_id"].astype(str).unique())
    model_entity_services = {str(record["entity_id"]) for record in context.entities}
    leakage_checks = [
        {
            "check_id": "model_entities_equal_sanitized_trace_vocabulary",
            "expected": True,
            "observed": model_entity_services == model_trace_services,
            "passed": model_entity_services == model_trace_services,
            "verification_method": "exact_set_equality",
        },
        {
            "check_id": "model_incident_has_no_fault_or_injection_time",
            "expected": [],
            "observed": sorted(
                set(model_incident)
                & {"inject_time", "inject_time_us", "anomaly_time", "anomaly_time_us", "fault_type"}
            ),
            "passed": not (
                set(model_incident)
                & {"inject_time", "inject_time_us", "anomaly_time", "anomaly_time_us", "fault_type"}
            ),
            "verification_method": "forbidden_key_intersection",
        },
        {
            "check_id": "pre_mask_artifacts_absent",
            "expected": [],
            "observed": sorted(
                str(path.relative_to(mask_root))
                for path in mask_root.rglob("*")
                if path.is_file() and "pre_mask" in path.name
            ),
            "passed": not any(
                path.is_file() and "pre_mask" in path.name
                for path in mask_root.rglob("*")
            ),
            "verification_method": "run_tree_filename_scan",
        },
    ]
    return {
        "mask_id": mask_id,
        "kind": specification["kind"],
        "evidence_level": mask.evaluator_manifest.evidence_level,
        "target_count": mask.evaluator_manifest.target_count,
        "redacted_boundary_spans": mask.evaluator_manifest.redacted_boundary_spans,
        "component_id": mask.evaluator_manifest.component_id,
        "model_entity_count": len(context.entities),
        "model_observed_edge_count": len(context.observed_edges),
        "evaluation_universe_count": len(suite.evaluation_universe),
        "a2_proposal_count": len(suite.proposals),
        "variants": statuses,
        "diagnostics": suite.diagnostics,
        "stage_gate": suite.gate,
        "leakage_checks": leakage_checks,
        "evaluation": {
            variant: {
                "candidate_recall": values["candidate_recall"],
                "masked_recall": values["masked_recall"],
                "mrr": values["ranking"]["mrr"],
                "hits": values["ranking"]["hits"],
                "silver_precision_lower_bound": values["silver_precision_lower_bound"],
            }
            for variant, values in evaluation.items()
        },
    }


def run(args: argparse.Namespace) -> Path:
    config_path = args.config.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    config = _load_config(config_path)
    _bind_backend_config(args, config)
    _verify_source_provenance(raw_root, config)
    bundle = _build_bundle(raw_root, config)
    graph = _build_graph(bundle, config)
    output.mkdir(parents=True, exist_ok=False)

    deberta, psl, backend_diagnostics = _load_backends(args)
    selected = [
        specification
        for specification in config["masks"]
        if not args.mask or specification["id"] in args.mask
    ]
    if not selected:
        raise ValueError("no configured v2 mask matched --mask")
    summaries = []
    for specification in selected:
        print(f"running v2 mask {specification['id']}...", flush=True)
        summaries.append(
            _run_mask(
                output=output,
                specification=specification,
                graph=graph,
                bundle=bundle,
                config=config,
                deberta_backend=deberta,
                psl_backend=psl,
                require_heavy=args.require_heavy,
            )
        )
    if deberta is not None and hasattr(deberta, "cache_info"):
        backend_diagnostics["deberta_cache"] = asdict(deberta.cache_info())

    summary = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "implementation_sha256": _implementation_fingerprint(config_path),
        "incident_id": bundle.incident["incident_id"],
        "dataset_revision": config["dataset"]["source_revision"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **backend_diagnostics,
        },
        "split": {
            "reference_traces": len(graph.trace_split.reference_trace_ids),
            "model_traces": len(graph.trace_split.model_trace_ids),
            "trace_overlap": len(
                graph.trace_split.reference_trace_ids
                & graph.trace_split.model_trace_ids
            ),
            "assignment": config["partition"]["assignment"],
            "reference_trace_ids_sha256": _fingerprint(
                graph.trace_split.reference_trace_ids
            ),
            "model_trace_ids_sha256": _fingerprint(graph.trace_split.model_trace_ids),
        },
        "silver_graph": {
            "reference_edges": len(graph.reference_edges),
            "semantics": "evaluator-only held-out runtime CALLS; not causal gold",
        },
        "pipeline_contract": {
            "a3_a4_a5_candidates": "exactly A2 proposals",
            "operational_path": "A2 -> A4 -> A5",
            "paired_control": "A3 and A4 score the same P2 independently",
            "direction_decision": "forward entailment + reverse ceiling + margin",
            "a4_effect_scope": "composite runtime context; not hierarchy-only",
            "hierarchy_scope": "partial System-Service plus operation-string summaries",
            "unknown_levels": ["Application", "Instance", "Host", "Deployment"],
            "psl_scope": "fixed pruning baseline; not proof or coupled global inference",
            "ranking_comparability": (
                "A0/A1 rank over U; A2-A5 rank within P2; cross-scope MRR/Hits "
                "are not directly comparable"
            ),
        },
        "threshold_contract": {
            **dict(config["pipeline"]["thresholds"]),
            "a2_formula": "1 - exp(-supporting_trace_count)",
            "direction_decision": "forward >= threshold AND reverse <= ceiling AND forward-reverse >= margin",
            "threshold_origin": "FROZEN_CONFIG_V2",
            "ranking_tie_policy": "deterministic_candidate_key_order",
        },
        "statistical_scope": {
            "ci_status": "NOT_ESTIMABLE_SINGLE_INCIDENT_SEED_NESTED_MASKS",
            "iid_masks_are_nested": True,
        },
        "leakage_boundary": {
            "api_separated": True,
            "evaluator_access_ordered_after_model_stages": True,
            "process_isolation": False,
            "note": "evaluator-private material exists in the same Python process",
        },
        "masks": summaries,
        "scope_warning": (
            "This remains one incident and one seed per mask. L2 removes the opaque "
            "orphan token shortcut, but does not establish cross-system generalization "
            "or LLM RCA improvement."
        ),
    }
    _write_json(output / "summary.json", summary)
    print(f"complete: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BUDGET_CONFIG_PATH,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/rcaeval/smoke"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", action="append")
    parser.add_argument("--deberta-model-dir", type=Path)
    parser.add_argument(
        "--deberta-model-sha256",
        default=None,
    )
    parser.add_argument("--enable-psl", action="store_true")
    parser.add_argument("--psl-seed", type=int, default=None)
    parser.add_argument("--psl-compatibility-override", action="store_true")
    parser.add_argument("--require-heavy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
