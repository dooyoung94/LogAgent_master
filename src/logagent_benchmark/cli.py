"""Executable RCAEval smoke pipeline.

The command keeps model inputs and evaluator-only artifacts in separate
directories.  Relation recovery receives neither the held-out graph nor the
private mask manifest; those are loaded only by the evaluation functions after
inference has completed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

from .graph import CANONICAL_TRACE_COLUMNS, SilverGraph, build_heldout_silver_graph
from .masking import (
    StructuralMaskResult,
    make_component_blackout,
    make_iid_mask,
)
from .metrics import evaluate_recovery
from .rcaeval import IncidentBundle, convert_rcaeval_case
from .recovery import (
    AblationConfig,
    DEFAULT_RELATION_SPECS,
    InferenceContext,
    RelationSpec,
    RecoveryResult,
    run_ablation_suite,
)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("experiment config schema_version must be 1")
    if config.get("dataset", {}).get("dataset_id") != "rcaeval":
        raise ValueError("this runner accepts only the RCAEval experiment contract")
    return config


def _verify_source_provenance(raw_root: Path, config: Mapping[str, Any]) -> None:
    provenance_path = raw_root / ".logagent-source.json"
    if not provenance_path.is_file():
        raise ValueError(
            "raw profile provenance is missing; acquire it with tools/datasets.py"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = config["dataset"]["source_revision"]
    if provenance.get("revision") != expected:
        raise ValueError(
            f"source revision mismatch: expected {expected}, "
            f"got {provenance.get('revision')}"
        )
    verified = provenance.get("verified", {})
    required = {
        "cases.parquet",
        f"{config['dataset']['raw_case']}/inject_time.txt",
        f"{config['dataset']['raw_case']}/metrics.parquet",
        f"{config['dataset']['raw_case']}/logs.parquet",
        f"{config['dataset']['raw_case']}/traces.parquet",
    }
    missing = sorted(required.difference(verified))
    if missing:
        raise ValueError(f"source provenance lacks verified files: {missing}")
    for relative_name in sorted(required):
        path = (raw_root / relative_name).resolve()
        try:
            path.relative_to(raw_root.resolve())
        except ValueError as exc:
            raise ValueError(f"unsafe source path: {relative_name}") from exc
        if not path.is_file():
            raise ValueError(f"verified source file is missing: {relative_name}")
        record = verified[relative_name]
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"source byte count changed: {relative_name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        expected_checksum = str(record["checksum"]).removeprefix("sha256:")
        if digest.hexdigest() != expected_checksum:
            raise ValueError(f"source checksum changed: {relative_name}")


def _build_bundle(raw_root: Path, config: Mapping[str, Any]) -> IncidentBundle:
    dataset = config["dataset"]
    case_dir = raw_root / dataset["raw_case"]
    return convert_rcaeval_case(
        case_dir,
        cases_index_path=raw_root / "cases.parquet",
        dataset_revision=dataset["source_revision"],
        incident_id=dataset["incident_id"],
    )


def _build_graph(bundle: IncidentBundle, config: Mapping[str, Any]) -> SilverGraph:
    return build_heldout_silver_graph(
        bundle.canonical_traces,
        revision=config["dataset"]["source_revision"],
        incident_id=str(bundle.incident["incident_id"]),
        inject_time_us=int(bundle.incident["anomaly_time_us"]),
        dataset_id=str(bundle.incident["dataset_id"]),
        system_id=str(bundle.incident["system_id"]),
        reference_ratio=float(config["partition"]["reference_percent"]) / 100.0,
        columns=CANONICAL_TRACE_COLUMNS,
    )


def _eligible_reference_keys(graph: SilverGraph) -> set[tuple[str, str, str]]:
    reference = graph.reference_edges
    if "attestation" in reference:
        reference = reference.loc[reference["attestation"].eq("A")]
    observed = {
        tuple(map(str, row))
        for row in graph.observed_edges[["subject", "predicate", "object"]].itertuples(
            index=False, name=None
        )
    }
    return {
        tuple(map(str, row))
        for row in reference[["subject", "predicate", "object"]].itertuples(
            index=False, name=None
        )
    } & observed


def _highest_degree_component(graph: SilverGraph) -> str:
    degree: Counter[str] = Counter()
    for subject, _predicate, object_id in _eligible_reference_keys(graph):
        degree[subject] += 1
        degree[object_id] += 1
    if not degree:
        raise ValueError("no attested component is available for blackout")
    return sorted(degree, key=lambda item: (-degree[item], item))[0]


def _make_mask(
    graph: SilverGraph,
    specification: Mapping[str, Any],
) -> StructuralMaskResult:
    kind = specification["kind"]
    common = {
        "seed": int(specification.get("seed", 0)),
        "dataset_id": "rcaeval",
        "system_id": "train-ticket",
        "columns": CANONICAL_TRACE_COLUMNS,
    }
    if kind == "iid":
        return make_iid_mask(
            graph,
            fraction=float(specification["ratio"]),
            **common,
        )
    if kind == "component_blackout":
        component_id = (
            _highest_degree_component(graph)
            if specification.get("selector") == "highest_total_degree"
            else str(specification["component_id"])
        )
        return make_component_blackout(
            graph,
            component_id=component_id,
            **common,
        )
    raise ValueError(f"unsupported mask kind: {kind}")


def _call_relation_specs() -> dict[str, RelationSpec]:
    return {"CALLS": DEFAULT_RELATION_SPECS["CALLS"]}


def _service_ids(entities: pd.DataFrame) -> list[str]:
    allowed = DEFAULT_RELATION_SPECS["CALLS"].domain_types
    selected = entities.loc[
        entities["entity_type"].astype(str).str.upper().isin(allowed), "entity_id"
    ]
    return sorted(selected.astype(str).unique())


def _trace_cooccurrence_evidence(
    traces: pd.DataFrame,
    service_ids: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Create label-blind, pair-local English evidence for DeBERTa.

    The statistic is deliberately non-directional.  Directional reconstruction
    remains the responsibility of A2 and the later PSL stage.
    """

    counts: Counter[tuple[str, str]] = Counter()
    per_trace = traces.groupby("trace_id", sort=False)["service_id"].unique()
    for values in per_trace:
        present = sorted(map(str, values))
        for subject in present:
            for object_id in present:
                if subject != object_id:
                    counts[(subject, object_id)] += 1

    records: list[dict[str, Any]] = []
    services = list(service_ids)
    for subject in services:
        for object_id in services:
            if subject == object_id:
                continue
            count = counts[(subject, object_id)]
            if count:
                statement = (
                    f"The services {subject} and {object_id} co-occurred in "
                    f"{count} whole distributed traces. Co-occurrence alone does "
                    "not reveal caller direction."
                )
            else:
                statement = (
                    f"No whole distributed trace in the model partition contained "
                    f"both services {subject} and {object_id}."
                )
            records.append(
                {
                    "evidence_id": f"trace-cooccurrence:{subject}|{object_id}",
                    "candidate_subject": subject,
                    "candidate_object": object_id,
                    "statement": statement,
                    "cooccurrence_trace_count": count,
                    "directional": False,
                }
            )
    return tuple(records)


def _model_context(
    bundle: IncidentBundle,
    mask: StructuralMaskResult,
) -> InferenceContext:
    traces = mask.model.traces[
        [
            "trace_id",
            "span_id",
            "parent_span_id",
            "service_id",
            "start_time_us",
            "end_time_us",
            "duration_us",
        ]
    ]
    trace_records = tuple(traces.itertuples(index=False, name="TraceRecord"))
    entity_records = tuple(bundle.entities.to_dict(orient="records"))
    observed_records = tuple(mask.model.observed_edges.to_dict(orient="records"))
    evidence = _trace_cooccurrence_evidence(traces, _service_ids(bundle.entities))
    return InferenceContext(
        incident_id=str(bundle.incident["incident_id"]),
        entities=entity_records,
        observed_edges=observed_records,
        traces=trace_records,
        evidence=evidence,
        decision_time=bundle.incident["anomaly_time_us"],
    )


def _load_backends(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    deberta = None
    psl = None
    diagnostics: dict[str, Any] = {}
    model_dir = args.deberta_model_dir or os.environ.get(
        "LOGAGENT_DEBERTA_MODEL_DIR"
    )
    if model_dir:
        try:
            from .onnx_deberta import OnnxDebertaNLIBackend

            deberta = OnnxDebertaNLIBackend(
                model_dir,
                expected_sha256=args.deberta_model_sha256,
            )
            diagnostics["deberta_requested"] = True
            availability = deberta.availability()
            diagnostics["deberta_availability"] = availability.status
            diagnostics["deberta_metadata"] = deberta.metadata()
            if availability.status == "READY":
                diagnostics["deberta_direction_contrast"] = deberta.direction_contrast(
                    (
                        "A span owned by ts-order-service is the parent of a span "
                        "owned by ts-payment-service."
                    ),
                    "ts-order-service calls ts-payment-service.",
                    "ts-payment-service calls ts-order-service.",
                ).to_dict()
                diagnostics["deberta_batch_composition"] = (
                    deberta.batch_composition_contrast(
                        (
                            "A span owned by ts-order-service is the parent of a "
                            "span owned by ts-payment-service.",
                            "ts-order-service calls ts-payment-service.",
                        ),
                        (
                            "A separate service emitted an unrelated database span.",
                            "The separate service uses a database.",
                        ),
                    ).to_dict()
                )
        except ImportError as exc:
            diagnostics["deberta_import_error"] = str(exc)
    else:
        diagnostics["deberta_requested"] = False

    if args.enable_psl:
        try:
            from .psl_backend import PslPythonBackend

            candidate_backend = PslPythonBackend(random_seed=args.psl_seed)
            availability = candidate_backend.availability()
            metadata = {}
            if availability.detail:
                try:
                    metadata = json.loads(availability.detail)
                except json.JSONDecodeError:
                    metadata = {"detail": availability.detail}
            override_detected = bool(metadata.get("compatibility_override"))
            diagnostics["psl_requested"] = True
            diagnostics["psl_availability"] = availability.status
            diagnostics["psl_runtime"] = metadata
            if override_detected and not args.psl_compatibility_override:
                diagnostics["psl_blocked_reason"] = (
                    "PSL_COMPATIBILITY_OVERRIDE_REQUIRES_EXPLICIT_FLAG"
                )
            else:
                psl = candidate_backend
        except ImportError as exc:
            diagnostics["psl_import_error"] = str(exc)
    else:
        diagnostics["psl_requested"] = False
    return deberta, psl, diagnostics


def _prediction_frame(result: RecoveryResult) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "variant": result.variant,
                "subject": item.subject,
                "predicate": item.predicate,
                "object": item.object,
                "score": item.score,
                "decision": item.decision,
                "evidence_ids_json": json.dumps(item.evidence_ids),
                "stage_scores_json": json.dumps(dict(item.stage_scores), sort_keys=True),
                "reason_codes_json": json.dumps(item.reason_codes),
            }
            for item in result.predictions
        ]
    )


def _write_graph_artifacts(
    output: Path,
    graph: SilverGraph,
    bundle: IncidentBundle,
) -> None:
    private = output / "graph" / "evaluator_private"
    model = output / "graph" / "model_base"
    private.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    graph.reference_edges.to_parquet(private / "reference_edges.parquet", index=False)
    graph.trace_split.reference.to_parquet(private / "reference_traces.parquet", index=False)
    _write_json(private / "labels.json", bundle.evaluator_labels)
    graph.observed_edges.to_parquet(model / "observed_edges.parquet", index=False)
    graph.trace_split.model.to_parquet(model / "model_traces.parquet", index=False)


def _run_mask(
    *,
    output: Path,
    specification: Mapping[str, Any],
    graph: SilverGraph,
    bundle: IncidentBundle,
    variants: Iterable[str],
    deberta_backend: Any,
    psl_backend: Any,
    require_variants: Iterable[str],
) -> dict[str, Any]:
    mask_id = str(specification["id"])
    mask = _make_mask(graph, specification)
    mask_root = output / "masks" / mask_id
    model_dir = mask_root / "model_input"
    evaluator_dir = mask_root / "evaluator_private"
    predictions_dir = mask_root / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    mask.model.traces.to_parquet(model_dir / "traces.parquet", index=False)
    mask.model.observed_edges.to_parquet(
        model_dir / "observed_edges.parquet", index=False
    )
    context = _model_context(bundle, mask)
    pd.DataFrame.from_records(context.evidence).to_parquet(
        model_dir / "candidate_evidence.parquet", index=False
    )
    _write_json(
        model_dir / "incident.json",
        {
            "incident_id": bundle.incident["incident_id"],
            "dataset_id": bundle.incident["dataset_id"],
            "system_id": bundle.incident["system_id"],
            "anomaly_time_us": bundle.incident["anomaly_time_us"],
            "mask_condition": {
                "kind": specification["kind"],
                "evidence_level": specification["level"],
            },
        },
    )

    suite = run_ablation_suite(
        context,
        variants=tuple(variants),
        relation_specs=_call_relation_specs(),
        config=AblationConfig(),
        deberta_backend=deberta_backend,
        psl_backend=psl_backend,
        require_variants=tuple(require_variants),
    )

    # The target list first enters here, after every inference variant has
    # completed.  It is never an InferenceContext field or recovery argument.
    private_manifest = mask.evaluator_manifest.to_private_dict()
    _write_json(evaluator_dir / "mask_manifest.json", private_manifest)
    evaluation: dict[str, Any] = {}
    statuses: dict[str, Any] = {}
    for variant, result in suite.results.items():
        statuses[variant] = {
            "status": result.status,
            "reason_code": result.reason_code,
            "research_valid": result.research_valid,
            "accepted_edge_count": len(result.accepted_edges),
        }
        if result.predictions:
            _prediction_frame(result).to_parquet(
                predictions_dir / f"{variant}.parquet", index=False
            )
        evaluation[variant] = evaluate_recovery(
            result,
            masked_edges=mask.evaluator_manifest.target_edges,
            silver_reference_edges=graph.reference_edges.to_dict(orient="records"),
            all_reference_edges=graph.reference_edges.to_dict(orient="records"),
        )
    _write_json(evaluator_dir / "evaluation.json", evaluation)
    _write_json(evaluator_dir / "activation.json", suite.activation)
    _write_json(evaluator_dir / "d3_gate.json", suite.gate)

    return {
        "mask_id": mask_id,
        "kind": specification["kind"],
        "target_count": mask.evaluator_manifest.target_count,
        "redacted_boundary_spans": mask.evaluator_manifest.redacted_boundary_spans,
        "component_id": mask.evaluator_manifest.component_id,
        "observed_edge_count": len(mask.model.observed_edges),
        "candidate_count": len(suite.candidates),
        "variants": statuses,
        "d3_gate": suite.gate,
        "evaluation": {
            variant: {
                "masked_recall": values["masked_recall"],
                "mrr": values["ranking"]["mrr"],
                "hits": values["ranking"]["hits"],
                "silver_precision_lower_bound": values[
                    "silver_precision_lower_bound"
                ],
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
    _verify_source_provenance(raw_root, config)
    bundle = _build_bundle(raw_root, config)
    graph = _build_graph(bundle, config)
    output.mkdir(parents=True, exist_ok=False)
    bundle.write(output / "standardized")
    _write_graph_artifacts(output, graph, bundle)

    deberta, psl, backend_diagnostics = _load_backends(args)
    selected = [
        specification
        for specification in config["masks"]
        if not args.mask or specification["id"] in args.mask
    ]
    if not selected:
        raise ValueError("no configured mask matched --mask")
    require_variants = tuple(config["optional_backends"]["deberta_required_for"])
    if not args.require_heavy:
        require_variants = ()

    mask_summaries = []
    for specification in selected:
        print(f"running mask {specification['id']}...", flush=True)
        mask_summaries.append(
            _run_mask(
                output=output,
                specification=specification,
                graph=graph,
                bundle=bundle,
                variants=config["ablations"],
                deberta_backend=deberta,
                psl_backend=psl,
                require_variants=require_variants,
            )
        )

    if deberta is not None and hasattr(deberta, "cache_info"):
        backend_diagnostics["deberta_cache"] = asdict(deberta.cache_info())

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "incident_id": bundle.incident["incident_id"],
        "dataset_revision": config["dataset"]["source_revision"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **backend_diagnostics,
        },
        "standardization": {
            "entities": len(bundle.entities),
            "metric_observations": len(bundle.metrics),
            "log_events": len(bundle.logs),
            "spans": len(bundle.canonical_traces),
        },
        "split": {
            "reference_traces": len(graph.trace_split.reference_trace_ids),
            "model_traces": len(graph.trace_split.model_trace_ids),
            "reference_spans": len(graph.trace_split.reference),
            "model_spans": len(graph.trace_split.model),
            "trace_overlap": len(
                graph.trace_split.reference_trace_ids
                & graph.trace_split.model_trace_ids
            ),
        },
        "silver_graph": {
            "reference_edges": len(graph.reference_edges),
            "model_base_edges": len(graph.observed_edges),
            "reference_attestation": graph.reference_edges["attestation"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "reference_nonroot_parent_coverage": graph.reference_stats.nonroot_parent_coverage,
            "model_nonroot_parent_coverage": graph.model_stats.nonroot_parent_coverage,
            "semantics": "held-out runtime CALLS; not causal or deployment gold",
        },
        "masks": mask_summaries,
        "scope_warning": (
            "The auth-service source case is an ETL/leakage smoke. Its injected "
            "service is isolated in the cross-service CALLS graph, so this run "
            "is not primary RCA-path evidence."
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
        default=Path("configs/experiment_rcaeval_smoke.json"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/rcaeval/smoke"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mask",
        action="append",
        help="Run only a configured mask ID; repeat for more than one",
    )
    parser.add_argument("--deberta-model-dir", type=Path)
    parser.add_argument(
        "--deberta-model-sha256",
        default="03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae",
    )
    parser.add_argument("--enable-psl", action="store_true")
    parser.add_argument("--psl-seed", type=int, default=7)
    parser.add_argument("--psl-compatibility-override", action="store_true")
    parser.add_argument(
        "--require-heavy",
        action="store_true",
        help="Fail D3 availability gate when A3/A4/A5 are not READY",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
