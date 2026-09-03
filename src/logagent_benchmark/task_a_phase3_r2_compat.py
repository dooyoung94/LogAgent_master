"""Compatibility and handoff wrapper for Task A Phase 3-R2.

Phase-2 evaluator artifacts can encode a relation triple either as a mapping
(``{"subject": ..., "predicate": ..., "object": ...}``) or as a JSON array
(``[subject, predicate, object]``).  The original R2 reader assumed mappings
only and failed before any scientific metric was produced.

The wrapper also adds an opaque incident token to model-visible output and
materializes one evaluator-private all-candidate table.  This avoids repeating
the expensive trace feature extraction when the next evidence-specific NLI
stage is executed.  Raw case/fault labels remain evaluator-private.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import task_a_phase3_r2 as _impl


EdgeKey = tuple[str, str, str]
CANDIDATE_KEY = ("subject", "predicate", "object")


def decode_edge_key(item: Any, *, field_name: str) -> EdgeKey:
    """Decode one edge from either object or three-element array form."""

    if isinstance(item, Mapping):
        required = ("subject", "predicate", "object")
        missing = [name for name in required if name not in item]
        if missing:
            raise _impl.Phase3R2Error(
                f"{field_name} edge mapping is missing fields: {missing}"
            )
        values = tuple(item[name] for name in required)
    elif isinstance(item, Sequence) and not isinstance(
        item, (str, bytes, bytearray)
    ):
        if len(item) != 3:
            raise _impl.Phase3R2Error(
                f"{field_name} edge array must contain exactly three values"
            )
        values = tuple(item)
    else:
        raise _impl.Phase3R2Error(
            f"{field_name} edge must be a mapping or three-element array"
        )

    if any(value is None for value in values):
        raise _impl.Phase3R2Error(f"{field_name} edge contains null values")
    return tuple(str(value) for value in values)  # type: ignore[return-value]


def decode_edge_set(items: Any, *, field_name: str) -> set[EdgeKey]:
    """Decode a JSON edge collection with explicit schema errors."""

    if items is None:
        return set()
    if not isinstance(items, Sequence) or isinstance(
        items, (str, bytes, bytearray)
    ):
        raise _impl.Phase3R2Error(f"{field_name} must be a JSON array")
    return {
        decode_edge_key(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(items)
    }


def evaluator_flags(
    cell_root: Path, candidate_keys: set[EdgeKey]
) -> pd.DataFrame:
    """Load evaluator labels after model-visible R2 features are frozen."""

    private = cell_root / "evaluator_private"
    manifest = json.loads(
        (private / "mask_manifest.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (private / "evaluation.json").read_text(encoding="utf-8")
    )

    targets = decode_edge_set(
        manifest.get("target_edges", ()), field_name="mask_manifest.target_edges"
    )
    try:
        precision = evaluation["A2"]["silver_precision_lower_bound"]
    except (KeyError, TypeError) as exc:
        raise _impl.Phase3R2Error(
            "evaluation.A2.silver_precision_lower_bound is missing"
        ) from exc
    unverified = decode_edge_set(
        precision.get("unverified_edges", ()),
        field_name="evaluation.A2.silver_precision_lower_bound.unverified_edges",
    )

    missing_targets = targets.difference(candidate_keys)
    if missing_targets:
        raise _impl.Phase3R2Error(
            "A2 candidate set no longer contains masked targets: "
            f"{sorted(missing_targets)[:3]}"
        )
    unknown_unverified = unverified.difference(candidate_keys)
    if unknown_unverified:
        raise _impl.Phase3R2Error(
            "evaluation contains unverified edges outside the A2 candidate set: "
            f"{sorted(unknown_unverified)[:3]}"
        )

    silver = candidate_keys - unverified
    expected_silver = precision.get("silver_matched_count")
    if expected_silver is not None and len(silver) != int(expected_silver):
        raise _impl.Phase3R2Error(
            "reconstructed silver-matched count differs from evaluation: "
            f"expected={expected_silver} observed={len(silver)}"
        )

    return pd.DataFrame.from_records(
        [
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_id,
                "is_masked_target": key in targets,
                "is_silver_matched": key in silver,
            }
            for key in sorted(candidate_keys)
            for subject, predicate, object_id in (key,)
        ]
    )


def incident_token(case_id: Any) -> str:
    """Return a stable opaque case identifier safe for model-side joins."""

    digest = hashlib.sha256(f"task-a-r2|{case_id}".encode("utf-8")).hexdigest()
    return f"incident:{digest[:24]}"


def _materialize_all_candidate_analysis(
    *, output: Path, phase2_root: Path
) -> None:
    """Join frozen model features to evaluator flags without recomputing traces."""

    model_path = output / "model_output" / "a3_r2_operational_features.parquet"
    summary_path = output / "published" / "task_a_phase3_r2_results.json"
    if not model_path.is_file() or not summary_path.is_file():
        raise _impl.Phase3R2Error("R2 output is incomplete for the NLI handoff")

    model = pd.read_parquet(model_path)
    required_model = {
        "incident_token",
        "seed",
        "mask_id",
        *CANDIDATE_KEY,
    }
    missing = sorted(required_model.difference(model.columns))
    if missing:
        raise _impl.Phase3R2Error(
            f"R2 model output lacks handoff columns: {missing}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration_cases = set(summary["split"]["calibration_cases"])
    heldout_cases = set(summary["split"]["heldout_cases"])
    cells = pd.read_csv(phase2_root / "cells.csv")
    records: list[pd.DataFrame] = []
    token_manifest: list[dict[str, Any]] = []

    for row in cells.to_dict(orient="records"):
        case_id = str(row["case"])
        token = incident_token(case_id)
        subset = model.loc[
            model["incident_token"].astype(str).eq(token)
            & pd.to_numeric(model["seed"], errors="raise").eq(int(row["seed"]))
            & model["mask_id"].astype(str).eq(str(row["mask_id"]))
        ].copy()
        if len(subset) != int(row["a2_proposal_count"]):
            raise _impl.Phase3R2Error(
                "R2 handoff candidate count mismatch for "
                f"{case_id} seed={row['seed']} mask={row['mask_id']}: "
                f"expected={row['a2_proposal_count']} observed={len(subset)}"
            )
        candidate_keys = {
            tuple(map(str, values))
            for values in subset[list(CANDIDATE_KEY)].itertuples(
                index=False, name=None
            )
        }
        cell_root = _impl._cell_root(phase2_root, row)
        flags = evaluator_flags(cell_root, candidate_keys)
        subset = subset.merge(
            flags, on=list(CANDIDATE_KEY), how="inner", validate="one_to_one"
        )
        subset["case"] = case_id
        subset["fault"] = str(row["fault"])
        subset["role"] = (
            "calibration"
            if case_id in calibration_cases
            else "heldout"
            if case_id in heldout_cases
            else "unknown"
        )
        if subset["role"].eq("unknown").any():
            raise _impl.Phase3R2Error(
                f"case is absent from frozen calibration/heldout split: {case_id}"
            )
        records.append(subset)
        token_manifest.append(
            {
                "incident_token": token,
                "case": case_id,
                "fault": str(row["fault"]),
                "seed": int(row["seed"]),
                "mask_id": str(row["mask_id"]),
                "role": str(subset.iloc[0]["role"]),
                "candidate_count": len(subset),
            }
        )

    analysis = pd.concat(records, ignore_index=True)
    if len(analysis) != int(cells["a2_proposal_count"].sum()):
        raise _impl.Phase3R2Error("all-candidate handoff row count changed")
    if analysis.duplicated(
        ["incident_token", "seed", "mask_id", *CANDIDATE_KEY]
    ).any():
        raise _impl.Phase3R2Error("duplicate rows in all-candidate R2 handoff")

    private = output / "evaluator_private"
    private.mkdir(parents=True, exist_ok=True)
    analysis.to_parquet(private / "all_candidate_analysis.parquet", index=False)
    (private / "incident_token_manifest.json").write_text(
        json.dumps(token_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def run_phase3_r2(**kwargs: Any) -> Path:
    """Execute R2 with compatible decoding and an R3-ready handoff."""

    original_flags = _impl._evaluator_flags
    original_process = _impl._process_cell

    def process_with_token(
        phase2_root: Path, row: Mapping[str, Any], role: str
    ) -> tuple[tuple[str, int, str], pd.DataFrame, dict[str, Any]]:
        key, frame, diagnostics = original_process(phase2_root, row, role)
        token = incident_token(row["case"])
        frame = frame.copy()
        frame["incident_token"] = token
        diagnostics = dict(diagnostics)
        diagnostics["incident_token"] = token
        return key, frame, diagnostics

    _impl._evaluator_flags = evaluator_flags
    _impl._process_cell = process_with_token
    try:
        output = _impl.run_phase3_r2(**kwargs)
        _materialize_all_candidate_analysis(
            output=output,
            phase2_root=Path(kwargs["phase2_root"]).expanduser().resolve(),
        )
        return output
    finally:
        _impl._evaluator_flags = original_flags
        _impl._process_cell = original_process


__all__ = [
    "decode_edge_key",
    "decode_edge_set",
    "evaluator_flags",
    "incident_token",
    "run_phase3_r2",
]
