"""Compatibility wrapper for A3-R2 evaluator edge serialization.

Phase-2 evaluator artifacts can encode a relation triple either as a mapping
(``{"subject": ..., "predicate": ..., "object": ...}``) or as a JSON array
(``[subject, predicate, object]``).  The original R2 reader assumed mappings
only and failed before any scientific metric was produced.  This wrapper keeps
model-side feature computation unchanged and replaces only the evaluator-side
artifact decoder while R2 is executed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import task_a_phase3_r2 as _impl


EdgeKey = tuple[str, str, str]


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


def run_phase3_r2(**kwargs: Any) -> Path:
    """Execute R2 with the serialization-compatible evaluator reader."""

    original = _impl._evaluator_flags
    _impl._evaluator_flags = evaluator_flags
    try:
        return _impl.run_phase3_r2(**kwargs)
    finally:
        _impl._evaluator_flags = original


__all__ = [
    "decode_edge_key",
    "decode_edge_set",
    "evaluator_flags",
    "run_phase3_r2",
]
