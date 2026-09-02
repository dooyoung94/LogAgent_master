"""Frozen contracts and leakage-safe helpers for Task A Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .onnx_deberta import (
    NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
    NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
    NLI_DEBERTA_V3_SMALL_REPO_ID,
    NLI_DEBERTA_V3_SMALL_REVISION,
)


DEFAULT_PHASE3_CONFIG = Path("configs/experiment_task_a_rcaeval_phase3.json")
RUNTIME_CONTEXT_MAX_LINES = 8
RUNTIME_CONTEXT_LINE_CHAR_LIMIT = 144
RUNTIME_CONTEXT_TOTAL_CHAR_LIMIT = 960

REQUIRED_CELL_COLUMNS = {
    "case", "fault", "seed", "mask_id", "mask_ratio",
    "a2_proposal_count", "candidate_recall", "mrr_within_a2",
    "silver_precision_lower_bound", "run_summary",
    "leakage_checks_all_pass", "cell_gate_passed",
}

class Phase3Error(RuntimeError):
    """Raised when the frozen A3 contract or a Phase-2 artifact is invalid."""


@dataclass(frozen=True)
class TriStateThresholds:
    corroborate_entailment_min: float = 0.67
    contradict_probability_min: float = 0.67
    evidence_margin_min: float = 0.05
    contradiction_margin_min: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.corroborate_entailment_min,
            self.contradict_probability_min,
            self.evidence_margin_min,
            self.contradiction_margin_min,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("tri-state thresholds must be finite probabilities")


@dataclass(frozen=True)
class ShortlistPolicy:
    retention_fraction: float
    minimum_keep: int
    nli_weight: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.retention_fraction) or not 0.0 < self.retention_fraction <= 1.0:
            raise ValueError("retention_fraction must be in (0, 1]")
        if (
            isinstance(self.minimum_keep, bool)
            or not isinstance(self.minimum_keep, int)
            or self.minimum_keep <= 0
        ):
            raise ValueError("minimum_keep must be a positive integer")
        if not math.isfinite(self.nli_weight) or not 0.0 <= self.nli_weight <= 1.0:
            raise ValueError("nli_weight must be in [0, 1]")


@dataclass(frozen=True)
class NliEvidence:
    state: str
    evidence_score: float
    forward_entailment: float
    reverse_entailment: float
    forward_contradiction: float
    forward_neutral: float
    direction_margin: float
    label_margin: float


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase3Error(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase3Error(f"invalid JSON in {path}: {exc}") from exc


def _json_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise Phase3Error(f"invalid {field_name} JSON: {value!r}") from exc
    if not isinstance(parsed, Mapping):
        raise Phase3Error(f"{field_name} must decode to an object")
    return dict(parsed)


def _json_sequence(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise Phase3Error(f"invalid {field_name} JSON: {value!r}") from exc
    if not isinstance(parsed, list):
        raise Phase3Error(f"{field_name} must decode to an array")
    return tuple(parsed)


def _edge_key(record: Mapping[str, Any] | Sequence[Any]) -> tuple[str, str, str]:
    if isinstance(record, Mapping):
        return str(record["subject"]), str(record["predicate"]), str(record["object"])
    if len(record) != 3:
        raise Phase3Error(f"edge must contain three fields: {record!r}")
    return str(record[0]), str(record[1]), str(record[2])


def _metric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        for key in ("value", "recall", "lower_bound"):
            if key in value:
                parsed = _metric_value(value[key])
                if parsed is not None:
                    return parsed
    return None


def validate_phase3_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise Phase3Error("Phase-3 config schema_version must be 1")
    if config.get("experiment_id") != "rcaeval-task-a-phase3-tristate-evidence":
        raise Phase3Error("unexpected Phase-3 experiment_id")
    split = config.get("calibration_split", {})
    if split.get("method") != "sha256_case_order":
        raise Phase3Error("calibration_split.method must be sha256_case_order")
    if int(split.get("calibration_incidents", 0)) != 2:
        raise Phase3Error("the frozen Phase-3 split requires exactly two calibration incidents")
    if split.get("salt") != "task-a-phase3-calibration":
        raise Phase3Error("unexpected Phase-3 calibration salt")
    if split.get("selection_uses_evaluator_metrics") is not False:
        raise Phase3Error("calibration case selection must be evaluator-label independent")

    backend = config.get("backend", {})
    frozen = {
        "repo_id": NLI_DEBERTA_V3_SMALL_REPO_ID,
        "revision": NLI_DEBERTA_V3_SMALL_REVISION,
        "onnx_filename": NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
        "onnx_sha256": NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
        "batch_size": 1,
    }
    mismatches = {
        key: (backend.get(key), expected)
        for key, expected in frozen.items()
        if backend.get(key) != expected
    }
    if mismatches:
        raise Phase3Error(f"backend differs from the frozen A3 contract: {mismatches}")

    TriStateThresholds(**dict(config.get("tri_state", {})))
    search = config.get("policy_search", {})
    for field in ("retention_fractions", "minimum_keep", "nli_weights"):
        values = tuple(search.get(field, ()))
        if not values:
            raise Phase3Error(f"policy_search.{field} cannot be empty")
    if any(float(value) <= 0.0 or float(value) > 1.0 for value in search["retention_fractions"]):
        raise Phase3Error("retention fractions must be in (0,1]")
    if any(int(value) <= 0 for value in search["minimum_keep"]):
        raise Phase3Error("minimum_keep values must be positive")
    if any(float(value) <= 0.0 or float(value) > 1.0 for value in search["nli_weights"]):
        raise Phase3Error("proposed A3 nli_weights must be in (0,1]")


def stable_case_split(
    cases: Iterable[str],
    *,
    revision: str,
    calibration_incidents: int,
) -> tuple[tuple[str, ...], tuple[str, ...], Mapping[str, str]]:
    unique = sorted(set(str(case) for case in cases))
    if not 0 < calibration_incidents < len(unique):
        raise Phase3Error("calibration split must leave at least one held-out incident")
    hashes = {
        case: hashlib.sha256(
            f"{revision}|task-a-phase3-calibration|{case}".encode("utf-8")
        ).hexdigest()
        for case in unique
    }
    ordered = sorted(unique, key=lambda case: (hashes[case], case))
    calibration = tuple(ordered[:calibration_incidents])
    heldout = tuple(ordered[calibration_incidents:])
    return calibration, heldout, hashes


__all__ = [
    "DEFAULT_PHASE3_CONFIG", "NliEvidence", "Phase3Error",
    "REQUIRED_CELL_COLUMNS", "RUNTIME_CONTEXT_LINE_CHAR_LIMIT",
    "RUNTIME_CONTEXT_MAX_LINES", "RUNTIME_CONTEXT_TOTAL_CHAR_LIMIT",
    "ShortlistPolicy", "TriStateThresholds", "stable_case_split",
    "validate_phase3_config",
]
