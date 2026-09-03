"""Task A Phase 4 PSL v2: direct confirmation with explicit abstention.

A2 remains the candidate generator.  PSL v2 is intentionally narrower than
v1: it may confirm a runtime ``CALLS`` candidate only when at least one
canonical direct telemetry channel is present.  Weak trace counts, direction
heuristics, operation similarity, endpoint compatibility, and graph-role
scores are retained for diagnostics only and cannot change confirmation.

Every candidate is preserved.  Candidates that do not satisfy the direct
evidence policy are emitted as ``ABSTAIN``; they are never emitted as negative
relations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .psl_direct_evidence_v2 import (
    DIRECT_EVIDENCE_COLUMNS,
    PslDirectEvidenceBackendV2,
    PslDirectRuleWeights,
)
from .psl_multi_evidence import (
    CANDIDATE_KEY,
    CELL_KEY,
    FORBIDDEN_EVALUATOR_COLUMNS,
)


SOURCE_CELL_KEY = ("incident_token", "seed", "mask_id")
EVALUATOR_COLUMNS = frozenset(
    {
        "case",
        "fault",
        "role",
        "is_masked_target",
        "is_silver_matched",
        "root_cause_service",
    }
)
REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        *SOURCE_CELL_KEY,
        "mask_ratio",
        *CANDIDATE_KEY,
        "a2_score",
        "a2_rank_normalized",
        "proposal_rank",
        "supporting_traces",
        "boundary_spans",
        "reverse_supporting_traces",
        "reverse_boundary_spans",
        "direct_evidence",
        "boundary_alignment",
        "direction_score",
        "operation_role_score",
        "operation_pair_concentration",
        "method_coverage",
        "method_match_rate",
        "route_coverage",
        "route_exact_rate",
        "route_jaccard_mean",
        "operation_jaccard_mean",
        "endpoint_compatibility_score",
        "graph_role_score",
        *EVALUATOR_COLUMNS.difference({"root_cause_service"}),
    }
)
WEAK_FIELDS_IGNORED_FOR_CONFIRMATION = (
    "a2_score",
    "a2_rank_normalized",
    "proposal_rank",
    "supporting_traces",
    "boundary_spans",
    "reverse_supporting_traces",
    "reverse_boundary_spans",
    "boundary_alignment",
    "direction_score",
    "operation_role_score",
    "operation_pair_concentration",
    "method_coverage",
    "method_match_rate",
    "route_coverage",
    "route_exact_rate",
    "route_jaccard_mean",
    "operation_jaccard_mean",
    "endpoint_compatibility_score",
    "graph_role_score",
)


class Phase4PslV2Error(RuntimeError):
    """Raised when the PSL v2 experiment contract is violated."""


@dataclass(frozen=True)
class DirectEvidencePolicy:
    channel_truth_min: float = 0.90
    psl_score_min: float = 0.90
    minimum_direct_channels: int = 1

    def __post_init__(self) -> None:
        for name in ("channel_truth_min", "psl_score_min"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if (
            isinstance(self.minimum_direct_channels, bool)
            or not isinstance(self.minimum_direct_channels, int)
            or not 1 <= self.minimum_direct_channels <= 3
        ):
            raise ValueError("minimum_direct_channels must be an integer in [1,3]")


@dataclass(frozen=True)
class GateConfig:
    direct_candidate_coverage_min: float = 0.001
    direct_target_coverage_min: float = 0.001
    confirmed_count_min: int = 1
    confirmed_precision_lower_bound_min: float = 0.90
    target_confirmation_recall_min: float = 0.50
    unsupported_confirmation_max: int = 0
    candidate_retention_min: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "direct_candidate_coverage_min",
            "direct_target_coverage_min",
            "confirmed_precision_lower_bound_min",
            "target_confirmation_recall_min",
            "candidate_retention_min",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.confirmed_count_min < 0:
            raise ValueError("confirmed_count_min must be non-negative")
        if self.unsupported_confirmation_max < 0:
            raise ValueError(
                "unsupported_confirmation_max must be non-negative"
            )


@dataclass(frozen=True)
class DirectEvidenceAliases:
    direct_trace: tuple[str, ...] = (
        "direct_trace_evidence",
        "trace_parent_child_evidence",
        "trace_parent_child_match",
        "direct_evidence",
    )
    client_server: tuple[str, ...] = (
        "client_server_evidence",
        "client_server_span_evidence",
        "span_kind_pair_evidence",
    )
    workload: tuple[str, ...] = (
        "workload_evidence",
        "workload_pair_evidence",
        "source_destination_workload_evidence",
    )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Sequence[str]]
    ) -> "DirectEvidenceAliases":
        expected = {"direct_trace", "client_server", "workload"}
        extra = sorted(set(value).difference(expected))
        missing = sorted(expected.difference(value))
        if extra or missing:
            raise ValueError(
                f"invalid direct evidence aliases; "
                f"missing={missing} extra={extra}"
            )
        converted = {
            name: tuple(str(item) for item in value[name])
            for name in expected
        }
        if any(not items for items in converted.values()):
            raise ValueError("every direct evidence channel needs an alias")
        return cls(**converted)


@dataclass(frozen=True)
class InferenceVariant:
    variant_id: str
    disabled_rules: tuple[str, ...] = ()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _truth_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return 1.0
    if text in {"false", "no", "n", "", "none", "nan"}:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"direct evidence truth is not numeric: {value!r}")
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(
            f"direct evidence truth must be finite and in [0,1]: {value!r}"
        )
    return number


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _cell_id(incident_token: Any, seed: Any, mask_id: Any) -> str:
    material = f"{incident_token}|{int(seed)}|{mask_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def validate_source_frame(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> None:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(frame.columns))
    if missing:
        raise Phase4PslV2Error(
            f"candidate analysis is missing columns: {missing}"
        )
    expected = config["source_contract"]
    if len(frame) != int(expected["candidate_rows"]):
        raise Phase4PslV2Error(
            f"candidate row count differs: "
            f"{len(frame)} != {expected['candidate_rows']}"
        )
    cells = frame.groupby(list(SOURCE_CELL_KEY), dropna=False).ngroups
    if cells != int(expected["candidate_cells"]):
        raise Phase4PslV2Error(
            f"candidate cell count differs: "
            f"{cells} != {expected['candidate_cells']}"
        )
    if frame["case"].astype(str).nunique() != int(expected["incidents"]):
        raise Phase4PslV2Error(
            "incident count differs from the frozen source contract"
        )
    if set(frame["role"].astype(str)) != {"calibration", "heldout"}:
        raise Phase4PslV2Error(
            "source must contain calibration and heldout roles"
        )
    calibration_cells = frame.loc[
        frame["role"].astype(str).eq("calibration")
    ].groupby(list(SOURCE_CELL_KEY), dropna=False).ngroups
    heldout_cells = frame.loc[
        frame["role"].astype(str).eq("heldout")
    ].groupby(list(SOURCE_CELL_KEY), dropna=False).ngroups
    if calibration_cells != int(expected["calibration_cells"]):
        raise Phase4PslV2Error(
            "calibration cell count differs from the contract"
        )
    if heldout_cells != int(expected["heldout_cells"]):
        raise Phase4PslV2Error(
            "heldout cell count differs from the contract"
        )
    duplicate = frame.duplicated(
        [*SOURCE_CELL_KEY, *CANDIDATE_KEY], keep=False
    )
    if bool(duplicate.any()):
        raise Phase4PslV2Error(
            "candidate analysis contains duplicate candidate keys"
        )
    if set(frame["predicate"].astype(str).str.upper()) != {"CALLS"}:
        raise Phase4PslV2Error(
            "Phase 4 PSL v2 accepts CALLS candidates only"
        )


def _channel_series(
    frame: pd.DataFrame, aliases: Sequence[str]
) -> tuple[pd.Series, list[str]]:
    present = [column for column in aliases if column in frame.columns]
    if not present:
        return pd.Series(0.0, index=frame.index, dtype=float), []
    values = pd.DataFrame(
        {
            column: frame[column].map(_truth_value).astype(float)
            for column in present
        },
        index=frame.index,
    )
    return values.max(axis=1).clip(0.0, 1.0), present


def build_direct_evidence(
    source: pd.DataFrame,
    aliases: DirectEvidenceAliases | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build canonical direct channels without using weak proxies."""

    aliases = aliases or DirectEvidenceAliases()
    model = source.drop(
        columns=[
            column
            for column in FORBIDDEN_EVALUATOR_COLUMNS
            if column in source.columns
        ]
    ).copy()
    forbidden = FORBIDDEN_EVALUATOR_COLUMNS.intersection(model.columns)
    if forbidden:
        raise Phase4PslV2Error(
            f"evaluator columns survived direct evidence split: "
            f"{sorted(forbidden)}"
        )

    model[CELL_KEY] = [
        _cell_id(row.incident_token, row.seed, row.mask_id)
        for row in model.itertuples(index=False)
    ]
    direct_trace, trace_columns = _channel_series(
        model, aliases.direct_trace
    )
    client_server, client_server_columns = _channel_series(
        model, aliases.client_server
    )
    workload, workload_columns = _channel_series(
        model, aliases.workload
    )

    evidence = model[[CELL_KEY, *CANDIDATE_KEY]].copy()
    evidence["candidate"] = 1.0
    evidence["direct_trace"] = direct_trace
    evidence["client_server"] = client_server
    evidence["workload"] = workload
    evidence = evidence[
        [CELL_KEY, *CANDIDATE_KEY, *DIRECT_EVIDENCE_COLUMNS]
    ]

    metadata = model[
        [
            CELL_KEY,
            *SOURCE_CELL_KEY,
            "mask_ratio",
            *CANDIDATE_KEY,
            "a2_score",
            "a2_rank_normalized",
            "proposal_rank",
        ]
    ].copy()
    metadata["a2_priority"] = (
        0.60
        * pd.to_numeric(
            metadata["a2_score"], errors="raise"
        ).clip(0.0, 1.0)
        + 0.40
        * pd.to_numeric(
            metadata["a2_rank_normalized"], errors="raise"
        ).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    metadata["direct_trace"] = direct_trace
    metadata["client_server"] = client_server
    metadata["workload"] = workload
    metadata["direct_evidence_max"] = metadata[
        ["direct_trace", "client_server", "workload"]
    ].max(axis=1)

    diagnostics = {
        "candidate_rows": len(evidence),
        "candidate_cells": int(evidence[CELL_KEY].nunique()),
        "evaluator_columns_removed_before_psl": True,
        "canonical_direct_channels": [
            "direct_trace",
            "client_server",
            "workload",
        ],
        "source_aliases_used": {
            "direct_trace": trace_columns,
            "client_server": client_server_columns,
            "workload": workload_columns,
        },
        "source_aliases_missing": {
            "direct_trace": not trace_columns,
            "client_server": not client_server_columns,
            "workload": not workload_columns,
        },
        "direct_trace_count": int((direct_trace > 0.0).sum()),
        "client_server_count": int((client_server > 0.0).sum()),
        "workload_count": int((workload > 0.0).sum()),
        "any_direct_count": int(
            (
                evidence[
                    ["direct_trace", "client_server", "workload"]
                ].max(axis=1)
                > 0.0
            ).sum()
        ),
        "weak_fields_ignored_for_confirmation": list(
            WEAK_FIELDS_IGNORED_FOR_CONFIRMATION
        ),
        "a2_prior_used_for_confirmation": False,
        "reverse_or_direction_conflict_used": False,
    }
    return evidence, metadata, diagnostics


def _mutate_weak_fields(source: pd.DataFrame) -> pd.DataFrame:
    output = source.copy()
    for column in WEAK_FIELDS_IGNORED_FOR_CONFIRMATION:
        if column not in output.columns:
            continue
        numeric = pd.to_numeric(output[column], errors="coerce")
        if numeric.notna().any():
            replacement = numeric.fillna(0.0).iloc[::-1].to_numpy()
            output.loc[:, column] = replacement
        else:
            output.loc[:, column] = (
                output[column].astype(str).iloc[::-1].to_numpy()
            )
    return output


def weak_evidence_invariance(
    source: pd.DataFrame,
    aliases: DirectEvidenceAliases,
    baseline: pd.DataFrame,
) -> bool:
    mutated, _metadata, _diagnostics = build_direct_evidence(
        _mutate_weak_fields(source), aliases
    )
    columns = [CELL_KEY, *CANDIDATE_KEY, *DIRECT_EVIDENCE_COLUMNS]
    left = baseline[columns].sort_values(
        [CELL_KEY, *CANDIDATE_KEY], kind="mergesort"
    ).reset_index(drop=True)
    right = mutated[columns].sort_values(
        [CELL_KEY, *CANDIDATE_KEY], kind="mergesort"
    ).reset_index(drop=True)
    return left.equals(right)


def _score_frame(
    evidence: pd.DataFrame,
    *,
    backend: PslDirectEvidenceBackendV2,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = backend.infer(evidence)
    records = [
        {
            CELL_KEY: cell,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            score_column: score,
        }
        for (cell, subject, predicate, obj), score in result.scores.items()
    ]
    frame = pd.DataFrame.from_records(records)
    expected = evidence[[CELL_KEY, *CANDIDATE_KEY]]
    if len(frame) != len(expected):
        raise Phase4PslV2Error(
            "PSL v2 score frame changed candidate count"
        )
    return frame, {
        "grounded_rule_count": result.grounded_rule_count,
        "grounded_atom_count": result.grounded_atom_count,
        "metadata": dict(result.metadata),
        "score_std": float(frame[score_column].std(ddof=0)),
        "score_min": float(frame[score_column].min()),
        "score_max": float(frame[score_column].max()),
        "unique_score_count": int(
            frame[score_column].round(12).nunique()
        ),
    }


def _labels(source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *SOURCE_CELL_KEY,
        "mask_ratio",
        *CANDIDATE_KEY,
        *sorted(EVALUATOR_COLUMNS.intersection(source.columns)),
    ]
    labels = source[columns].copy()
    labels[CELL_KEY] = [
        _cell_id(row.incident_token, row.seed, row.mask_id)
        for row in labels.itertuples(index=False)
    ]
    return labels


def _analysis_frame(
    source: pd.DataFrame,
    metadata: pd.DataFrame,
    score_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = metadata.merge(
        score_frame,
        on=[CELL_KEY, *CANDIDATE_KEY],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        _labels(source),
        on=[
            CELL_KEY,
            *SOURCE_CELL_KEY,
            "mask_ratio",
            *CANDIDATE_KEY,
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(source):
        raise Phase4PslV2Error(
            "evaluator label join changed candidate count"
        )
    return merged


def apply_abstention_policy(
    frame: pd.DataFrame,
    policy: DirectEvidencePolicy,
    *,
    score_column: str = "psl_direct_score",
) -> pd.DataFrame:
    """Assign only ``CONFIRMED`` or ``ABSTAIN`` states."""

    output = frame.copy().reset_index(drop=True)
    channel_columns = ["direct_trace", "client_server", "workload"]
    threshold = float(policy.channel_truth_min)
    output["direct_channel_count"] = (
        output[channel_columns].ge(threshold).sum(axis=1).astype(int)
    )
    output["direct_evidence_max"] = output[channel_columns].max(axis=1)
    output["direct_eligible"] = (
        output["direct_channel_count"]
        >= int(policy.minimum_direct_channels)
    )
    output["psl_direct_score"] = pd.to_numeric(
        output[score_column], errors="raise"
    ).clip(0.0, 1.0)
    confirmed = output["direct_eligible"] & output[
        "psl_direct_score"
    ].ge(float(policy.psl_score_min))
    output["decision_state"] = "ABSTAIN"
    output.loc[confirmed, "decision_state"] = "CONFIRMED"
    output["decision_reason"] = "NO_DIRECT_EVIDENCE"
    weak_direct = (
        output["direct_evidence_max"].gt(0.0)
        & ~output["direct_eligible"]
    )
    output.loc[
        weak_direct, "decision_reason"
    ] = "DIRECT_EVIDENCE_BELOW_CHANNEL_POLICY"
    low_psl = output["direct_eligible"] & ~output[
        "psl_direct_score"
    ].ge(float(policy.psl_score_min))
    output.loc[
        low_psl, "decision_reason"
    ] = "DIRECT_EVIDENCE_BELOW_PSL_POLICY"
    output.loc[
        confirmed, "decision_reason"
    ] = "DIRECT_EVIDENCE_CONFIRMED"
    output["review_priority"] = pd.to_numeric(
        output["a2_priority"], errors="raise"
    ).clip(0.0, 1.0)
    return output


def evaluate_cell(
    decided: pd.DataFrame,
    policy: DirectEvidencePolicy,
) -> dict[str, Any]:
    states = set(decided["decision_state"].astype(str))
    invalid_states = states.difference({"CONFIRMED", "ABSTAIN"})
    if invalid_states:
        raise Phase4PslV2Error(
            f"invalid PSL v2 decision states: {sorted(invalid_states)}"
        )
    confirmed = decided.loc[
        decided["decision_state"].astype(str).eq("CONFIRMED")
    ]
    abstained = decided.loc[
        decided["decision_state"].astype(str).eq("ABSTAIN")
    ]
    targets = decided.loc[decided["is_masked_target"].map(_truthy)]
    direct_supported = decided.loc[
        decided["direct_channel_count"]
        >= int(policy.minimum_direct_channels)
    ]
    unsupported_confirmed = confirmed.loc[
        confirmed["direct_channel_count"]
        < int(policy.minimum_direct_channels)
    ]
    confirmed_silver = confirmed.loc[
        confirmed["is_silver_matched"].map(_truthy)
    ]
    confirmed_targets = confirmed.loc[
        confirmed["is_masked_target"].map(_truthy)
    ]
    direct_targets = direct_supported.loc[
        direct_supported["is_masked_target"].map(_truthy)
    ]
    total = len(decided)
    confirmed_count = len(confirmed)
    target_count = len(targets)
    return {
        "candidate_count": total,
        "confirmed_count": confirmed_count,
        "abstained_count": len(abstained),
        "confirmation_rate": confirmed_count / total if total else None,
        "abstention_rate": len(abstained) / total if total else None,
        "candidate_retention_rate": (
            (confirmed_count + len(abstained)) / total
            if total
            else None
        ),
        "direct_supported_count": len(direct_supported),
        "direct_candidate_coverage": (
            len(direct_supported) / total if total else None
        ),
        "target_count": target_count,
        "confirmed_target_count": len(confirmed_targets),
        "target_confirmation_recall": (
            len(confirmed_targets) / target_count
            if target_count
            else None
        ),
        "direct_supported_target_count": len(direct_targets),
        "direct_target_coverage": (
            len(direct_targets) / target_count
            if target_count
            else None
        ),
        "confirmed_silver_count": len(confirmed_silver),
        "confirmed_precision_lower_bound": (
            len(confirmed_silver) / confirmed_count
            if confirmed_count
            else None
        ),
        "unsupported_confirmation_count": len(
            unsupported_confirmed
        ),
        "decision_state_count": len(states),
        "no_negative_relation_state": not bool(
            states.intersection({"REJECTED", "NEGATIVE", "FALSE"})
        ),
    }


def _mean_available(
    rows: Sequence[Mapping[str, Any]], field: str
) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    return statistics.fmean(values) if values else None


def _aggregate(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not rows:
        return {}
    totals = {
        field: sum(int(row[field]) for row in rows)
        for field in (
            "candidate_count",
            "confirmed_count",
            "abstained_count",
            "direct_supported_count",
            "target_count",
            "confirmed_target_count",
            "direct_supported_target_count",
            "confirmed_silver_count",
            "unsupported_confirmation_count",
        )
    }
    confirmed = totals["confirmed_count"]
    candidates = totals["candidate_count"]
    targets = totals["target_count"]
    return {
        "cell_count": len(rows),
        **totals,
        "confirmation_rate": (
            confirmed / candidates if candidates else None
        ),
        "abstention_rate": (
            totals["abstained_count"] / candidates
            if candidates
            else None
        ),
        "candidate_retention_rate": (
            (
                totals["confirmed_count"]
                + totals["abstained_count"]
            )
            / candidates
            if candidates
            else None
        ),
        "direct_candidate_coverage": (
            totals["direct_supported_count"] / candidates
            if candidates
            else None
        ),
        "target_confirmation_recall": (
            totals["confirmed_target_count"] / targets
            if targets
            else None
        ),
        "direct_target_coverage": (
            totals["direct_supported_target_count"] / targets
            if targets
            else None
        ),
        "confirmed_precision_lower_bound": (
            totals["confirmed_silver_count"] / confirmed
            if confirmed
            else None
        ),
        "confirmation_rate_macro": _mean_available(
            rows, "confirmation_rate"
        ),
        "abstention_rate_macro": _mean_available(
            rows, "abstention_rate"
        ),
        "target_confirmation_recall_macro": _mean_available(
            rows, "target_confirmation_recall"
        ),
        "direct_candidate_coverage_macro": _mean_available(
            rows, "direct_candidate_coverage"
        ),
        "direct_target_coverage_macro": _mean_available(
            rows, "direct_target_coverage"
        ),
        "confirmed_precision_lower_bound_macro": _mean_available(
            rows, "confirmed_precision_lower_bound"
        ),
        "all_cells_no_negative_relation_state": all(
            bool(row["no_negative_relation_state"]) for row in rows
        ),
    }


def _cells(
    frame: pd.DataFrame, role: str
) -> dict[str, pd.DataFrame]:
    selected = frame.loc[
        frame["role"].astype(str).eq(role)
    ].copy()
    return {
        str(cell): group.reset_index(drop=True)
        for cell, group in selected.groupby(
            CELL_KEY, sort=True, dropna=False
        )
    }


def _evaluate_role(
    cells: Mapping[str, pd.DataFrame],
    policy: DirectEvidencePolicy,
    *,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    outputs: list[pd.DataFrame] = []
    for cell_id, group in sorted(cells.items()):
        decided = apply_abstention_policy(
            group, policy, score_column=score_column
        )
        metric = evaluate_cell(decided, policy)
        first = group.iloc[0]
        common = {
            CELL_KEY: cell_id,
            "case": str(first["case"]),
            "fault": str(first["fault"]),
            "role": str(first["role"]),
            "seed": int(first["seed"]),
            "mask_id": str(first["mask_id"]),
            "mask_ratio": float(first["mask_ratio"]),
        }
        rows.append({**common, **metric})
        for name, value in common.items():
            decided[name] = value
        outputs.append(decided)
    row_frame = pd.DataFrame.from_records(rows)
    return (
        row_frame,
        _aggregate(rows),
        (
            pd.concat(outputs, ignore_index=True)
            if outputs
            else pd.DataFrame()
        ),
    )


def _rule_weights(
    config: Mapping[str, Any]
) -> PslDirectRuleWeights:
    return PslDirectRuleWeights.from_mapping(
        config["direct_rule_weights"]
    )


def _policy(
    config: Mapping[str, Any]
) -> DirectEvidencePolicy:
    values = config["abstention_policy"]
    return DirectEvidencePolicy(
        channel_truth_min=float(values["channel_truth_min"]),
        psl_score_min=float(values["psl_score_min"]),
        minimum_direct_channels=int(
            values["minimum_direct_channels"]
        ),
    )


def _gate_config(
    config: Mapping[str, Any]
) -> GateConfig:
    values = config["gate"]
    return GateConfig(
        direct_candidate_coverage_min=float(
            values["direct_candidate_coverage_min"]
        ),
        direct_target_coverage_min=float(
            values["direct_target_coverage_min"]
        ),
        confirmed_count_min=int(values["confirmed_count_min"]),
        confirmed_precision_lower_bound_min=float(
            values["confirmed_precision_lower_bound_min"]
        ),
        target_confirmation_recall_min=float(
            values["target_confirmation_recall_min"]
        ),
        unsupported_confirmation_max=int(
            values["unsupported_confirmation_max"]
        ),
        candidate_retention_min=float(
            values["candidate_retention_min"]
        ),
    )


def _aliases(
    config: Mapping[str, Any]
) -> DirectEvidenceAliases:
    return DirectEvidenceAliases.from_mapping(
        config["direct_evidence_aliases"]
    )


def _ablation_variants() -> tuple[InferenceVariant, ...]:
    return (
        InferenceVariant(
            "single_channel_only",
            (
                "TRACE_CLIENT_SERVER",
                "TRACE_WORKLOAD",
                "CLIENT_SERVER_WORKLOAD",
                "ALL_DIRECT",
            ),
        ),
        InferenceVariant(
            "trace_only",
            (
                "CLIENT_SERVER",
                "WORKLOAD",
                "TRACE_CLIENT_SERVER",
                "TRACE_WORKLOAD",
                "CLIENT_SERVER_WORKLOAD",
                "ALL_DIRECT",
            ),
        ),
        InferenceVariant(
            "client_server_only",
            (
                "DIRECT_TRACE",
                "WORKLOAD",
                "TRACE_CLIENT_SERVER",
                "TRACE_WORKLOAD",
                "CLIENT_SERVER_WORKLOAD",
                "ALL_DIRECT",
            ),
        ),
        InferenceVariant(
            "workload_only",
            (
                "DIRECT_TRACE",
                "CLIENT_SERVER",
                "TRACE_CLIENT_SERVER",
                "TRACE_WORKLOAD",
                "CLIENT_SERVER_WORKLOAD",
                "ALL_DIRECT",
            ),
        ),
    )


def _write_markdown(
    path: Path, summary: Mapping[str, Any]
) -> None:
    held = summary["heldout"]
    evidence = summary["evidence_diagnostics"]
    precision = held["confirmed_precision_lower_bound"]
    precision_text = (
        "N/A" if precision is None else f"{precision:.6f}"
    )
    lines = [
        "# Task A Phase 4 — Direct-evidence PSL v2",
        "",
        f"- Scientific status: **{summary['status']}**",
        f"- Mechanism gate: **{summary['mechanism_gate']['status']}**",
        f"- Data eligibility: **{summary['data_eligibility']['status']}**",
        f"- Gate: `{summary['gate_id']}`",
        f"- Reason codes: "
        f"`{', '.join(summary['reason_codes']) or 'none'}`",
        "",
        "## Direct evidence contract",
        "",
        "- Only direct Trace, CLIENT/SERVER, or Workload evidence "
        "can confirm a relation.",
        "- Reverse traces, direction conflicts, topology similarity, "
        "operation similarity, and A2 prior cannot confirm or reject "
        "a relation.",
        "- Every unsupported candidate is preserved as `ABSTAIN`.",
        "- No negative `CALLS` relation is emitted.",
        "",
        "## Held-out result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Candidate rows | {held['candidate_count']} |",
        f"| Direct-supported candidates | "
        f"{held['direct_supported_count']} |",
        f"| Confirmed | {held['confirmed_count']} |",
        f"| Abstained | {held['abstained_count']} |",
        f"| Confirmation rate | "
        f"{held['confirmation_rate']:.6f} |",
        f"| Abstention rate | "
        f"{held['abstention_rate']:.6f} |",
        f"| Candidate retention | "
        f"{held['candidate_retention_rate']:.6f} |",
        f"| Target confirmation recall | "
        f"{held['target_confirmation_recall']:.6f} |",
        f"| Confirmed P-LB | {precision_text} |",
        f"| Unsupported confirmations | "
        f"{held['unsupported_confirmation_count']} |",
        "",
        "## Source evidence availability",
        "",
        "| Channel | Positive candidates | Source aliases used |",
        "|---|---:|---|",
        f"| Direct Trace | {evidence['direct_trace_count']} | "
        f"`{', '.join(evidence['source_aliases_used']['direct_trace']) or 'none'}` |",
        f"| CLIENT/SERVER | {evidence['client_server_count']} | "
        f"`{', '.join(evidence['source_aliases_used']['client_server']) or 'none'}` |",
        f"| Workload | {evidence['workload_count']} | "
        f"`{', '.join(evidence['source_aliases_used']['workload']) or 'none'}` |",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Claim boundary",
        "",
        summary["claim_limit"],
        "",
        "The output is a two-state confirmation decision "
        "(`CONFIRMED` / `ABSTAIN`) over runtime `CALLS` candidates. "
        "It is not a causal `CAUSES` graph and does not establish "
        "RCA or LLM improvement.",
    ]
    path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_phase4_psl_v2(
    *,
    candidate_analysis_path: Path,
    config_path: Path,
    output: Path,
) -> Path:
    candidate_analysis_path = (
        candidate_analysis_path.expanduser().resolve()
    )
    config_path = config_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise Phase4PslV2Error(
            f"refusing to overwrite existing output: {output}"
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    if config.get("schema_version") != 2:
        raise Phase4PslV2Error(
            "PSL v2 config schema_version must be 2"
        )
    source_sha = _sha256_file(candidate_analysis_path)
    expected_sha = str(
        config["source_contract"].get(
            "candidate_analysis_sha256", ""
        )
    )
    if expected_sha and source_sha != expected_sha:
        raise Phase4PslV2Error(
            f"candidate analysis SHA-256 mismatch: "
            f"{source_sha} != {expected_sha}"
        )
    source = pd.read_parquet(candidate_analysis_path)
    validate_source_frame(source, config)

    output.mkdir(parents=True, exist_ok=False)
    published = output / "published"
    model_output = output / "model_output"
    evaluator_private = output / "evaluator_private"
    for directory in (
        published,
        model_output,
        evaluator_private,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    aliases = _aliases(config)
    policy = _policy(config)
    gate_config = _gate_config(config)
    weights = _rule_weights(config)

    evidence, metadata, evidence_diagnostics = (
        build_direct_evidence(source, aliases)
    )
    weak_invariant = weak_evidence_invariance(
        source, aliases, evidence
    )
    evidence.to_parquet(
        model_output / "psl_v2_direct_evidence.parquet",
        index=False,
    )

    random_seed = int(
        config["psl_runtime"]["random_seed"]
    )
    jvm_options = tuple(
        str(value)
        for value in config["psl_runtime"]["jvm_options"]
    )
    backend = PslDirectEvidenceBackendV2(
        weights=weights,
        profile_id="direct_union",
        random_seed=random_seed,
        jvm_options=jvm_options,
    )
    score_frame, psl_diagnostics = _score_frame(
        evidence,
        backend=backend,
        score_column="psl_direct_score",
    )
    analysis = _analysis_frame(
        source, metadata, score_frame
    )
    analysis.to_parquet(
        evaluator_private / "candidate_analysis_v2.parquet",
        index=False,
    )

    calibration_cells = _cells(analysis, "calibration")
    heldout_cells = _cells(analysis, "heldout")
    (
        calibration_rows,
        calibration_summary,
        calibration_decisions,
    ) = _evaluate_role(
        calibration_cells,
        policy,
        score_column="psl_direct_score",
    )
    (
        heldout_rows,
        heldout_summary,
        heldout_decisions,
    ) = _evaluate_role(
        heldout_cells,
        policy,
        score_column="psl_direct_score",
    )
    calibration_rows.to_csv(
        published
        / "task_a_phase4_psl_v2_calibration_cells.csv",
        index=False,
    )
    heldout_rows.to_csv(
        published / "task_a_phase4_psl_v2_heldout_cells.csv",
        index=False,
    )
    all_decisions = pd.concat(
        [calibration_decisions, heldout_decisions],
        ignore_index=True,
    )
    all_decisions.to_parquet(
        model_output / "psl_v2_candidate_decisions.parquet",
        index=False,
    )

    ablation_summaries: dict[str, Any] = {
        "direct_union": heldout_summary
    }
    ablation_diagnostics: dict[str, Any] = {}
    ablation_rows = []
    for variant in _ablation_variants():
        variant_backend = PslDirectEvidenceBackendV2(
            weights=weights,
            profile_id=variant.variant_id,
            disabled_rules=variant.disabled_rules,
            random_seed=random_seed,
            jvm_options=jvm_options,
        )
        score_column = f"psl_direct_score__{variant.variant_id}"
        variant_score, variant_diag = _score_frame(
            evidence,
            backend=variant_backend,
            score_column=score_column,
        )
        variant_analysis = _analysis_frame(
            source, metadata, variant_score
        )
        _rows, aggregate, _decisions = _evaluate_role(
            _cells(variant_analysis, "heldout"),
            policy,
            score_column=score_column,
        )
        ablation_summaries[
            variant.variant_id
        ] = aggregate
        ablation_diagnostics[
            variant.variant_id
        ] = variant_diag
        ablation_rows.append(
            {"variant": variant.variant_id, **aggregate}
        )
    pd.DataFrame.from_records(ablation_rows).to_csv(
        published
        / "task_a_phase4_psl_v2_ablation_results.csv",
        index=False,
    )

    direct_candidate_coverage = float(
        heldout_summary["direct_candidate_coverage"]
    )
    direct_target_coverage = float(
        heldout_summary["direct_target_coverage"]
    )
    data_eligibility_conditions = {
        "direct_candidate_coverage": (
            direct_candidate_coverage
            >= gate_config.direct_candidate_coverage_min
        ),
        "direct_target_coverage": (
            direct_target_coverage
            >= gate_config.direct_target_coverage_min
        ),
    }
    data_eligible = all(
        data_eligibility_conditions.values()
    )

    psl_metadata = psl_diagnostics["metadata"]
    mechanism_conditions = {
        "candidate_handoff_complete": (
            len(source)
            == int(
                config["source_contract"]["candidate_rows"]
            )
        ),
        "candidate_retention": (
            float(
                heldout_summary[
                    "candidate_retention_rate"
                ]
            )
            >= gate_config.candidate_retention_min
        ),
        "unsupported_confirmation_zero": (
            int(
                heldout_summary[
                    "unsupported_confirmation_count"
                ]
            )
            <= gate_config.unsupported_confirmation_max
        ),
        "no_negative_relation_rules": (
            int(
                psl_metadata[
                    "negative_relation_rule_count"
                ]
            )
            == 0
        ),
        "a2_prior_not_used_for_confirmation": (
            not bool(
                psl_metadata[
                    "uses_a2_prior_for_confirmation"
                ]
            )
        ),
        "only_confirmed_or_abstain": (
            set(
                all_decisions[
                    "decision_state"
                ].astype(str)
            )
            <= {"CONFIRMED", "ABSTAIN"}
        ),
        "no_negative_relation_state": bool(
            heldout_summary[
                "all_cells_no_negative_relation_state"
            ]
        ),
        "weak_evidence_invariant": weak_invariant,
        "evaluator_labels_joined_after_psl": True,
        "deberta_not_used": True,
    }
    mechanism_passed = all(
        mechanism_conditions.values()
    )

    precision = heldout_summary[
        "confirmed_precision_lower_bound"
    ]
    utility_conditions = {
        "confirmed_count": (
            int(heldout_summary["confirmed_count"])
            >= gate_config.confirmed_count_min
        ),
        "confirmed_precision_lower_bound": (
            precision is not None
            and float(precision)
            >= gate_config.confirmed_precision_lower_bound_min
        ),
        "target_confirmation_recall": (
            float(
                heldout_summary[
                    "target_confirmation_recall"
                ]
            )
            >= gate_config.target_confirmation_recall_min
        ),
    }
    utility_passed = all(utility_conditions.values())

    if not mechanism_passed:
        status = "FAIL"
        reason_codes = [
            name.upper()
            for name, passed in mechanism_conditions.items()
            if not passed
        ]
    elif not data_eligible:
        status = "INELIGIBLE"
        reason_codes = [
            "SOURCE_" + name.upper()
            for name, passed in (
                data_eligibility_conditions.items()
            )
            if not passed
        ]
    elif utility_passed:
        status = "PASS"
        reason_codes = []
    else:
        status = "FAIL"
        reason_codes = [
            name.upper()
            for name, passed in utility_conditions.items()
            if not passed
        ]

    if status == "INELIGIBLE":
        interpretation = (
            "The mechanism contract passed, but the frozen RCAEval "
            "handoff contains no positive canonical direct telemetry. "
            "All candidates were therefore preserved as ABSTAIN. "
            "This dataset cannot validate direct relation confirmation; "
            "the next confirmatory dataset must expose parent/child trace, "
            "CLIENT/SERVER span-kind, or workload-pair evidence."
        )
    elif status == "PASS":
        interpretation = (
            "Direct telemetry was available and the fixed confirmation "
            "policy satisfied the preregistered safety and utility gates."
        )
    else:
        interpretation = (
            "The direct-evidence mechanism or its utility gates failed. "
            "No unsupported candidate was converted into a negative edge; "
            "the complete decision table is retained for diagnosis."
        )

    summary = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "status": status,
        "gate_id": config["gate_id"],
        "reason_codes": reason_codes,
        "mechanism_gate": {
            "status": (
                "PASS" if mechanism_passed else "FAIL"
            ),
            "passed": mechanism_passed,
            "conditions": mechanism_conditions,
        },
        "data_eligibility": {
            "status": (
                "ELIGIBLE"
                if data_eligible
                else "INELIGIBLE"
            ),
            "eligible": data_eligible,
            "conditions": data_eligibility_conditions,
            "required": {
                "direct_candidate_coverage_min": (
                    gate_config.direct_candidate_coverage_min
                ),
                "direct_target_coverage_min": (
                    gate_config.direct_target_coverage_min
                ),
            },
        },
        "utility_gate": {
            "status": (
                "PASS" if utility_passed else "FAIL"
            ),
            "passed": utility_passed,
            "conditions": utility_conditions,
            "required": {
                "confirmed_count_min": (
                    gate_config.confirmed_count_min
                ),
                "confirmed_precision_lower_bound_min": (
                    gate_config.confirmed_precision_lower_bound_min
                ),
                "target_confirmation_recall_min": (
                    gate_config.target_confirmation_recall_min
                ),
            },
            "evaluated_as_overall_gate": data_eligible,
        },
        "source": {
            "candidate_analysis": str(
                candidate_analysis_path
            ),
            "candidate_analysis_sha256": source_sha,
            "candidate_rows": len(source),
            "candidate_cells": int(
                source.groupby(
                    list(SOURCE_CELL_KEY)
                ).ngroups
            ),
            "incidents": int(source["case"].nunique()),
            "calibration_cells": len(
                calibration_cells
            ),
            "heldout_cells": len(heldout_cells),
        },
        "abstention_policy": asdict(policy),
        "direct_rule_weights": asdict(weights),
        "calibration": calibration_summary,
        "heldout": heldout_summary,
        "ablations": ablation_summaries,
        "psl": psl_diagnostics,
        "psl_ablation_diagnostics": (
            ablation_diagnostics
        ),
        "evidence_diagnostics": (
            evidence_diagnostics
        ),
        "weak_evidence_invariance": weak_invariant,
        "leakage_boundary": {
            "evaluator_columns_removed_before_feature_build": True,
            "evaluator_columns_removed_before_psl": True,
            "evaluator_labels_joined_after_psl_score_freeze": True,
            "fault_or_root_label_used_for_scoring": False,
            "mask_or_silver_label_used_for_scoring": False,
            "a2_prior_used_for_confirmation": False,
            "deberta_score_used": False,
        },
        "decision_semantics": {
            "states": ["CONFIRMED", "ABSTAIN"],
            "negative_relation_state": None,
            "candidate_deletion": False,
            "unsupported_candidate_state": "ABSTAIN",
        },
        "protocol_status": {
            "phase": "DEVELOPMENT_PSL_V2_DIRECT_EVIDENCE",
            "confirmatory_direct_telemetry_required": (
                not data_eligible
            ),
            "recommended_direct_fields": [
                "direct_trace_evidence",
                "client_server_evidence",
                "workload_evidence",
            ],
        },
        "interpretation": interpretation,
        "claim_limit": config["claim_limit"],
    }

    result_json = (
        published
        / "task_a_phase4_psl_v2_results.json"
    )
    result_md = (
        published
        / "task_a_phase4_psl_v2_results.md"
    )
    _write_json(result_json, summary)
    _write_markdown(result_md, summary)
    (
        published / "task_a_phase4_psl_v2_status.txt"
    ).write_text(status + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256_file(config_path),
        "candidate_analysis_sha256": source_sha,
        "output_files": {},
    }
    for path in sorted(published.iterdir()):
        if (
            path.is_file()
            and path.name
            != "task_a_phase4_psl_v2_manifest.json"
        ):
            manifest["output_files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _write_json(
        published
        / "task_a_phase4_psl_v2_manifest.json",
        manifest,
    )
    print(
        f"Task A Phase 4 PSL v2 status: "
        f"{status} -> {output}",
        flush=True,
    )
    return output


__all__ = [
    "DirectEvidenceAliases",
    "DirectEvidencePolicy",
    "GateConfig",
    "Phase4PslV2Error",
    "REQUIRED_SOURCE_COLUMNS",
    "WEAK_FIELDS_IGNORED_FOR_CONFIRMATION",
    "apply_abstention_policy",
    "build_direct_evidence",
    "evaluate_cell",
    "run_phase4_psl_v2",
    "validate_source_frame",
    "weak_evidence_invariance",
]
