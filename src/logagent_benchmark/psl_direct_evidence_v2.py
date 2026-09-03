"""Direct-evidence-only PSL backend for runtime ``CALLS`` confirmation.

PSL v2 never interprets reverse traces, direction conflicts, sparse observations,
or weak topology compatibility as evidence that a relation is false.  It only
aggregates explicit direct telemetry channels:

* parent/child trace evidence,
* CLIENT -> SERVER span evidence,
* source -> destination workload evidence.

Candidate acceptance is still guarded by a deterministic decision policy in
``task_a_phase4_psl_v2``.  Therefore an unsupported candidate is preserved as
``ABSTAIN`` rather than emitted as a negative relation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping, Sequence

import pandas as pd

from .psl_backend import PslPythonBackend
from .psl_multi_evidence import (
    CANDIDATE_KEY,
    CELL_KEY,
    FORBIDDEN_EVALUATOR_COLUMNS,
)
from .recovery import Availability, ERROR, READY, SKIPPED


_PSL_DIRECT_LOCK = threading.Lock()
DIRECT_EVIDENCE_COLUMNS = (
    "candidate",
    "direct_trace",
    "client_server",
    "workload",
)


class PslDirectEvidenceError(RuntimeError):
    """Raised when the PSL v2 direct-evidence contract cannot be satisfied."""


@dataclass(frozen=True)
class PslDirectRuleWeights:
    """Positive-only PSL rule weights.

    There are deliberately no ``!ConfirmedCallsV2`` rules.  Absence of direct
    evidence is handled as abstention by the decision policy, not as a negative
    edge assertion.
    """

    direct_trace: float = 10.0
    client_server: float = 12.0
    workload: float = 12.0
    trace_client_server: float = 4.0
    trace_workload: float = 4.0
    client_server_workload: float = 5.0
    all_direct: float = 8.0

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in asdict(self).items()
            if not math.isfinite(float(value)) or float(value) < 0.0
        }
        if invalid:
            raise ValueError(
                f"PSL direct rule weights must be finite and non-negative: {invalid}"
            )
        if max(float(value) for value in asdict(self).values()) <= 0.0:
            raise ValueError("at least one direct-evidence PSL rule must be active")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PslDirectRuleWeights":
        expected = set(asdict(cls()))
        extra = sorted(set(value).difference(expected))
        missing = sorted(expected.difference(value))
        if extra or missing:
            raise ValueError(
                f"invalid PSL v2 weight profile; missing={missing} extra={extra}"
            )
        return cls(**{name: float(value[name]) for name in expected})


@dataclass(frozen=True)
class PslDirectRuleSpec:
    rule_id: str
    weight: float
    body: str

    @property
    def expression(self) -> str:
        return f"{self.weight:.12g}: {self.body} ^2"


@dataclass(frozen=True)
class PslDirectEvidenceResult:
    scores: Mapping[tuple[str, str, str, str], float]
    grounded_rule_count: int
    grounded_atom_count: int
    metadata: Mapping[str, Any]


def _rule_specs(weights: PslDirectRuleWeights) -> tuple[PslDirectRuleSpec, ...]:
    guarded = "CandidateDirectV2(C, S, O)"
    target = "ConfirmedCallsV2(C, S, O)"
    return (
        PslDirectRuleSpec(
            "DIRECT_TRACE",
            weights.direct_trace,
            f"{guarded} & DirectTraceV2(C, S, O) -> {target}",
        ),
        PslDirectRuleSpec(
            "CLIENT_SERVER",
            weights.client_server,
            f"{guarded} & ClientServerV2(C, S, O) -> {target}",
        ),
        PslDirectRuleSpec(
            "WORKLOAD",
            weights.workload,
            f"{guarded} & WorkloadPairV2(C, S, O) -> {target}",
        ),
        PslDirectRuleSpec(
            "TRACE_CLIENT_SERVER",
            weights.trace_client_server,
            f"{guarded} & DirectTraceV2(C, S, O) & ClientServerV2(C, S, O) -> {target}",
        ),
        PslDirectRuleSpec(
            "TRACE_WORKLOAD",
            weights.trace_workload,
            f"{guarded} & DirectTraceV2(C, S, O) & WorkloadPairV2(C, S, O) -> {target}",
        ),
        PslDirectRuleSpec(
            "CLIENT_SERVER_WORKLOAD",
            weights.client_server_workload,
            f"{guarded} & ClientServerV2(C, S, O) & WorkloadPairV2(C, S, O) -> {target}",
        ),
        PslDirectRuleSpec(
            "ALL_DIRECT",
            weights.all_direct,
            f"{guarded} & DirectTraceV2(C, S, O) & ClientServerV2(C, S, O) "
            f"& WorkloadPairV2(C, S, O) -> {target}",
        ),
    )


def _safe_atom(value: Any) -> str:
    text = str(value)
    if not text:
        raise ValueError("PSL atom identifiers cannot be empty")
    if any(character in text for character in ("\t", "\r", "\n")):
        raise ValueError("PSL atom identifiers cannot contain tabs or newlines")
    return text


def _clip_truth(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1], got {value!r}")
    return number


def validate_direct_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize leakage-safe direct telemetry evidence."""

    forbidden = FORBIDDEN_EVALUATOR_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise PslDirectEvidenceError(
            "PSL v2 input contains evaluator columns: "
            + ", ".join(sorted(forbidden))
        )
    required = (CELL_KEY, *CANDIDATE_KEY, *DIRECT_EVIDENCE_COLUMNS)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise PslDirectEvidenceError(
            f"PSL v2 direct evidence is missing columns: {missing}"
        )
    if frame.empty:
        raise PslDirectEvidenceError("PSL v2 direct evidence cannot be empty")

    output = frame[list(required)].copy()
    output[CELL_KEY] = output[CELL_KEY].map(_safe_atom)
    output["subject"] = output["subject"].map(_safe_atom)
    output["predicate"] = output["predicate"].astype(str).str.upper()
    output["object"] = output["object"].map(_safe_atom)
    if set(output["predicate"]) != {"CALLS"}:
        raise PslDirectEvidenceError("PSL v2 accepts CALLS candidates only")

    duplicate = output.duplicated([CELL_KEY, *CANDIDATE_KEY], keep=False)
    if bool(duplicate.any()):
        examples = (
            output.loc[duplicate, [CELL_KEY, *CANDIDATE_KEY]]
            .head(5)
            .to_dict("records")
        )
        raise PslDirectEvidenceError(f"duplicate PSL v2 candidates: {examples}")
    for column in DIRECT_EVIDENCE_COLUMNS:
        output[column] = output[column].map(
            lambda value, c=column: _clip_truth(value, name=c)
        )
    return output.sort_values(
        [CELL_KEY, *CANDIDATE_KEY], kind="mergesort"
    ).reset_index(drop=True)


class PslDirectEvidenceBackendV2:
    """Official pslpython backend with positive direct-evidence rules only."""

    research_valid = True
    relation = "CALLS"
    target_predicate = "ConfirmedCallsV2"

    _predicate_names = {
        "candidate": "CandidateDirectV2",
        "direct_trace": "DirectTraceV2",
        "client_server": "ClientServerV2",
        "workload": "WorkloadPairV2",
    }

    def __init__(
        self,
        *,
        weights: PslDirectRuleWeights | None = None,
        profile_id: str = "direct_union",
        disabled_rules: Sequence[str] = (),
        random_seed: int = 7,
        jvm_options: Sequence[str] = ("-Xms128m", "-Xmx1024m"),
        temporary_parent: str | Path | None = None,
    ) -> None:
        self.weights = weights or PslDirectRuleWeights()
        self.profile_id = str(profile_id)
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        self.disabled_rules = frozenset(
            str(value).upper() for value in disabled_rules
        )
        known = {rule.rule_id for rule in _rule_specs(self.weights)}
        unknown = sorted(self.disabled_rules.difference(known))
        if unknown:
            raise ValueError(f"unknown disabled PSL v2 rules: {unknown}")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        if not jvm_options:
            raise ValueError("jvm_options must be non-empty")
        self.random_seed = random_seed
        self.jvm_options = tuple(str(value) for value in jvm_options)
        self.temporary_parent = (
            None if temporary_parent is None else Path(temporary_parent).resolve()
        )

    def active_rules(self) -> tuple[PslDirectRuleSpec, ...]:
        return tuple(
            rule
            for rule in _rule_specs(self.weights)
            if rule.weight > 0.0 and rule.rule_id not in self.disabled_rules
        )

    def availability(self) -> Availability:
        try:
            jpype, _model, _partition, _predicate, _rule, jar_path = (
                PslPythonBackend._load_runtime()
            )
            metadata = PslPythonBackend._runtime_metadata(jpype, jar_path)
        except (ImportError, ModuleNotFoundError) as exc:
            return Availability(SKIPPED, "PSL_DEPENDENCY_MISSING", str(exc))
        except FileNotFoundError as exc:
            return Availability(SKIPPED, "PSL_RUNTIME_MISSING", str(exc))
        except Exception as exc:
            if exc.__class__.__name__ == "JVMNotFoundException":
                return Availability(SKIPPED, "PSL_RUNTIME_MISSING", str(exc))
            return Availability(ERROR, "PSL_RUNTIME_INVALID", str(exc))
        metadata.update(
            {
                "backend": "PslDirectEvidenceBackendV2",
                "profile_id": self.profile_id,
                "target_predicate": self.target_predicate,
                "active_rules": [
                    rule.expression for rule in self.active_rules()
                ],
                "disabled_rules": sorted(self.disabled_rules),
                "negative_relation_rule_count": 0,
                "uses_a2_prior_for_confirmation": False,
            }
        )
        return Availability(READY, detail=json.dumps(metadata, sort_keys=True))

    @staticmethod
    def _write_observation(
        path: Path, frame: pd.DataFrame, column: str
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            for row in frame.itertuples(index=False):
                stream.write(
                    f"{row.cell_id}\t{row.subject}\t{row.object}\t"
                    f"{float(getattr(row, column)):.17g}\n"
                )

    def infer(self, frame: pd.DataFrame) -> PslDirectEvidenceResult:
        evidence = validate_direct_evidence(frame)
        active_rules = self.active_rules()
        if not active_rules:
            raise PslDirectEvidenceError("no active PSL v2 direct rules")

        jpype, Model, Partition, Predicate, Rule, jar_path = (
            PslPythonBackend._load_runtime()
        )
        runtime_metadata = PslPythonBackend._runtime_metadata(jpype, jar_path)
        runtime_metadata.pop("rules", None)
        runtime_metadata.pop("rule_weights", None)
        psl_options = {"random.seed": self.random_seed}
        temporary_path: Path | None = None
        scores: dict[tuple[str, str, str, str], float] = {}
        grounded_rule_count = 0
        grounded_atom_count = 0
        grounded_by_rule: dict[str, int] = {}
        jvm_was_started = False

        with _PSL_DIRECT_LOCK:
            jvm_was_started = bool(jpype.isJVMStarted())
            if jvm_was_started:
                jpype.JClass("org.linqs.psl.util.RandUtils").seed(
                    self.random_seed
                )
            with tempfile.TemporaryDirectory(
                prefix="logagent-psl-v2-",
                dir=(
                    None
                    if self.temporary_parent is None
                    else str(self.temporary_parent)
                ),
            ) as temporary:
                temporary_path = Path(temporary)
                model = Model(f"logagent-psl-v2-{self.profile_id}")
                for column, predicate_name in self._predicate_names.items():
                    predicate = Predicate(predicate_name, size=3)
                    model.add_predicate(predicate)
                    data_path = temporary_path / f"{column}.tsv"
                    self._write_observation(data_path, evidence, column)
                    predicate.add_data_file(
                        Partition.OBSERVATIONS, str(data_path)
                    )

                calls = Predicate(self.target_predicate, size=3)
                model.add_predicate(calls)
                target_path = temporary_path / "confirmed_calls_targets.tsv"
                with target_path.open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    for row in evidence.itertuples(index=False):
                        stream.write(
                            f"{row.cell_id}\t{row.subject}\t{row.object}\n"
                        )
                calls.add_data_file(Partition.TARGETS, str(target_path))

                for rule in active_rules:
                    if "!" in rule.body:
                        raise PslDirectEvidenceError(
                            f"negative relation rule is forbidden in PSL v2: "
                            f"{rule.rule_id}"
                        )
                    model.add_rule(Rule(rule.expression))

                grounded = model.ground(
                    psl_options=psl_options,
                    jvm_options=list(self.jvm_options),
                )
                ground_rules = grounded.get("groundRules", ())
                ground_atoms = grounded.get("atoms", ())
                grounded_rule_count = len(ground_rules)
                grounded_atom_count = len(ground_atoms)
                for index, rule in enumerate(active_rules):
                    grounded_by_rule[rule.rule_id] = sum(
                        int(item.get("ruleIndex", -1)) == index
                        for item in ground_rules
                    )

                expected = {
                    (
                        str(row.cell_id),
                        str(row.subject),
                        "CALLS",
                        str(row.object),
                    )
                    for row in evidence.itertuples(index=False)
                }
                atom_items = (
                    ground_atoms.items()
                    if isinstance(ground_atoms, Mapping)
                    else enumerate(ground_atoms)
                )
                grounded_calls = {
                    (
                        str(atom["arguments"][0]),
                        str(atom["arguments"][1]),
                        "CALLS",
                        str(atom["arguments"][2]),
                    )
                    for _atom_id, atom in atom_items
                    if str(atom.get("predicate", "")).upper()
                    == self.target_predicate.upper()
                }
                if grounded_calls != expected:
                    raise PslDirectEvidenceError(
                        "PSL v2 target atom coverage mismatch: "
                        f"missing={len(expected - grounded_calls)} "
                        f"extra={len(grounded_calls - expected)}"
                    )

                jpype.JClass("org.linqs.psl.util.RandUtils").seed(
                    self.random_seed
                )
                inferred = model.infer(
                    psl_options=psl_options,
                    jvm_options=list(self.jvm_options),
                )
                if calls not in inferred:
                    raise PslDirectEvidenceError(
                        "PSL did not return ConfirmedCallsV2 targets"
                    )
                result_frame = inferred[calls]
                for cell, subject, obj, truth in result_frame.itertuples(
                    index=False, name=None
                ):
                    key = (
                        str(cell),
                        str(subject),
                        "CALLS",
                        str(obj),
                    )
                    value = float(truth)
                    if (
                        not math.isfinite(value)
                        or not 0.0 <= value <= 1.0
                    ):
                        raise PslDirectEvidenceError(
                            f"invalid PSL v2 posterior {value} for {key}"
                        )
                    scores[key] = value
                if set(scores) != expected:
                    raise PslDirectEvidenceError(
                        "PSL v2 posterior coverage mismatch: "
                        f"missing={len(expected - set(scores))} "
                        f"extra={len(set(scores) - expected)}"
                    )

        assert temporary_path is not None
        metadata = {
            **runtime_metadata,
            "backend": "PslDirectEvidenceBackendV2",
            "profile_id": self.profile_id,
            "relation": "CALLS",
            "target_predicate": self.target_predicate,
            "candidate_count": len(evidence),
            "cell_count": int(evidence[CELL_KEY].nunique()),
            "random_seed": self.random_seed,
            "random_seed_reset_before_inference": True,
            "jvm_options": self.jvm_options,
            "jvm_options_applied_by_call": not jvm_was_started,
            "active_rules": [
                {
                    "rule_id": rule.rule_id,
                    "weight": rule.weight,
                    "expression": rule.expression,
                }
                for rule in active_rules
            ],
            "disabled_rules": sorted(self.disabled_rules),
            "grounded_rule_count_by_id": grounded_by_rule,
            "negative_relation_rule_count": 0,
            "broad_negative_rules_removed": True,
            "uses_a2_prior_for_confirmation": False,
            "unsupported_candidates_are_negative_edges": False,
            "unsupported_candidates_are_abstained_by_policy": True,
            "temporary_data_cleaned": not temporary_path.exists(),
            "java_runtime_version": str(
                jpype.JClass("java.lang.System").getProperty(
                    "java.runtime.version"
                )
            ),
        }
        return PslDirectEvidenceResult(
            scores=scores,
            grounded_rule_count=grounded_rule_count,
            grounded_atom_count=grounded_atom_count,
            metadata=metadata,
        )


__all__ = [
    "DIRECT_EVIDENCE_COLUMNS",
    "PslDirectEvidenceBackendV2",
    "PslDirectEvidenceError",
    "PslDirectEvidenceResult",
    "PslDirectRuleSpec",
    "PslDirectRuleWeights",
    "validate_direct_evidence",
]
