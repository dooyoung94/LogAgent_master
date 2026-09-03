"""Multi-evidence PSL backend for bounded ``CALLS`` relation candidates.

The backend consumes only model-visible candidate evidence.  Evaluator labels,
mask targets, reference edges, fault labels and root-cause labels are forbidden
at the API boundary.  Every rule is a soft rule; a negative rule adjusts the
posterior but never hard-deletes an A2 candidate.
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
from .recovery import Availability, ERROR, READY, SKIPPED


_PSL_MULTI_LOCK = threading.Lock()
CELL_KEY = "cell_id"
CANDIDATE_KEY = ("subject", "predicate", "object")
MODEL_EVIDENCE_COLUMNS = (
    "candidate",
    "a2_prior",
    "trace_support",
    "boundary_support",
    "repeated_support",
    "direction_support",
    "operation_match",
    "endpoint_match",
    "role_compatibility",
    "direct_observed",
    "reverse_support",
    "direction_conflict",
    "self_loop",
)
FORBIDDEN_EVALUATOR_COLUMNS = frozenset(
    {
        "case",
        "fault",
        "role",
        "is_masked_target",
        "is_silver_matched",
        "root_cause_service",
        "ground_truth",
        "reference_graph",
        "mask_target",
    }
)


class PslMultiEvidenceError(RuntimeError):
    """Raised when the multi-evidence PSL contract cannot be satisfied."""


@dataclass(frozen=True)
class PslRuleWeights:
    """One preregistered PSL rule-weight profile."""

    a2_prior: float = 5.0
    trace_support: float = 3.0
    boundary_support: float = 1.5
    repeated_support: float = 1.5
    direction_support: float = 3.0
    operation_match: float = 1.5
    endpoint_match: float = 1.0
    role_compatibility: float = 1.0
    direct_observed: float = 12.0
    reverse_support_negative: float = 3.0
    direction_conflict_negative: float = 4.0
    self_loop_negative: float = 20.0
    sparsity: float = 0.75

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in asdict(self).items()
            if not math.isfinite(float(value)) or float(value) < 0.0
        }
        if invalid:
            raise ValueError(f"PSL rule weights must be finite and non-negative: {invalid}")
        if self.a2_prior <= 0.0:
            raise ValueError("A2 prior must retain positive weight")
        if self.sparsity <= 0.0:
            raise ValueError("sparsity prior must retain positive weight")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PslRuleWeights":
        expected = set(asdict(cls()))
        extra = sorted(set(value).difference(expected))
        missing = sorted(expected.difference(value))
        if extra or missing:
            raise ValueError(f"invalid PSL weight profile; missing={missing} extra={extra}")
        return cls(**{name: float(value[name]) for name in expected})


@dataclass(frozen=True)
class PslRuleSpec:
    rule_id: str
    weight: float
    body: str

    @property
    def expression(self) -> str:
        return f"{self.weight:.12g}: {self.body} ^2"


@dataclass(frozen=True)
class PslMultiEvidenceResult:
    scores: Mapping[tuple[str, str, str, str], float]
    grounded_rule_count: int
    grounded_atom_count: int
    metadata: Mapping[str, Any]


def _rule_specs(weights: PslRuleWeights) -> tuple[PslRuleSpec, ...]:
    guarded = "Candidate(C, S, O)"
    return (
        PslRuleSpec(
            "A2_PRIOR",
            weights.a2_prior,
            f"{guarded} & A2Prior(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "TRACE_SUPPORT",
            weights.trace_support,
            f"{guarded} & TraceSupport(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "BOUNDARY_SUPPORT",
            weights.boundary_support,
            f"{guarded} & BoundarySupport(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "REPEATED_SUPPORT",
            weights.repeated_support,
            f"{guarded} & RepeatedSupport(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "DIRECTION_SUPPORT",
            weights.direction_support,
            f"{guarded} & DirectionSupport(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "OPERATION_MATCH",
            weights.operation_match,
            f"{guarded} & OperationMatch(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "ENDPOINT_MATCH",
            weights.endpoint_match,
            f"{guarded} & EndpointMatch(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "ROLE_COMPATIBILITY",
            weights.role_compatibility,
            f"{guarded} & RoleCompatibility(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "DIRECT_OBSERVED",
            weights.direct_observed,
            f"{guarded} & DirectObserved(C, S, O) -> Calls(C, S, O)",
        ),
        PslRuleSpec(
            "REVERSE_SUPPORT_NEGATIVE",
            weights.reverse_support_negative,
            f"{guarded} & ReverseSupport(C, S, O) -> !Calls(C, S, O)",
        ),
        PslRuleSpec(
            "DIRECTION_CONFLICT_NEGATIVE",
            weights.direction_conflict_negative,
            f"{guarded} & DirectionConflict(C, S, O) -> !Calls(C, S, O)",
        ),
        PslRuleSpec(
            "SELF_LOOP_NEGATIVE",
            weights.self_loop_negative,
            f"{guarded} & SelfLoop(C, S, O) -> !Calls(C, S, O)",
        ),
        PslRuleSpec(
            "SPARSITY",
            weights.sparsity,
            "!Calls(C, S, O)",
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


def validate_model_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize the leakage-safe PSL input table."""

    forbidden = FORBIDDEN_EVALUATOR_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise PslMultiEvidenceError(
            "PSL model input contains evaluator columns: " + ", ".join(sorted(forbidden))
        )
    required = (CELL_KEY, *CANDIDATE_KEY, *MODEL_EVIDENCE_COLUMNS)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise PslMultiEvidenceError(f"PSL evidence is missing columns: {missing}")
    if frame.empty:
        raise PslMultiEvidenceError("PSL evidence cannot be empty")

    output = frame[list(required)].copy()
    output[CELL_KEY] = output[CELL_KEY].map(_safe_atom)
    output["subject"] = output["subject"].map(_safe_atom)
    output["predicate"] = output["predicate"].astype(str).str.upper()
    output["object"] = output["object"].map(_safe_atom)
    if set(output["predicate"]) != {"CALLS"}:
        raise PslMultiEvidenceError("PSL v1 accepts CALLS candidates only")

    duplicate = output.duplicated([CELL_KEY, *CANDIDATE_KEY], keep=False)
    if bool(duplicate.any()):
        examples = output.loc[duplicate, [CELL_KEY, *CANDIDATE_KEY]].head(5).to_dict("records")
        raise PslMultiEvidenceError(f"duplicate PSL candidates: {examples}")
    for column in MODEL_EVIDENCE_COLUMNS:
        output[column] = output[column].map(lambda value, c=column: _clip_truth(value, name=c))
    return output.sort_values([CELL_KEY, *CANDIDATE_KEY], kind="mergesort").reset_index(drop=True)


class PslMultiEvidenceBackendV1:
    """Official pslpython backend for one pooled, cell-isolated CALLS program."""

    research_valid = True
    relation = "CALLS"

    _predicate_names = {
        "candidate": "Candidate",
        "a2_prior": "A2Prior",
        "trace_support": "TraceSupport",
        "boundary_support": "BoundarySupport",
        "repeated_support": "RepeatedSupport",
        "direction_support": "DirectionSupport",
        "operation_match": "OperationMatch",
        "endpoint_match": "EndpointMatch",
        "role_compatibility": "RoleCompatibility",
        "direct_observed": "DirectObserved",
        "reverse_support": "ReverseSupport",
        "direction_conflict": "DirectionConflict",
        "self_loop": "SelfLoop",
    }

    def __init__(
        self,
        *,
        weights: PslRuleWeights | None = None,
        profile_id: str = "balanced",
        disabled_rules: Sequence[str] = (),
        random_seed: int = 7,
        jvm_options: Sequence[str] = ("-Xms128m", "-Xmx1024m"),
        temporary_parent: str | Path | None = None,
    ) -> None:
        self.weights = weights or PslRuleWeights()
        self.profile_id = str(profile_id)
        self.disabled_rules = frozenset(str(value).upper() for value in disabled_rules)
        known = {rule.rule_id for rule in _rule_specs(self.weights)}
        unknown = sorted(self.disabled_rules.difference(known))
        if unknown:
            raise ValueError(f"unknown disabled PSL rules: {unknown}")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        if not jvm_options:
            raise ValueError("jvm_options must be non-empty")
        self.random_seed = random_seed
        self.jvm_options = tuple(str(value) for value in jvm_options)
        self.temporary_parent = (
            None if temporary_parent is None else Path(temporary_parent).resolve()
        )

    def active_rules(self) -> tuple[PslRuleSpec, ...]:
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
                "backend": "PslMultiEvidenceBackendV1",
                "profile_id": self.profile_id,
                "active_rules": [rule.expression for rule in self.active_rules()],
                "disabled_rules": sorted(self.disabled_rules),
            }
        )
        return Availability(READY, detail=json.dumps(metadata, sort_keys=True))

    @staticmethod
    def _write_observation(path: Path, frame: pd.DataFrame, column: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            for row in frame.itertuples(index=False):
                stream.write(
                    f"{row.cell_id}\t{row.subject}\t{row.object}\t"
                    f"{float(getattr(row, column)):.17g}\n"
                )

    def infer(self, frame: pd.DataFrame) -> PslMultiEvidenceResult:
        evidence = validate_model_evidence(frame)
        active_rules = self.active_rules()
        if not active_rules:
            raise PslMultiEvidenceError("no active PSL rules")

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

        with _PSL_MULTI_LOCK:
            jvm_was_started = bool(jpype.isJVMStarted())
            if jvm_was_started:
                jpype.JClass("org.linqs.psl.util.RandUtils").seed(self.random_seed)
            with tempfile.TemporaryDirectory(
                prefix="logagent-psl-v1-",
                dir=None if self.temporary_parent is None else str(self.temporary_parent),
            ) as temporary:
                temporary_path = Path(temporary)
                model = Model(f"logagent-psl-v1-{self.profile_id}")
                predicates: dict[str, Any] = {}
                for column, predicate_name in self._predicate_names.items():
                    predicate = Predicate(predicate_name, size=3)
                    predicates[column] = predicate
                    model.add_predicate(predicate)
                    path = temporary_path / f"{column}.tsv"
                    self._write_observation(path, evidence, column)
                    predicate.add_data_file(Partition.OBSERVATIONS, str(path))

                calls = Predicate("Calls", size=3)
                model.add_predicate(calls)
                target_path = temporary_path / "calls_targets.tsv"
                with target_path.open("w", encoding="utf-8", newline="") as stream:
                    for row in evidence.itertuples(index=False):
                        stream.write(f"{row.cell_id}\t{row.subject}\t{row.object}\n")
                calls.add_data_file(Partition.TARGETS, str(target_path))

                for rule in active_rules:
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
                        int(item.get("ruleIndex", -1)) == index for item in ground_rules
                    )

                expected = {
                    (str(row.cell_id), str(row.subject), "CALLS", str(row.object))
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
                    if str(atom.get("predicate", "")).upper() == "CALLS"
                }
                if grounded_calls != expected:
                    raise PslMultiEvidenceError(
                        "PSL target atom coverage mismatch: "
                        f"missing={len(expected - grounded_calls)} extra={len(grounded_calls - expected)}"
                    )

                jpype.JClass("org.linqs.psl.util.RandUtils").seed(self.random_seed)
                inferred = model.infer(
                    psl_options=psl_options,
                    jvm_options=list(self.jvm_options),
                )
                if calls not in inferred:
                    raise PslMultiEvidenceError("PSL did not return Calls targets")
                result_frame = inferred[calls]
                for cell, subject, obj, truth in result_frame.itertuples(
                    index=False, name=None
                ):
                    key = (str(cell), str(subject), "CALLS", str(obj))
                    value = float(truth)
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise PslMultiEvidenceError(f"invalid PSL posterior {value} for {key}")
                    scores[key] = value
                if set(scores) != expected:
                    raise PslMultiEvidenceError(
                        "PSL posterior coverage mismatch: "
                        f"missing={len(expected - set(scores))} extra={len(set(scores) - expected)}"
                    )

        assert temporary_path is not None
        metadata = {
            **runtime_metadata,
            "backend": "PslMultiEvidenceBackendV1",
            "profile_id": self.profile_id,
            "relation": "CALLS",
            "candidate_count": len(evidence),
            "cell_count": int(evidence[CELL_KEY].nunique()),
            "random_seed": self.random_seed,
            "random_seed_reset_before_inference": True,
            "jvm_options": self.jvm_options,
            "jvm_options_applied_by_call": not jvm_was_started,
            "active_rules": [
                {"rule_id": rule.rule_id, "weight": rule.weight, "expression": rule.expression}
                for rule in active_rules
            ],
            "disabled_rules": sorted(self.disabled_rules),
            "grounded_rule_count_by_id": grounded_by_rule,
            "temporary_data_cleaned": not temporary_path.exists(),
            "java_runtime_version": str(
                jpype.JClass("java.lang.System").getProperty("java.runtime.version")
            ),
        }
        return PslMultiEvidenceResult(
            scores=scores,
            grounded_rule_count=grounded_rule_count,
            grounded_atom_count=grounded_atom_count,
            metadata=metadata,
        )


__all__ = [
    "CANDIDATE_KEY",
    "CELL_KEY",
    "FORBIDDEN_EVALUATOR_COLUMNS",
    "MODEL_EVIDENCE_COLUMNS",
    "PslMultiEvidenceBackendV1",
    "PslMultiEvidenceError",
    "PslMultiEvidenceResult",
    "PslRuleSpec",
    "PslRuleWeights",
    "validate_model_evidence",
]
