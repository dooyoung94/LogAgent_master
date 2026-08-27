"""Official PSL backend for the A5 ``CALLS`` relation-recovery stage.

This module deliberately wraps :mod:`pslpython`; it does not contain a local
optimizer or a mock fallback.  Each candidate's upstream score is supplied as
the truth value of ``Evidence(subject, object)`` and PSL infers the open
``Calls(subject, object)`` target.  Other predicates are rejected because
mixing relation semantics into these fixed rules would make the A5 ablation
ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import re
import tempfile
import threading
from typing import Any, Mapping, Sequence
import zipfile

from .recovery import Availability, Candidate, InferenceContext, READY, SKIPPED, ERROR


_PSL_LOCK = threading.Lock()
_JPYPE_PIN = re.compile(r"^JPype1\s*==\s*([^;\s]+)", re.IGNORECASE)


@dataclass(frozen=True)
class PslInferenceResult:
    """Machine-readable result accepted by :class:`recovery.PslBackend`."""

    scores: Mapping[tuple[str, str, str], float]
    grounded_rule_count: int
    grounded_atom_count: int
    metadata: Mapping[str, Any]


def _properties(raw: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        output[key.strip()] = value.strip().replace("\\:", ":")
    return output


class PslPythonBackend:
    """Run fixed, squared-hinge PSL rules for ``CALLS`` candidates only.

    The two weights are intentionally class constants rather than constructor
    parameters.  Changing them creates a different A5 model and must therefore
    be introduced as a separately versioned experiment.
    """

    research_valid = True
    relation = "CALLS"
    evidence_weight = 5.0
    sparsity_weight = 0.5
    evidence_rule = "5.0: Evidence(S, O) -> Calls(S, O) ^2"
    sparsity_rule = "0.5: !Calls(S, O) ^2"

    def __init__(
        self,
        *,
        random_seed: int = 7,
        jvm_options: Sequence[str] = ("-Xms128m", "-Xmx512m"),
        temporary_parent: str | Path | None = None,
    ) -> None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        if not jvm_options:
            raise ValueError("jvm_options must include deterministic memory bounds")
        self.random_seed = random_seed
        self.jvm_options = tuple(str(option) for option in jvm_options)
        self.temporary_parent = (
            None if temporary_parent is None else Path(temporary_parent).resolve()
        )

    @staticmethod
    def _load_runtime() -> tuple[Any, Any, Any, Any, Any, Path]:
        import jpype  # type: ignore
        from pslpython.model import Model  # type: ignore
        from pslpython.partition import Partition  # type: ignore
        from pslpython.predicate import Predicate  # type: ignore
        from pslpython.rule import Rule  # type: ignore
        import pslpython.runtime as runtime  # type: ignore

        jar_path = Path(runtime.JAR_PATH).resolve()
        if not jar_path.is_file():
            raise FileNotFoundError(f"PSL runtime jar is missing: {jar_path}")
        return jpype, Model, Partition, Predicate, Rule, jar_path

    @staticmethod
    def _runtime_metadata(jpype: Any, jar_path: Path) -> dict[str, Any]:
        pslpython_version = importlib_metadata.version("pslpython")
        jpype_version = importlib_metadata.version("JPype1")
        declared_pin = ""
        for requirement in importlib_metadata.requires("pslpython") or ():
            match = _JPYPE_PIN.match(requirement.strip())
            if match:
                declared_pin = match.group(1)
                break

        runtime_version = "unknown"
        runtime_commit = "unknown"
        try:
            with zipfile.ZipFile(jar_path) as archive:
                pom = _properties(
                    archive.read(
                        "META-INF/maven/org.linqs/psl-runtime/pom.properties"
                    ).decode("utf-8")
                )
                git = _properties(archive.read("git.properties").decode("utf-8"))
            runtime_version = pom.get("version", git.get("git.build.version", "unknown"))
            runtime_commit = git.get(
                "git.commit.id.abbrev", git.get("git.commit.id", "unknown")
            )
        except (KeyError, OSError, UnicodeDecodeError, zipfile.BadZipFile):
            # The jar still remains runnable; unknown provenance is surfaced in
            # metadata instead of being silently fabricated.
            pass

        return {
            "backend": "pslpython",
            "relation": "CALLS",
            "pslpython_version": pslpython_version,
            "psl_runtime_version": runtime_version,
            "psl_runtime_commit": runtime_commit,
            "psl_runtime_artifact": str(jar_path),
            "psl_runtime_sha256": hashlib.sha256(jar_path.read_bytes()).hexdigest(),
            "jpype_version": jpype_version,
            "declared_jpype_pin": declared_pin or None,
            "compatibility_override": bool(declared_pin and jpype_version != declared_pin),
            "python_version": platform.python_version(),
            "jvm_path": str(jpype.getDefaultJVMPath()),
            "rules": (PslPythonBackend.evidence_rule, PslPythonBackend.sparsity_rule),
            "rule_weights": {
                "evidence": PslPythonBackend.evidence_weight,
                "sparsity": PslPythonBackend.sparsity_weight,
            },
        }

    def availability(self) -> Availability:
        """Check the official wheel, bundled runtime jar, and local JVM."""

        try:
            jpype, _model, _partition, _predicate, _rule, jar_path = self._load_runtime()
            metadata = self._runtime_metadata(jpype, jar_path)
        except (ImportError, ModuleNotFoundError) as exc:
            return Availability(SKIPPED, "PSL_DEPENDENCY_MISSING", str(exc))
        except FileNotFoundError as exc:
            return Availability(SKIPPED, "PSL_RUNTIME_MISSING", str(exc))
        except Exception as exc:
            if exc.__class__.__name__ == "JVMNotFoundException":
                return Availability(SKIPPED, "PSL_RUNTIME_MISSING", str(exc))
            return Availability(ERROR, "PSL_RUNTIME_INVALID", str(exc))
        return Availability(READY, detail=json.dumps(metadata, sort_keys=True))

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[Candidate],
        local_scores: Mapping[tuple[str, str, str], float],
    ) -> tuple[tuple[Candidate, float], ...]:
        validated: list[tuple[Candidate, float]] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            if candidate.predicate.upper() != "CALLS":
                raise ValueError("PslPythonBackend supports CALLS candidates only")
            if candidate.key in seen:
                raise ValueError(f"duplicate PSL candidate: {candidate.candidate_id}")
            seen.add(candidate.key)
            if candidate.key not in local_scores:
                raise KeyError(f"missing local score for candidate: {candidate.candidate_id}")
            score = float(local_scores[candidate.key])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"local score must be finite and in [0, 1]: {candidate.candidate_id}"
                )
            for atom in (candidate.subject, candidate.object):
                if any(character in atom for character in ("\t", "\r", "\n")):
                    raise ValueError("PSL atom identifiers cannot contain tabs or newlines")
            validated.append((candidate, score))
        return tuple(validated)

    def infer(
        self,
        *,
        context: InferenceContext,
        candidates: Sequence[Candidate],
        local_scores: Mapping[tuple[str, str, str], float],
    ) -> PslInferenceResult:
        """Ground and infer fixed ``Evidence -> Calls`` rules.

        ``context`` is accepted to satisfy the shared backend protocol but is
        intentionally not inspected; this prevents evaluator-only incident
        fields from entering A5.  Empty candidate sets return without starting
        a JVM.
        """

        del context
        validated = self._validate_candidates(candidates, local_scores)
        if not validated:
            return PslInferenceResult(
                scores={},
                grounded_rule_count=0,
                grounded_atom_count=0,
                metadata={
                    "backend": "pslpython",
                    "relation": "CALLS",
                    "pslpython_version": None,
                    "psl_runtime_version": None,
                    "psl_runtime_commit": None,
                    "jpype_version": None,
                    "compatibility_override": None,
                    "random_seed": self.random_seed,
                    "jvm_options": self.jvm_options,
                    "empty_candidate_set": True,
                    "temporary_data_cleaned": True,
                },
            )

        jpype, Model, Partition, Predicate, Rule, jar_path = self._load_runtime()
        base_metadata = self._runtime_metadata(jpype, jar_path)
        psl_options = {"random.seed": self.random_seed}
        temporary_path: Path | None = None
        scores: dict[tuple[str, str, str], float] = {}
        grounded_rule_count = 0
        grounded_atom_count = 0
        grounded_evidence_rule_count = 0
        grounded_prior_rule_count = 0
        jvm_was_started = False

        with _PSL_LOCK:
            jvm_was_started = bool(jpype.isJVMStarted())
            if jvm_was_started:
                jpype.JClass("org.linqs.psl.util.RandUtils").seed(self.random_seed)
            with tempfile.TemporaryDirectory(
                prefix="logagent-psl-",
                dir=None if self.temporary_parent is None else str(self.temporary_parent),
            ) as temporary:
                temporary_path = Path(temporary)
                evidence_path = temporary_path / "evidence.tsv"
                targets_path = temporary_path / "calls_targets.tsv"
                with evidence_path.open("w", encoding="utf-8", newline="") as stream:
                    for candidate, score in validated:
                        stream.write(
                            f"{candidate.subject}\t{candidate.object}\t{score:.17g}\n"
                        )
                with targets_path.open("w", encoding="utf-8", newline="") as stream:
                    for candidate, _score in validated:
                        stream.write(f"{candidate.subject}\t{candidate.object}\n")

                model = Model("logagent-a5-calls")
                evidence = Predicate("Evidence", size=2)
                calls = Predicate("Calls", size=2)
                model.add_predicate(evidence)
                model.add_predicate(calls)
                model.add_rule(Rule(self.evidence_rule))
                model.add_rule(Rule(self.sparsity_rule))
                evidence.add_data_file(Partition.OBSERVATIONS, str(evidence_path))
                calls.add_data_file(Partition.TARGETS, str(targets_path))

                grounded = model.ground(
                    psl_options=psl_options,
                    jvm_options=list(self.jvm_options),
                )
                ground_rules = grounded.get("groundRules", ())
                ground_atoms = grounded.get("atoms", ())
                grounded_rule_count = len(ground_rules)
                grounded_atom_count = len(ground_atoms)
                grounded_evidence_rule_count = sum(
                    int(rule.get("ruleIndex", -1)) == 0 for rule in ground_rules
                )
                grounded_prior_rule_count = sum(
                    int(rule.get("ruleIndex", -1)) == 1 for rule in ground_rules
                )
                if grounded_prior_rule_count != len(validated):
                    raise RuntimeError(
                        "PSL target grounding coverage mismatch: "
                        f"expected {len(validated)} prior rules, "
                        f"got {grounded_prior_rule_count}"
                    )
                if grounded_evidence_rule_count > len(validated):
                    raise RuntimeError("PSL produced duplicate evidence ground rules")
                known_ground_rules = (
                    grounded_evidence_rule_count + grounded_prior_rule_count
                )
                if known_ground_rules != grounded_rule_count:
                    raise RuntimeError("PSL produced a ground rule outside the fixed A5 model")

                atom_items = (
                    ground_atoms.items()
                    if isinstance(ground_atoms, Mapping)
                    else enumerate(ground_atoms)
                )
                grounded_calls = {
                    (str(atom["arguments"][0]), "CALLS", str(atom["arguments"][1]))
                    for _atom_id, atom in atom_items
                    if str(atom.get("predicate", "")).upper() == "CALLS"
                }
                expected_keys = {candidate.key for candidate, _score in validated}
                if grounded_calls != expected_keys:
                    raise RuntimeError(
                        "PSL target atom coverage mismatch: "
                        f"missing={sorted(expected_keys - grounded_calls)}, "
                        f"extra={sorted(grounded_calls - expected_keys)}"
                    )

                # ``random.seed`` initializes PSL's process-global RNG only on
                # the first runtime call.  Grounding or a previous inference in
                # the same JVM may advance it, so reset immediately before each
                # optimization to make repeated backend calls reproducible.
                jpype.JClass("org.linqs.psl.util.RandUtils").seed(self.random_seed)
                inferred = model.infer(
                    psl_options=psl_options,
                    jvm_options=list(self.jvm_options),
                )
                if calls not in inferred:
                    raise RuntimeError("PSL did not return the Calls target predicate")
                frame = inferred[calls]
                for subject, obj, truth in frame.itertuples(index=False, name=None):
                    scores[(str(subject), "CALLS", str(obj))] = float(truth)

                if set(scores) != expected_keys:
                    missing = sorted(expected_keys - set(scores))
                    extra = sorted(set(scores) - expected_keys)
                    raise RuntimeError(
                        f"PSL target coverage mismatch: missing={missing}, extra={extra}"
                    )

        assert temporary_path is not None
        metadata = {
            **base_metadata,
            "random_seed": self.random_seed,
            "random_seed_reset_before_inference": True,
            "jvm_options": self.jvm_options,
            "jvm_options_applied_by_call": not jvm_was_started,
            "java_runtime_version": str(
                jpype.JClass("java.lang.System").getProperty("java.runtime.version")
            ),
            "candidate_count": len(validated),
            "grounded_evidence_rule_count": grounded_evidence_rule_count,
            "grounded_prior_rule_count": grounded_prior_rule_count,
            "pruned_evidence_rule_count": (
                len(validated) - grounded_evidence_rule_count
            ),
            "maximum_grounded_rule_count": 2 * len(validated),
            "temporary_data_cleaned": not temporary_path.exists(),
            "empty_candidate_set": False,
        }
        return PslInferenceResult(
            scores=scores,
            grounded_rule_count=grounded_rule_count,
            grounded_atom_count=grounded_atom_count,
            metadata=metadata,
        )


__all__ = ["PslInferenceResult", "PslPythonBackend"]
