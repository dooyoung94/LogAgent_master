"""Frozen, CPU-friendly ONNX backend for the DeBERTa NLI ablations.

This module is intentionally independent from PyTorch and Optimum.  It loads a
local Hugging Face tokenizer together with an author-published ONNX sequence
classifier, and implements the :class:`recovery.DebertaBackend` protocol.

No network fallback is allowed: experiments must download and checksum the
model artifact before constructing :class:`OnnxDebertaNLIBackend`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from .recovery import ERROR, READY, SKIPPED, Availability


NLI_DEBERTA_V3_SMALL_REPO_ID = "cross-encoder/nli-deberta-v3-small"
NLI_DEBERTA_V3_SMALL_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
NLI_DEBERTA_V3_SMALL_AVX2_FILENAME = "onnx/model_quint8_avx2.onnx"
NLI_DEBERTA_V3_SMALL_AVX2_SHA256 = (
    "03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae"
)

_EXPECTED_LABELS = frozenset({"contradiction", "entailment", "neutral"})
_REQUIRED_INPUTS = frozenset({"input_ids", "attention_mask"})


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the artifact in RAM."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_softmax(logits: Any) -> np.ndarray:
    """Compute row-wise softmax in float64 and reject invalid model output."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("NLI logits must be a non-empty rank-2 array")
    if not np.isfinite(values).all():
        raise ValueError("NLI logits contain a non-finite value")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    denominators = np.sum(exponentials, axis=1, keepdims=True)
    if not np.isfinite(denominators).all() or np.any(denominators <= 0.0):
        raise ValueError("NLI softmax denominator is invalid")
    return exponentials / denominators


def _normal_label(label: Any) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _validated_label_indices(id2label: Mapping[Any, Any]) -> Mapping[str, int]:
    label_to_index: dict[str, int] = {}
    for raw_index, raw_label in id2label.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"NLI id2label has a non-integer index: {raw_index!r}") from exc
        label = _normal_label(raw_label)
        if label in label_to_index:
            raise ValueError(f"NLI id2label repeats label {label!r}")
        label_to_index[label] = index
    if frozenset(label_to_index) != _EXPECTED_LABELS:
        raise ValueError(
            "NLI id2label must contain exactly contradiction, entailment, and neutral"
        )
    if set(label_to_index.values()) != set(range(3)):
        raise ValueError("NLI id2label indices must be exactly 0, 1, and 2")
    return label_to_index


def _import_runtime() -> tuple[Any, Any, Any]:
    """Import optional runtime dependencies lazily for honest availability."""

    # ONNX Runtime binaries may otherwise emit process/device telemetry to a
    # Microsoft collection endpoint.  Relation-recovery experiments are fully
    # local and disable that channel before importing the runtime.
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime as ort  # type: ignore
    from transformers import AutoConfig, AutoTokenizer  # type: ignore

    return ort, AutoConfig, AutoTokenizer


@dataclass(frozen=True)
class PairCacheInfo:
    hits: int
    misses: int
    size: int
    inference_batches: int


@dataclass(frozen=True)
class DirectionContrastDiagnostic:
    """Counterfactual direction check for a relation-verbalization template."""

    premise: str
    forward_hypothesis: str
    reverse_hypothesis: str
    forward_entailment: float
    reverse_entailment: float
    entailment_margin: float
    minimum_margin: float
    gate_passed: bool
    reason_code: str

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchCompositionDiagnostic:
    """Detect score drift caused only by a companion item in an INT8 batch."""

    probe_pair: tuple[str, str]
    companion_pair: tuple[str, str]
    isolated_probabilities: Mapping[str, float]
    batched_probabilities: Mapping[str, float]
    max_absolute_delta: float
    tolerance: float
    gate_passed: bool
    reason_code: str

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class OnnxDebertaNLIBackend:
    """Actual DeBERTa NLI backend using a frozen local ONNX artifact.

    Parameters
    ----------
    model_dir:
        Local directory containing ``config.json``, tokenizer assets, and the
        ONNX artifact.  Hugging Face is always invoked with
        ``local_files_only=True``.
    expected_sha256:
        Optional full SHA-256 of the ONNX file.  A mismatch makes
        :meth:`availability` return ``ERROR`` rather than silently running a
        different model.
    batch_size / performance_mode:
        Paper-mode defaults to one pair per ONNX call.  ``batch_size > 1`` is
        accepted only with ``performance_mode=True`` and marks the backend
        ``research_valid=False`` because the official INT8 artifact has
        measurable batch-composition sensitivity.
    """

    research_valid = True
    local_files_only = True

    def __init__(
        self,
        model_dir: str | Path,
        *,
        onnx_filename: str = NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
        expected_sha256: str | None = None,
        revision: str | None = NLI_DEBERTA_V3_SMALL_REVISION,
        batch_size: int = 1,
        performance_mode: bool = False,
        max_length: int = 512,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > 1 and not performance_mode:
            raise ValueError(
                "batch_size > 1 requires performance_mode=True because the official "
                "INT8 artifact is batch-composition sensitive"
            )
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if not providers:
            raise ValueError("at least one ONNX Runtime provider is required")
        if intra_op_num_threads is not None and intra_op_num_threads <= 0:
            raise ValueError("intra_op_num_threads must be positive")
        if inter_op_num_threads is not None and inter_op_num_threads <= 0:
            raise ValueError("inter_op_num_threads must be positive")

        normalized_digest: str | None = None
        if expected_sha256 is not None:
            normalized_digest = str(expected_sha256).strip().lower()
            if len(normalized_digest) != 64 or any(
                character not in "0123456789abcdef" for character in normalized_digest
            ):
                raise ValueError("expected_sha256 must be a full hexadecimal SHA-256")

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.onnx_path = self.model_dir / onnx_filename
        self.expected_sha256 = normalized_digest
        self.revision = revision
        self.batch_size = int(batch_size)
        self.performance_mode = bool(performance_mode)
        # The official dynamic-INT8 export has measurable co-batch score drift.
        # Research runs therefore use one pair per inference call.
        self.research_valid = self.batch_size == 1 and not self.performance_mode
        self.max_length = int(max_length)
        self.providers = tuple(str(provider) for provider in providers)
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads

        self._tokenizer: Any = None
        self._session: Any = None
        self._output_name: str | None = None
        self._label_to_index: Mapping[str, int] | None = None
        self._actual_sha256: str | None = None
        self._cache: dict[tuple[str, str], tuple[float, float, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._inference_batches = 0
        self._lock = threading.RLock()

    @property
    def artifact_sha256(self) -> str:
        """Return and memoize the digest of the exact ONNX artifact."""

        with self._lock:
            if self._actual_sha256 is None:
                self._require_local_artifacts()
                self._actual_sha256 = sha256_file(self.onnx_path)
            return self._actual_sha256

    @property
    def label_to_index(self) -> Mapping[str, int]:
        self._load()
        assert self._label_to_index is not None
        return dict(self._label_to_index)

    def cache_info(self) -> PairCacheInfo:
        with self._lock:
            return PairCacheInfo(
                hits=self._cache_hits,
                misses=self._cache_misses,
                size=len(self._cache),
                inference_batches=self._inference_batches,
            )

    def metadata(self) -> Mapping[str, Any]:
        """Return provenance flags that must accompany an experiment result."""

        return {
            "backend": type(self).__name__,
            "model_dir": str(self.model_dir),
            "onnx_path": str(self.onnx_path),
            "revision": self.revision,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self._actual_sha256,
            "batch_size": self.batch_size,
            "performance_mode": self.performance_mode,
            "research_valid": self.research_valid,
            "local_files_only": self.local_files_only,
            "providers": self.providers,
            "max_length": self.max_length,
            "truncation_policy": "reject_over_budget",
            "label_to_index": dict(self._label_to_index or {}),
            "telemetry_disabled": True,
        }

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._inference_batches = 0

    def _require_local_artifacts(self) -> None:
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"local DeBERTa model directory is missing: {self.model_dir}")
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"local DeBERTa ONNX artifact is missing: {self.onnx_path}")
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"local DeBERTa config is missing: {self.model_dir / 'config.json'}"
            )

    def _validate_digest(self) -> None:
        if self.expected_sha256 is None:
            return
        if not hmac.compare_digest(self.artifact_sha256, self.expected_sha256):
            raise ValueError(
                "DeBERTa ONNX SHA-256 mismatch: "
                f"expected {self.expected_sha256}, got {self.artifact_sha256}"
            )

    def _load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            self._require_local_artifacts()
            self._validate_digest()
            ort, auto_config, auto_tokenizer = _import_runtime()
            disable_telemetry = getattr(ort, "disable_telemetry_events", None)
            if callable(disable_telemetry):
                disable_telemetry()

            config = auto_config.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            labels = _validated_label_indices(getattr(config, "id2label", {}))
            tokenizer = auto_tokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )

            options = ort.SessionOptions()
            if self.intra_op_num_threads is not None:
                options.intra_op_num_threads = self.intra_op_num_threads
            if self.inter_op_num_threads is not None:
                options.inter_op_num_threads = self.inter_op_num_threads
            session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=options,
                providers=list(self.providers),
            )

            inputs = {item.name: item for item in session.get_inputs()}
            if frozenset(inputs) != _REQUIRED_INPUTS:
                raise ValueError(
                    "DeBERTa ONNX inputs must be exactly input_ids and attention_mask"
                )
            for name in sorted(_REQUIRED_INPUTS):
                if str(getattr(inputs[name], "type", "")) != "tensor(int64)":
                    raise ValueError(f"DeBERTa ONNX input {name} must be tensor(int64)")

            outputs = list(session.get_outputs())
            if len(outputs) != 1:
                raise ValueError("DeBERTa ONNX model must expose exactly one logits output")

            self._tokenizer = tokenizer
            self._session = session
            self._output_name = str(outputs[0].name)
            self._label_to_index = labels

    def availability(self) -> Availability:
        try:
            self._load()
        except (ImportError, ModuleNotFoundError) as exc:
            return Availability(
                SKIPPED,
                "DEBERTA_DEPENDENCY_MISSING",
                str(exc),
                self.research_valid,
            )
        except FileNotFoundError as exc:
            return Availability(
                SKIPPED,
                "DEBERTA_MODEL_ARTIFACT_MISSING",
                str(exc),
                self.research_valid,
            )
        except Exception as exc:
            return Availability(
                ERROR, "DEBERTA_MODEL_INVALID", str(exc), self.research_valid
            )
        return Availability(READY, research_valid=self.research_valid)

    @staticmethod
    def _normal_pair(pair: Any) -> tuple[str, str]:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("each NLI item must be a (premise, hypothesis) pair")
        premise, hypothesis = pair
        if not isinstance(premise, str) or not isinstance(hypothesis, str):
            raise TypeError("NLI premise and hypothesis must both be strings")
        return premise, hypothesis

    def _infer_uncached(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[float, float, float]]:
        assert self._tokenizer is not None
        assert self._session is not None
        assert self._output_name is not None
        assert self._label_to_index is not None

        premises = [pair[0] for pair in pairs]
        hypotheses = [pair[1] for pair in pairs]
        encoded = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=False,
            return_tensors="np",
        )
        missing = _REQUIRED_INPUTS.difference(encoded)
        if missing:
            raise ValueError(f"tokenizer omitted required ONNX inputs: {sorted(missing)}")
        attention_mask = np.asarray(encoded["attention_mask"])
        token_lengths = tuple(int(value) for value in attention_mask.sum(axis=1))
        over_budget = [
            (index, length)
            for index, length in enumerate(token_lengths)
            if length > self.max_length
        ]
        if over_budget:
            raise ValueError(
                "NLI input exceeds the frozen token budget; silent truncation is "
                f"forbidden (max_length={self.max_length}, items={over_budget})"
            )
        feed = {
            name: np.asarray(encoded[name]).astype(np.int64, copy=False)
            for name in sorted(_REQUIRED_INPUTS)
        }
        logits = self._session.run([self._output_name], feed)[0]
        probabilities = stable_softmax(logits)
        if probabilities.shape != (len(pairs), 3):
            raise ValueError(
                "DeBERTa ONNX logits shape must be [batch, 3], got "
                f"{tuple(probabilities.shape)}"
            )

        contradiction = self._label_to_index["contradiction"]
        entailment = self._label_to_index["entailment"]
        neutral = self._label_to_index["neutral"]
        return [
            (
                float(row[contradiction]),
                float(row[entailment]),
                float(row[neutral]),
            )
            for row in probabilities
        ]

    def pair_token_lengths(
        self, pairs: Sequence[tuple[str, str]]
    ) -> tuple[int, ...]:
        """Return untruncated pair lengths for experiment diagnostics."""

        normalized = tuple(self._normal_pair(pair) for pair in pairs)
        if not normalized:
            return ()
        with self._lock:
            self._load()
            assert self._tokenizer is not None
            encoded = self._tokenizer(
                [pair[0] for pair in normalized],
                [pair[1] for pair in normalized],
                padding=False,
                truncation=False,
            )
            input_ids = encoded["input_ids"]
            return tuple(len(row) for row in input_ids)

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[Mapping[str, float]]:
        """Score pairs in batches, deduplicating identical pairs across calls."""

        normalized = tuple(self._normal_pair(pair) for pair in pairs)
        if not normalized:
            return ()
        with self._lock:
            self._load()
            pending: list[tuple[str, str]] = []
            seen_pending: set[tuple[str, str]] = set()
            for pair in normalized:
                if pair in self._cache or pair in seen_pending:
                    self._cache_hits += 1
                else:
                    self._cache_misses += 1
                    seen_pending.add(pair)
                    pending.append(pair)

            for offset in range(0, len(pending), self.batch_size):
                batch = pending[offset : offset + self.batch_size]
                scores = self._infer_uncached(batch)
                self._inference_batches += 1
                self._cache.update(zip(batch, scores))

            return tuple(
                {
                    "contradiction": self._cache[pair][0],
                    "entailment": self._cache[pair][1],
                    "neutral": self._cache[pair][2],
                }
                for pair in normalized
            )

    def direction_contrast(
        self,
        premise: str,
        forward_hypothesis: str,
        reverse_hypothesis: str,
        *,
        minimum_margin: float = 0.05,
    ) -> DirectionContrastDiagnostic:
        """Evaluate whether NLI support distinguishes a relation's direction.

        This is a diagnostic gate, not an inference fallback.  A failed gate
        should be recorded in the experiment report and handled by structural
        constraints rather than hidden or converted to a passing unit test.
        """

        if not math.isfinite(minimum_margin) or minimum_margin < 0.0:
            raise ValueError("minimum_margin must be finite and non-negative")
        rows = self.score_pairs(
            ((premise, forward_hypothesis), (premise, reverse_hypothesis))
        )
        forward = float(rows[0]["entailment"])
        reverse = float(rows[1]["entailment"])
        margin = forward - reverse
        gate_passed = margin >= minimum_margin
        return DirectionContrastDiagnostic(
            premise=premise,
            forward_hypothesis=forward_hypothesis,
            reverse_hypothesis=reverse_hypothesis,
            forward_entailment=forward,
            reverse_entailment=reverse,
            entailment_margin=margin,
            minimum_margin=float(minimum_margin),
            gate_passed=gate_passed,
            reason_code=(
                "DIRECTION_CONTRAST_PASS"
                if gate_passed
                else "DIRECTION_CONTRAST_INSUFFICIENT_MARGIN"
            ),
        )

    def batch_composition_contrast(
        self,
        probe_pair: tuple[str, str],
        companion_pair: tuple[str, str],
        *,
        tolerance: float = 1e-6,
    ) -> BatchCompositionDiagnostic:
        """Measure whether a probe's score changes when another pair is batched.

        The diagnostic bypasses the pair cache by design and does not mutate it.
        A failed gate is why paper-mode runs fix ``batch_size=1`` even though the
        backend retains an explicitly invalid performance batching mode.
        """

        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        probe = self._normal_pair(probe_pair)
        companion = self._normal_pair(companion_pair)
        with self._lock:
            self._load()
            isolated = self._infer_uncached((probe,))[0]
            batched = self._infer_uncached((probe, companion))[0]
        names = ("contradiction", "entailment", "neutral")
        isolated_map = {name: float(value) for name, value in zip(names, isolated)}
        batched_map = {name: float(value) for name, value in zip(names, batched)}
        maximum = max(
            abs(isolated_map[name] - batched_map[name]) for name in names
        )
        gate_passed = maximum <= tolerance
        return BatchCompositionDiagnostic(
            probe_pair=probe,
            companion_pair=companion,
            isolated_probabilities=isolated_map,
            batched_probabilities=batched_map,
            max_absolute_delta=maximum,
            tolerance=float(tolerance),
            gate_passed=gate_passed,
            reason_code=(
                "BATCH_COMPOSITION_INVARIANT"
                if gate_passed
                else "BATCH_COMPOSITION_SENSITIVE"
            ),
        )


__all__ = [
    "BatchCompositionDiagnostic",
    "DirectionContrastDiagnostic",
    "NLI_DEBERTA_V3_SMALL_AVX2_FILENAME",
    "NLI_DEBERTA_V3_SMALL_AVX2_SHA256",
    "NLI_DEBERTA_V3_SMALL_REPO_ID",
    "NLI_DEBERTA_V3_SMALL_REVISION",
    "OnnxDebertaNLIBackend",
    "PairCacheInfo",
    "sha256_file",
    "stable_softmax",
]
