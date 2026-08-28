from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


import logagent_benchmark.onnx_deberta as onnx_deberta  # noqa: E402
from logagent_benchmark.onnx_deberta import (  # noqa: E402
    NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
    OnnxDebertaNLIBackend,
    stable_softmax,
)
from logagent_benchmark.recovery import ERROR, READY, SKIPPED  # noqa: E402


class _FakeSession:
    instances: list["_FakeSession"] = []

    def __init__(self, path: str, *, sess_options: object, providers: list[str]) -> None:
        self.path = path
        self.sess_options = sess_options
        self.providers = providers
        self.feeds: list[dict[str, np.ndarray]] = []
        self.__class__.instances.append(self)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="logits")]

    def run(
        self, output_names: list[str], feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        if output_names != ["logits"]:
            raise AssertionError(f"unexpected output selection: {output_names}")
        self.feeds.append({key: value.copy() for key, value in feed.items()})
        batch = feed["input_ids"].shape[0]
        signal = feed["input_ids"][:, 0].astype(np.float64) / 10.0
        return [np.column_stack((-signal, signal + 1.0, np.zeros(batch)))]


class _FakeSessionOptions:
    pass


def _fake_runtime(
    id2label: dict[int, str] | None = None,
) -> tuple[object, type, type]:
    labels = id2label or {0: "contradiction", 1: "entailment", 2: "neutral"}

    class FakeAutoConfig:
        calls: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            cls.calls.append((path, kwargs))
            return SimpleNamespace(id2label=labels)

    class FakeTokenizer:
        load_calls: list[tuple[str, dict[str, object]]] = []
        encode_calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> "FakeTokenizer":
            cls.load_calls.append((path, kwargs))
            return cls()

        def __call__(
            self,
            premises: list[str],
            hypotheses: list[str],
            **kwargs: object,
        ) -> dict[str, np.ndarray]:
            self.__class__.encode_calls.append(
                {
                    "premises": premises,
                    "hypotheses": hypotheses,
                    **kwargs,
                }
            )
            batch = len(premises)
            return {
                # Deliberately non-int64 to prove the backend casts both inputs.
                "input_ids": np.arange(batch * 4, dtype=np.int32).reshape(batch, 4),
                "attention_mask": np.ones((batch, 4), dtype=np.int16),
            }

    fake_ort = SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        InferenceSession=_FakeSession,
    )
    return fake_ort, FakeAutoConfig, FakeTokenizer


def _fake_runtime_with_pair_lengths() -> tuple[object, type, type]:
    """Return a tokenizer whose exact, untruncated pair lengths are observable.

    The fake follows DeBERTa's pair convention for this test: four special
    tokens plus the whitespace-token counts of premise and hypothesis.  When
    truncation is requested it behaves like a real tokenizer and clips to
    ``max_length``; this makes a silent-truncation regression observable.
    """

    fake_ort, fake_config, base_tokenizer = _fake_runtime()

    class LengthAwareTokenizer(base_tokenizer):
        load_calls: list[tuple[str, dict[str, object]]] = []
        encode_calls: list[dict[str, object]] = []

        @staticmethod
        def _pair_length(premise: str, hypothesis: str) -> int:
            return len(premise.split()) + len(hypothesis.split()) + 4

        def encode(
            self,
            premise: str,
            hypothesis: str,
            **kwargs: object,
        ) -> list[int]:
            self.__class__.encode_calls.append(
                {"premise": premise, "hypothesis": hypothesis, **kwargs}
            )
            length = self._pair_length(premise, hypothesis)
            if kwargs.get("truncation"):
                length = min(length, int(kwargs.get("max_length", length)))
            return list(range(length))

        def __call__(
            self,
            premises: list[str],
            hypotheses: list[str],
            **kwargs: object,
        ) -> dict[str, object]:
            self.__class__.encode_calls.append(
                {
                    "premises": premises,
                    "hypotheses": hypotheses,
                    **kwargs,
                }
            )
            lengths = [
                self._pair_length(premise, hypothesis)
                for premise, hypothesis in zip(premises, hypotheses)
            ]
            if kwargs.get("truncation"):
                maximum = int(kwargs.get("max_length", max(lengths)))
                lengths = [min(length, maximum) for length in lengths]

            # An unpadded call intentionally returns ragged Python lists, as
            # Hugging Face tokenizers do without tensor conversion.
            if not kwargs.get("padding"):
                rows = [list(range(length)) for length in lengths]
                return {
                    "input_ids": rows,
                    "attention_mask": [[1] * length for length in lengths],
                    "length": lengths,
                }

            width = max(lengths)
            input_ids = np.zeros((len(lengths), width), dtype=np.int32)
            attention_mask = np.zeros((len(lengths), width), dtype=np.int16)
            for row_index, length in enumerate(lengths):
                input_ids[row_index, :length] = np.arange(length, dtype=np.int32)
                attention_mask[row_index, :length] = 1
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "length": lengths,
            }

    return fake_ort, fake_config, LengthAwareTokenizer


class OnnxDebertaUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSession.instances.clear()

    def _model_dir(self, root: str, payload: bytes = b"fake-onnx") -> Path:
        model_dir = Path(root) / "model"
        (model_dir / "onnx").mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "onnx" / "model_quint8_avx2.onnx").write_bytes(payload)
        return model_dir

    def test_stable_softmax_handles_extreme_logits(self) -> None:
        probabilities = stable_softmax(
            np.asarray([[10_000.0, 0.0, -10_000.0], [-10_000.0, 0.0, 10_000.0]])
        )
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(probabilities.argmax(axis=1).tolist(), [0, 2])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            stable_softmax([[float("nan"), 0.0, 1.0]])

    def test_batching_int64_inputs_local_only_and_identical_pair_cache(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            payload = b"fake-onnx"
            model_dir = self._model_dir(root, payload)
            digest = hashlib.sha256(payload).hexdigest()
            fake_ort, fake_config, fake_tokenizer = _fake_runtime()
            with mock.patch.object(
                onnx_deberta,
                "_import_runtime",
                return_value=(fake_ort, fake_config, fake_tokenizer),
            ):
                backend = OnnxDebertaNLIBackend(
                    model_dir,
                    expected_sha256=digest,
                    batch_size=2,
                    performance_mode=True,
                )
                availability = backend.availability()
                self.assertEqual(availability.status, READY)
                self.assertFalse(availability.research_valid)
                pairs = (
                    ("premise-1", "hypothesis-1"),
                    ("premise-1", "hypothesis-1"),
                    ("premise-2", "hypothesis-2"),
                    ("premise-3", "hypothesis-3"),
                )
                scores = backend.score_pairs(pairs)
                repeated = backend.score_pairs(pairs[:2])

            self.assertEqual(len(scores), len(pairs))
            self.assertEqual(scores[0], scores[1])
            self.assertEqual(repeated[0], scores[0])
            for row in scores:
                self.assertEqual(set(row), {"contradiction", "entailment", "neutral"})
                self.assertAlmostEqual(sum(row.values()), 1.0)

            self.assertEqual(len(fake_config.calls), 1)
            self.assertEqual(fake_config.calls[0][1], {"local_files_only": True})
            self.assertEqual(len(fake_tokenizer.load_calls), 1)
            self.assertEqual(
                fake_tokenizer.load_calls[0][1], {"local_files_only": True}
            )
            self.assertEqual(len(_FakeSession.instances), 1)
            session = _FakeSession.instances[0]
            # Three unique misses at batch size two require two ONNX calls.
            self.assertEqual(len(session.feeds), 2)
            for feed in session.feeds:
                self.assertEqual(feed["input_ids"].dtype, np.dtype(np.int64))
                self.assertEqual(feed["attention_mask"].dtype, np.dtype(np.int64))

            cache = backend.cache_info()
            self.assertEqual(cache.misses, 3)
            self.assertEqual(cache.hits, 3)
            self.assertEqual(cache.size, 3)
            self.assertEqual(cache.inference_batches, 2)
            metadata = backend.metadata()
            self.assertTrue(metadata["performance_mode"])
            self.assertFalse(metadata["research_valid"])

    def test_research_mode_rejects_implicit_multi_pair_batching(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            model_dir = self._model_dir(root)
            with self.assertRaisesRegex(ValueError, "performance_mode=True"):
                OnnxDebertaNLIBackend(model_dir, batch_size=2)
            backend = OnnxDebertaNLIBackend(model_dir)
            self.assertEqual(backend.batch_size, 1)
            self.assertTrue(backend.research_valid)

    def test_pair_token_lengths_are_exact_untruncated_counts_in_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            model_dir = self._model_dir(root)
            fake_ort, fake_config, fake_tokenizer = (
                _fake_runtime_with_pair_lengths()
            )
            pairs = (
                ("alpha beta", "gamma"),       # 2 + 1 + 4 special = 7
                ("one", "two three four"),     # 1 + 3 + 4 special = 8
                ("a b c", "d e f"),            # 3 + 3 + 4 special = 10
            )
            with mock.patch.object(
                onnx_deberta,
                "_import_runtime",
                return_value=(fake_ort, fake_config, fake_tokenizer),
            ):
                backend = OnnxDebertaNLIBackend(model_dir, max_length=8)
                self.assertEqual(backend.pair_token_lengths(pairs), (7, 8, 10))

            # The diagnostic must measure the original sequence, not the
            # sequence after applying the configured inference limit.
            self.assertTrue(fake_tokenizer.encode_calls)
            self.assertTrue(
                any(
                    call.get("truncation") is False
                    for call in fake_tokenizer.encode_calls
                )
            )

    def test_over_max_length_pair_is_rejected_before_onnx_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            model_dir = self._model_dir(root)
            fake_ort, fake_config, fake_tokenizer = (
                _fake_runtime_with_pair_lengths()
            )
            with mock.patch.object(
                onnx_deberta,
                "_import_runtime",
                return_value=(fake_ort, fake_config, fake_tokenizer),
            ):
                backend = OnnxDebertaNLIBackend(model_dir, max_length=8)
                with self.assertRaisesRegex(
                    ValueError,
                    r"(?i)(max_length|maximum token|token length)",
                ):
                    backend.score_pairs(
                        (
                            ("a b c", "d e f"),  # exactly 10: reject
                        )
                    )

            self.assertEqual(len(_FakeSession.instances), 1)
            self.assertEqual(
                _FakeSession.instances[0].feeds,
                [],
                "an over-length batch must be rejected before ONNX inference",
            )

    def test_missing_artifact_skips_but_checksum_or_label_error_fails(self) -> None:
        missing = OnnxDebertaNLIBackend("/definitely/not/a/local/model")
        availability = missing.availability()
        self.assertEqual(availability.status, SKIPPED)
        self.assertEqual(availability.reason_code, "DEBERTA_MODEL_ARTIFACT_MISSING")

        with tempfile.TemporaryDirectory() as root:
            model_dir = self._model_dir(root)
            mismatch = OnnxDebertaNLIBackend(
                model_dir,
                expected_sha256="0" * 64,
            ).availability()
            self.assertEqual(mismatch.status, ERROR)
            self.assertEqual(mismatch.reason_code, "DEBERTA_MODEL_INVALID")
            self.assertIn("SHA-256 mismatch", mismatch.detail)

            fake_ort, fake_config, fake_tokenizer = _fake_runtime(
                {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
            )
            with mock.patch.object(
                onnx_deberta,
                "_import_runtime",
                return_value=(fake_ort, fake_config, fake_tokenizer),
            ):
                invalid_labels = OnnxDebertaNLIBackend(model_dir).availability()
            self.assertEqual(invalid_labels.status, ERROR)
            self.assertEqual(invalid_labels.reason_code, "DEBERTA_MODEL_INVALID")
            self.assertIn("id2label", invalid_labels.detail)

    def test_direction_contrast_is_an_explicit_research_gate(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            backend = OnnxDebertaNLIBackend(self._model_dir(root))
            synthetic_scores = (
                {"contradiction": 0.1, "entailment": 0.7, "neutral": 0.2},
                {"contradiction": 0.2, "entailment": 0.6, "neutral": 0.2},
            )
            with mock.patch.object(backend, "score_pairs", return_value=synthetic_scores):
                diagnostic = backend.direction_contrast(
                    "normalized evidence",
                    "service A calls service B.",
                    "service B calls service A.",
                    minimum_margin=0.05,
                )

        self.assertAlmostEqual(
            diagnostic.entailment_margin,
            diagnostic.forward_entailment - diagnostic.reverse_entailment,
        )
        self.assertEqual(
            diagnostic.gate_passed,
            diagnostic.entailment_margin >= diagnostic.minimum_margin,
        )
        expected_reason = (
            "DIRECTION_CONTRAST_PASS"
            if diagnostic.gate_passed
            else "DIRECTION_CONTRAST_INSUFFICIENT_MARGIN"
        )
        self.assertEqual(diagnostic.reason_code, expected_reason)

    def test_batch_composition_sensitivity_is_an_explicit_research_gate(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            backend = OnnxDebertaNLIBackend(self._model_dir(root))
            isolated = [(0.1, 0.7, 0.2)]
            co_batched = [(0.2, 0.6, 0.2), (0.1, 0.8, 0.1)]
            with (
                mock.patch.object(backend, "_load"),
                mock.patch.object(
                    backend,
                    "_infer_uncached",
                    side_effect=[isolated, co_batched],
                ),
            ):
                diagnostic = backend.batch_composition_contrast(
                    ("probe premise", "probe hypothesis"),
                    ("companion premise", "companion hypothesis"),
                )

        measured = max(
            abs(
                diagnostic.isolated_probabilities[label]
                - diagnostic.batched_probabilities[label]
            )
            for label in ("contradiction", "entailment", "neutral")
        )
        self.assertAlmostEqual(diagnostic.max_absolute_delta, measured)
        self.assertEqual(
            diagnostic.gate_passed,
            diagnostic.max_absolute_delta <= diagnostic.tolerance,
        )
        self.assertEqual(
            diagnostic.reason_code,
            (
                "BATCH_COMPOSITION_INVARIANT"
                if diagnostic.gate_passed
                else "BATCH_COMPOSITION_SENSITIVE"
            ),
        )


INTEGRATION_MODEL_DIR = Path(
    os.environ.get("LOGAGENT_DEBERTA_MODEL_DIR", "/tmp/logagent-deberta-model")
)
INTEGRATION_READY = (
    INTEGRATION_MODEL_DIR.joinpath("config.json").is_file()
    and INTEGRATION_MODEL_DIR.joinpath("onnx/model_quint8_avx2.onnx").is_file()
    and importlib.util.find_spec("onnxruntime") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(
    INTEGRATION_READY,
    "actual local DeBERTa ONNX artifact/runtime is not installed",
)
class OnnxDebertaIntegrationTests(unittest.TestCase):
    def test_real_model_smoke_and_direction_contrast_diagnostic(self) -> None:
        backend = OnnxDebertaNLIBackend(
            INTEGRATION_MODEL_DIR,
            expected_sha256=NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
        )
        self.assertEqual(backend.availability().status, READY)
        premise = (
            "A span owned by ts-order-service is the parent of a span owned by "
            "ts-payment-service."
        )
        forward = "ts-order-service calls ts-payment-service."
        reverse = "ts-payment-service calls ts-order-service."
        diagnostic = backend.direction_contrast(
            premise,
            forward,
            reverse,
            minimum_margin=float(os.environ.get("DEBERTA_DIRECTION_MIN_MARGIN", "0.05")),
        )

        # Emit the measured gate result; do not bless or hardcode a model score.
        print("DEBERTA_DIRECTION_CONTRAST=" + json.dumps(diagnostic.to_dict(), sort_keys=True))
        self.assertTrue(math.isfinite(diagnostic.forward_entailment))
        self.assertTrue(math.isfinite(diagnostic.reverse_entailment))
        self.assertEqual(
            diagnostic.gate_passed,
            diagnostic.entailment_margin >= diagnostic.minimum_margin,
        )
        self.assertIn(
            diagnostic.reason_code,
            {"DIRECTION_CONTRAST_PASS", "DIRECTION_CONTRAST_INSUFFICIENT_MARGIN"},
        )

        batch_diagnostic = backend.batch_composition_contrast(
            (premise, forward),
            (
                "A separate service emitted an unrelated database span.",
                "The separate service uses a database.",
            ),
        )
        print(
            "DEBERTA_BATCH_COMPOSITION="
            + json.dumps(batch_diagnostic.to_dict(), sort_keys=True)
        )
        self.assertEqual(
            batch_diagnostic.gate_passed,
            batch_diagnostic.max_absolute_delta <= batch_diagnostic.tolerance,
        )
        self.assertIn(
            batch_diagnostic.reason_code,
            {"BATCH_COMPOSITION_INVARIANT", "BATCH_COMPOSITION_SENSITIVE"},
        )


if __name__ == "__main__":
    unittest.main()
