from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from logagent_benchmark.cli_v2 import (  # noqa: E402
    DEFAULT_BUDGET_CONFIG_PATH,
    _bind_backend_config,
    _implementation_fingerprint,
    _load_config,
    build_parser,
)


class V2ProvenanceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = ROOT / "configs" / "experiment_rcaeval_smoke_v2.json"
        cls.config = _load_config(cls.config_path)

    def test_config_supplies_backend_sha_and_psl_seed(self) -> None:
        args = build_parser().parse_args(["--output", "unused-output"])
        self.assertIsNone(args.deberta_model_sha256)
        self.assertIsNone(args.psl_seed)

        _bind_backend_config(args, self.config)

        backend = self.config["optional_backends"]
        self.assertEqual(
            args.deberta_model_sha256,
            backend["deberta_artifact_sha256"],
        )
        self.assertEqual(args.psl_seed, backend["psl_seed"])

    def test_cli_default_is_the_budget_profile(self) -> None:
        args = build_parser().parse_args(["--output", "unused-output"])
        self.assertEqual(args.config, DEFAULT_BUDGET_CONFIG_PATH)

        budget = _load_config(ROOT / args.config)
        self.assertNotIn("iid60_l2", {mask["id"] for mask in budget["masks"]})

    def test_cli_or_config_backend_drift_is_rejected(self) -> None:
        args = build_parser().parse_args(
            [
                "--output",
                "unused-output",
                "--deberta-model-sha256",
                "0" * 64,
            ]
        )
        with self.assertRaisesRegex(ValueError, "differs from the v2 config"):
            _bind_backend_config(args, self.config)

        altered = deepcopy(self.config)
        altered["optional_backends"]["deberta_revision"] = "unregistered"
        args = build_parser().parse_args(["--output", "unused-output"])
        with self.assertRaisesRegex(ValueError, "frozen implementation"):
            _bind_backend_config(args, altered)

    def test_implementation_fingerprint_binds_config_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / "first.json"
            second = Path(root) / "second.json"
            first.write_text(json.dumps({"value": 1}), encoding="utf-8")
            second.write_text(json.dumps({"value": 2}), encoding="utf-8")

            first_digest = _implementation_fingerprint(first)
            self.assertEqual(first_digest, _implementation_fingerprint(first))
            self.assertNotEqual(first_digest, _implementation_fingerprint(second))

    def test_budget_profile_excludes_only_iid60_from_the_full_mask_set(self) -> None:
        budget_path = ROOT / "configs" / "experiment_rcaeval_smoke_v2_budget.json"
        budget = _load_config(budget_path)
        full_ids = {mask["id"] for mask in self.config["masks"]}
        budget_ids = {mask["id"] for mask in budget["masks"]}

        self.assertEqual(
            budget_ids,
            {"iid20_l2", "iid40_l2", "component_l2"},
        )
        self.assertEqual(full_ids - budget_ids, {"iid60_l2"})
        self.assertEqual(
            budget["execution_policy"]["excluded_masks"],
            ["iid60_l2"],
        )

    def test_historical_full_config_digest_remains_reported_digest(self) -> None:
        report = json.loads(
            (ROOT / "reports" / "rcaeval_smoke_results_v2.json").read_text(
                encoding="utf-8"
            )
        )
        digest = hashlib.sha256(self.config_path.read_bytes()).hexdigest()

        self.assertEqual(digest, report["config_sha256"])


if __name__ == "__main__":
    unittest.main()
