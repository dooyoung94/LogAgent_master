from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from tools.datasets import (
    DEFAULT_REGISTRY,
    RegistryError,
    _checksum,
    dataset_by_id,
    load_registry,
    profile_by_name,
    validate_registry,
    verify_local_profile,
)


class DatasetRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(DEFAULT_REGISTRY)

    def test_registry_is_valid(self) -> None:
        self.assertEqual([], validate_registry(self.registry))

    def test_dataset_ids_are_unique(self) -> None:
        ids = [dataset["id"] for dataset in self.registry["datasets"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_ready_profiles_have_verified_licenses(self) -> None:
        automatic = {"huggingface", "zenodo", "github"}
        for dataset in self.registry["datasets"]:
            if dataset["readiness"] != "ready":
                continue
            for profile in dataset["profiles"].values():
                if profile["adapter"] in automatic:
                    license_info = profile.get("license", dataset["license"])
                    self.assertEqual("verified", license_info["status"], dataset["id"])

    def test_duplicate_id_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.registry)
        invalid["datasets"].append(copy.deepcopy(invalid["datasets"][0]))
        self.assertTrue(any("duplicate dataset id" in error for error in validate_registry(invalid)))

    def test_lookup_errors_are_explicit(self) -> None:
        with self.assertRaises(RegistryError):
            dataset_by_id(self.registry, "missing")
        with self.assertRaises(RegistryError):
            profile_by_name(dataset_by_id(self.registry, "rcaeval"), "missing")

    def test_checksum_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"logagent")
            self.assertEqual(
                "sha256:704409583ff3513dfb108694908218241b84ab4abb4b5154628de07e4d624294",
                _checksum(
                    path,
                    "sha256:704409583ff3513dfb108694908218241b84ab4abb4b5154628de07e4d624294",
                ),
            )

    def test_local_profile_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            path = target / "sample.bin"
            path.write_bytes(b"logagent")
            verified = verify_local_profile(
                {
                    "checksums": {
                        "sample.bin": "sha256:704409583ff3513dfb108694908218241b84ab4abb4b5154628de07e4d624294"
                    },
                    "expected_bytes": {"sample.bin": 8},
                },
                target,
                parse_parquet=False,
            )
            self.assertEqual(8, verified["sample.bin"]["bytes"])

    def test_local_profile_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RegistryError):
                verify_local_profile(
                    {"checksums": {"../escape": "md5:d41d8cd98f00b204e9800998ecf8427e"}},
                    Path(directory),
                    parse_parquet=False,
                )


if __name__ == "__main__":
    unittest.main()
