#!/usr/bin/env python3
"""Download and verify the frozen RCABench OPS-Lite confirmatory subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _case_id(row: dict[str, Any]) -> str:
    for key in ("case_id", "name", "id", "source"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError("manifest row lacks a case identifier")


def prepare(config_path: Path, destination: Path) -> Path:
    try:
        import requests
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub and requests are required") from exc

    config_path = config_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config = _load_json(config_path)
    dataset = config["dataset"]
    selection = config["selection_contract"]

    repo_id = str(dataset["repo_id"])
    revision = str(dataset["revision"])
    manifest_path = str(dataset["manifest_path"])
    manifest = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=str(dataset.get("repo_type", "dataset")),
            revision=revision,
            filename=manifest_path,
            local_dir=destination,
        )
    )
    observed_manifest_sha = _sha256(manifest)
    expected_manifest_sha = str(dataset["manifest_sha256"])
    if observed_manifest_sha != expected_manifest_sha:
        raise ValueError(
            f"manifest SHA mismatch: {observed_manifest_sha} != {expected_manifest_sha}"
        )

    split_url = (
        f"https://raw.githubusercontent.com/{dataset['leaderboard_repo']}/"
        f"{dataset['leaderboard_commit']}/{dataset['test_split_path']}"
    )
    response = requests.get(split_url, timeout=60)
    response.raise_for_status()
    split_text = response.text
    observed_split_sha = hashlib.sha256(split_text.encode("utf-8")).hexdigest()
    if observed_split_sha != str(dataset["test_split_sha256"]):
        raise ValueError(
            f"test split SHA mismatch: {observed_split_sha} != {dataset['test_split_sha256']}"
        )
    test_cases = [
        line.strip()
        for line in split_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(test_cases) != int(dataset["expected_test_cases"]):
        raise ValueError("unexpected official test split size")

    manifest_rows: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest row {line_no} is not an object")
        key = _case_id(value)
        if key in manifest_rows:
            raise ValueError(f"duplicate manifest case: {key}")
        manifest_rows[key] = value
    if len(manifest_rows) != int(dataset["expected_manifest_cases"]):
        raise ValueError("unexpected manifest case count")

    selected = [str(item["case_id"]) for item in selection["selected_cases"]]
    selected_text = "\n".join(selected) + "\n"
    selected_sha = hashlib.sha256(selected_text.encode("utf-8")).hexdigest()
    if selected_sha != str(selection["selected_case_ids_sha256"]):
        raise ValueError("selected case list SHA mismatch")
    if len(selected) != len(set(selected)):
        raise ValueError("selected case identifiers repeat")
    if not set(selected).issubset(set(test_cases)):
        missing = sorted(set(selected) - set(test_cases))
        raise ValueError(f"selected cases are outside the official test split: {missing}")

    eligibility = selection["structural_eligibility"]
    n_svc_min = int(eligibility["n_svc_min"])
    n_edge_min = int(eligibility["n_edge_min"])
    invalid = []
    selected_metadata = {}
    for case_id in selected:
        row = manifest_rows[case_id]
        selected_metadata[case_id] = row
        if int(row.get("n_svc", 0)) < n_svc_min or int(row.get("n_edge", 0)) < n_edge_min:
            invalid.append(case_id)
    if invalid:
        raise ValueError(f"selected cases violate frozen structural eligibility: {invalid}")

    allow_patterns = [manifest_path]
    for case_id in selected:
        allow_patterns.extend(
            [
                f"cases/{case_id}/normal_traces.parquet",
                f"cases/{case_id}/abnormal_traces.parquet",
                f"cases/{case_id}/injection.json",
                f"cases/{case_id}/env.json",
                f"cases/{case_id}/causal_graph.json",
            ]
        )
    snapshot_download(
        repo_id=repo_id,
        repo_type=str(dataset.get("repo_type", "dataset")),
        revision=revision,
        local_dir=destination,
        allow_patterns=allow_patterns,
        max_workers=8,
    )

    file_rows = []
    for case_id in selected:
        case_root = destination / "cases" / case_id
        for required_name in ("normal_traces.parquet", "abnormal_traces.parquet"):
            if not (case_root / required_name).is_file():
                raise FileNotFoundError(f"missing confirmatory source: {case_root / required_name}")
        for path in sorted(case_root.glob("*")):
            if path.is_file():
                file_rows.append(
                    {
                        "case_id": case_id,
                        "path": str(path.relative_to(destination)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )

    source = {
        "schema_version": 1,
        "dataset": {
            "repo_id": repo_id,
            "revision": revision,
            "manifest_sha256": observed_manifest_sha,
            "test_split_sha256": observed_split_sha,
        },
        "selection": {
            "selected_case_ids": selected,
            "selected_case_ids_sha256": selected_sha,
            "case_replacement_allowed": False,
            "metadata": selected_metadata,
        },
        "files": file_rows,
        "config_sha256": _sha256(config_path),
    }
    (destination / "official_test_split.txt").write_text(split_text, encoding="utf-8")
    output = destination / "source_manifest.json"
    output.write_text(
        json.dumps(source, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"OPS_LITE_CONFIRMATORY_SOURCE_READY cases={len(selected)} files={len(file_rows)} "
        f"bytes={sum(item['bytes'] for item in file_rows)}"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_task_a_ops_lite_r3_confirmatory.json"),
    )
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(args.config, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
