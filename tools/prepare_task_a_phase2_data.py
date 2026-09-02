#!/usr/bin/env python3
"""Acquire the deterministic RCAEval subset for Task A Phase 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd  # noqa: E402

from logagent_benchmark.task_a_phase2 import (  # noqa: E402
    DEFAULT_PHASE2_CONFIG,
    Phase2Error,
    select_phase2_cases,
    write_verified_provenance,
)


def run(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise Phase2Error(
            "huggingface-hub is required: pip install 'huggingface-hub>=0.26,<1'"
        ) from exc

    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = args.dest.expanduser().resolve()
    if target.exists():
        if not args.force:
            raise Phase2Error(f"refusing to overwrite existing destination: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)

    dataset = config["dataset"]
    repo_id = str(dataset["repo_id"])
    revision = str(dataset["source_revision"])
    print(f"fetching cases.parquet from {repo_id}@{revision}", flush=True)
    cases_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="cases.parquet",
            revision=revision,
            local_dir=target,
        )
    )
    selected = select_phase2_cases(pd.read_parquet(cases_path), config)
    patterns = [f"{record['case']}/*" for record in selected]
    print("selected cases:", flush=True)
    for record in selected:
        print(
            f"- {record['fault']}: {record['case']} "
            f"(service={record['root_cause_service']}, traces={record['n_traces']})",
            flush=True,
        )

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=target,
    )
    provenance = write_verified_provenance(
        target,
        config=config,
        selected_cases=selected,
    )
    (target / "selected_cases.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"complete: {target} | cases={len(selected)} | "
        f"verified_files={len(provenance['verified'])} | "
        f"bytes={provenance['total_verified_bytes']}",
        flush=True,
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE2_CONFIG)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
