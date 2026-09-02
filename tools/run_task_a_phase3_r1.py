#!/usr/bin/env python3
"""Run Task A Phase 3-R1 structured-evidence validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from logagent_benchmark.task_a_phase3_r1 import run_phase3_r1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-analysis", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_task_a_rcaeval_phase3_r1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_phase3_r1(
        candidate_analysis=args.candidate_analysis,
        output=args.output,
        config_path=args.config,
    )
    print(output)
    status = (
        output / "published" / "task_a_phase3_r1_status.txt"
    ).read_text(encoding="utf-8").strip()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
