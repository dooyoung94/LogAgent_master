#!/usr/bin/env python3
"""Run Task A Phase 4 multi-evidence PSL v1 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.task_a_phase4_psl import run_phase4_psl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-analysis", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "experiment_task_a_rcaeval_phase4_psl_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_phase4_psl(
        candidate_analysis_path=args.candidate_analysis,
        config_path=args.config,
        output=args.output,
    )
    result_path = output / "published" / "task_a_phase4_psl_results.json"
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    print("TASK_A_PHASE4_PSL_STATUS=" + str(summary["status"]), flush=True)
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
