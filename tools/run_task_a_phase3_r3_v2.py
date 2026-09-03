#!/usr/bin/env python3
"""Run Task A Phase 3-R3 channel-wise DeBERTa validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.task_a_phase3_r3_channel_v2 import run_phase3_r3  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment_task_a_rcaeval_phase3_r3_v2.json",
    )
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_phase3_r3(
        phase2_root=args.phase2_root,
        model_dir=args.model_dir,
        output=args.output,
        config_path=args.config,
        max_workers=args.max_workers,
    )
    summary = json.loads(
        (output / "published" / "task_a_phase3_r3_results.json").read_text(
            encoding="utf-8"
        )
    )
    print("TASK_A_PHASE3_R3_STATUS=" + str(summary["status"]), flush=True)
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
