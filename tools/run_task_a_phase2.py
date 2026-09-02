#!/usr/bin/env python3
"""Run Task A Phase 2 multi-case/multi-seed validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.task_a_phase2 import (  # noqa: E402
    DEFAULT_PHASE2_CONFIG,
    run_phase2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE2_CONFIG)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--keep-heavy-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_phase2(
        config_path=args.config,
        raw_root=args.raw_root,
        output=args.output,
        max_workers=args.max_workers,
        keep_heavy_artifacts=args.keep_heavy_artifacts,
    )
    import json

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
