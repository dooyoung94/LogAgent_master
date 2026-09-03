#!/usr/bin/env python3
"""Run Task A Phase 3-R2 operational-evidence validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.task_a_phase3_r2_compat import (  # noqa: E402
    run_phase3_r2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_task_a_rcaeval_phase3_r2.json"),
    )
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_phase3_r2(
        phase2_root=args.phase2_root,
        output=args.output,
        config_path=args.config,
        max_workers=args.max_workers,
    )
    summary = json.loads(
        (output / "published" / "task_a_phase3_r2_results.json").read_text(
            encoding="utf-8"
        )
    )
    heldout = summary["heldout"]
    print("TASK_A_PHASE3_R2_SUMMARY_START")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "protocol_status": summary["protocol_status"],
                "selected_policy": summary["selected_policy"],
                "feature_diagnostics": summary["feature_diagnostics"],
                "calibration_feasible": summary["calibration"]["feasible"],
                "feasible_policy_count": summary["calibration"][
                    "feasible_policy_count"
                ],
                "baseline_a2_full": heldout["baseline_a2_full"],
                "proposed_a3_r2": heldout["proposed_a3_r2"],
                "delta_vs_equal_size_a2": heldout[
                    "delta_vs_equal_size_a2"
                ],
                "gate": summary["gate"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print("TASK_A_PHASE3_R2_SUMMARY_END")
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
