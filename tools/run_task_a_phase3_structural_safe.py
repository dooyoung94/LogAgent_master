#!/usr/bin/env python3
"""Run risk-limited Task A Phase 3 structural development validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.phase3_structural_risk_limited import (  # noqa: E402
    DEFAULT_CONFIG,
    run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run(
        candidate_analysis=args.candidate_analysis,
        output=args.output,
        config_path=args.config,
    )
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    compact = {
        "status": summary["status"],
        "gate": summary["gate"],
        "selected_policy": summary["selected_policy"],
        "baseline_heldout": summary["baseline_heldout"],
        "proposed_heldout": summary["proposed_heldout"],
        "delta_vs_full_a2": summary["delta_vs_full_a2"],
        "delta_vs_exact_budget_a2": summary[
            "delta_vs_exact_budget_a2"
        ],
        "calibration": {
            "status": summary["calibration"]["status"],
            "feasible_policy_count": summary["calibration"][
                "feasible_policy_count"
            ],
            "searched_policy_count": summary["calibration"][
                "searched_policy_count"
            ],
        },
    }
    print("TASK_A_PHASE3_SAFE_SUMMARY_START")
    print(
        json.dumps(
            compact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print("TASK_A_PHASE3_SAFE_SUMMARY_END")
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
