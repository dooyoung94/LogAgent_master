#!/usr/bin/env python3
"""Run Task A Phase 3-R1 structural evidence validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.task_a_phase3_structural import (  # noqa: E402
    DEFAULT_STRUCTURAL_CONFIG,
    run_structural_phase3,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_STRUCTURAL_CONFIG)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run_structural_phase3(
        phase2_root=args.phase2_root,
        output=args.output,
        config_path=args.config,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print("TASK_A_PHASE3_STRUCTURAL_SUMMARY_START")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "selected_policy": summary["selected_policy"],
                "feature_alignment": summary["feature_alignment"],
                "baseline_heldout": summary["baseline"]["heldout"],
                "proposed_heldout": summary["proposed"]["heldout"],
                "delta": summary["proposed"]["heldout_delta_vs_a2"],
                "structural_additive_delta": summary["matched_budget_control"][
                    "heldout_delta_proposed_minus_control"
                ],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print("TASK_A_PHASE3_STRUCTURAL_SUMMARY_END")
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
