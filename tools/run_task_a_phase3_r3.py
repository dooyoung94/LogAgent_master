#!/usr/bin/env python3
"""Run Task A Phase 3-R3 channel-specific DeBERTa validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.task_a_phase3_r3 import run_phase3_r3  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_task_a_rcaeval_phase3_r3.json"),
    )
    parser.add_argument("--candidate-analysis", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_phase3_r3(
        candidate_analysis=args.candidate_analysis,
        model_dir=args.model_dir,
        output=args.output,
        config_path=args.config,
    )
    summary = json.loads(
        (output / "published" / "task_a_phase3_r3_results.json").read_text(
            encoding="utf-8"
        )
    )
    heldout = summary["heldout"]
    compact = {
        "status": summary["status"],
        "protocol_status": summary["protocol_status"],
        "selected_policy": summary["selected_policy"],
        "channel_coverage": summary["nli_diagnostics"][
            "channel_candidate_coverage"
        ],
        "tri_state_counts": summary["nli_diagnostics"]["tri_state_counts"],
        "baseline_a2_full": heldout["baseline_a2_full"],
        "proposed_a3_r3": heldout["proposed_a3_r3"],
        "delta_vs_equal_size_operational": heldout[
            "delta_vs_equal_size_operational"
        ],
        "delta_vs_equal_size_a2": heldout["delta_vs_equal_size_a2"],
        "gate": summary["gate"],
    }
    print("TASK_A_PHASE3_R3_SUMMARY_START")
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    print("TASK_A_PHASE3_R3_SUMMARY_END")
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
