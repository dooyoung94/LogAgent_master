#!/usr/bin/env python3
"""Run the frozen R3 policy on the RCABench OPS-Lite confirmatory subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.ops_lite_r3_confirmatory import (  # noqa: E402
    run_ops_lite_confirmatory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_task_a_ops_lite_r3_confirmatory.json"),
    )
    parser.add_argument(
        "--frozen-policy",
        type=Path,
        default=Path("configs/frozen/task_a_phase3_r3_policy.json"),
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_ops_lite_confirmatory(
        data_root=args.data_root,
        output=args.output,
        config_path=args.config,
        frozen_policy_path=args.frozen_policy,
        model_dir=args.model_dir,
    )
    path = output / "published" / "ops_lite_r3_confirmatory_results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    print("OPS_LITE_R3_CONFIRMATORY_SUMMARY_START")
    print(
        json.dumps(
            {
                "status": result["status"],
                "execution": result["execution"],
                "a2_candidate_recovery": result.get("a2_candidate_recovery"),
                "confirmatory": result.get("confirmatory"),
                "paired_bootstrap": result.get("paired_bootstrap"),
                "gate": result["gate"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print("OPS_LITE_R3_CONFIRMATORY_SUMMARY_END")
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
