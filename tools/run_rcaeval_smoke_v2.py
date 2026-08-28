#!/usr/bin/env python3
"""Run the RCAEval A2-gated cumulative relation-recovery smoke."""

from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.cli_v2 import main  # noqa: E402


DEFAULT_BUDGET_CONFIG = PROJECT_ROOT / "configs" / "experiment_rcaeval_smoke_v2_budget.json"


def with_default_budget_config(argv: Sequence[str]) -> list[str]:
    """Select the low-cost development profile unless a config is explicit."""

    arguments = list(argv)
    has_explicit_config = any(
        value == "--config" or value.startswith("--config=")
        for value in arguments
    )
    if has_explicit_config:
        return arguments
    return ["--config", str(DEFAULT_BUDGET_CONFIG), *arguments]


if __name__ == "__main__":
    raise SystemExit(main(with_default_budget_config(sys.argv[1:])))
