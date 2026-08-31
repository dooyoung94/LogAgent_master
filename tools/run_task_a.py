#!/usr/bin/env python3
"""Run Task A phase 1 with IID20/IID40 and bounded abductive candidates."""

from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.cli_task_a import main  # noqa: E402


DEFAULT_TASK_A_CONFIG = PROJECT_ROOT / "configs" / "experiment_task_a_rcaeval.json"


def with_default_task_a_config(argv: Sequence[str]) -> list[str]:
    arguments = list(argv)
    has_explicit_config = any(
        value == "--config" or value.startswith("--config=")
        for value in arguments
    )
    if has_explicit_config:
        return arguments
    return ["--config", str(DEFAULT_TASK_A_CONFIG), *arguments]


if __name__ == "__main__":
    raise SystemExit(main(with_default_task_a_config(sys.argv[1:])))
