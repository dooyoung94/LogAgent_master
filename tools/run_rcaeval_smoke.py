#!/usr/bin/env python3
"""Run the leakage-controlled RCAEval relation-recovery smoke."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

