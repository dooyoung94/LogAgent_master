from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run_rcaeval_smoke_v2.py"
SPEC = importlib.util.spec_from_file_location("run_rcaeval_smoke_v2_tool", TOOL_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"cannot load {TOOL_PATH}")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class BudgetRunnerTests(unittest.TestCase):
    def test_default_injects_budget_config(self) -> None:
        arguments = TOOL.with_default_budget_config(["--output", "unused-output"])

        self.assertEqual(arguments[:2], ["--config", str(TOOL.DEFAULT_BUDGET_CONFIG)])
        self.assertEqual(arguments[2:], ["--output", "unused-output"])

    def test_explicit_config_pair_is_preserved(self) -> None:
        arguments = [
            "--output",
            "unused-output",
            "--config",
            "configs/experiment_rcaeval_smoke_v2.json",
        ]
        self.assertEqual(TOOL.with_default_budget_config(arguments), arguments)

    def test_explicit_config_equals_form_is_preserved(self) -> None:
        arguments = [
            "--config=configs/experiment_rcaeval_smoke_v2.json",
            "--output",
            "unused-output",
        ]
        self.assertEqual(TOOL.with_default_budget_config(arguments), arguments)


if __name__ == "__main__":
    unittest.main()
