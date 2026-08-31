import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_task_a.py"
SPEC = importlib.util.spec_from_file_location("run_task_a_tool", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TaskAToolTests(unittest.TestCase):
    def test_default_config_is_inserted(self):
        args = MODULE.with_default_task_a_config(["--output", "out"])
        self.assertEqual(args[0], "--config")
        self.assertEqual(Path(args[1]).name, "experiment_task_a_rcaeval.json")

    def test_explicit_config_is_preserved(self):
        args = MODULE.with_default_task_a_config(
            ["--config", "custom.json", "--output", "out"]
        )
        self.assertEqual(args.count("--config"), 1)
        self.assertIn("custom.json", args)


if __name__ == "__main__":
    unittest.main()
