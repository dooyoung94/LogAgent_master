import importlib.util
import json
from pathlib import Path
import tempfile
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

    def test_phase2_case_encoded_incident_id_is_replaced_with_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "profile": "task_a_phase2_dynamic_subset",
                            "incident_id": "rcaeval_re2tt_ts-order-service_disk_3",
                        }
                    }
                ),
                encoding="utf-8",
            )
            MODULE.normalize_phase2_incident_id(["--config", str(path)])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(payload["dataset"]["incident_id"])

    def test_phase1_incident_id_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase1.json"
            original = {
                "dataset": {
                    "profile": "smoke",
                    "incident_id": "rcaeval_tt_smoke_0001",
                }
            }
            path.write_text(json.dumps(original), encoding="utf-8")
            MODULE.normalize_phase2_incident_id([f"--config={path}"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, original)


if __name__ == "__main__":
    unittest.main()
