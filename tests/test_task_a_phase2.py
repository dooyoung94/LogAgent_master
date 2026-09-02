import json
from pathlib import Path
import random
import unittest

import pandas as pd

from logagent_benchmark.task_a_phase2 import (
    aggregate_phase2,
    build_cell_config,
    select_phase2_cases,
    validate_phase2_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_task_a_rcaeval_phase2.json"


class TaskAPhase2SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _row(
        self,
        case: str,
        fault: str,
        *,
        service: str = "svc",
        repetition: int = 1,
        logs: bool = True,
        traces: bool = True,
    ):
        return {
            "case": case,
            "dataset": "RE2-TT",
            "root_cause_service": service,
            "fault": fault,
            "repetition": repetition,
            "has_logs": logs,
            "n_logs": 10 if logs else 0,
            "has_traces": traces,
            "n_traces": 100 if traces else 0,
        }

    def _index(self):
        rows = [
            self._row(
                "re2tt_ts-auth-service_cpu_2",
                "cpu",
                service="ts-auth-service",
                repetition=2,
            )
        ]
        for fault in ("mem", "disk", "delay", "loss", "socket"):
            rows.append(self._row(f"re2tt_a_{fault}_1", fault, service=f"svc-{fault}-a"))
            rows.append(self._row(f"re2tt_b_{fault}_2", fault, service=f"svc-{fault}-b", repetition=2))
            rows.append(
                self._row(
                    f"re2tt_ineligible_{fault}_3",
                    fault,
                    service=f"svc-{fault}-bad",
                    repetition=3,
                    logs=False,
                )
            )
        rows.append(self._row("re2ob_unrelated_cpu_1", "cpu"))
        rows[-1]["dataset"] = "RE2-OB"
        return pd.DataFrame.from_records(rows)

    def test_phase2_config_contract(self):
        validate_phase2_config(self.config)

    def test_selection_is_fault_stratified_and_stable_under_row_order(self):
        index = self._index()
        first = select_phase2_cases(index, self.config)
        shuffled_rows = index.to_dict(orient="records")
        random.Random(37).shuffle(shuffled_rows)
        second = select_phase2_cases(pd.DataFrame.from_records(shuffled_rows), self.config)

        self.assertEqual(first, second)
        self.assertEqual(
            [record["fault"] for record in first],
            ["cpu", "mem", "disk", "delay", "loss", "socket"],
        )
        self.assertEqual(first[0]["case"], "re2tt_ts-auth-service_cpu_2")
        self.assertTrue(
            all("ineligible" not in record["case"] for record in first)
        )
        self.assertEqual(len({record["case"] for record in first}), 6)

    def test_cell_config_keeps_incident_split_stable_across_seeds(self):
        base = json.loads(
            (PROJECT_ROOT / "configs" / "experiment_task_a_rcaeval.json").read_text(
                encoding="utf-8"
            )
        )
        case = select_phase2_cases(self._index(), self.config)[1]
        seed_11 = build_cell_config(base, self.config, case, 11)
        seed_47 = build_cell_config(base, self.config, case, 47)

        self.assertEqual(
            seed_11["dataset"]["incident_id"],
            seed_47["dataset"]["incident_id"],
        )
        self.assertNotEqual(seed_11["masks"][0]["id"], seed_47["masks"][0]["id"])
        self.assertEqual([mask["seed"] for mask in seed_11["masks"]], [11, 11])
        self.assertEqual([mask["ratio"] for mask in seed_11["masks"]], [0.2, 0.4])


class TaskAPhase2AggregateTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.selected = [
            {
                "case": f"case-{fault}",
                "fault": fault,
                "root_cause_service": f"svc-{fault}",
            }
            for fault in ("cpu", "mem", "disk", "delay", "loss", "socket")
        ]

    def _grid(self, recall: float = 1.0):
        cells = []
        runs = []
        for case in self.selected:
            for seed in self.config["seeds"]:
                runs.append(
                    {
                        "status": "READY",
                        "case": case["case"],
                        "fault": case["fault"],
                        "seed": seed,
                    }
                )
                for ratio, proposals in ((0.2, 13), (0.4, 26)):
                    cells.append(
                        {
                            "case": case["case"],
                            "fault": case["fault"],
                            "seed": seed,
                            "mask_ratio": ratio,
                            "candidate_recall": recall,
                            "a2_proposal_count": proposals,
                            "silver_precision_lower_bound": 0.8,
                            "compression_ratio": 0.97,
                            "budget_saturated": False,
                            "dropped_by_budget": 0,
                            "leakage_checks_all_pass": True,
                        }
                    )
        return cells, runs

    def test_complete_high_recall_grid_passes(self):
        cells, runs = self._grid()
        gate = aggregate_phase2(
            cells=cells,
            run_records=runs,
            selected_cases=self.selected,
            config=self.config,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["observed"]["cell_count"], 60)
        self.assertEqual(gate["observed"]["candidate_recall_min"], 1.0)
        self.assertEqual(gate["observed"]["proposal_count_max"], 26)

    def test_one_low_recall_cell_fails_worst_case_gate(self):
        cells, runs = self._grid()
        cells[0] = {**cells[0], "candidate_recall": 0.8}
        gate = aggregate_phase2(
            cells=cells,
            run_records=runs,
            selected_cases=self.selected,
            config=self.config,
        )
        self.assertFalse(gate["passed"])
        self.assertIn("CANDIDATE_RECALL_EACH_CELL", gate["reason_codes"])

    def test_excessive_budget_saturation_fails(self):
        cells, runs = self._grid()
        for index in range(16):
            cells[index] = {**cells[index], "budget_saturated": True}
        gate = aggregate_phase2(
            cells=cells,
            run_records=runs,
            selected_cases=self.selected,
            config=self.config,
        )
        self.assertFalse(gate["passed"])
        self.assertIn("BUDGET_SATURATION_RATE", gate["reason_codes"])


if __name__ == "__main__":
    unittest.main()
