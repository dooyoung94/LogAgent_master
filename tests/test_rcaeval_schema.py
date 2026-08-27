from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.rcaeval import (  # noqa: E402
    RCAEvalSchemaError,
    convert_rcaeval_case,
    write_incident_bundle,
)


class RCAEvalSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.case_name = "re2tt_ts-auth-service_cpu_2"
        self.case_dir = self.root / self.case_name
        self.case_dir.mkdir()

        pd.DataFrame(
            [
                {
                    "case": self.case_name,
                    "dataset": "RE2-TT",
                    "suite": "RE2",
                    "system": "tt",
                    "system_name": "Train Ticket",
                    "root_cause_service": "ts-auth-service",
                    "fault": "cpu",
                    "fault_description": "CPU stress",
                    "repetition": 2,
                    "inject_time": 101,
                    "n_metrics": 4,
                    "n_timesteps": 3,
                    "time_start": 100,
                    "time_end": 102,
                }
            ]
        ).to_parquet(self.root / "cases.parquet", index=False)
        pd.DataFrame(
            {
                "time": [100, 101, 102],
                "ts-auth-service_cpu": [1.0, 9.0, 10.0],
                "ts-auth-mongo_mem": [100.0, 101.0, 102.0],
                "ts-user-service_cpu": [2.0, None, 3.0],
                "ts-ui-dashboard_workload": [4.0, 4.0, 4.0],
            }
        ).to_parquet(self.case_dir / "metrics.parquet", index=False)
        pd.DataFrame(
            {
                "timestamp": [100, 101],
                "container_name": ["ts-auth-service", "ts-ui-dashboard"],
                "message": [
                    "INFO LOGIN USER :fdse_microservice __ 111111 __ null "
                    "eyJhbGciOiJIUzI1NiJ9.abc.signature "
                    "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f",
                    '127.0.0.6 INFO password=secret',
                ],
            }
        ).to_parquet(self.case_dir / "logs.parquet", index=False)
        pd.DataFrame(
            {
                "time": ["00:01", "00:01"],
                "traceID": ["0" * 32, "0" * 32],
                "spanID": ["1" * 16, "2" * 16],
                "serviceName": ["ts-auth-service", "ts-user-service"],
                "methodName": [None, None],
                "operationName": ["POST /api/v1/users/login", "GET /api/v1/users/{id}"],
                "parentSpanID": [None, "1" * 16],
                "startTimeMillis": [100_000, 100_001],
                "startTime": [100_000_123, 100_001_456],
                "duration": [2_000, 500],
                "statusCode": [None, None],
            }
        ).to_parquet(self.case_dir / "traces.parquet", index=False)
        (self.case_dir / "inject_time.txt").write_text("101", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def convert(self):
        return convert_rcaeval_case(
            self.case_dir,
            dataset_revision="fixture-revision",
        )

    def test_model_input_separates_labels_and_exact_parent_answers(self) -> None:
        bundle = self.convert()
        model_input = bundle.model_input()

        self.assertNotIn(self.case_name, bundle.incident["incident_id"])
        self.assertNotIn("ts-auth-service", bundle.incident["incident_id"])
        self.assertNotIn("cpu", bundle.incident["incident_id"])
        self.assertFalse(
            {"case", "root_cause_service", "fault", "fault_description"}
            .intersection(model_input["incident"])
        )
        self.assertNotIn("parent_span_id", model_input["traces"].columns)
        self.assertNotIn("is_root_span", model_input["traces"].columns)
        self.assertIn("parent_span_id", bundle.canonical_traces.columns)
        self.assertEqual(
            "ts-auth-service", bundle.evaluator_labels["root_cause_service"]
        )
        self.assertEqual("cpu", bundle.evaluator_labels["fault_type"])
        self.assertEqual("train-ticket", bundle.incident["system_id"])
        self.assertEqual("tt", bundle.restricted_provenance["source_system_code"])
        self.assertTrue(
            bundle.entities["entity_id"].str.startswith("rcaeval:train-ticket:").all()
        )
        self.assertEqual(
            "derived_from_root_service_and_fault;not_source_ground_truth",
            bundle.evaluator_labels["root_cause_indicator_provenance"],
        )

    def test_accepts_safe_incident_id_override_and_rejects_label_ids(self) -> None:
        bundle = convert_rcaeval_case(
            self.case_dir,
            dataset_revision="fixture-revision",
            incident_id="rcaeval_tt_smoke_0001",
        )
        self.assertEqual("rcaeval_tt_smoke_0001", bundle.incident["incident_id"])
        self.assertEqual(
            "caller-supplied-opaque",
            bundle.restricted_provenance["incident_id_strategy"],
        )
        self.assertTrue(
            bundle.entities["incident_id"].eq("rcaeval_tt_smoke_0001").all()
        )

        with self.assertRaisesRegex(RCAEvalSchemaError, "must not encode"):
            convert_rcaeval_case(
                self.case_dir,
                incident_id="ts-auth-service-cpu-result",
            )

    def test_normalizes_entities_metrics_logs_and_trace_time(self) -> None:
        bundle = self.convert()

        self.assertEqual(4, len(bundle.entities))
        self.assertEqual(
            {"Service", "DataSource", "WebApplication"},
            set(bundle.entities["entity_type"]),
        )
        self.assertEqual(11, len(bundle.metrics))  # one source null is omitted
        self.assertEqual(100_000_000, int(bundle.metrics["event_time_us"].min()))
        self.assertEqual({"cpu", "mem", "workload"}, set(bundle.metrics["metric_name"]))

        redacted = "\n".join(bundle.logs["body"].dropna().astype(str))
        for secret in (
            "eyJhbGciOiJIUzI1NiJ9.abc.signature",
            "fdse_microservice",
            "111111",
            "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f",
            "127.0.0.6",
            "password=secret",
        ):
            self.assertNotIn(secret, redacted)
        self.assertGreater(int(bundle.logs["redaction_count"].sum()), 0)

        traces = bundle.canonical_traces
        self.assertEqual(100_000_123, int(traces.iloc[0]["start_time_us"]))
        self.assertEqual(100_002_123, int(traces.iloc[0]["end_time_us"]))
        self.assertEqual([True, False], traces["is_root_span"].tolist())
        self.assertEqual("0" * 32, traces.iloc[0]["trace_id"])

    def test_writer_physically_separates_evaluator_truth(self) -> None:
        bundle = self.convert()
        output = self.root / "normalized"
        paths = write_incident_bundle(bundle, output)

        self.assertTrue(paths["evaluator_labels"].is_file())
        self.assertEqual(output / "evaluator" / "labels.json", paths["evaluator_labels"])
        self.assertEqual(
            output / "restricted" / "canonical_traces.parquet",
            paths["canonical_traces"],
        )
        incident = json.loads(paths["incident"].read_text(encoding="utf-8"))
        self.assertNotIn("root_cause_service", incident)
        self.assertNotIn("fault", incident)
        model_traces = pd.read_parquet(paths["traces"])
        self.assertNotIn("parent_span_id", model_traces.columns)

        with self.assertRaises(RCAEvalSchemaError):
            write_incident_bundle(bundle, output)

    def test_rejects_inconsistent_trace_time_units(self) -> None:
        traces = pd.read_parquet(self.case_dir / "traces.parquet")
        traces.loc[0, "startTime"] = 123
        traces.to_parquet(self.case_dir / "traces.parquet", index=False)
        with self.assertRaisesRegex(RCAEvalSchemaError, "microsecond refinement"):
            self.convert()


SMOKE_ROOT = Path(
    os.environ.get(
        "LOGAGENT_RCAEVAL_SMOKE_ROOT",
        PROJECT_ROOT / "data" / "raw" / "rcaeval" / "smoke",
    )
)
SMOKE_CASE = SMOKE_ROOT / "re2tt_ts-auth-service_cpu_2"


@unittest.skipUnless(
    (SMOKE_ROOT / "cases.parquet").is_file() and SMOKE_CASE.is_dir(),
    "downloaded RCAEval smoke case is not present",
)
class RCAEvalDownloadedSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = convert_rcaeval_case(
            SMOKE_CASE,
            cases_index_path=SMOKE_ROOT / "cases.parquet",
            dataset_revision="afeacb11bcc94dadfd1c8f483ee4377b2b8b614e",
        )

    def test_measured_shapes_and_safe_split(self) -> None:
        self.assertEqual(68, len(self.bundle.entities))
        self.assertEqual(520_817, len(self.bundle.metrics))
        self.assertEqual(271_919, len(self.bundle.logs))
        self.assertEqual(838_936, len(self.bundle.canonical_traces))
        self.assertEqual(6_475, int(self.bundle.canonical_traces["is_root_span"].sum()))
        self.assertNotIn("parent_span_id", self.bundle.traces.columns)
        self.assertEqual(
            "ts-auth-service", self.bundle.evaluator_labels["root_cause_service"]
        )
        self.assertGreater(int(self.bundle.logs["redaction_count"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
