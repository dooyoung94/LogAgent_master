#!/usr/bin/env python3
"""Run Task A with IID20/IID40 and bounded abductive candidates."""

import json
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


def _config_path(arguments: Sequence[str]) -> Path | None:
    for index, value in enumerate(arguments):
        if value == "--config" and index + 1 < len(arguments):
            return Path(arguments[index + 1]).expanduser().resolve()
        if value.startswith("--config="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return None


def normalize_phase2_incident_id(arguments: Sequence[str]) -> None:
    """Delegate Phase-2 case IDs to RCAEval's opaque SHA-256 generator.

    Phase-2 cell configs are generated artifacts.  The source case name must not
    appear in a model-visible incident identifier, while the identifier still
    has to remain stable across masking seeds.  Setting the override to ``None``
    lets ``convert_rcaeval_case`` derive the same opaque revision+case hash for
    every seed without exposing the case, root label, or fault.
    """

    path = _config_path(arguments)
    if path is None or not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload.get("dataset", {})
    if dataset.get("profile") != "task_a_phase2_dynamic_subset":
        return
    if dataset.get("incident_id") is None:
        return
    dataset["incident_id"] = None
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    arguments = with_default_task_a_config(sys.argv[1:])
    normalize_phase2_incident_id(arguments)
    raise SystemExit(main(arguments))
