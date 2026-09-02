#!/usr/bin/env python3
"""Download and verify the frozen DeBERTa-v3-small ONNX research artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logagent_benchmark.onnx_deberta import (  # noqa: E402
    NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
    NLI_DEBERTA_V3_SMALL_AVX2_SHA256,
    NLI_DEBERTA_V3_SMALL_REPO_ID,
    NLI_DEBERTA_V3_SMALL_REVISION,
    sha256_file,
)


def prepare(destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=NLI_DEBERTA_V3_SMALL_REPO_ID,
        revision=NLI_DEBERTA_V3_SMALL_REVISION,
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.model",
            "tokenizer.*",
            "spm.model",
            "sentencepiece.bpe.model",
            NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
        ],
    )
    onnx_path = destination / NLI_DEBERTA_V3_SMALL_AVX2_FILENAME
    digest = sha256_file(onnx_path)
    if digest != NLI_DEBERTA_V3_SMALL_AVX2_SHA256:
        raise RuntimeError(
            "DeBERTa ONNX SHA-256 mismatch: "
            f"expected {NLI_DEBERTA_V3_SMALL_AVX2_SHA256}, got {digest}"
        )
    manifest = {
        "repo_id": NLI_DEBERTA_V3_SMALL_REPO_ID,
        "revision": NLI_DEBERTA_V3_SMALL_REVISION,
        "onnx_filename": NLI_DEBERTA_V3_SMALL_AVX2_FILENAME,
        "onnx_sha256": digest,
        "local_files_only": True,
    }
    (destination / ".logagent-model.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/models/nli-deberta-v3-small"),
    )
    args = parser.parse_args(argv)
    prepare(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
