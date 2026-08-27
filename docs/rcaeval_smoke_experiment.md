# RCAEval relation-recovery smoke

This executable smoke test validates the path from raw RCAEval telemetry to a
leakage-controlled relation-recovery experiment. It is intentionally narrower
than the final paper experiment.

## Contract

- The source case is pinned by dataset revision and file checksums.
- The model receives an opaque incident ID. Root service, injected fault, and
  the label-bearing source folder name remain evaluator-only.
- Whole traces are deterministically partitioned: 40% evaluator-only reference
  evidence and 60% model evidence.
- `Silver-G_ref^CALLS` contains only exact cross-service parent/child joins from
  held-out traces. `CALLS` is a runtime relation, not a causal claim.
- The primary smoke mask is `L1_boundary_hidden`: both the observed edge and
  the exact parent link that can regenerate it are hidden from model evidence.
- Predictions outside the held-out silver graph remain `unknown`. A
  closed-world precision/F1 is reported only as a conservative lower bound.

## Ablations

| ID | Executable meaning |
|---|---|
| A0 | Sanitized observed graph only |
| A1 | Deterministic exact-parent extraction from sanitized model traces |
| A2 | Temporal-containment abduction over parent-unmatched spans |
| A3 | Actual DeBERTa NLI over the same typed candidate universe |
| A4 | Abduction plus actual DeBERTa |
| A5 | Actual PSL joint inference plus calibration/abstention |

A3/A4/A5 are marked `SKIPPED`, never replaced by a lexical or mocked stand-in,
when their real model/runtime artifact is unavailable. A research result cannot
claim activation until the relevant variants produce non-zero score or decision
changes.

The pinned A3/A4 artifact is the official quantized AVX2 ONNX file from
`cross-encoder/nli-deberta-v3-small` at revision
`fa2804872c3b4bd748f38c0185cc85775361e735` (SHA-256
`03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae`).
Its configured label order is contradiction, entailment, neutral. Directional
contrast is a required diagnostic because generic NLI scores do not by
themselves establish caller direction.

The official `pslpython==2.4.0` package pins `JPype1==1.4.0`, whose published
wheel support does not include Python 3.12. Paper runs should use the supported
Python 3.10 environment. A Python 3.12 run with a newer JPype may be used only
when its manifest explicitly records `compatibility_override=true`; it is not
an exact dependency reproduction.

## Current scope limitation

`re2tt_ts-auth-service_cpu_2` is useful for schema, split, mask, leakage, and
runner regression testing. Its injected service is isolated in the observed
cross-service `CALLS` graph, so it is not a defensible primary impact-path case.
The next representative run should use `re2tt_ts-travel-service_cpu_2`, followed
by `ts-order-service`, after pinning their raw files with the same controls.
