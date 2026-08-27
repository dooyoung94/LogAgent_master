# RCAEval relation-recovery smoke results

Run date: 2026-08-27  
Dataset revision: `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`  
Experiment: `rcaeval-call-recovery-smoke-v1`

## What was executed

The checksummed RCAEval Train Ticket smoke case was converted to the standard
incident contract, split by whole `trace_id`, converted into an evaluator-only
silver `CALLS` graph, structurally masked, and evaluated with A0--A5.

- Standardized: 68 entities, 520,817 metric observations, 271,919 redacted log
  events, and 838,936 spans.
- Evaluator/model trace split: 2,539/3,936 traces and 332,867/506,069 spans;
  overlap is zero.
- Held-out silver graph: 55 directed `CALLS`; 52 attestation A and 3
  attestation B.
- Model-side base graph: the same 55 relations before masking.
- The experiment used only attestation-A edges as primary mask targets.
- Every model variant in one condition received the same typed candidate
  universe. Reference traces and target manifests entered only the evaluator.

The silver graph means exact parent/child runtime support in held-out traces.
It is neither deployment gold nor a causal graph.

## Results

`P-LB` is conservative silver precision: predictions outside the incomplete
silver graph are counted as unverified in the denominator, not asserted false.

### IID 20% — 10 masked edges

| Variant | Accepted | Masked recall | MRR | Hits@1 | P-LB | Unverified |
|---|---:|---:|---:|---:|---:|---:|
| A0 observed only | 0 | 0.000 | 0.030 | 0.000 | N/A | 0 |
| A1 direct rules | 0 | 0.000 | 0.030 | 0.000 | N/A | 0 |
| A2 abduction | 13 | **1.000** | **1.000** | **1.000** | 0.769 | 3 |
| A3 DeBERTa only | 1,272 | 0.000 | 0.036 | 0.000 | 0.000 | 1,272 |
| A4 abduction + DeBERTa | 12 | **1.000** | **1.000** | **1.000** | 0.833 | 2 |
| A5 A4 + PSL | 10 | **1.000** | **1.000** | **1.000** | **1.000** | 0 |

### IID missingness sensitivity

| Mask | Target | Boundary spans hidden | Variant | Accepted | Recall | MRR | Hits@1 | P-LB | Unverified |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| IID 40% | 21 | 21,906 | A2 | 26 | 1.000 | 1.000 | 1.000 | 0.808 | 5 |
| IID 40% | 21 | 21,906 | A4 | 23 | 1.000 | 1.000 | 1.000 | 0.913 | 2 |
| IID 40% | 21 | 21,906 | A5 | 21 | 1.000 | 1.000 | 1.000 | **1.000** | 0 |
| IID 60% | 31 | 60,294 | A2 | 48 | 1.000 | 1.000 | 1.000 | 0.646 | 17 |
| IID 60% | 31 | 60,294 | A4 | 36 | 1.000 | 0.968 | 0.935 | 0.861 | 5 |
| IID 60% | 31 | 60,294 | A5 | 33 | 1.000 | 0.968 | 0.935 | **0.939** | 2 |

A3 accepted 1,272 candidates at every IID level while recovering none of the
masked targets. The generic NLI model is therefore not a valid standalone edge
classifier in this setup.

### Component blackout

The frozen selector chose the attestation-A component with highest total
degree: `ts-preserve-service`. All ten incoming/outgoing `CALLS` relations were
hidden, including 568 direct boundary spans.

| Variant | Accepted | Masked recall | MRR | Hits@1 | P-LB |
|---|---:|---:|---:|---:|---:|
| A2 | 10 | **1.000** | **1.000** | **1.000** | **1.000** |
| A4 | 10 | **1.000** | **1.000** | **1.000** | **1.000** |
| A5 | 10 | **1.000** | **1.000** | **1.000** | **1.000** |

This structured condition is solved by the abductive temporal rule alone.
A2-to-A4 and A4-to-A5 decision flips are zero, so a strict paper activation
gate for those stages fails even though the general smoke gate passes.

## Actual backend diagnostics

- DeBERTa artifact: official quantized AVX2 ONNX at revision
  `fa2804872c3b4bd748f38c0185cc85775361e735`, SHA-256
  `03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae`.
- Research inference used `batch_size=1` and local-only model loading.
- Direction contrast failed: forward entailment 0.9161, reverse entailment
  0.9487, margin -0.0326.
- Quantized batch-composition invariance failed: maximum probability change
  0.0401. This is why multi-pair batching is not used in research mode.
- PSL used the official 2.4.0 runtime (`aad67da`) with 5.0 evidence and 0.5
  sparsity rules, seed 7, and actual grounding/inference.
- This host used `JPype1==1.7.1` although `pslpython==2.4.0` declares 1.4.0.
  `compatibility_override=true` is recorded. A paper reproduction must use a
  supported Python 3.10/exact-pin environment.
- A5 currently has no learned calibration and no functional-relation
  abstention for `CALLS`. Its improvement is caused by the fixed PSL evidence
  and sparsity objective crossing the frozen threshold; it is not a proof.

## Allowed conclusion

The executable path and leakage controls are active. Temporal-containment
abduction recovers all masked silver relations in this one incident/seed, and
the fixed PSL sparsity stage reduces unverified additions under IID masks.

## Conclusions that are not supported

- Off-the-shelf DeBERTa improves relation recovery.
- PSL proves that an inferred relation is logically true.
- A5 generalizes across systems, seeds, faults, or collectors.
- The completed graph improves RCA or an LLM cause/impact path.
- The held-out runtime graph is a gold causal topology.

The source incident's injected `ts-auth-service` is isolated in the observed
cross-service graph. This case is therefore an ETL/leakage smoke, not primary
RCA-path evidence. The next representative case should be
`re2tt_ts-travel-service_cpu_2`, followed by `ts-order-service`, after pinning
their raw artifacts with the same controls.

## Reproduction entry point

```bash
python tools/datasets.py verify rcaeval \
  --profile smoke \
  --path data/raw/rcaeval/smoke \
  --parse-parquet

python tools/run_rcaeval_smoke.py \
  --output outputs/rcaeval_smoke/run \
  --deberta-model-dir /path/to/pinned/deberta \
  --enable-psl \
  --require-heavy
```

The runner refuses to overwrite an existing output directory. Raw data,
models, and run artifacts remain ignored by Git.

