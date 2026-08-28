# RCAEval cumulative relation-recovery smoke v2

This experiment freezes the v1 outputs and tests a leakage-safer cumulative
contract for recovering missing runtime `CALLS` relations.

## Research question

Can model-visible trace evidence propose a small set of missing relations, and
can directional NLI plus runtime-role context verify those proposals before a
fixed PSL pruning stage?

This is a topology-recovery smoke test. The held-out graph is silver runtime
parent/child support, not deployment gold and not a causal RCA graph.

## Why v2 exists

The v1 standalone DeBERTa result accepted 1,272 edges, but 1,272 was not the
number of masked relations. For IID20, only 10 relations were masked. The v1
legacy control scored 1,595 all-pairs candidates and accepted every one of the
1,272 candidates with zero trace co-occurrence, while accepting none of the
323 candidates with positive co-occurrence. That result is retained only as
`D0_LEGACY` and is not comparable to the v2 A-series.

v2 also replaces the L1 opaque orphan token with `L2_PARENT_DROPPED`: a masked
child's `parent_span_id` becomes null, the same surface form as an ordinary
root span. No mask marker enters model input.

## Executable contract

| ID | Role | Candidate scope | Acceptance meaning |
|---|---|---|---|
| A0 | Observed graph baseline | Typed universe `U` | No inferred edge |
| A1 | Direct-evidence layer | `U` | Deterministic declared evidence only |
| A2 | Abductive proposal layer | Compressed proposal set `P2` | Temporal containment support |
| A3 | Flat directional control | Exactly `P2` | Forward/reverse DeBERTa hard gate |
| A4 | Runtime-context verifier | Exactly `P2` | Same gate with composite model-side context |
| A5 | Fixed PSL pruning baseline | Exactly `P2` | May prune A4-accepted `CALLS`; cannot add an edge |

The operational path is `A2 -> A4 -> A5`. A3 and A4 are paired verifiers over
the same ordered `P2`, not two different candidate generators. Direct evidence
is protected through A3--A5. Empty or direct-only proposal sets do not touch
the heavy backends.

Directional acceptance requires all three conditions:

- forward entailment >= 0.67;
- reverse entailment <= 0.33;
- forward minus reverse entailment >= 0.05.

The last condition is redundant under the current forward/reverse thresholds,
but is retained and reported for diagnostic continuity. It must be changed
only in a newly versioned experiment.

## Runtime context boundary

A4 receives balanced source/target summaries derived only from sanitized model
traces and the masked observed graph:

- Service identity and model-trace type basis;
- in/out degree and continuous orchestrator/provider role proxies;
- bounded upstream/downstream examples;
- HTTP/data-access ratios;
- bounded operation-string examples.

RCAEval does not supply reliable Application, Instance, Host, Deployment, or
ownership levels for this case. A4 therefore tests a composite runtime-context
effect, not an isolated upper-hierarchy effect. Operation examples are strings,
not asserted ontology nodes.

Source and target use the same serialization fields and bounds. The frozen
backend tokenizes without truncation, records candidate-level token lengths,
and returns `DEBERTA_INPUT_TOO_LONG` before inference if a pair exceeds 512
tokens. ONNX telemetry is disabled before runtime import.

## Split, masking, and leakage controls

- Whole-trace assignment is
  `sha256(revision|incident_id|trace_id) mod 100`; no unused salt is claimed.
- Reference and model trace-ID fingerprints are written to `summary.json`.
- Model entities come from the sanitized model trace partition only.
- Reference traces, target edges, mask manifest, injection time, and fault label
  are absent from every inference API.
- Pre-mask graph/trace artifacts are not written to a mask run tree.
- Evaluator access is API-separated and ordered after model stages, but the
  evaluator and model stages still share one Python process. This is not
  process isolation.

## Metric interpretation

A0/A1 ranking is over `U`; A2--A5 ranking is within `P2`. Cross-scope MRR and
Hits values are not directly comparable. Decision metrics and candidate recall
must be read separately. Predictions outside the incomplete silver graph are
`unverified`, not declared false; `P-LB` is only a conservative lower bound.

## Reproduction

```bash
python tools/run_rcaeval_smoke_v2.py \
  --deberta-model-dir data/models/nli-deberta-v3-small-fa280487 \
  --enable-psl \
  --psl-compatibility-override \
  --require-heavy \
  --output outputs/rcaeval_smoke_v2/run
```

The runner refuses to overwrite an existing directory. The Python 3.12 run
records the explicit `pslpython==2.4.0` / `JPype1==1.7.1` compatibility
override; an exact paper reproduction should use a supported exact-pin
environment.
