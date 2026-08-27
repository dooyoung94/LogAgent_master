# Research protocol

## 1. Claim and non-claims

**Primary claim to test**

> Under realistic observability blind spots, symptom-conditioned abductive candidate generation, local textual evidence scoring, and calibrated soft-logic inference can recover useful operational relations; the recovered paths improve an otherwise fixed LLM RCA system.

**Non-claims**

- Logs automatically invent a correct ontology/TBox.
- A structural dependency is automatically a causal edge.
- DeBERTa or PSL proves a relation.
- An unobserved edge is false.
- More graph context necessarily improves an LLM.

The first paper fixes the entity set and completes typed edges. Open-set node discovery and entity resolution are separate extensions.

## 2. Formalization

Let the reviewed ontology be

```text
O = (C, R, domain, range, A)
```

where `C` is the class set, `R` is the relation set, and `A` contains typing, cardinality, inverse, provenance, temporal, and traversal constraints.

For incident `i`:

```text
X_i       = {logs, metrics, traces, collector metadata}
G_obs_i   = phi(X_i; O) = (V_i, E_confirmed_i)
G_ref_i   = (V_ref_i, E_ref_i)
E_miss_i  = E_ref_i - E_observed_i
```

`G_ref` must be built from evidence independent of the model input: deployment manifests, service-mesh configuration, source-level calls, controlled complete traces, or a reviewed experimental topology. A CMDB or trace source cannot serve as both input evidence and hidden ground truth in the same condition.

Every edge must carry:

```json
{
  "subject": "...",
  "predicate": "...",
  "object": "...",
  "relation_layer": "structural | runtime | causal_hypothesis",
  "status": "confirmed | inferred | unresolved",
  "source": "...",
  "evidence_ids": [],
  "observed_at": "...",
  "extractor": "...",
  "confidence": 0.0
}
```

## 3. Research questions

- **RQ1:** How well are missing typed operational relations recovered?
- **RQ2:** Does the neuro-symbolic pipeline outperform rules, text-only scoring, and generic KG completion while remaining calibrated?
- **RQ3:** Does adding recovered relations improve root, cause-path, and impact-path RCA under a fixed LLM and budget?
- **RQ4:** Does the result persist under collector, component, relation, and path-critical blind spots rather than IID edge loss?
- **RQ5:** Does the system abstain when the evidence cannot identify a relation?

## 4. Relation vocabulary

The initial schema is deliberately small.

| Relation | Domain → range | Layer | Typical evidence |
|---|---|---|---|
| `INSTANCE_OF` | Instance → Application/Service | structural | deployment metadata, process identity |
| `CALLS` | Service/Instance → Service/Instance | runtime | parent-child spans, correlated transactions |
| `EXPOSES` | Application → Endpoint | structural | route config, access logs |
| `ROUTES_TO` | WebPage/URL → Endpoint | runtime | crawler observations, gateway logs |
| `USES_DATASOURCE` | Service/Application → DataSource | runtime | DB spans, connection pools, SQL logs |
| `EXECUTES` | Transaction → SQLPattern | runtime | APM transaction and SQL correlation |
| `LOCATED_ON` | Instance → Host | structural | collector/process/host identity |

`CAUSES` and `PROPAGATES_TO` are causal hypotheses and are not silently inferred from `CALLS` or `USES_DATASOURCE`.

## 5. Missingness benchmark

Random masks are retained for continuity, but are not the primary evidence.

| Scenario | Removal unit | Operational analogue |
|---|---|---|
| IID relation-stratified | 20/40/60% per relation | comparability baseline |
| Relation block | one relation/source family | absent collector or integration |
| Component blackout | all incident edges for selected nodes | unregistered/new component |
| Path critical | one or more root-to-symptom bridge edges | RCA-breaking blind spot |
| Trace dropout | trace-derived facts and direct identifiers | sampling/export failure |
| Identity-link dropout | cross-source aliases | inconsistent CMDB/APM names |
| Temporal window dropout | contiguous evidence window | outage or retention gap |
| Natural missingness | observed production gaps | external case study |

Each mask has three evidence levels:

- **Direct:** the graph edge is hidden, but the raw evidence explicitly exposes both endpoints.
- **Indirect:** direct identifiers and equivalent inverse/alias facts are also hidden.
- **No evidence:** no discriminating support remains; calibrated abstention is the desired behavior.

Masks must remove target-edge aliases, inverses, derived duplicates, fault-injection commands, root labels in file/folder names, and future telemetry.

## 6. Candidate generation by abduction

Given observations `O_i` and background facts `B_i = O ∪ G_obs_i ∪ rules_train`, propose a minimal typed hypothesis `H` such that:

```text
B_i ∪ H explains O_i
B_i ∪ H is constraint-consistent
no proper subset of H explains the same observations
```

Example:

```text
DBWait(d) & DependsOn(a,d) -> AppLatency(a)

Observed: DBWait(payment-db), AppLatency(payment-app)
Candidate: DependsOn(payment-app, payment-db)
```

Candidate priority:

```text
score_A(H) =
  observation_coverage
  - lambda_size * |H|
  - lambda_violation * hard_violations
  - lambda_time * temporal_cost
```

Start with backward depth at most 3 and at most 2 missing literals, then freeze values on validation. Persist the rule, observations, and evidence IDs that produced each candidate. Report `Candidate Recall@K` separately: later stages cannot recover an edge absent from the candidate set.

The incident root label is forbidden as an abductive observation.

## 7. DeBERTa local evidence model

Use DeBERTa as a cross-encoder over an evidence bundle and a verbalized triple.

```text
Premise:
  tx-17 ran on payment-01
  tx-17 called /payments/approve
  tx-17 executed SQL on payment-db
Hypothesis:
  payment-application uses payment-db
```

The model returns entailment, contradiction, and neutral probabilities. The soft-logic feature is initially:

```text
score_D(h) = P(entailment | h) - P(contradiction | h)
```

Training labels:

- positive: hidden reference edge or independently reviewed relation;
- verified negative: type/time/cardinality contradiction in a scope known to be complete;
- unlabeled: all other unobserved edges.

Use positive-unlabeled/open-world controls; never convert every non-edge into a negative. Hard negatives include type-compatible corruption, relation corruption, incident-local aliases, topological near misses, and temporal contradictions. Fit temperature scaling only on validation and report Brier score and ECE.

DeBERTa is a replaceable component, not the novelty claim. Compare with at least a small cross-encoder and a non-text score.

## 8. PSL joint inference

For candidate `h`, infer a soft truth value `y_h in [0,1]` by minimizing weighted hinge losses under grounded rules:

```text
y* = argmin_y sum_j w_j * phi_j(x, y)
```

Weights and thresholds are frozen using training/validation only. Example rules:

```text
DebertaUses(A,D)                  -> UsesDataSource(A,D)
AbductiveSupport(A,D)             -> UsesDataSource(A,D)

TxHitsEndpoint(T,E)
& EndpointOf(E,A)
& TxUsesDatabase(T,D)             -> UsesDataSource(A,D)

ParentSpan(P,C)
& ExecutesOn(P,I)
& ExecutesOn(C,J)                 -> CallsInstance(I,J)

CallsInstance(I,J)
& InstanceOf(I,A)
& InstanceOf(J,B)                 -> CallsApplication(A,B)
```

High-weight/hard constraints are allowed only when the system guarantees them:

```text
UsesDataSource(A,D) -> IsApplication(A)
UsesDataSource(A,D) -> IsDataSource(D)
Causes(X,Y) & NotBefore(X,Y) -> contradiction
```

Do not impose antisymmetry on service calls. Do not impose single ownership in multi-tenant systems. PSL scores grounded candidates; it cannot create a missing entity.

A relation is accepted only when its relation-specific posterior and top-two margin exceed frozen thresholds. Otherwise it remains `unresolved`, not false.

## 9. Experiment matrix

### Task A — relation recovery

| ID | System |
|---|---|
| A0 | observed graph only |
| A1 | deterministic trace/time rules |
| A2 | abduction only |
| A3 | DeBERTa only over typed candidates |
| A4 | abduction + DeBERTa without PSL |
| A5 | abduction + DeBERTa + PSL + calibration/abstention |
| AO | oracle reference graph |
| AW | A5 plus controlled wrong-edge perturbation |

Generic KGC baselines should include at least a translational/bilinear model and one graph model where feasible. An ontology-prompted LLM extractor is an additional baseline.

### Task B — controlled LLM RCA

Keep model, prompt, temperature, token budget, retrieval window, and candidate budget identical.

| ID | LLM evidence |
|---|---|
| B0 | raw telemetry only |
| B1 | raw telemetry + observed graph |
| B2 | raw telemetry + strongest baseline-completed graph |
| B3 | raw telemetry + A5 graph |
| BO | raw telemetry + oracle graph |
| BW | raw telemetry + same-count shuffled/false edges |
| BG | graph-free blind-spot method such as TORAI |

Intervention controls add separately: only recovered gold path edges, the same count of non-path edges, and the same count of false edges. These distinguish useful path recovery from simply supplying more tokens.

Required LLM output:

```json
{
  "root_candidates": [],
  "cause_path": [],
  "impact_path": [],
  "supporting_evidence": [],
  "uncertainty": [],
  "abstain": false
}
```

RCAEval supplies root-cause labels, not a complete gold causal path for every case. Path labels must be derived from an independently held-out topology and explicitly reviewed; they must not be invented from the model output.

## 10. Metrics and statistics

**Task A**

- Candidate Recall@K
- relation-wise precision/recall/macro-F1 and AUPRC
- filtered MRR and Hits@1/3/10
- Brier score and ECE
- risk-coverage/abstention curve
- constraint-violation and false-edge rates
- isolated-node recovery, root-to-symptom reachability, path edge F1, shortest-path distortion

**Task B**

- root Top-1/3/5 and MRR
- multi-root exact match when applicable
- cause-path edge F1 and exact match
- impact-component F1
- evidence citation precision and unsupported-claim rate
- latency, peak memory, grounded-rule count, tokens, and cost

Use at least five mask seeds. Group repeated injections from the same campaign into the same split. Report system and relation macro-averages. Use incident-level paired bootstrap confidence intervals; McNemar/permutation tests for correctness; paired bootstrap or Wilcoxon for continuous metrics; Holm correction for multiple comparisons.

Report both:

- **transductive:** test-system entities/normal telemetry may be observed;
- **inductive:** system/service/entity is held out, including leave-one-system-out.

An anonymized-entity run checks name and endpoint memorization.

## 11. Leakage controls

- A topology/CMDB source has exactly one role in a condition: input or ground truth.
- Never evaluate an edge extracted from a log against a “gold” edge extracted from the same log.
- Remove target aliases, inverses, equivalent paths, injection metadata, incident titles, and folder-name labels.
- Freeze normalization, anomaly thresholds, calibration, PSL rules, and relation thresholds before test.
- Never use the root label to generate completion candidates.
- Never use telemetry after the decision timestamp.
- Split repeated topology edges and campaigns together; incident-random split alone is insufficient.
- Natural production gaps without gold truth are qualitative or dual-reviewed, with inter-rater agreement reported.

## 12. Pre-registered decision gates

| Gate | Pass condition | Failure action |
|---|---|---|
| D0 — data | source URL, immutable version/checksum, schema, license, and GT audit recorded | do not run benchmark |
| D1 — extraction | hidden target and aliases are absent from model input; audit tests pass | fix masking before modeling |
| D2 — candidates | validation Candidate Recall@K reaches the pre-registered target (initial target 0.90) | improve candidate generation; do not tune PSL |
| D3 — activation | ablations produce materially different predictions; model flip counts are non-zero | implementation is inactive; no result claim |
| D4 — oracle utility | oracle graph improves path/RCA over observed graph under fixed LLM | stop: topology is not the demonstrated bottleneck |
| D5 — completion | A5 beats the strongest completion baseline with paired uncertainty and acceptable calibration | reject H1 |
| D6 — downstream | A5 graph improves RCA/path metrics and does not exceed the false-edge risk budget | reject H2/H3 |
| D7 — generalization | improvement persists under structured blind spots and held-out systems | limit claim to the observed setting |

The previous pattern in which multiple ablations are identical and DeBERTa/PSL have zero flips fails D3 and is explicitly non-paper-ready.

## 13. Failure conditions to report

The hypothesis is not supported when any of the following holds:

- oracle topology does not improve the fixed RCA agent;
- gains exist only under IID masking;
- a simple rule/KGC/LLM extractor matches A5;
- edge F1 rises while causal-path and RCA metrics do not;
- confidence remains high in the no-evidence condition;
- false inferred edges degrade RCA;
- DeBERTa relies on service-name memorization;
- PSL spreads an invalid domain rule;
- a missing entity, multiple consecutive bridge edges, clock skew, or multi-root incident violates assumptions;
- candidate grounding is operationally intractable.

## 14. Hypotheses

- **H1:** A5 improves calibrated relation recovery over the strongest baseline under structured blind spots.
- **H2:** The A5 graph improves fixed-LLM root and path RCA over the observed graph.
- **H3:** Calibration and abstention reduce false-edge RCA degradation compared with uncalibrated completion.

The main novelty is the realistic missingness task, symptom-conditioned minimal hypotheses, open-world calibrated completion, and graph-intervention evidence—not the individual use of DeBERTa or PSL.
