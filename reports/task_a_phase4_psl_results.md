# Task A Phase 4 — Multi-evidence PSL v1

- Scientific status: **FAIL**
- Gate: `D4D_A4_MULTI_EVIDENCE_PSL_DEVELOPMENT`
- Selected policy: `{"minimum_keep": 8, "profile_id": "conservative", "retention_fraction": 0.85}`
- Gate reasons: `CALIBRATION_POLICY_FEASIBLE, MATCHED_A2_RECALL_NONINFERIOR, MATCHED_A2_P_LB_IMPROVED, MATCHED_A2_MRR_NONINFERIOR, RULE_ABLATION_HAS_EFFECT`

## Held-out result

| Metric | A2 full | Equal-size A2 | PSL v1 | PSL vs equal-size A2 |
|---|---:|---:|---:|---:|
| Recall macro | 1.000000 | 0.997727 | 0.994318 | -0.003409 |
| Recall minimum | 1.000000 | 0.954545 | 0.909091 | - |
| Recall pooled | 1.000000 | 0.996296 | 0.990741 | - |
| Mean candidates | 19.375 | 17.025 | 17.025 | +0.000 |
| P-LB macro | 0.723387 | 0.807928 | 0.804883 | -0.003045 |
| MRR macro | 1.000000 | 0.997727 | 0.993723 | -0.004004 |

## Rule ablation

| Variant | Recall | P-LB | MRR | Mean candidates |
|---|---:|---:|---:|---:|
| full | 0.994318 | 0.804883 | 0.993723 | 17.025 |
| prior_only | 0.997727 | 0.807928 | 0.997727 | 17.025 |
| no_negative | 0.997727 | 0.807928 | 0.997132 | 17.025 |
| no_operation | 0.994318 | 0.804883 | 0.994318 | 17.025 |
| no_structure | 0.997727 | 0.807928 | 0.997727 | 17.025 |
| permuted_evidence | 0.936280 | 0.756271 | 0.912818 | 17.025 |

## Claim boundary

Development-only validation of multi-evidence PSL reranking for runtime CALLS candidates on six previously inspected RCAEval TrainTicket incidents. It does not establish causal CAUSES edges, production generalization, RCA improvement, or LLM improvement.

PSL output is a probability-ranked runtime `CALLS` hypothesis set. It is not a causal `CAUSES` graph and does not establish RCA/LLM improvement.
