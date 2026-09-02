# Task A Phase 3 결과 — Tri-state DeBERTa 가설 검증

- 최종 Gate: **FAIL**
- Calibration 선택 상태: **DIAGNOSTIC_FALLBACK_NO_FEASIBLE_POLICY**
- Calibration feasible 정책 수: **0**
- Held-out Cell: **40**
- 미통과 조건: `CALIBRATION_POLICY_FEASIBLE, MATCHED_BUDGET_ADDITIVE_GAIN`

## 선택 정책

| 항목 | 값 |
|---|---:|
| retention_fraction | 0.8500 |
| minimum_keep | 8 |
| nli_weight | 0.0500 |
| calibration_feasible | False |

## Held-out 결과

| 지표 | A2 전체 후보 | A3 Shortlist | 변화 |
|---|---:|---:|---:|
| Candidate Recall Macro | 1.0000 | 0.9977 | -0.0023 |
| Candidate Recall Minimum | 1.0000 | 0.9545 | - |
| 후보 수 평균 | 19.38 | 17.02 | -2.35 |
| P-LB Macro | 0.7234 | 0.8079 | 0.0845 |
| MRR Macro | 0.9919 | 0.9977 | 0.0058 |

## 동일 후보 수 A2-only 대조군 대비

| 지표 | A3 - A2 matched-budget |
|---|---:|
| Recall Macro | 0.0000 |
| P-LB Macro | 0.0000 |
| MRR Macro | 0.0000 |

## NLI 상태 분포

- corroborates: 0
- ambiguous: 7
- contradicts: 1243

## 해석

- 사전 정의된 과학적 Gate를 통과하지 못했으므로 A3 개선 주장을 하지 않는다.
- 결과 파일은 실패 원인 분석과 다음 설계 변경을 위한 진단 산출물이다.
- Calibration에서 feasible 정책이 없더라도, Calibration-only 기준으로 고정한 진단용 정책을 Held-out에 적용하여 선택 편향 없이 실패 양상을 기록했다.

## 범위 제한

- A3 evaluates tri-state NLI-assisted shortlisting of frozen A2 CALLS proposals on six RCAEval TrainTicket incidents. It does not establish causal-edge recovery or RCA/LLM improvement.
- `CALLS` 복원은 runtime 구조 관계이며 causal `CAUSES` 복원을 의미하지 않는다.
