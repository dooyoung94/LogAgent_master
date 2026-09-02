# Task A Phase 1 결과 — 제한된 귀추 관계 후보 복원

실행일: **2026-09-02**  
결론: **PASS — D2 bounded candidate-recall gate 통과**

## 1. 연구 질문

불완전한 runtime `CALLS` 그래프에서 정답 관계를 직접 보지 않고도, 시간 포함관계 기반 귀추추론이 누락 관계를 **작은 후보 집합**으로 유지할 수 있는지를 검증했다.

이번 단계는 `A0 → A1 → A2`만 실행한다. DeBERTa, PSL, causal path, RCA, LLM 평가는 의도적으로 제외했다.

## 2. 고정 실험 계약

| 항목 | 값 |
|---|---|
| Dataset | RCAEval RE2 TrainTicket |
| Case | `re2tt_ts-auth-service_cpu_2` |
| Dataset revision | `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e` |
| Relation | `Service -[CALLS]-> Service` |
| Masks | IID 20%, IID 40% |
| Evidence level | `L2_PARENT_DROPPED` |
| Seed | 17 |
| Candidate cap | 전체 귀추 후보 32, source/target별 8 |
| Active stages | A0, A1, A2 |
| Deferred stages | A3, A4, A5 |

후보는 `abduction score → supporting trace 수 → boundary span 수 → edge key` 순으로 정렬했다. Reference graph, mask manifest, injection time, fault/root label은 모델 입력에 전달하지 않았다.

## 3. 입력 데이터 및 재현성 검증

| 데이터 | Shape | SHA-256 |
|---|---:|---|
| `cases.parquet` | 735 cases index | `c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb` |
| `logs.parquet` | 271,919 × 3 | `0cc2ed5ff13a20cf776a42d9d4c3914981afe025fbaa69109ac652d9c7502537` |
| `metrics.parquet` | 1,441 × 368 | `16597725c18258ce0a3bdedc1833fb52ad6638fdc5068e944dce23c1bbde6d93` |
| `traces.parquet` | 838,936 × 11 | `3d704979b684c5450a3ddcd48bb91a71d07c485edc878a2969b04ff152b5857c` |

- 전체 Trace: 6,475
- Reference Trace: 2,539
- Model Trace: 3,936
- Trace ID overlap: 0
- Evaluator-only Silver `CALLS`: 55 edges
- 단위테스트: **88 passed, 4 skipped, 10 subtests passed**

## 4. 핵심 결과

| Mask | 숨긴 관계 | Typed universe U | A2 후보 P2 | 정답 포함 | Candidate / Masked Recall | P-LB | Unverified | 압축률 | Budget drop | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| IID20 | 10 | 657 | **13** | 10 | **1.000** | 0.7692 | 3 | **98.02%** | 0 | **PASS** |
| IID40 | 21 | 668 | **26** | 21 | **1.000** | 0.8077 | 5 | **96.11%** | 0 | **PASS** |

추가 관찰:

- IID20과 IID40 모두 `Hits@1/3/10 = 1.0`, `MRR = 1.0`을 기록했다. 이 값은 A2 후보집합 내부 랭킹 기준이다.
- 두 조건 모두 `budget_saturated=false`이며 전역·source·target 상한으로 제거된 후보는 0개다.
- `P-LB`는 불완전한 Silver Graph에 대한 보수적 정밀도 하한이다. Silver 밖의 3개·5개 관계는 false로 확정하지 않고 `unverified`로 유지한다.

## 5. Gate 판정

| Gate | 기준 | 결과 |
|---|---|---|
| D0 Data | revision, checksum, schema audit 통과 | **PASS** |
| D1 Leakage | 모델 entity 일치, fault/injection 미노출, pre-mask artifact 부재 | **PASS** |
| D2 Candidate | 각 Mask Candidate Recall ≥ 0.90 | **PASS: 1.00 / 1.00** |
| Budget | 각 Mask 귀추 후보 ≤ 32 | **PASS: 13 / 26** |

초기 실행에서 결과 요약기가 구조화된 `candidate_recall` 객체를 scalar로 읽어 종료된 결함이 발견됐다. 귀추 실행 결과에는 영향이 없었으며, scalar 추출 로직과 PASS/FAIL 회귀 테스트를 추가한 뒤 전체 워크플로를 재실행해 성공을 확인했다.

## 6. 해석

현재 결과는 다음 한정된 주장을 지지한다.

> 한 RCAEval TrainTicket 사건과 seed 17의 IID20/IID40 `L2_PARENT_DROPPED` 조건에서, 시간 포함관계 기반 A2는 657~668개의 가능한 관계를 13~26개로 압축하면서 모든 masked Silver `CALLS` 관계를 후보로 보존했다.

다음 주장은 아직 허용되지 않는다.

- 일반적인 운영환경에서도 동일 성능을 낸다.
- 복원 관계가 causal relation이다.
- DeBERTa 또는 PSL이 성능을 개선했다.
- 복원 그래프가 RCA 또는 LLM 결과를 개선했다.

특히 이번 마스크와 귀추 Evidence가 모두 Trace 시간·parent 구조에 기반하므로, 다른 incident·seed·structured blind spot에서 재현되지 않으면 과도한 결론을 내릴 수 없다.

## 7. 후속 결정

A3~A5로 바로 넘어가지 않는다. 다음 Task A Phase 2는 동일한 20%·40% 마스킹에서:

1. 최소 5개 seed;
2. 다중 TrainTicket incident;
3. 후보 32개 상한과 endpoint별 8개 상한 유지;
4. incident별·seed별 Candidate Recall과 후보 수 분포;
5. macro recall, worst-case recall, P-LB, compression, budget saturation 비율

을 먼저 검증한다. Phase 2에서도 D2가 유지된 후에만 DeBERTa를 hard veto가 아닌 `corroborates / contradicts / ambiguous` 보조 증거로 결합한다.

## 8. 실행 근거

- Validated PR head: `8de904664beb56ae65efefeb6c6af866f986ec28`
- Merged commit: `82693fcfa815496e495519f5fd86dd6d5a0fc7bd`
- GitHub Actions validation run: `33574023760`
- Implementation SHA-256: `e7caaf57fa7310cae09c425796dd94e38924cc3413143c34b08ef388d160531a`
- Config SHA-256: `12241046c5a5b464d00b5683418a62424eec756e6b72ad4e3a57d70d77d8f036`
- Artifact ID: `9826002986`
- Artifact ZIP SHA-256: `ad7d1b7a05b36d2ed7f8e79eaca485ffeee9372ff344e88c57f19659d35f92b1`

Raw RCAEval 데이터와 전체 Parquet 출력은 Git에 커밋하지 않는다. 저장소에는 설정, 코드, compact 결과와 checksum만 유지한다.
