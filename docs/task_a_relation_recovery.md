# Task A 1차 실행안 — 제한된 귀추 관계 후보 복원

## 목적

첫 단계는 RCA나 LLM 평가가 아니라, 불완전한 관측 그래프에서 누락된 runtime
`CALLS` 관계를 **작고 재현 가능한 후보 집합으로 복원할 수 있는지** 확인하는
것이다. 기존 v2에서 A2 후보 재현율은 높았지만 DeBERTa hard gate가 후보를
제거했으므로, 검증기와 PSL을 다시 조정하기 전에 A2 후보 품질과 후보 수부터
동결한다.

## 이번 실행 범위

| 항목 | 고정값 |
|---|---|
| Dataset | RCAEval RE2 TrainTicket smoke incident |
| Relation | `Service -[CALLS]-> Service` |
| Mask | IID 20%, IID 40% |
| Evidence level | `L2_PARENT_DROPPED` |
| Seed | 17 |
| Active stages | A0, A1, A2 |
| Deferred stages | A3, A4, A5 |
| Task B / LLM RCA | 실행하지 않음 |

IID60과 component blackout은 삭제한 것이 아니라 이번 1차 실행에서 제외한다.
20%와 40% 조건에서 후보 생성 계약이 통과한 뒤 다중 seed와 다중 incident로
확장한다.

## IN → OUT

| 단계 | IN | 처리 | OUT |
|---|---|---|---|
| A0 | Masked traces, observed graph, Service entities | typed `CALLS` universe 생성 | 관측 그래프 baseline |
| A1 | Model-visible direct evidence | 명시 관계만 채택 | direct proposal |
| A2 | parent가 누락된 span, 시간 포함관계 | 귀추 후보 생성·랭킹·예산 적용 | 최대 32개 abductive proposal |
| 평가 | evaluator-only masked edge | A2와 정답 비교 | Candidate Recall, Masked Recall, P-LB |

Reference graph, mask manifest, injection time, fault/root label은 모델 API에 전달하지
않는다.

## 후보 폭증 방지 계약

귀추 후보는 다음 순서로만 정렬한다.

1. Abduction score 내림차순
2. Supporting trace 수 내림차순
3. Boundary span 수 내림차순
4. Candidate edge key 오름차순 — 동점 재현성용

다음 예산을 동시에 적용한다.

| 제약 | 값 |
|---|---:|
| 전체 abductive 후보 | 최대 32 |
| 동일 source 후보 | 최대 8 |
| 동일 target 후보 | 최대 8 |
| 최소 supporting trace | 1 |
| 최소 boundary | 1 |

명시적으로 관측된 direct evidence는 버리지 않으며 32개 귀추 예산에도 포함하지
않는다. 후보 선택 함수는 reference graph나 masked target을 입력받지 않으므로
정답을 이용한 Top-K 선택이 아니다.

32개 상한은 기존 IID40 smoke의 A2 후보 26개보다 높다. 따라서 과거 결과에
맞춰 정답을 잘라낸 값이 아니라, 현재 결과를 보존하면서 더 큰 데이터에서의
후보 폭증을 막는 초기 운영 예산이다.

## 실행

```bash
python tools/run_task_a.py \
  --raw-root data/raw/rcaeval/smoke \
  --output outputs/task_a/phase1
```

A3~A5를 의도적으로 실행하지 않으므로 DeBERTa 모델 경로, PSL 옵션,
`--require-heavy`는 사용하지 않는다.

## 1차 통과 기준

| Gate | 통과 기준 |
|---|---|
| D0 Data | 고정 revision/checksum 및 schema audit 통과 |
| D1 Leakage | 모든 leakage check 통과 |
| D2 Candidate | IID20과 IID40 각각 Candidate Recall ≥ 0.90 |
| Budget | 각 mask의 abductive proposal ≤ 32 |
| Compression | 후보 수와 universe 대비 압축률을 함께 보고 |

`silver_precision_lower_bound`는 참고 지표로 보고한다. Silver graph 밖의 예측은
실제 false가 아니라 unverified일 수 있으므로, 이를 단독 hard gate로 사용하지
않는다.

## 다음 단계

D2가 통과하면 같은 20%·40% 마스킹에서 seed를 최소 5개로 늘리고 incident를
확장한다. 그 후 DeBERTa는 A2를 제거하는 hard veto가 아니라
`corroborates / contradicts / ambiguous` 보조 증거로 재설계하고, 마지막에 PSL
및 calibration을 결합한다. Task B의 LLM RCA 비교는 Task A 관계복원과 oracle
utility가 통과한 이후에만 수행한다.
