# Task A Phase 4 — Multi-evidence PSL v1

## 목적

A2 귀추추론은 RCAEval 60개 Cell에서 숨긴 runtime `CALLS` 관계를 모두 후보군에 포함했지만, 후보 수가 평균 20.83개이고 P-LB 최저가 0.4231이었다. Phase 4는 DeBERTa를 주 경로에서 제거하고, A2 후보를 유지한 채 구조·운영·온톨로지 Evidence를 PSL의 Soft Rule로 결합해 더 타당한 후보를 상위에 배치하는지 검증한다.

```text
A2 bounded abduction
        ↓
A2 prior + Trace + Direction + Operation + Endpoint + Role
        ↓
PSL soft-logic inference
        ↓
Probability-ranked CALLS hypotheses
        ↓
Calibration에서 정책 선택
        ↓
고정 정책으로 Held-out 평가
```

## 범위

- 데이터: RCAEval RE2-TT
- 입력: Phase 3-R2에서 생성한 1,250개 A2 후보 / 60개 Cell
- Calibration: 20개 Cell
- Held-out: 40개 Cell
- 관계: `Service -[CALLS]-> Service`
- DeBERTa: 사용하지 않음
- PSL 실행체: `pslpython==2.4.0`, Java 17, Python 3.10 격리 실행환경
- 최종 외부검증: 별도 RCABench/Aegis 단계에서 수행

## PSL Predicate

| Predicate | 의미 | 성격 |
|---|---|---|
| `A2Prior` | 귀추 점수와 A2 내부 순위 | 양의 Evidence |
| `TraceSupport` | 여러 Trace에서의 지지 강도 | 양의 Evidence |
| `BoundarySupport` | 시간 포함 경계 Span 지지 | 양의 Evidence |
| `RepeatedSupport` | 독립 Trace 반복성 | 양의 Evidence |
| `DirectionSupport` | 순방향 대비 역방향 우세 | 양의 Evidence |
| `OperationMatch` | Operation·역할·Endpoint 호환성 | 양의 Evidence |
| `EndpointMatch` | HTTP Method·Route 일치 | 양의 Evidence |
| `RoleCompatibility` | 관측 그래프와 Operation 역할 | 양의 Evidence |
| `DirectObserved` | 직접 관측된 관계 | 강한 양의 Evidence |
| `ReverseSupport` | 반대 방향 Trace 지지 | 음의 Evidence |
| `DirectionConflict` | 역방향이 순방향보다 강함 | 음의 Evidence |
| `SelfLoop` | 동일 Service 자기 관계 | 강한 음의 Evidence |
| `RecoveredCallsV1` | PSL이 추론할 Cell별 관계 확률 | 출력 Target |

모든 규칙은 Soft Rule이며 후보를 즉시 삭제하지 않는다. `CALLS`의 전이성은 사용하지 않는다. `A→B`, `B→C`가 관측되어도 직접 `A→C`를 생성하지 않는다.

`RecoveredCallsV1(Cell, Source, Target)`는 기존 2항 `Calls(Source, Target)` Predicate와 JVM 전역 이름·Arity가 충돌하지 않도록 별도 이름으로 격리한다. 출력은 평가 단계에서 다시 `(Source, CALLS, Target)` 관계로 매핑한다.

## Rule profile

세 개의 사전 정의 프로파일만 Calibration에서 비교한다.

1. `conservative`: A2 Prior를 강하게 보존
2. `balanced`: Prior·구조·운영 Evidence 균형
3. `evidence_heavy`: 구조·운영 Evidence 비중 강화

Held-out 결과를 본 뒤 Weight나 Threshold를 변경하지 않는다.

## 비교군

| 비교군 | 목적 |
|---|---|
| A2 Full | 귀추 후보 전체 기준값 |
| Equal-size A2 | 후보 수 감소 효과 통제 |
| A2 + PSL | 제안 방식 |
| Prior-only PSL | PSL 최적화 자체와 다중 Evidence 효과 분리 |
| No-negative | 역방향·충돌 규칙 기여 확인 |
| No-operation | Operation·Endpoint 규칙 기여 확인 |
| No-structure | Trace·Direction·Role 규칙 기여 확인 |
| Permuted Evidence | Evidence와 Edge의 정렬을 깨뜨린 음성 통제 |

## 성공 Gate

- Held-out Recall Macro/Pooled ≥ 0.95
- 각 Cell Recall ≥ 0.90
- A2 대비 평균 후보 수 10% 이상 감소
- 동일 후보 수 A2 대비 Recall 비열등
- 동일 후보 수 A2 대비 P-LB +0.01 이상
- 동일 후보 수 A2 대비 MRR 비열등
- PSL Score 분산 존재
- 다중 규칙 제거 시 성능 변화 존재
- Evidence Permutation 시 성능 하락
- 모든 1,250개 후보가 PSL 입력·출력에서 보존
- Evaluator Label은 PSL 점수 고정 후 결합

## 실행 상태

- PSL 공식 런타임 의존성은 Python 3.10 격리 환경에서 설치한다.
- 기존 2항 `Calls`와 신규 3항 Target의 충돌을 제거하기 위해 출력 Predicate를 `RecoveredCallsV1`로 분리했다.
- 과학적 PASS/FAIL과 Workflow 실행 성공 여부는 별도로 기록한다.

## 해석 제한

이번 결과는 이전에 관찰한 RCAEval Incident를 사용한 개발 검증이다. PASS해도 runtime `CALLS` 관계 재랭킹 효과만 주장할 수 있다. 인과관계 `CAUSES`, Root Cause, Cause Path, Impact Path, RCA·LLM 개선은 별도의 후속 실험이 필요하다.
