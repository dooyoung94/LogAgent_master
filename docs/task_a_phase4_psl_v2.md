# Task A Phase 4 — Direct-evidence PSL v2

## 목적

PSL v1은 A2 후보를 12.13% 줄였지만 동일 후보 수 A2보다 Recall, P-LB,
MRR이 모두 낮았다. 특히 `ReverseSupport`와 `DirectionConflict` 음의
규칙이 정답 후보를 억제했고, Operation·구조 규칙은 추가 순위 이득을
만들지 못했다.

PSL v2는 관계를 더 많이 맞히는 범용 재랭커가 아니다. **명시적인 직접
Telemetry가 있는 관계만 확정하고, 나머지는 삭제하거나 거짓으로 판단하지
않고 `ABSTAIN`으로 보존하는 안전한 확인 계층**이다.

```text
A2 bounded abduction
        ↓
CALLS candidate set
        ↓
Canonical direct evidence
  ├─ Parent/child Trace
  ├─ CLIENT → SERVER Span kind
  └─ Source → Destination Workload
        ↓
Positive-only PSL aggregation
        ↓
Direct evidence policy
  ├─ CONFIRMED
  └─ ABSTAIN
```

## 핵심 변경

| 구분 | PSL v1 | PSL v2 |
|---|---|---|
| A2 prior | PSL 점수에 포함 | 확인 점수에서 제외 |
| 약한 Trace/Boundary | 양·음 규칙에 사용 | 확인에 사용하지 않음 |
| ReverseSupport | 음의 관계 규칙 | 제거 |
| DirectionConflict | 음의 관계 규칙 | 제거 |
| Sparsity | 전체 후보 음의 Prior | 제거 |
| Operation/Endpoint 유사도 | 확인 점수에 사용 | 확인에 사용하지 않음 |
| 직접 증거 없음 | 낮은 점수 또는 후보 탈락 | `ABSTAIN` |
| 출력 상태 | 선택/비선택 | `CONFIRMED` / `ABSTAIN` |
| 음의 CALLS | 암묵적 억제 가능 | 생성하지 않음 |

## Canonical direct evidence

PSL v2는 후보 단위의 다음 세 컬럼만 관계 확인에 사용한다.

| 컬럼 | 의미 | 생성 기준 |
|---|---|---|
| `direct_trace_evidence` | 직접 부모/자식 Trace 연결 | 동일 Trace에서 parent service와 child service가 후보 방향과 일치 |
| `client_server_evidence` | CLIENT→SERVER Span 연결 | source span kind가 CLIENT, target span kind가 SERVER이며 trace/endpoint가 연결 |
| `workload_evidence` | Workload pair 연결 | 관측 source workload와 destination workload가 후보 방향과 일치 |

각 값은 `[0,1]` 범위다. 현재 RCAEval Phase 3-R2 handoff의
`direct_evidence`는 하위 호환을 위해 `direct_trace_evidence` alias로
읽지만, CLIENT/SERVER와 Workload 채널은 해당 원천 컬럼이 실제로 존재할
때만 활성화한다.

다음 값은 직접 증거로 승격하지 않는다.

- `supporting_traces`
- `boundary_spans`
- `direction_score`
- `operation_role_score`
- `endpoint_compatibility_score`
- HTTP Method/Route 유사도
- `graph_role_score`
- A2 score/rank
- Reverse trace 또는 direction conflict

## PSL Rule

모든 규칙은 양의 Soft Rule이다.

```text
DirectTrace(C,S,O)               → ConfirmedCallsV2(C,S,O)
ClientServer(C,S,O)              → ConfirmedCallsV2(C,S,O)
Workload(C,S,O)                  → ConfirmedCallsV2(C,S,O)

DirectTrace ∧ ClientServer       → ConfirmedCallsV2
DirectTrace ∧ Workload           → ConfirmedCallsV2
ClientServer ∧ Workload          → ConfirmedCallsV2
DirectTrace ∧ ClientServer
            ∧ Workload           → ConfirmedCallsV2
```

다음 형태의 규칙은 금지한다.

```text
ReverseSupport      → ¬CALLS
DirectionConflict   → ¬CALLS
WeakTopologyMismatch → ¬CALLS
NoEvidence           → ¬CALLS
```

## Abstention 정책

고정 정책:

```text
channel_truth_min       = 0.90
psl_score_min           = 0.90
minimum_direct_channels = 1
```

판정:

```text
직접 채널 ≥ 1개 AND PSL score ≥ 0.90
    → CONFIRMED

그 외
    → ABSTAIN
```

`ABSTAIN`은 관계가 거짓이라는 뜻이 아니다. 현재 증거만으로 자동 확정할 수
없다는 뜻이며 A2 후보, 순위, 근거는 모두 보존한다.

## 평가 상태

전체 상태는 세 가지다.

| 상태 | 의미 |
|---|---|
| `PASS` | 직접 증거가 존재하고 안전성·효용 Gate를 통과 |
| `INELIGIBLE` | 구현 안전성은 통과했지만 데이터에 직접 증거가 없어 효용 검증 불가 |
| `FAIL` | 직접 증거가 있는데도 안전성 또는 효용 Gate 실패 |

RCAEval handoff에 직접 증거가 없다면 `INELIGIBLE`이 올바른 결과다. 약한
점수를 직접 증거로 변환해 억지로 PASS시키지 않는다.

## Gate

### Mechanism Gate

- 후보 1,250개 전부 보존
- 출력은 `CONFIRMED` 또는 `ABSTAIN`만 사용
- 직접 증거 없는 후보 확정 0건
- 음의 관계 규칙 0개
- A2 Prior 확인 규칙 0개
- 약한 Feature를 변경해도 직접 증거와 판정 불변
- Evaluator label은 PSL 점수 고정 후 결합
- DeBERTa 미사용

### Data eligibility

- Held-out 직접 증거 후보 Coverage > 0
- Held-out 정답 관계 중 직접 증거 Coverage > 0

### Utility Gate

- 확인 관계 1건 이상
- 확인 관계 P-LB ≥ 0.90
- 정답 관계 Confirmation Recall ≥ 0.50

## 해석 제한

PSL v2의 `CONFIRMED`는 Runtime `CALLS` 관계에 대한 직접 Telemetry 확인이다.
인과관계 `CAUSES`, Root Cause, Cause Path, Impact Path, RCA 또는 LLM 성능
개선을 의미하지 않는다.

현재 RCAEval 데이터가 `INELIGIBLE`이면 다음 검증은 parent span id,
`span.kind`, source/destination workload 또는 Jennifer Profile의 명시적
호출 방향을 제공하는 데이터셋에서 수행해야 한다.
