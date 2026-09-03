# Task A Phase 3-R2 — Operation·HTTP·Role Evidence 검증

## 목적

기존 A3의 가장 큰 문제는 일반 NLI 문장이 후보별 차이를 만들지 못해 거의 모든 관계를 `contradicts`로 분류한 것이다. A3-R2는 DeBERTa를 다시 조정하기 전에, `CALLS` 관계와 직접 관련된 운영 Evidence가 동일 후보 수의 A2-only 순위보다 실제 추가 판별력을 제공하는지 검증한다.

```text
A2 Bounded Abduction
        ↓
Trace Direction
+ Operation / Method
+ Endpoint / Route
+ Caller·Callee Role
+ optional Span Kind / Workload
        ↓
Operational Evidence Reranking
        ↓
Equal-size A2 Control
```

## Evidence 원칙

| 채널 | 직접값 | 직접값이 없을 때의 Proxy |
|---|---|---|
| 호출 방향 | Parent/Child Span | Forward·Reverse 귀추 지지 비대칭 |
| Operation | `operation_name`, `method_name` | Operation token overlap |
| HTTP | `http_method`, `http_route` | Operation 문자열에서 Method·Path 파싱 |
| Endpoint | 명시 Route | 동적 ID를 정규화한 Path 호환성 |
| Span 역할 | `CLIENT/SERVER`, `PRODUCER/CONSUMER` | Operation의 parent/child 역할 빈도 |
| Service 역할 | Source/Destination workload | 관측 `CALLS`의 in/out-degree 역할 Proxy |

RCAEval에 존재하지 않는 `span.kind`, `http.route`, workload를 존재한다고 가정하지 않는다. 필드가 없으면 해당 Coverage를 0으로 기록하고 Proxy 사용 사실을 결과에 남긴다.

## 누수 방지

Feature 계산 입력은 다음으로 제한한다.

- 마스킹된 Model Trace
- 마스킹 후 Observed Graph
- 고정된 A2 후보와 Evidence ID

다음 정보는 Feature 계산 이후 Evaluator 단계에서만 결합한다.

- Mask target
- Silver matched 여부
- Fault label
- Root Cause label
- Reference Graph

## 비교 설계

- Dataset: RCAEval RE2 TrainTicket
- Incident: 기존 6건
- Seed: 11, 17, 23, 31, 47
- Mask: IID20, IID40
- Calibration: 2 Incident, 20 Cell
- Held-out: 4 Incident, 40 Cell
- 후보: A2의 1,250개 후보를 변경 없이 인계
- 대조군: 각 Cell에서 A3-R2와 정확히 같은 후보 수를 A2 순위만으로 선택

기존 6개 Incident는 이미 A3와 A3-R1에서 확인했으므로 이번 결과는 **개발 재검증**이다. 성공하더라도 새로운 Incident 집합에서 별도 확인해야 한다.

## 성공 조건

| 지표 | 기준 |
|---|---:|
| Held-out Recall Macro/Pooled | ≥ 0.95 |
| 각 Cell Recall | ≥ 0.90 |
| 후보 수 | A2 대비 최소 5% 감소 |
| P-LB / MRR | A2 전체 대비 비열등 |
| Equal-size A2 대비 Recall/P-LB/MRR | 비열등 |
| 추가 판별력 | Equal-size A2 대비 P-LB 또는 MRR +0.000001 이상 |
| Boundary 재구성 정렬률 | ≥ 0.95 |
| Operation Pair 후보 Coverage | ≥ 0.95 |

## 해석 범위

PASS는 Operation·HTTP·Role Evidence가 **같은 후보 예산에서 A2 기본 순위보다 더 좋은 CALLS 후보를 선택했다**는 뜻이다. 이는 causal `CAUSES`, RCA, LLM 성능 개선을 의미하지 않는다.

FAIL이면 Threshold를 사후 조정하지 않고 다음을 확인한다.

1. RCAEval의 Operation/Method가 실제 Endpoint 의미를 충분히 담는가
2. 직접 `span.kind`·HTTP attribute 부재가 판별력 한계인가
3. A2 Evidence ID로 재구성한 Boundary Pair가 충분히 정확한가
4. CALLS 관계 검증에 Trace 원문보다 Source/K8s Route Reference가 필요한가
