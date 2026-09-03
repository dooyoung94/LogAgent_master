# Task A Phase 3-R3 — Evidence별 DeBERTa Tri-state와 해석 가능한 재랭커

## 목적

기존 A3는 모든 정보를 하나의 합성 문장에 넣어 `corroborates=0`,
`contradicts=1,243`으로 붕괴했다. R3는 모델을 개조하거나 Threshold만 다시
맞추기 전에 다음 질문을 검증한다.

> Trace direction, Operation, HTTP, Runtime role을 독립 Evidence로 평가하면,
> 동일 후보 수의 Operational-only 재랭커보다 추가 판별력이 생기는가?

## 입력과 Leakage 경계

- 입력: A3-R2가 보존한 A2 후보 1,250개와 model-visible operational feature
- Cell: 6 Incident × 5 Seed × IID20/IID40 = 60
- Calibration: 기존 2 Incident·20 Cell
- Held-out: 기존 4 Incident·40 Cell
- NLI 입력 전 제거: `case`, `fault`, `role`, `is_masked_target`,
  `is_silver_matched`
- 서비스 이름은 NLI 문장에서 사용하지 않고 `Service A/B`로 익명화
- 정답 Label은 NLI 점수 계산이 끝난 뒤 불변 Candidate Key로 다시 결합

기존 Incident는 이미 분석에 사용됐으므로 이 결과는 개발 검증이다. 최종 주장을
위해서는 새로운 독립 Incident가 필요하다.

## 독립 Evidence 채널

| 채널 | 내용 |
|---|---|
| Direction | 순·역방향 Trace 수와 Boundary Span 수 |
| Operation | 대표 부모·자식 Operation, Token overlap, 역할 prior |
| HTTP | Method coverage/match, Route coverage/overlap/exact match |
| Role | 관측 그래프 in/out degree, span.kind와 workload 가용 증거 |

각 채널은 Forward와 Reverse 가설을 별도로 평가한다.

```text
Premise(Direction) → A CALLS B / B CALLS A
Premise(Operation) → A CALLS B / B CALLS A
Premise(HTTP)      → A CALLS B / B CALLS A
Premise(Role)      → A CALLS B / B CALLS A
```

Tri-state(`corroborates`, `ambiguous`, `contradicts`)는 진단 지표일 뿐 후보를
제거하지 않는다. 재랭커는 연속 NLI 방향성 점수를 사용한다.

## 재랭커

```text
A3-R3 score
= A2 Prior
+ Operational Evidence Rank
+ Channel-specific NLI Rank
```

- A2 가중치는 항상 0.20 이상 보존
- Calibration에서만 선형 가중치와 후보 유지율 선택
- Direct Evidence는 Shortlist에서 보호

## 대조군

1. A2 전체 후보
2. A3-R3와 동일 개수의 A2-only 후보
3. A3-R3와 동일 개수·동일 Operational 설정의 Operational-only 후보

세 번째 대조군이 핵심이다. A3-R3 성공은 후보 수 축소나 Operation Feature가
아니라 **DeBERTa NLI의 추가 효과**를 의미해야 한다.

## 성공조건

- Held-out Recall Macro/Pooled ≥ 0.95
- 각 Cell Recall ≥ 0.90
- 후보 수 A2 대비 최소 5% 감소
- A2 전체 대비 P-LB·MRR 비열등
- 동일 크기 A2-only 대비 Recall·P-LB·MRR 비열등
- 동일 크기 Operational-only 대비 Recall·P-LB·MRR 비열등
- Operational-only 대비 P-LB 또는 MRR 실제 증가
- NLI 가중치 > 0, A2 Prior ≥ 0.20
- 전체 후보 NLI 점수화, Token truncation 0

## 해석 제한

R3는 runtime `CALLS` 후보 재랭킹만 검증한다. `CALLS`를 causal `CAUSES`로
해석하지 않으며 RCA·LLM 성능 개선도 아직 주장하지 않는다.
