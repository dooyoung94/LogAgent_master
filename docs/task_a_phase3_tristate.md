# Task A Phase 3 — DeBERTa 3상태 증거와 보수적 후보 축소

## 목적

Phase 2에서 A2 귀추는 6개 Incident × 5개 Seed × IID20/IID40의 60개 Cell에서
Candidate Recall 1.0을 유지했지만, 후보는 평균 20.83개이고 IID40의 12개 Cell이
32개 상한에 도달했다. A3는 A2 후보를 다시 생성하지 않고 **후보 타당성 증거를
추가해 shortlist를 줄이는 단계**다.

핵심 원칙은 다음과 같다.

- A2 후보 전체는 불변 입력과 감사 산출물로 보존한다.
- DeBERTa는 단일 관계를 삭제하는 hard veto로 사용하지 않는다.
- NLI 결과는 `corroborates / ambiguous / contradicts` 3상태로 기록한다.
- `contradicts`도 자동 삭제하지 않고 재랭킹 점수로만 사용한다.
- Calibration Incident와 Held-out Incident를 분리한다.
- 동일 후보 수의 A2-only control을 함께 측정해 후보 예산 효과와 NLI 효과를 구분한다.

## IN → OUT

| 단계 | IN | 처리 | OUT |
|---|---|---|---|
| A2 고정 | 최대 32개 `CALLS` 후보 | 변경하지 않음 | Frozen P2 |
| 방향 증거 | Forward/Reverse containment 수 | 문장화 | Flat premise |
| 문맥 증거 | 서비스 역할·이웃·Operation | 문장화 | Runtime premise |
| DeBERTa | 두 premise × 정/역방향 | NLI 확률 | 3상태 증거 |
| Calibration | 해시 선정 2개 Incident | 후보 유지율·NLI 가중치 선택 | Frozen policy |
| Held-out | 나머지 4개 Incident | 정책 1회 적용 | A3 shortlist |

## NLI 증거

각 후보 `A → B`에 대해 다음 네 쌍을 평가한다.

1. Flat premise → `A directly invokes B`
2. Flat premise → `B directly invokes A`
3. Runtime premise → `A directly invokes B`
4. Runtime premise → `B directly invokes A`

Flat premise에는 다음 정보만 포함한다.

- A가 outer span이고 B가 inner span이라는 시간 위치
- Forward supporting trace / boundary 수
- Reverse supporting trace / boundary 수
- 해당 정보가 확정 관계가 아닌 귀추 후보라는 제한

Runtime premise에는 model partition에서 계산된 다음 정보가 추가된다.

- In/Out degree와 orchestrator/provider proxy
- 관측 upstream/downstream
- HTTP·data-access 비율
- Operation 예시

Runtime context는 모델 토큰 절단에 의존하지 않는다. 비어 있지 않은 앞 8개 줄을
줄당 144자, 전체 960자로 명시적으로 직렬화한 뒤 토큰 길이를 검사하며, 512토큰을
초과하면 해당 실행을 실패 처리한다.

Root label, Fault label, Injection time, Mask target, Silver graph는 NLI 입력에 사용하지
않는다. 모델용 후보 산출물에서도 원본 Case명, Fault, Calibration/Held-out 역할과
평가 정답 컬럼을 제외하고 불투명 Incident ID만 제공한다.

## 3상태 판정

- `corroborates`: 정방향 entailment와 방향·label margin이 모두 충분함
- `contradicts`: 역방향 entailment 또는 정방향 contradiction이 명확히 우세함
- `ambiguous`: 위 두 조건을 만족하지 않음

3상태는 설명 및 재랭킹용이다. 어떤 상태도 단독으로 후보를 제거하지 않는다.

## Calibration / Held-out

6개 Incident는 `sha256(revision|task-a-phase3-calibration|case)` 순서로 정렬한다.
앞의 2개만 Calibration에 사용하고, 나머지 4개는 정책이 고정된 뒤 평가한다.

정책 탐색값:

- Retention fraction: 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
- Minimum keep: 8, 10, 12, 14
- NLI weight: 0.05, 0.10, 0.20, 0.30

Calibration에서 Recall과 MRR 비열등 기준을 만족하는 정책 중 평균 후보 수가 가장
작은 정책을 선택한다. 동률이면 P-LB, MRR, 작은 NLI 가중치 순으로 결정한다.
Calibration에서도 동일 후보 수의 A2-only 대조군보다 Recall·P-LB·MRR이 열화되지
않고 P-LB 또는 MRR에 양의 개선이 있는 정책만 선택한다.

## Held-out Gate

| 지표 | 기준 |
|---|---:|
| Recall Macro | ≥ 0.95 |
| Pooled Recall | ≥ 0.95 |
| Cell 최저 Recall | ≥ 0.90 |
| 평균 후보 수 비율 | A2 대비 ≤ 0.95 |
| P-LB Macro | A2 이상 |
| MRR Macro | A2 이상 |
| A2 전체 증거 보존 | 필수 |

A3와 동일한 shortlist 크기를 A2 점수만으로 선택한 `matched-budget A2 control`도
별도로 계산한다. A3는 이 대조군보다 Recall·P-LB·MRR이 열화되지 않아야 하고,
P-LB 또는 MRR 중 최소 하나에서 양의 개선이 있어야 통과한다. 따라서 단순 후보
예산 효과를 DeBERTa의 효과로 오인하지 않는다.

## 해석 제한

A3 통과는 `CALLS` 관계 후보의 타당성 shortlist가 개선됐다는 의미다. `CALLS`를
`CAUSES`로 해석하거나 RCA·LLM 성능 개선을 주장하지 않는다. 해당 주장은 이후
PSL/Calibration과 Task B 평가에서 별도로 검증한다.
