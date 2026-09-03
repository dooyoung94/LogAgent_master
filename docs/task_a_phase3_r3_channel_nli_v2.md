# Task A Phase 3-R3 — Evidence별 DeBERTa Tri-state 검증

## 1. 목적

기존 A3는 Trace·Operation·Runtime Context를 하나의 합성 문장으로 입력해 대부분의 후보를 `contradicts`로 분류했고, 동일 후보 수의 A2-only 대조군보다 추가 판별력을 보이지 못했다. R3는 모델 자체를 개조하기 전에 입력 Evidence를 관계별 독립 채널로 분리해 다음 질문을 검증한다.

> A2 귀추 Prior와 A3-R2 운영 Evidence를 보존한 상태에서, Trace·Operation·HTTP·Role별 DeBERTa NLI 점수가 동일 크기 대조군보다 실제로 더 좋은 `CALLS` 후보를 선택하는가?

## 2. 고정 입력

| 항목 | 값 |
|---|---|
| Dataset | RCAEval RE2 TrainTicket |
| Incident | 6건 |
| Seed | 11, 17, 23, 31, 47 |
| Mask | IID20, IID40 |
| 평가 Cell | 60 |
| A2 후보 | 1,250개 |
| Relation | `Service -[CALLS]-> Service` |
| Calibration / Held-out | 20 / 40 Cell |
| DeBERTa | `cross-encoder/nli-deberta-v3-small` ONNX INT8 |
| Batch | 1 — batch composition drift 방지 |

현재 6개 Incident는 이전 A3 단계에서 이미 관찰됐으므로 이번 결과는 개발 재검증이다. 최종 확증에는 새로운 Incident 또는 다른 시스템이 필요하다.

## 3. Evidence 채널

| 채널 | 사용 정보 | 데이터 부재 시 처리 |
|---|---|---|
| Trace | 순·역방향 Supporting Trace, Boundary Span, 정합도 | 후보에 Trace 증거가 없으면 unavailable |
| Operation | Operation token overlap, pair concentration, parent/child role | Boundary Operation pair가 없으면 unavailable |
| HTTP | Method·Route coverage와 일치도 | 실제 속성이 없으면 추정 사실을 만들지 않고 unavailable |
| Role | 관측 CALLS의 in/out degree, Span kind, workload | 직접 Span kind/workload가 없으면 coverage 0으로 기록 |

각 채널은 독립 Premise와 순방향·역방향 Hypothesis로 평가한다.

```text
Trace premise      ─┐
Operation premise  ─┼─ DeBERTa forward / reverse ─ tri-state score
HTTP premise       ─┤
Role premise       ─┘
```

- `corroborates`: 순방향 Entailment와 방향 Margin이 충분함
- `contradicts`: 역방향 또는 Contradiction이 우세함
- `ambiguous`: 어느 쪽도 충분하지 않음
- `unavailable`: 해당 Evidence가 실제 입력에 없음

`contradicts`는 후보를 단독 삭제하지 않고 재랭킹 Feature로만 사용한다.

## 4. 최종 점수

\[
S_{R3}(h)=
\alpha S_{A2}(h)
+\beta S_{Operational}(h)
+\gamma S_{NLI}(h)
\]

- `A2`: 귀추 후보 Prior
- `Operational`: R2의 Operation·HTTP·Role 기반 순위
- `NLI`: 채널별 DeBERTa 점수의 가용성·신뢰도 가중 결합
- `\alpha > 0`, `\gamma > 0`을 강제한다.

## 5. Leakage 통제

```text
Sanitized Trace + Observed Graph + A2 Candidate
        ↓
Operational Feature 계산
        ↓
Evidence별 DeBERTa 점수 계산·고정
        ↓
Evaluator Target / Silver Label 결합
        ↓
Calibration 정책 선택
        ↓
고정 정책으로 Held-out 평가
```

Feature와 NLI 계산 시 다음 값은 접근하지 않는다.

- Mask Target
- Silver Match
- Fault Label
- Root Cause Service
- Reference Graph

## 6. 대조군

| 대조군 | 목적 |
|---|---|
| A2 Full | 후보 축소 전 절대 기준 |
| Equal-size A2 | 후보 개수 감소 효과와 NLI 효과 분리 |
| Equal-size R2 | Operation·HTTP·Role만 사용했을 때보다 NLI가 추가 가치가 있는지 검증 |

R3의 핵심 성공조건은 **동일 후보 수의 R2보다 P-LB 또는 MRR이 실제 증가하는 것**이다.

## 7. 과학적 Gate

| 조건 | 기준 |
|---|---:|
| Held-out Recall Macro / Pooled | ≥ 0.95 |
| 각 Cell Recall | ≥ 0.90 |
| 후보 수 | A2 평균의 95% 이하 |
| A2 Full 대비 P-LB·MRR | 비열등 |
| Equal-size A2 대비 Recall·P-LB·MRR | 비열등 |
| Equal-size R2 대비 Recall·P-LB·MRR | 비열등 |
| DeBERTa 추가 이득 | Equal-size R2 대비 P-LB 또는 MRR ≥ 1e-6 |
| NLI 후보 Coverage | ≥ 0.95 |
| NLI Score 표준편차 | ≥ 1e-6 |
| ONNX 실행 | batch=1, checksum 일치, silent truncation 0 |

과학적 Gate가 실패해도 결과·정책 Grid·Held-out Cell·NLI 상태 분포를 모두 보존한다.

## 8. 실행

```bash
python tools/prepare_deberta_model.py \
  --dest data/models/nli-deberta-v3-small

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python tools/run_task_a_phase3_r3_v2.py \
  --phase2-root outputs/task_a/phase2_r3_source \
  --model-dir data/models/nli-deberta-v3-small \
  --output outputs/task_a/phase3_r3
```

## 9. 해석 제한

- PASS는 Runtime `CALLS` 후보 재랭킹의 개발 단계 효용을 의미한다.
- `CALLS` 복원은 `CAUSES` 인과관계 복원이 아니다.
- RCA·LLM 성능 개선은 Task B에서 별도로 검증한다.
- 같은 6개 Incident를 재사용하므로 새로운 Incident 기반 확증 실험 전에는 일반화 결론을 내리지 않는다.
