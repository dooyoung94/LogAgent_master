# Task A Phase 3-R1 — 구조 Evidence 우선 검증

## 목적

기존 A3의 가장 큰 문제는 DeBERTa 자체의 크기보다 후보별 입력 Evidence가 충분히 구분되지 않았다는 점이다.

```text
corroborates = 0
ambiguous    = 7
contradicts  = 1,243
```

따라서 NLI 문장이나 threshold를 다시 조정하기 전에, A2 handoff에 이미 존재하는 model-visible 구조 Evidence가 동일 크기 A2-only shortlist보다 추가 판별력을 제공하는지 먼저 검증한다.

## 검증 순서

```text
A2 bounded candidates
       ↓
Forward / Reverse trace support
Forward / Reverse boundary support
Support density / self-loop constraint
       ↓
Structured reranking
       ↓
Calibration 20 Cell에서 정책 고정
       ↓
Held-out 40 Cell 검증
       ↓
Equal-size A2-only control 비교
```

## 사용 Feature

| Feature | 의미 |
|---|---|
| `trace_direction_margin` | 순방향과 역방향 supporting trace의 비대칭 |
| `boundary_direction_margin` | 순방향과 역방향 boundary span의 비대칭 |
| `trace_direction_ratio` | smoothing을 적용한 순방향 trace 비율 |
| `boundary_direction_ratio` | smoothing을 적용한 순방향 boundary 비율 |
| `forward_support_density` | boundary 수 대비 독립 trace 지지 밀도 |
| `reverse_support_density` | 역방향 후보의 지지 밀도 |
| `self_loop` | `Service → same Service` 구조 위험 |
| `a2_rank_normalized` | 기존 귀추 prior |

평가 정답인 `is_masked_target`, `is_silver_matched`, `case`, `fault`, `role`은 Feature 계산 전에 제거한다. Feature 계산이 완료된 뒤 immutable candidate key로만 evaluator 정보와 재결합한다.

## 정책

```text
A3-R1 score
  = (1 - structure_weight) × normalized A2 prior
  + structure_weight × structured evidence score
```

- A2 후보 전체는 삭제하지 않고 보존한다.
- 직접 관측 관계는 shortlist 예산과 무관하게 보호한다.
- Calibration Incident에서만 정책을 선택한다.
- Held-out label은 정책 동결 이후에만 평가에 사용한다.

## 성공조건

| 조건 | 기준 |
|---|---:|
| Held-out Recall Macro | ≥ 0.95 |
| Held-out Pooled Recall | ≥ 0.95 |
| 각 Cell Recall | ≥ 0.90 |
| 후보 수 | A2 대비 최소 5% 감소 |
| P-LB | A2 전체보다 비열등 |
| MRR | A2 전체보다 비열등 |
| Equal-size A2 대비 Recall | 비열등 |
| Equal-size A2 대비 P-LB/MRR | 비열등 |
| 추가 판별력 | Equal-size A2보다 P-LB 또는 MRR가 실제 증가 |

마지막 조건이 핵심이다. 후보 수를 줄여 절대 지표가 좋아지는 것만으로는 구조 Evidence의 효과로 인정하지 않는다.

## 결과 해석

- **PASS**: 방향 비대칭·지지 밀도가 A2 순위에 실제 추가 판별력을 제공한다. 다음 단계에서 Evidence별 DeBERTa NLI를 추가한다.
- **FAIL**: 현재 A2 handoff의 count 정보만으로는 부족하다. 다음 우선순위는 `span_kind`, `HTTP method/route`, `operation`, `endpoint compatibility`를 독립 Feature로 materialize하는 것이다.

어떤 결과에서도 Runtime `CALLS`를 장애 인과관계 `CAUSES`로 해석하지 않는다.

## 실행

```bash
python tools/run_task_a_phase3_r1.py \
  --candidate-analysis <A3_ARTIFACT>/evaluator_private/a3_candidate_analysis.parquet \
  --config configs/experiment_task_a_rcaeval_phase3_r1.json \
  --output outputs/task_a/phase3_r1
```
