# Task A Phase 2 — 다중 Incident·다중 Seed 일반화 검증

## 목적

Phase 1에서는 RCAEval TrainTicket 사건 1건, Seed 17에서 IID20·IID40 모두
Candidate Recall 1.0을 확인했다. 그러나 같은 사건과 단일 Seed 결과만으로는
귀추 후보 생성 방식이 다른 장애·마스킹에서도 유지된다고 주장할 수 없다.

Phase 2의 질문은 다음과 같다.

> `CALLS` 관계가 20% 또는 40% 누락된 여러 TrainTicket 사건에서, 제한된 귀추
> 후보 수를 유지하면서 숨긴 관계를 반복적으로 후보집합에 보존할 수 있는가?

## 사전 고정 범위

| 항목 | 값 |
|---|---|
| Dataset | RCAEval RE2-TT |
| Incident | 6건 |
| Fault strata | cpu, mem, disk, delay, loss, socket |
| Seed | 11, 17, 23, 31, 47 |
| Mask | IID20, IID40 |
| 총 평가 Cell | 6 × 5 × 2 = **60** |
| Relation | `Service -[CALLS]-> Service` |
| Active | A0, A1, A2 |
| Deferred | A3, A4, A5, RCA, LLM |

## Incident 선정

- CPU는 Phase 1 연속성 확인을 위해 `re2tt_ts-auth-service_cpu_2`로 고정한다.
- mem·disk·delay·loss·socket은 고정 Revision에서 다음 필터를 통과한 사건만 사용한다.
  - `dataset == RE2-TT`
  - Log 존재 및 1건 이상
  - Trace 존재 및 1건 이상
- 각 Fault에서 아래 값이 가장 작은 사건 1건을 선택한다.

```text
sha256(dataset_revision | task-a-phase2 | case_id)
```

따라서 실험 결과, Mask target, 관계복원 점수로 사건을 선택하지 않는다. Fault와
Root Service는 표본 층화 및 결과 집계에만 사용하며 관계추론 입력에는 전달하지
않는다.

## Leakage 통제

- Incident ID는 Case별로 고정하며 Seed를 포함하지 않는다.
- 따라서 같은 Case의 5개 Seed에서 Reference/Model whole-trace split은 동일하다.
- Seed는 IID 마스킹 위치에만 영향을 준다.
- Reference Graph, Mask Manifest, Injection Time, Fault/Root Label은 모델 입력에서 제외한다.
- 다운로드한 파일은 모두 SHA-256과 Byte Count를 `.logagent-source.json`에 기록한다.

## 후보 예산

| 제약 | 값 |
|---|---:|
| 전체 귀추 후보 | 최대 32 |
| 동일 Source | 최대 8 |
| 동일 Target | 최대 8 |
| Supporting Trace | 최소 1 |
| Boundary Span | 최소 1 |

후보 정렬은 Phase 1과 동일하다.

1. Abduction score 내림차순
2. Supporting Trace 수 내림차순
3. Boundary Span 수 내림차순
4. Edge key 오름차순

## D3 통과 기준

| 항목 | 기준 |
|---|---:|
| Grid 완결성 | 6 Incident × 5 Seed × 2 Mask 전부 완료 |
| Cell별 Candidate Recall | 모두 ≥ 0.90 |
| Macro Candidate Recall | ≥ 0.95 |
| 후보 수 | 모두 ≤ 32 |
| Budget saturation rate | ≤ 0.25 |
| Leakage Check | 전 Cell 통과 |

`silver_precision_lower_bound`, 압축률, 후보 수 분포, 예산으로 제거된 후보 수는
보조지표다. Silver Graph 밖의 관계는 곧바로 false로 간주하지 않는다.

## 실행

```bash
python tools/prepare_task_a_phase2_data.py \
  --config configs/experiment_task_a_rcaeval_phase2.json \
  --dest data/raw/rcaeval/phase2

python tools/run_task_a_phase2.py \
  --config configs/experiment_task_a_rcaeval_phase2.json \
  --raw-root data/raw/rcaeval/phase2 \
  --output outputs/task_a/phase2
```

기본 병렬 수는 2다. 각 Cell 실행 후 대용량 Model Trace 사본은 제거하고, Summary,
A2 Prediction, Evaluation, Manifest와 집계 CSV만 보존한다.

## 산출물

```text
outputs/task_a/phase2/
├── selected_cases.json
├── phase2_config.json
├── cells.csv
├── runs.json
├── summary.json
├── cell_configs/
└── runs/<case>/seed-<seed>/
```

## 해석 제한

Phase 2가 통과해도 검증 범위는 RCAEval RE2-TT의 runtime `CALLS` 후보복원이다.
복원 관계가 인과관계라는 의미가 아니며, DeBERTa·PSL·RCA·LLM 개선을 증명하지
않는다. D3 통과 후 A2 후보를 제거하지 않는 보조 증거 방식으로 DeBERTa를
설계한다.
