# RCABench OPS-Lite 독립 확인시험 사전등록

## 1. 목적

RCAEval TrainTicket에서 개발한 A3-R3 정책이, 개발 중 사용하지 않은 RCABench 계열 Incident와 다른 시스템에서도 `CALLS` 후보의 Recall을 보존하면서 동일 크기 A2/R2보다 추가 판별력을 제공하는지 확인한다.

이 시험은 **정책 탐색이 없는 외부 확인시험**이다. 확인 데이터 결과를 본 뒤 Case, Seed, Threshold, 정책 가중치 또는 성공 기준을 바꾸지 않는다.

## 2. 데이터 고정

| 항목 | 고정값 |
|---|---|
| Dataset | RCABench OPS-Lite |
| Repository | `anon-ops/ops-lite` |
| Revision | `9ac09981c08ab02a0b923eab7830d778934851a8` |
| Manifest SHA-256 | `5d4d3960446408a6e43ea87c51a255aa7a39d43a1db6aeb731cf20b18f4fb7cd` |
| Split | 공식 RCABench leaderboard seed-42 Test 100건 |
| Split commit | `8eae8ca62d425437f010f78afde0a4e7606e6da6` |
| Split SHA-256 | `f129fe8414985ac35b16aa0c22c4afe8d1ae132c9e16b46681ae009a7d178280` |

공식 FSE 데이터 저장소는 실제 파일 다운로드 시 인증이 요구되어, 공식 RCABench leaderboard가 공개 데이터로 등록한 checksum-locked OPS-Lite를 사용한다. 이는 데이터 가용성에 따른 변경이며, 성능값을 보고 Dataset을 교체한 것이 아니다.

## 3. Incident 선택 계약

Telemetry·Fault Label·모델 결과를 읽기 전에 Manifest의 구조 규모만 사용한다.

1. 공식 Test 100건으로 한정한다.
2. `n_svc >= 5`, `n_edge >= 5`인 Incident만 구조적으로 적격으로 정의한다.
3. 시스템별로 `revision|independent-confirmatory-v1|case_id`의 SHA-256 순서를 계산한다.
4. TrainTicket 4건, Hotel Reservation 2건, OTel Demo 2건을 선택한다.
5. 선택 후에는 Incident 교체를 금지한다.

선택 Case:

| 시스템 | Case ID |
|---|---|
| TrainTicket | `ts2-ts-order-other-service-container-kill-48rlds` |
| TrainTicket | `ts9-ts-order-service-container-kill-bsh6lx` |
| TrainTicket | `batch-01KQJWSGJ099M4CZYEP916XDZ4` |
| TrainTicket | `ts3-ts-travel-service-response-delay-7c9494` |
| Hotel Reservation | `hs1-frontend-delay-qhklld` |
| Hotel Reservation | `hs7-geo-pod-failure-fpxsqz` |
| OTel Demo | `otel-demo3-product-catalog-pod-failure-kp7pdt` |
| OTel Demo | `batch-01KQKYZVMC7JPGSX5S16TABEVF` |

선택 목록 SHA-256:

```text
f881f2018fcf4dd5455cd7bea3d2a84f6f8091283c8924b215fc16e96bcc8bc6
```

## 4. 실험 Cell

| 항목 | 고정값 |
|---|---|
| Incident | 8건 |
| 시스템 | 3종 |
| Seed | 101, 211, 307 |
| Mask | IID20, IID40 |
| 총 Cell | 48 |
| Relation | `Service -[CALLS]-> Service` |
| Evidence level | L2 parent dropped |
| Reference/Model Trace split | 40% / 60%, whole-trace hash split |

정상·장애 Trace ID와 Span ID는 Phase prefix를 부여해 충돌을 막는다. Trace의 Parent–Child, Service, Span Kind, HTTP Method는 Model-visible Evidence로 사용한다. Root Cause와 Fault Label은 후보 점수 생성에 사용하지 않는다.

## 5. 동결 대상

RCAEval R3의 실제 과학적 Gate가 PASS한 경우에만 다음을 파일로 고정한다.

- 선택 R3 정책: retention, minimum keep, operational weight, NLI weight
- DeBERTa 채널 가중치
- Tri-state Threshold
- DeBERTa Repository, Revision, ONNX SHA-256
- 개발 실행 Run ID, Commit SHA, 결과 파일 SHA-256

확인시험에서는 위 값을 읽기만 하며 정책 Grid Search를 실행하지 않는다.

## 6. 대조군

같은 Cell과 같은 후보 수에서 다음을 비교한다.

1. A2 전체 후보
2. Equal-size A2-only
3. Equal-size R2 operational evidence
4. Frozen A3-R3

후보 수를 줄인 효과와 NLI의 추가 효과를 분리하기 위한 구성이다.

## 7. 성공 Gate

- 선택 Incident 8건과 48 Cell 전부 실행 가능
- TrainTicket, Hotel Reservation, OTel Demo 모두 포함
- A2 Candidate Recall 평균 `>= 0.95`, Cell 최저 `>= 0.90`
- Frozen R3 Recall 평균·Pooled `>= 0.95`, Cell 최저 `>= 0.90`
- A2 전체 대비 평균 후보 수 5% 이상 감소
- A2 전체 대비 P-LB 비열등, MRR 저하 0.01 이내
- 동일 크기 A2 대비 Recall·P-LB·MRR 비열등
- 동일 크기 R2 대비 Recall·P-LB·MRR 비열등
- 동일 크기 R2 대비 P-LB 또는 MRR의 추가 개선 `>= 1e-6`
- 10,000회 paired bootstrap에서 P-LB 또는 MRR 개선의 95% 하한 `>= 0`
- NLI Candidate Coverage `>= 0.95`
- NLI Score 표준편차 `>= 1e-6`

## 8. 실패 처리

- 구조적으로 Masking이 불가능한 Case를 다른 Case로 교체하지 않는다.
- A2가 누락한 정답 관계는 R3 Recall과 MRR에서 0으로 반영한다.
- Engineering 오류는 수정할 수 있으나 정책·Case·Seed·Gate는 수정하지 않는다.
- 과학적 Gate가 FAIL해도 결과·Cell Matrix·오류 원인·Artifact를 게시한다.

## 9. 주장 범위

PASS하더라도 다음만 주장한다.

> 동결된 R3 정책이 사전 선택한 RCABench OPS-Lite의 신규 Incident와 3개 시스템에서 runtime `CALLS` 후보 재랭킹 성능을 외부 확인했다.

다음은 이 실험의 범위가 아니다.

- `CALLS`가 장애의 인과관계 `CAUSES`라는 주장
- RCA Root 또는 Cause Path 성능 향상
- LLM 장애분석 성능 향상
- Production Terabyte Scale 일반화
