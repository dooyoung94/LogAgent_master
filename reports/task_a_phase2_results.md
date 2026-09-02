# Task A Phase 2 결과 — 다중 Incident·다중 Seed 일반화 검증

실행일: **2026-09-02**  
최종 판정: **PASS — D3_MULTICASE_MULTISEED_GENERALIZATION**

## 1. 검증 목적

Phase 1의 단일 사건·단일 Seed 결과를 넘어, 여러 TrainTicket 장애와 마스킹 Seed에서도 귀추 후보 생성이 다음 조건을 동시에 만족하는지 검증했다.

- 누락된 runtime `CALLS` 관계를 후보집합에 보존
- 전체 후보를 최대 32개로 제한
- Reference/Model Trace와 평가 정답 간 Leakage 차단
- DeBERTa·PSL 없이 A2 자체의 일반화 성능 측정

## 2. 실험 범위

| 항목 | 값 |
|---|---|
| Dataset | RCAEval RE2-TT |
| Incident | 6건 |
| Fault | CPU, MEM, DISK, DELAY, LOSS, SOCKET |
| Seed | 11, 17, 23, 31, 47 |
| Mask | IID20, IID40 |
| 총 Run | 30 |
| 총 평가 Cell | **60** |
| 관계 | `Service -[CALLS]-> Service` |
| 활성 단계 | A0, A1, A2 |
| 후보 상한 | 전체 32, Source별 8, Target별 8 |
| 보류 단계 | A3, A4, A5, RCA, LLM |

Incident ID는 Case명을 노출하지 않는 Revision+Case 기반 SHA-256 불투명 ID로 생성했다. 같은 Incident의 모든 Seed는 동일한 Reference/Model whole-trace split을 사용하고 Seed는 마스킹 위치에만 적용했다.

## 3. 선택된 Incident

| Fault | Case | Root Service | Trace Row |
|---|---|---|---:|
| CPU | `re2tt_ts-auth-service_cpu_2` | `ts-auth-service` | 838,936 |
| MEM | `re2tt_ts-travel-service_mem_3` | `ts-travel-service` | 587,772 |
| DISK | `re2tt_ts-order-service_disk_3` | `ts-order-service` | 126,429 |
| DELAY | `re2tt_ts-auth-service_delay_3` | `ts-auth-service` | 671,979 |
| LOSS | `re2tt_ts-train-service_loss_3` | `ts-train-service` | 1,042,622 |
| SOCKET | `re2tt_ts-travel-service_socket_3` | `ts-travel-service` | 783,847 |

CPU는 Phase 1 연속성 Anchor로 고정했고 나머지는 Fault별 적격 사건 중 `sha256(revision|task-a-phase2|case)` 최솟값으로 결정했다. 총 25개 파일, 121,565,953 Byte의 SHA-256 Provenance를 검증했다.

## 4. 종합 결과

| 지표 | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| 완료 Cell | **60/60** | 60 | PASS |
| Candidate Recall Macro | **1.0000** | ≥ 0.95 | PASS |
| Candidate Recall Minimum | **1.0000** | 각 Cell ≥ 0.90 | PASS |
| Masked Recall Macro | **1.0000** | 참고 | PASS |
| 후보 수 평균 | **20.83** | — | 기록 |
| 후보 수 중앙값 | **21** | — | 기록 |
| 후보 수 최대 | **32** | ≤ 32 | PASS |
| 후보 압축률 평균 | **96.70%** | — | 기록 |
| 후보 압축률 최소 | **95.20%** | — | 기록 |
| Budget 포화 | **12/60 = 20%** | ≤ 25% | PASS |
| Budget 제거 후보 | **117개** | 참고 | 기록 |
| Silver Precision Lower Bound 평균 | **0.7066** | 참고 | 기록 |
| Silver Precision Lower Bound 최소 | **0.4231** | 참고 | 주의 |
| A2 내부 MRR 평균 | **0.9872** | 참고 | 양호 |
| A2 내부 MRR 최소 | **0.8500** | 참고 | 주의 |
| Leakage Check | **60/60 통과** | 전부 통과 | PASS |

## 5. 마스킹 비율별 결과

| Mask | Cell | Recall 평균/최저 | 후보 평균/최대 | P-LB 평균 | 포화 Cell | 제거 후보 |
|---|---:|---:|---:|---:|---:|---:|
| IID20 | 30 | **1.0 / 1.0** | 14.77 / 26 | 0.6966 | **0** | 0 |
| IID40 | 30 | **1.0 / 1.0** | 26.90 / 32 | 0.7165 | **12** | 117 |

IID40에서도 모든 누락 관계를 보존했지만, 30개 중 12개 Cell이 32개 상한에 도달했다. 즉 현재 상한은 Recall을 지키면서 작동했으나 40% 누락 조건에서는 여유가 크지 않다.

## 6. Fault별 결과

| Fault | Recall 평균/최저 | 후보 평균/최대 | P-LB 평균 | 포화 Cell | 제거 후보 |
|---|---:|---:|---:|---:|---:|
| CPU | 1.0 / 1.0 | 21.9 / 32 | 0.7700 | 1 | 1 |
| MEM | 1.0 / 1.0 | 25.6 / 32 | 0.5758 | 5 | 42 |
| DISK | 1.0 / 1.0 | 8.2 / 14 | 0.7658 | 0 | 0 |
| DELAY | 1.0 / 1.0 | 27.7 / 32 | 0.5690 | 5 | 72 |
| LOSS | 1.0 / 1.0 | 20.2 / 30 | **0.8246** | 0 | 0 |
| SOCKET | 1.0 / 1.0 | 21.4 / 32 | 0.7342 | 1 | 2 |

MEM과 DELAY가 후보 수와 Unverified 관계 측면에서 가장 어려운 조건이었다. 두 Fault의 IID40은 모든 Seed에서 32개 상한에 도달했지만 Recall 1.0을 유지했다.

## 7. 최악 Cell

- 최저 P-LB: DELAY, Seed 31, IID20
  - Target 11개, 후보 26개, Recall 1.0, P-LB **0.4231**
- 최저 MRR: MEM, Seed 31, IID40
  - Target 20개, 후보 32개, Recall 1.0, MRR **0.85**, P-LB 0.625
- 최대 단일 Budget Drop: DELAY, Seed 31, IID40
  - 후보 32개 유지, 추가 후보 **20개 제거**, Recall 1.0

Silver Graph 밖의 후보는 실제 False가 아니라 Reference에서 확인되지 않은 `unverified` 관계일 수 있으므로 P-LB는 보수적 하한으로만 해석한다.

## 8. 핵심 해석

### 확인된 결과

- A2는 6개 Fault·5개 Seed·20/40% 마스킹의 **모든 60 Cell에서 누락 관계를 100% 후보로 보존**했다.
- 가능한 전체 Typed Universe를 평균 약 3.3% 크기로 압축했다.
- 32개 상한으로 후보 폭증을 막으면서도 이번 범위에서는 Recall 손실이 없었다.
- 따라서 현재 병목은 **후보 생성 Recall이 아니라 후보의 타당성 검증과 축소**다.

### 남은 위험

- IID40의 Budget 포화율은 40%(12/30)이며 MEM·DELAY에서 집중됐다.
- P-LB 최소 0.4231은 일부 Cell에서 후보 절반 이상이 Silver Reference 밖임을 뜻한다.
- 현재 마스킹과 귀추 Evidence가 모두 Trace parent/시간 구조에 기반하므로, 이것만으로 실제 운영의 다양한 Blind Spot 일반화를 주장할 수 없다.
- 복원된 `CALLS`는 runtime 구조 관계이지 `CAUSES` 인과관계가 아니다.

## 9. 실행 중 발견·수정한 문제

1. 생성된 Incident ID에 원본 Case명이 포함되어 Leakage 검증기가 차단
   - Revision+Case 기반 불투명 SHA-256 ID로 수정
2. Phase 2 후보 예산 설정이 Phase 1의 `ranking` 계약을 덮어써 요약 단계에서 실패
   - Phase 1 Ranking 계약을 그대로 복원
3. 실패 시 대용량 Trace 사본이 Artifact에 남음
   - 단일 Cell Preflight와 업로드 전 Trace 강제 제거 추가

수정 후 전체 테스트 **95 passed, 5 skipped, 10 subtests passed**와 60 Cell을 다시 실행했다.

## 10. 다음 연구 단계

다음 단계는 A2 후보를 다시 생성하는 것이 아니라, 현재 후보를 유지한 채 타당성을 축소하는 것이다.

1. A2 후보 32개를 불변 입력으로 고정
2. DeBERTa를 Hard Veto가 아닌 `corroborates / contradicts / ambiguous` 증거로 적용
3. Forward/Reverse 방향성, Operation, Runtime Role, Log·Metric Evidence를 별도 점수로 기록
4. Recall 0.95 이상을 유지하면서 P-LB·MRR·후보 수 개선 검증
5. 이후 PSL과 Calibration/Abstention 결합

**강한 결론:** A2 관계 후보 생성은 다음 단계로 진행할 수준이다. 다만 MEM·DELAY의 후보 포화와 낮은 P-LB 때문에, A3부터는 정답 관계를 제거하지 않으면서 Unverified 후보를 줄이는 설계가 핵심이다.

## 11. 재현 정보

- 성공 Branch Head: `87ba2925f91317eacb3e6d68ff499f6a2afecd35`
- GitHub Actions Run: `33577556148`
- Artifact ID: `9827625110`
- Artifact Size: 31,470,263 Byte
- Artifact ZIP SHA-256: `f8ffe957693eccbcc57718753ae878d1ec1f01af863a1a72569d34754b65b257`
- Provenance Manifest SHA-256: `fb3532771b2e52be0e0429efcf6dba9e08368ef066365c92125ebf8ffb9ad8cf`
- Selected Cases SHA-256: `f913991ae09a487bf6c8b8f0cf75b466eca9e89be100cfe6569ee6d611cb39c7`
