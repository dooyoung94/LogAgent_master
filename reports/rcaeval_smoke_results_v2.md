# RCAEval 누적 관계복원 smoke v2 결과

실행일: 2026-08-28  
실험: `rcaeval-call-recovery-cumulative-smoke-v2`  
데이터 리비전: `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`  
결론 상태: **미확정** — 관계복원 smoke 결과이며 RCA 원인·영향 경로 개선은 아직 측정하지 않음

## 후속 기본 실행 정책

이 문서는 IID60을 포함한 `v2-full`의 역사적 결과를 그대로 보존한다. 결과를
확인한 이후의 알고리즘 개발 실행은 계산 예산을 위해 IID20, IID40,
component만 포함하는 `v2-budget`을 기본으로 사용한다. 이는 IID60 결과를
삭제하거나 IID60이 부적절한 평가 조건임을 뜻하지 않는다. budget 실행에서
허용되는 주장은 최대 40% IID 누락과 component blackout 범위로 제한한다.
`v2-full`의 정확한 구현 snapshot은 Git commit
`e5b362f92e3438f28153a7c34e8c24ec9a87d5e5`로 고정한다.

## 핵심 결론

- A2는 네 마스크 모두에서 숨긴 silver `CALLS`를 후보로 포함하고 채택하여
  masked recall 1.0을 기록했다. 다만 한 incident, 한 seed, parent ID만 제거한
  synthetic mask이며 causal gold가 아니다. `[E-V2-SUMMARY]`
- A3/A4는 A2의 **후보 집합 `P2`**는 정확히 이어받았지만 A2의 채택 신뢰는
  이어받지 않았다. DeBERTa hard gate가 A4에서 모든 후보를 제거하여 네
  조건 모두 masked recall 0.0이 됐다. `[E-V2-PRED]`
- A4의 복합 runtime context는 A3 대비 target relation을 한 건도 구제하지
  못했다. 현재 데이터에는 Application/Instance/Host/Deployment 관계가 없어
  “상위 ontology node 역할 부족이 원인”이라는 가설을 직접 검증할 수 없다.
  `[E-V2-ACT]`
- A5는 모든 조건에서 A4-accepted `CALLS`가 0개여서 실제 PSL grounding을
  실행하지 않은 no-op이었다. 따라서 v2에서 PSL 효과를 주장할 수 없다.
  `[E-V2-ACT]`
- 가장 강한 다음 설계는 A2를 prior/proposal layer로 보존하고, DeBERTa를
  hard veto가 아닌 `corroborates / contradicts / ambiguous` 보조 증거로
  사용하는 것이다. 이 설계는 v3로 사전 등록해야 하며 v2 임계값을 결과에
  맞춰 사후 조정하면 안 된다.

## 실행 근거

| Evidence ID | 내용 |
|---|---|
| `E-V2-SUMMARY` | `outputs/rcaeval_smoke_v2/full_real_offline_20260828_v4/summary.json` |
| `E-V2-PRED` | 각 mask의 `predictions/A2.parquet` ~ `A5.parquet` |
| `E-V2-ACT` | 각 mask의 `evaluator_private/stage_activation.json` 및 `stage_gate.json` |
| `E-V2-MASK` | 각 mask의 evaluator-private `mask_manifest.json` |
| `E-V2-CODE` | v2 config, cumulative runner, L2 masking, balanced runtime context 구현 |
| `E-V2-COMPACT` | 추적되는 `reports/rcaeval_smoke_results_v2.json`; summary 및 24개 source artifact SHA-256 포함 |

최종 실행의 `config_sha256`은
`e5faa9692997dc224125937e375106c80832e2de7fc6ec1e12105fc3bf8396f6`,
구현 fingerprint는
`8837d54b0fccb12f4486d72226f8f9a233c47d964c13d6516f4bddb2c2e06d0a`,
summary SHA-256은
`f99c63740d1d871574a4f37c97fd5b80e7f15f91bb6de9b6c0d9f9bc1876d63a`다.
Raw output은 Git에 넣지 않으며 compact JSON의 집계와 artifact hash로 재생성
결과를 대조한다. `[E-V2-COMPACT]`

## 결과표

`U`는 typed 미관측 후보 전체, `P2`는 A2가 제안하여 A3~A5가 공유한 후보다.
`P-LB`는 불완전한 silver graph 밖의 예측을 false가 아니라 unverified로
분모에 둔 보수적 하한이다.

| Mask | 숨긴 edge | 지운 boundary span | U | P2 | 후보 압축 | A2 accepted / recall / P-LB | A3 accepted / recall | A4 accepted / recall | A5 accepted | Gate |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---|
| IID20 L2 | 10 | 9,408 | 657 | 13 | 98.02% | 13 / **1.000** / 0.769 | 0 / 0.000 | 0 / 0.000 | 0 | FAIL |
| IID40 L2 | 21 | 21,906 | 668 | 26 | 96.11% | 26 / **1.000** / 0.808 | 0 / 0.000 | 0 / 0.000 | 0 | FAIL |
| IID60 L2 | 31 | 60,294 | 678 | 48 | 92.92% | 48 / **1.000** / 0.646 | 2 / 0.000 | 0 / 0.000 | 0 | FAIL |
| Component L2 | 10 | 568 | 657 | 10 | 98.48% | 10 / **1.000** / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 | FAIL |

IID60 A3의 2개 채택은 모두 silver 미일치 unverified였고 masked target은
한 건도 복원하지 못했다. A4는 두 건을 제거했지만 target recall은 여전히
0이었다. `[E-V2-SUMMARY]`

## DeBERTa 관찰

| Mask | A3 mean forward | A3 mean reverse | A3 mean margin | A4 mean forward | A4 mean reverse | A4 mean margin | A4 margin 개선/악화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID20 | 0.0520 | 0.0763 | -0.0243 | 0.0715 | 0.0701 | 0.0015 | 8 / 5 |
| IID40 | 0.0615 | 0.0721 | -0.0106 | 0.0515 | 0.0520 | -0.0005 | 14 / 12 |
| IID60 | 0.1140 | 0.0858 | 0.0282 | 0.0652 | 0.0701 | -0.0049 | 23 / 25 |
| Component | 0.0579 | 0.0759 | -0.0180 | 0.0439 | 0.0556 | -0.0118 | 4 / 6 |

A4는 IID20/40/component에서 평균 margin을 소폭 높였지만 decision rescue는
0건이었다. IID60에서는 평균 margin이 0.0331 감소하고 A3 채택 2건도
제거됐다. 따라서 현재 결과는 role context의 유효성을 지지하지 않는다.

고정 diagnostic에서도 generic direction contrast가 실패했다.

- forward entailment: 0.916105
- reverse entailment: 0.948678
- margin: -0.032573, 요구치 0.05 미만
- INT8 batch-composition 최대 변화: 0.040110, 허용치 0.000001 초과

연구 실행은 이 민감도를 피하기 위해 `batch_size=1`을 사용했다.
`[E-V2-SUMMARY]`

forward NLI argmax를 mask-condition 단위로 집계하면 A3는 neutral 93건,
entailment 4건이었고, A4는 neutral 92건, contradiction 5건이었다. A4의
contradiction 5건은 모두 evaluator-private masked silver target이었다.
조건 간 후보 중복이 있어 독립 표본 수는 아니지만, 이 결과만으로도
`low entailment`나 `contradiction argmax`를 관계 부재의 hard veto로 쓰면
안 된다는 점은 분명하다. `[E-V2-PRED] [E-V2-MASK]`

실험 prompt도 별도 검증이 필요하다. 현재 premise는 temporal-containment를
“candidate evidence, not a confirmed dependency”로 명시하므로, 사실 명제를
판정하도록 학습된 NLI 모델에서 낮은 entailment가 나올 수 있다. 이는 코드상
확인된 confound이지만 원인으로 확정하려면 사전 등록된 prompt-robustness
실험이 필요하다. `[E-V2-CODE]`

## 1,272개의 정확한 의미

1,272는 마스킹한 관계 수도, 복원 정답 수도 아니다.

| 항목 | 수 | 의미 |
|---|---:|---|
| IID20 실제 masked edge | 10 | 숨긴 silver relation |
| IID20 redacted boundary span | 9,408 | 위 관계의 span occurrence |
| v1 legacy all-pairs 후보 | 1,595 | 41 Service의 미관측 ordered pair |
| trace co-occurrence 0 후보 | 1,272 | 같은 trace에서 함께 관측되지 않은 pair |
| 그중 legacy D0 accepted | 1,272 | 전부 잘못 채택 |
| trace co-occurrence 양수 후보 | 323 | 같은 trace에서 관측된 pair |
| 그중 legacy D0 accepted | 0 | 전부 거절 |
| v2 IID20 P2 | 13 | 실제 A3/A4 검증 대상 |

따라서 v1의 1,272는 대량 관계복원이 아니라 legacy premise/score의
all-pairs artifact다. v2에서는 `D0_LEGACY` control로만 취급한다.

## A2를 “이어받는다”의 두 의미

| 구분 | v2 구현 | 결과 |
|---|---|---|
| 후보 상속 | `A3.candidates = A4.candidates = A5.candidates = P2` | 성공, 불변조건 테스트 통과 |
| 신뢰/채택 상속 | A3/A4 hard gate 통과 시에만 채택 | 실패, A4에서 전체 소실 |

사용자 가설대로 운영 pipeline에서는 두 번째 상속도 필요하다. 다만 A2를
무조건 최종 확정하면 A2의 unverified edge도 그대로 남는다. 다음 실험은
A2 prior를 보존하되 DeBERTa가 명백한 반대 방향을 보일 때만 감점하고,
ambiguous한 경우에는 “미확정이지만 유지”하도록 상태를 분리해야 한다.

## PSL 판단

PSL runtime 2.4.0과 실제 grounding/inference 구현은 별도 통합 테스트에서
동작했다. 그러나 본 v2 run에서는 A4 eligible이 0이므로 조건마다 다음과
같이 기록됐다.

- `stage_activated=false`
- `no_op_reason=NO_ELIGIBLE_A4_RELATIONS`
- `grounded_rule_count=0`
- `grounded_atom_count=0`

또한 현재 두 PSL 규칙은 edge 간 결합이 없는 독립 shrinkage다. 대략
`Calls ≈ 10/11 × Evidence`이므로, 활성화되더라도 “논리적 증명”이나
global joint reasoning이 아니라 고정 pruning/calibration baseline이다.

## 누수·재현성 점검

- model/reference trace: 3,936 / 2,539, overlap 0
- split formula: `sha256(revision|incident_id|trace_id) mod 100`
- DeBERTa ONNX SHA-256:
  `03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae`
- local-only model, ONNX telemetry disabled, batch size 1
- A3 최대 93 tokens, A4 최대 446 tokens, truncation 0
- 세 가지 실행 점검은 네 mask 모두 통과: sanitized trace vocabulary와
  entity 일치, fault/injection time 금지, pre-mask artifact 부재
- evaluator와 model stage는 API·시간 순서로 분리됐지만 같은 Python
  process를 사용하므로 process isolation은 아니다.
- IID20은 IID40의 부분집합이고 IID40은 IID60의 부분집합인 exact nested
  mask다. 한 incident·한 seed의 중첩 조건이므로 네 mask를 독립 반복으로
  평균 내거나 신뢰구간을 추정할 수 없다(`CI=NOT_ESTIMABLE`).

## 허용되는 주장

- L2 parent-drop 조건에서 A2가 657~678개 후보를 10~48개로 줄이면서 이
  incident/seed의 masked silver edge를 모두 포함했다.
- off-the-shelf DeBERTa hard gate는 A2 성능을 보존하지 못했다.
- 현재 복합 runtime context는 DeBERTa decision을 유의미하게 구제하지
  못했다.
- A2 prior를 보존하는 누적 추론 설계가 다음 연구 가설이다.

## 금지되는 주장

- 상위 ontology 역할 문맥이 DeBERTa를 개선했다.
- PSL이 관계를 증명하거나 본 run에서 성능을 개선했다.
- A2가 일반 CMDB 누락이나 자연 발생 collector 누락을 해결했다.
- v2 graph가 LLM RCA 원인/영향 경로 성능을 높였다.
- `ts-auth-service` 장애의 실제 root cause가 확인됐다. 원인 증거와
  supported impact path가 없으므로 **확인 필요**다.

## 다음 사전등록 실험

1. A2 prior-preserving verifier를 신규 v3로 고정한다.
2. DeBERTa 출력은 `corroborates`, `contradicts`, `ambiguous`로 분리하고
   ambiguous를 자동 거절하지 않는다.
3. `A4-H/R/N/O/F` factorial과 anonymized service-label control로 hierarchy,
   role, neighbor, operation 효과를 분리한다.
4. A4와 A5 사이에 PSL과 동일 effective threshold의 비-PSL control을 둔다.
5. 동일 `U`에서 global ranking, `P2`에서 verifier ranking을 따로 보고한다.
6. `travel-service`, `order-service` 대표 incident와 여러 seed로 반복한 뒤
   ontology graph 유무에 따른 LLM RCA path 성능을 평가한다.
