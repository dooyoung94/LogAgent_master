# LogAgent 연구 개요 — 불완전한 운영 토폴로지에서 귀추 기반 관계·원인 가설 복원과 LLM RCA

## 0. 한 문장 연구 목표

> **불완전하게 수집된 Web Topology와 Logs/Metrics/Traces로 운영 Ontology/KG를 구성하고, 관측되지 않은 관계와 알려지지 않은 장애 원인 가설을 귀추추론(Abduction)으로 보완한 뒤, 복원된 그래프와 RCA 근거를 LLM에 제공했을 때 Root Cause·Cause Path·Impact Path 분석이 실제로 개선되는지를 검증한다.**

이 연구의 핵심은 "로그에서 온톨로지 스키마를 자동 생성한다"가 아니다. 검토된 TBox/스키마는 고정하고, 실제 운영에서 불완전하게 관측되는 **ABox/Operational KG의 관계와 RCA 가설을 복원**하는 문제를 다룬다.

---

## 1. 실제 프로젝트에서 출발한 문제

현재 Log RCA 프로젝트는 다음 운영 흐름을 목표로 한다.

```text
Web Topology 수집
    web.csv
      ↓
Log/APM 수집
    Transaction / Trace / Profile / Metric / Error
      ↓
Operational Ontology / Knowledge Graph
    WebPage → API → Service → Instance/Pod → Server/Node
                         ↓
                     Database
                         ↓
                 CPU / Memory / Heap ...
      ↓
RCA
    장애 현상 → 원인 후보 → 원인 경로 → 영향 경로 → 근거
      ↓
LLM
    사람이 이해할 수 있는 원인·영향·조치 설명
```

그러나 실제 운영환경에서는 이 그래프가 완전하게 만들어진다는 보장이 없다.

- 웹 크롤러가 모든 동적 API 호출을 관측하지 못할 수 있다.
- APM/Trace가 샘플링되거나 특정 서비스가 계측 대상이 아닐 수 있다.
- 신규 서비스·외부 솔루션·Black-box 컴포넌트는 호출 관계가 보이지 않을 수 있다.
- 서비스명, Pod명, 애플리케이션명, DB 연결명 등 서로 다른 수집원의 식별자가 일치하지 않을 수 있다.
- 로그 보존시간, Collector 장애, 시간 동기화 문제로 특정 구간의 증거가 사라질 수 있다.
- 구조적 의존관계(`CALLS`, `USES_DATASOURCE`)가 존재한다고 해서 그 관계가 곧 장애의 인과관계(`CAUSES`)인 것은 아니다.

즉 실제 RCA는 **불완전한 Topology + 불완전한 Telemetry + 알려지지 않은 장애 원인**을 동시에 다뤄야 한다.

---

## 2. 기존 Log/Microservice RCA가 어려운 이유

### 2.1 Topology를 알고 있다는 가정

많은 Microservice RCA 방법은 Service Call Graph, Dependency Graph 또는 Trace-derived graph가 충분히 관측된다고 가정한다. 그러나 실제 시스템에는 trace blind spot, unsupported agent, black-box service가 존재한다.

TORAI(2026)는 이 문제를 직접 지적하고, call graph가 없어도 multi-source telemetry만으로 RCA를 수행하는 graph-free 접근을 제안한다. 이는 blind spot이 실제 RCA의 중요한 문제임을 보여주지만, **누락된 운영 관계 자체를 복원하지는 않는다.**

### 2.2 Telemetry가 완전하지 않음

RCA benchmark에 대한 최근 분석에서는 기존 공개 데이터가 현실의 관측 조건을 충분히 반영하지 못하는 문제가 지적되었다. RCABench/Aegis 계열 연구는 observability blind spot, 얕은 call graph, 정적인 workload가 기존 벤치마크의 핵심 한계라고 분석한다.

### 2.3 알려진 원인만으로는 현실 장애를 모두 설명하기 어려움

Threshold, 고정 Rule, 사전 정의 Cause Taxonomy만으로 RCA를 구성하면 이미 알고 있는 장애 유형에는 강하지만 새로운 조합의 장애나 미등록 의존관계에 취약하다.

실제 운영에서는 다음과 같은 형태가 필요하다.

```text
관측 결과 + 운영 지식/규칙
        ↓
"이 결과를 설명하려면 어떤 원인 또는 누락 관계가 존재해야 하는가?"
        ↓
가능한 가설 생성
```

따라서 본 연구는 연역적으로 "규칙에 맞는 알려진 원인을 선택"하는 것만이 아니라, **관측 결과를 가장 잘 설명하는 원인/관계 가설을 귀추적으로 생성**하는 방향을 사용한다.

### 2.4 잘못 복원한 관계는 오히려 RCA를 악화시킬 수 있음

관계가 안 보인다는 이유만으로 무조건 관계를 추가하면 false edge가 늘어나고, 잘못된 경로가 LLM에 전달될 수 있다. 따라서 연구 목표는 단순한 graph completion이 아니라 다음을 동시에 만족해야 한다.

- 필요한 누락 관계는 높은 Recall로 후보에 포함한다.
- 근거가 약한 관계를 과도하게 확정하지 않는다.
- 구조/시간/타입 제약과 모순되는 관계를 억제한다.
- 불충분한 증거에서는 `unresolved/abstain`을 허용한다.
- 최종적으로 추가된 관계가 실제 RCA 성능을 높이는지 별도로 검증한다.

---

## 3. 왜 Ontology / Knowledge Graph를 사용하는가

Ontology는 불완전한 데이터를 자동으로 정답으로 바꾸는 도구가 아니다. 본 연구에서 Ontology의 역할은 **추론 가능한 운영 의미 공간과 제약조건을 제공하는 것**이다.

예시 TBox/관계 vocabulary:

| Subject | Relation | Object | 의미 |
|---|---|---|---|
| WebPage | `ROUTES_TO` | APIEndpoint | 사용자 화면과 API 연결 |
| Application/Service | `EXPOSES` | APIEndpoint | API 제공 주체 |
| Service/Instance | `CALLS` | Service/Instance | Runtime 호출 관계 |
| Instance | `INSTANCE_OF` | Application/Service | 실행 인스턴스 소속 |
| Instance/Pod | `LOCATED_ON` | Host/Node | 물리/논리 실행 위치 |
| Service/Application | `USES_DATASOURCE` | DataSource/Database | DB 의존성 |
| Transaction | `EXECUTES` | SQLPattern | 실행 SQL 관계 |
| Entity | `HAS_METRIC` | Metric | CPU/Memory/Heap 등 관측 |

중요한 원칙:

- `CALLS(A,B)`를 자동으로 `CAUSES(A,B)`로 간주하지 않는다.
- `structural`, `runtime`, `causal_hypothesis` 관계 레이어를 구분한다.
- 모든 inferred edge는 `evidence_ids`, `source`, `time`, `confidence`, `status`를 보존한다.
- 근거가 부족하면 `confirmed`가 아니라 `inferred` 또는 `unresolved` 상태로 남긴다.

---

## 4. 기존 연구와 본 연구의 위치

| 연구 | 해결하는 문제 | 강점 | 본 연구와의 차이 |
|---|---|---|---|
| **OntoLogX (2025)** | Raw Log → Ontology-grounded KG | 비정형 로그를 구조화된 KG로 변환 | 누락 운영 Topology 복원과 RCA 개선이 주목적은 아님 |
| **RCAEval (2024)** | Microservice RCA benchmark | Logs/Metrics/Traces, 735 failure cases, 다양한 baseline | 평가 인프라이며 누락 관계 복원 방법 자체는 아님 |
| **Causal RCA 계열** | 관측 데이터에서 causal/root ranking | 원인 후보 순위화 | 입력 graph/telemetry 품질에 민감하고 모든 상황에서 우월한 방법이 없음 |
| **TraceDiag (2023)** | 대규모 trace graph 기반 RCA | Graph pruning + causal RCA | trace/service dependency graph를 활용하며 blind spot 관계 복원이 핵심은 아님 |
| **TORAI (2026)** | Call graph blind spot 상황의 RCA | graph 없이 multi-source telemetry로 fine-grained RCA | graph를 복원하지 않음 |
| **RCABench/Aegis** | 현실적인 RCA benchmark 생성·평가 | TrainTicket, 깊은 call path, realistic telemetry/fault | 본 연구의 주 데이터/평가 기반으로 사용 |
| **본 연구** | **불완전 Operational KG + open-world RCA** | **귀추 가설 → 증거 검증 → soft logic → RCA → LLM intervention** | **복원된 관계가 실제 RCA/LLM을 개선하는지까지 인과적으로 비교** |

### 연구 Gap

기존 연구를 연결하면 다음 빈 구간이 존재한다.

```text
Log → KG            : 기존 연구 존재
Telemetry → RCA     : 기존 연구 다수
Blind spot → RCA    : TORAI 등 존재

그러나

불완전 KG
   ↓
누락 관계 + 원인 가설 생성
   ↓
증거 기반 타당성 검증/축소
   ↓
복원 KG 기반 RCA
   ↓
동일 LLM의 RCA 성능 개선 검증

을 하나의 실험 프로토콜로 검증하는 것은 별도의 연구 문제다.
```

본 연구의 novelty는 DeBERTa나 PSL이라는 개별 모델 자체가 아니라, **현실적인 missingness 문제 정의, 귀추 기반 최소 가설 생성, open-world 불확실성 제어, 그리고 graph intervention을 통한 downstream RCA 개선 검증**에 둔다.

---

## 5. 제안 방법론

### 5.1 단계 1 — Topology + Telemetry로 Observed KG 생성

입력:

```text
Topology
- web.csv
- source/deployment/Kubernetes metadata

Telemetry
- logs
- metrics
- traces / transactions / profiles
- alerts/errors
```

관측 그래프:

```text
G_obs = φ(X ; Ontology)
```

예시:

```text
WebPage
  └─ROUTES_TO→ APIEndpoint
                    └─EXPOSED_BY→ Service
                                      ├─INSTANCE→ Pod
                                      ├─CALLS→ OtherService
                                      └─USES_DATASOURCE→ Database

Pod
  ├─LOCATED_ON→ Node
  ├─HAS_METRIC→ CPU
  └─HAS_METRIC→ Memory
```

### 5.2 단계 2 — 실제 운영 Blind Spot 재현

단순 Random edge mask만으로 성능을 주장하지 않는다. 주요 실험은 실제 운영에서 발생할 수 있는 누락을 재현한다.

| Mask | 실제 상황 |
|---|---|
| IID relation 20/40/60% | 기존 연구와 비교하기 위한 control |
| Component blackout | 특정 신규/미계측 서비스 전체 관계 누락 |
| Relation block | 특정 collector/source 전체 누락 |
| Trace dropout | sampling/export 장애 |
| Identity-link dropout | APM/CMDB/Web 간 이름 매핑 실패 |
| Temporal-window dropout | 보존/collector 장애로 특정 시간대 증거 소실 |
| Path-critical mask | Root→Symptom 경로의 bridge relation 누락 |

### 5.3 단계 3 — 귀추 기반 관계·원인 가설 생성

관측 결과 `O`, 배경지식/규칙 `B`, 후보 가설 `H`에 대해 다음 조건을 만족하는 최소 가설을 찾는다.

```text
B ∪ H explains O
B ∪ H is constraint-consistent
no proper subset of H explains the same observations
```

개념적으로:

```text
H* = argmax_H P(H | O, B)
```

예:

```text
관측:
- payment-api latency 급증
- payment-db wait 증가
- 직접 Service→DB 관계는 관측되지 않음

운영 규칙:
DBWait(D) ∧ UsesDataSource(A,D) → AppLatency(A)

귀추 가설:
UsesDataSource(payment-api, payment-db) ?
```

귀추 단계의 목적은 정답을 즉시 확정하는 것이 아니라 **전체 typed candidate universe를 설명력 있는 작은 원인/관계 후보 집합으로 압축하는 것**이다.

### 5.4 단계 4 — 원인 가설의 타당성 검증

가설마다 서로 다른 증거를 결합한다.

```text
S(H) =
  α · S_abduction
+ β · S_telemetry
+ γ · S_text/NLI
+ δ · S_graph_constraint
+ ε · S_temporal
```

- `S_abduction`: 관측 현상을 얼마나 잘 설명하는가
- `S_telemetry`: trace/log/metric의 직접·간접 증거
- `S_text/NLI`: 로그 및 evidence bundle이 가설을 corroborate/contradict하는 정도
- `S_graph_constraint`: Ontology domain/range, cardinality, path consistency
- `S_temporal`: 원인 후보가 현상보다 시간적으로 앞서거나 일관적인가

### DeBERTa/NLI 사용 원칙

DeBERTa는 **관계의 최종 증명기 또는 hard veto가 아니라 보조 증거 scorer**로 사용한다.

현재 smoke v2에서는 Abduction A2가 masked target을 후보로 모두 포함했지만 off-the-shelf DeBERTa hard gate가 해당 후보를 보존하지 못했다. 따라서 후속 설계에서는:

```text
Abduction prior 유지
    +
NLI = corroborates / contradicts / ambiguous
    +
Ontology/Telemetry evidence
```

처럼 누적 증거 방식으로 설계한다.

### 5.5 단계 5 — PSL / Soft Logic Joint Inference

가설별 local score만 보는 것이 아니라 서로 연결된 운영 규칙을 soft constraint로 함께 계산한다.

예:

```text
TxHitsEndpoint(T,E)
& EndpointOf(E,A)
& TxUsesDatabase(T,D)
    -> UsesDataSource(A,D)

ParentSpan(P,C)
& ExecutesOn(P,I)
& ExecutesOn(C,J)
    -> CallsInstance(I,J)

CallsInstance(I,J)
& InstanceOf(I,A)
& InstanceOf(J,B)
    -> CallsApplication(A,B)
```

최종 관계는 단순 true/false가 아니라:

```text
confirmed / inferred / unresolved
confidence ∈ [0,1]
```

로 관리하며, 증거가 충분하지 않으면 abstain한다.

### 5.6 단계 6 — RCA 생성

복원된 Operational KG와 장애 Evidence를 이용해 다음을 생성한다.

```json
{
  "root_candidates": [],
  "cause_path": [],
  "impact_path": [],
  "supporting_evidence": [],
  "relation_hypotheses": [],
  "confidence": 0.0,
  "abstain": false
}
```

RCA의 핵심은 단순 Root service 하나가 아니라:

```text
장애 현상
   ↓
원인 후보
   ↓
Root Cause
   ↓
Cause Path
   ↓
Impact Path
   ↓
Evidence
```

를 구조적으로 제공하는 것이다.

### 5.7 단계 7 — LLM에 구조화된 RCA 전달

LLM이 원시 로그 전체에서 무제한으로 원인을 추측하게 하지 않는다.

```text
Raw Telemetry
+ Observed / Recovered KG
+ RCA Candidate/Path
+ Evidence ID
+ Confidence
      ↓
Fixed LLM
      ↓
Root / Cause Path / Impact / Explanation / Action
```

최종 연구 질문은 다음이다.

```text
ΔRCA = Metric(LLM(X, G_recovered))
     - Metric(LLM(X, G_observed))
```

즉 **관계 복원이 좋은가**와 **그 관계가 실제 RCA를 좋게 만드는가**를 분리해 검증한다.

---

## 6. 데이터 전략

단일 공개 데이터셋이 Web topology, logs, metrics, traces, 완전한 topology gold, root cause, cause/impact path를 모두 제공하지 않는다. 따라서 데이터셋마다 역할을 분리한다.

### 6.1 Primary — RCABench/Aegis TrainTicket

주 실험 데이터로 사용한다.

주요 관측 데이터:

| Ontology 대상 | RCABench/Aegis 데이터 예 |
|---|---|
| Transaction | `trace_id` |
| Span | `span_id`, `parent_span_id` |
| APIEndpoint | `span_name`, HTTP method/status |
| Service | `service_name`, `attr.k8s.service.name` |
| Instance/Pod | `attr.k8s.pod.name` |
| Node/Server | `attr.k8s.node.name` 등 metric attributes |
| Runtime CALLS | parent/child span 및 workload source/destination |
| Error | trace status, HTTP status, log level/message |
| CPU/Memory | container/Kubernetes metrics |
| Root/Fault GT | fault injection / ground-truth metadata |

역할:

- Logs/Metrics/Traces를 이용한 `G_obs` 생성
- Component/Trace/Relation/Path blind spot 실험
- 실제 fault injection을 이용한 Root RCA 검증
- TrainTicket source/Kubernetes 구조를 모델 입력에서 숨긴 independent reference topology로 활용

### 6.2 RCAEval RE2/RE3

- 현재 executable smoke 및 빠른 반복 실험
- 735 failure cases의 standardized RCA benchmark
- Root service/root indicator/fault label 기반 비교
- 관계복원 알고리즘 개발과 재현성 확인

### 6.3 DejaVu

- 명시적인 fault dependency graph를 이용한 topology masking control
- text/log relation extraction의 주 데이터가 아니라 **정확한 구조 관계 복원 sanity check**로 사용

### 6.4 OntoLogX

- Raw Log → Ontology/Event KG front-end 추출 baseline
- 로그의 entity/relation extraction 성능 검증용
- Operational topology completion이나 RCA benchmark로 과대해석하지 않음

### 6.5 External validation

- Nezha / LO2v2: topology evidence와 cross-source robustness
- OpenRCA: downstream LLM RCA 외부 검증 후보
- TORAI: graph-free blind-spot RCA baseline

---

## 7. 실제 프로젝트와 연구 데이터 대응

| 운영 LogAgent | 연구 Benchmark |
|---|---|
| `web.csv` Topology | TrainTicket UI/source/deployment + telemetry-derived route |
| Jennifer Domain/Instance | K8s Service / Pod / Container |
| Jennifer Transaction | Trace `trace_id` |
| Jennifer Profile | Span / operation |
| Jennifer PERF | Metrics |
| `log.file` | Application Logs |
| AGENT_ID / Instance ID | K8s Pod/Service identity |
| Endpoint | span operation + HTTP attributes |
| CPU/Memory/Heap | Resource/JVM metrics where available |
| Trigger | fault injection timestamp |
| Neo4j Semantic Graph | Normalized Operational KG |
| RCA report | Root + cause path + impact path + evidence |
| LLM report | Fixed LLM downstream evaluation |

따라서 연구용 pipeline과 실제 프로젝트 pipeline의 구조를 동일하게 유지할 수 있다.

```text
[운영]
web.csv + Jennifer/log.file
       ↓
Operational KG
       ↓
Relation/Root Hypothesis
       ↓
RCA
       ↓
LLM

[연구]
TrainTicket topology + RCABench telemetry
       ↓
Operational KG
       ↓
Masked relation + Abductive hypothesis
       ↓
RCA
       ↓
Fixed LLM evaluation
```

---

## 8. 실험 비교군

### Task A — Relation / Hypothesis Recovery

| ID | 입력/방법 | 목적 |
|---|---|---|
| A0 | Observed graph only | 복원 없음 |
| A1 | Deterministic trace/time rule | 단순 Rule baseline |
| A2 | Abduction | 귀추 후보 생성 효과 |
| A3 | NLI/text scorer over same candidates | text evidence 단독 효과 |
| A4 | **Abductive prior + NLI corroboration + ontology/runtime evidence** | 제안 누적 검증 |
| A5 | **A4 + PSL + calibration/abstention** | 최종 제안 방식 |
| KGC | TransE/DistMult/graph model 등 | generic KG completion 비교 |
| LLM-KG | ontology-prompted LLM extractor | LLM relation extraction 비교 |
| AO | Oracle reference graph | 상한선 |
| AW | Proposed graph + controlled false edges | 잘못된 복원 위험 측정 |

### Task B — Fixed LLM RCA

LLM model, prompt, temperature, token budget, telemetry window를 동일하게 고정한다.

| ID | LLM 입력 |
|---|---|
| B0 | Raw telemetry only |
| B1 | Raw telemetry + Observed KG |
| B2 | Raw telemetry + strongest baseline-completed KG |
| B3 | Raw telemetry + Proposed recovered KG + RCA evidence |
| BO | Raw telemetry + Oracle graph |
| BW | Raw telemetry + 동일 개수의 shuffled/false edges |
| BG | Graph-free blind-spot baseline (예: TORAI) |

이 비교를 통해 단순히 "graph token을 더 줬기 때문에 좋아졌다"는 설명을 배제한다.

---

## 9. 검증 지표

### 9.1 관계 복원 지표

| 지표 | 목적 |
|---|---|
| Candidate Recall@K | 귀추 단계가 정답 관계를 후보로 포함하는가 |
| Relation Precision / Recall / Macro-F1 | 타입별 복원 정확도 |
| AUPRC | class imbalance 환경의 관계 복원 품질 |
| MRR / Hits@1/3/10 | 정답 관계 ranking |
| False-edge rate | 잘못 추가한 관계 비율 |
| Brier / ECE | confidence calibration |
| Risk-Coverage | abstention했을 때 위험 감소 |
| Constraint violation | Ontology 규칙 위반 여부 |
| Root→Symptom Reachability | 복원으로 단절된 RCA 경로가 연결되는가 |
| Path Edge F1 | 정답 경로의 relation을 얼마나 복원하는가 |
| Shortest-path distortion | 잘못된 shortcut/우회 경로 생성 여부 |

### 9.2 RCA/LLM 지표

| 지표 | 목적 |
|---|---|
| Root Top-1 / Top-3 / Top-5 | Root Cause 적중 |
| Root MRR | Root ranking 품질 |
| Cause-path Edge F1 | 원인 전파 경로 정확도 |
| Cause-path Exact Match | 전체 경로 완전 일치 |
| Impact-component F1 | 영향 서비스/컴포넌트 정확도 |
| Evidence citation precision | 실제 근거와 연결된 설명인가 |
| Unsupported-claim rate | 근거 없는 LLM 주장 비율 |
| Abstention accuracy | 판단 불가능한 경우 올바르게 보류하는가 |
| Latency / Memory / Tokens / Cost | 실제 운영 가능성 |

### 9.3 통계 검증

- 최소 5개 masking seed
- 동일 incident에 대한 paired comparison
- incident-level paired bootstrap confidence interval
- correctness: McNemar/permutation test
- continuous metric: paired bootstrap 또는 Wilcoxon
- multiple comparison: Holm correction
- IID random mask와 structured blind spot 결과를 분리 보고
- 동일 campaign/topology는 같은 split에 배치하여 leakage 방지

---

## 10. 핵심 연구가설

### H1 — 관계 복원

> 현실적인 observability blind spot에서 귀추 prior + multi-source evidence + soft logic 기반 방법은 Rule, NLI-only, generic KG completion보다 누락 관계를 더 정확하고 calibration된 형태로 복원한다.

### H2 — RCA 개선

> 복원된 관계를 포함한 Operational KG는 불완전한 Observed KG보다 동일 LLM의 Root Cause, Cause Path, Impact Path 성능을 향상시킨다.

### H3 — 안전성

> Calibration과 abstention을 적용한 복원은 무조건적인 graph completion보다 false edge로 인한 RCA 성능 저하를 줄인다.

---

## 11. 반드시 통과해야 하는 연구 Gate

| Gate | 질문 | 실패 시 |
|---|---|---|
| D0 Data | 데이터 provenance/license/schema/GT가 명확한가 | 실험 중단 |
| D1 Leakage | 숨긴 관계/정답 alias가 입력에 남아 있지 않은가 | masking 수정 |
| D2 Candidate | 귀추 후보가 정답을 충분히 포함하는가 | candidate 생성부터 개선 |
| D3 Activation | 각 ablation이 실제 prediction을 변화시키는가 | inactive component로 판정 |
| D4 Oracle Utility | 완전한 topology가 실제 RCA를 개선하는가 | topology 복원 연구 주장 재검토 |
| D5 Completion | 제안 방식이 strongest completion baseline보다 좋은가 | H1 기각 |
| D6 RCA | 복원 KG가 fixed LLM RCA를 개선하는가 | H2/H3 기각 |
| D7 Generalization | structured blind spot/held-out system에서도 유지되는가 | 주장 범위 제한 |

특히 **D4 Oracle Utility가 중요하다.** 완전한 topology를 넣어도 LLM/RCA가 개선되지 않는다면, 관계 복원 정확도를 높여도 본 연구의 최종 목적을 증명할 수 없다.

---

## 12. 현재 진행 위치

현재 RCAEval smoke에서는 다음까지 구현/측정했다.

```text
RCAEval telemetry
      ↓
Reference / Model trace split
      ↓
Held-out Silver CALLS graph
      ↓
Blind-spot masking
      ↓
A2 Abductive candidate generation
      ↓
DeBERTa directional/context verification
      ↓
PSL integration
```

예비 결과에서는 A2가 한 incident/한 seed의 IID20/40/60 및 component masking에서 masked target을 모두 candidate/accepted set에 포함했지만, off-the-shelf DeBERTa hard gate가 이를 보존하지 못했다. A5는 A4에서 eligible relation이 사라져 해당 run에서는 실질적인 PSL 효과를 검증하지 못했다.

따라서 다음 단계는 결과에 맞춰 threshold를 사후 조정하는 것이 아니라, 새로운 실험 버전에서 다음 누적 구조를 사전 정의하는 것이다.

```text
Abduction = proposal/prior
      ↓
Telemetry + NLI = corroborate / contradict / ambiguous
      ↓
Ontology + PSL = joint soft inference
      ↓
Calibration / Abstention
      ↓
Recovered KG
      ↓
RCA
      ↓
Fixed LLM B0~B3 / Oracle comparison
```

이후 RCAEval에서 algorithm contract를 안정화한 뒤 RCABench/Aegis TrainTicket로 확장한다.

---

## 13. 논문에서 주장해야 할 것 / 주장하지 말아야 할 것

### 주장 목표

- 실제와 유사한 topology/telemetry blind spot을 명시적으로 모델링했다.
- 귀추는 누락 관계와 원인 가설의 작은 candidate set을 생성한다.
- 다양한 관측 증거와 Ontology constraint를 이용해 후보의 타당성을 평가한다.
- 불충분한 증거에는 abstention한다.
- 복원된 관계가 단순 edge score뿐 아니라 **실제 downstream RCA/LLM 성능을 개선하는지 intervention으로 검증**한다.

### 주장하지 않음

- 로그가 자동으로 완전한 TBox/Ontology를 발명한다.
- 구조적 dependency가 곧 causal relation이다.
- DeBERTa가 관계를 증명한다.
- PSL이 형식논리적 proof를 제공한다.
- 관측되지 않은 edge는 false이다.
- graph context가 많을수록 LLM이 항상 좋아진다.

---

## 14. 최종 연구 흐름

```text
1. Web Topology 수집
   web.csv / source / deployment
            ↓
2. Log/APM 수집
   logs + metrics + traces
            ↓
3. Operational Ontology/KG
   Web → API → Service → Instance → Node
                    ↓
                  DB / Metric
            ↓
4. 현실적 Blind Spot 발생
   관계/trace/component/identity/time 누락
            ↓
5. 귀추 가설 생성
   결과 + 규칙 → 가능한 관계/원인
            ↓
6. 가설 타당성 평가
   Telemetry + NLI + Ontology + Time
            ↓
7. 후보 축소 / Soft Logic
   PSL + calibration + abstention
            ↓
8. RCA
   Root + Cause Path + Impact Path + Evidence
            ↓
9. LLM
   구조화된 RCA 근거 기반 장애 설명
            ↓
10. 검증
   "Observed KG보다 Recovered KG가 실제 RCA를 개선했는가?"
```

---

## References / Related Work

- Pham et al., **RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data**, 2024. https://arxiv.org/abs/2412.17015
- Cotti et al., **OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models**, 2025. https://arxiv.org/abs/2510.01409
- Pham et al., **TORAI: Unsupervised Fine-grained RCA using Multi-Source Telemetry Data**, 2026. https://arxiv.org/abs/2604.13522
- Pham et al., **Root Cause Analysis for Microservice System based on Causal Inference: How Far Are We?**, 2024. https://arxiv.org/abs/2408.13729
- Ding et al., **TraceDiag: Adaptive, Interpretable, and Efficient Root Cause Analysis on Large-Scale Microservice Systems**, 2023. https://arxiv.org/abs/2310.18740
- AegisLab / RCABench documentation, **Data Formats**. https://operationspai.github.io/AegisLab-doc/algorithm-developers/development-guide/data-formats/
- AegisLab / RCABench, **Rethinking the Evaluation of Microservice RCA with a Fault Propagation-Aware Benchmark**. https://operationspai.github.io/revisiting-rca-evaluation/
- TrainTicket benchmark system. https://github.com/FudanSELab/train-ticket

관련 세부 실험 계약은 `docs/research_protocol.md`, 데이터 준비상태는 `docs/dataset_readiness.md`, 현재 RCAEval smoke 결과는 `reports/rcaeval_smoke_results_v2.md`를 함께 참조한다.
