# LogAgent 연구 개요 v3 — 통합본
## 불완전한 운영 온톨로지에서 귀추 기반 관계·원인 가설 복원과 경로 검증형 LLM RCA

- 문서 상태: `research_overview_ko.md`와 `research_overview_ko2.md`의 통합 기준 문서
- 기준일: 2026-09-02
- 적용 프로젝트: `web.csv + Jennifer/log.file → Operational KG → RCA → LLM`
- 핵심 원칙: **Ontology는 완전한 정답 그래프가 아니라 의미·관계·제약을 제공하는 공간이며, Abduction은 불완전한 지식에서 누락 관계와 원인 가설을 생성하는 전단계이고, RCA는 이를 증거와 경로로 검증하여 조치 가능한 진단으로 만드는 후속 단계다.**

이 문서를 기준 개요로 사용하고, 기존 `research_overview_ko.md`와 `research_overview_ko2.md`는 설계 변화 이력을 확인하는 참고 문서로 유지한다.

---

## 0. 한 문장 연구 목표

> **불완전하게 수집된 Web Topology와 Logs/Metrics/Traces로 Operational Ontology/KG를 구성하고, 관측되지 않은 관계와 알려지지 않은 장애 원인 후보를 귀추추론으로 생성한 뒤, Telemetry·Ontology 제약·NLI·Soft Logic으로 가설을 검증·축소하여 RCA 경로를 만들고, 이 구조화된 근거를 LLM에 제공했을 때 Root Cause·Cause Path·Impact Path·Evidence Grounding 성능이 향상되는지를 검증한다.**

본 연구의 핵심 질문은 다음과 같다.

> **현실의 불완전한 Operational Ontology에서 귀추법으로 누락 관계와 복수의 원인 가설을 생성하고 증거로 검증하면, LLM이 단순히 원인 서비스 이름을 맞히는 수준을 넘어 검증 가능한 원인 전파 경로와 영향 경로를 더 정확하게 설명할 수 있는가?**

본 연구는 다음 네 계층을 명확히 구분한다.

| 계층 | 역할 |
|---|---|
| **TBox / Ontology Schema** | `WebPage`, `APIEndpoint`, `Service`, `Instance`, `Host`, `Database`, `Metric`과 허용 관계 정의 |
| **ABox / Operational KG** | `web.csv`, APM, Log, Metric, Trace에서 실제로 관측된 개체와 관계 적재 |
| **Hypothesis Layer** | 관측되지 않았지만 현상을 설명할 수 있는 누락 관계와 원인 후보 관리 |
| **Causal / RCA Layer** | 검증된 증거를 따라 Root→Propagation→Symptom→Impact 경로 구성 |

핵심은 로그로 TBox를 무제한 생성하는 것이 아니라, **검토된 Ontology 안에서 불완전한 ABox를 보완하고 그 보완이 실제 RCA에 유용한지 검증하는 것**이다.

---

## 1. 실제 LogAgent 프로젝트에서 출발한 문제

현재 프로젝트의 목표 흐름은 다음과 같다.

```text
1. Web Topology 수집
   web.csv / source / deployment metadata
                  ↓
2. Log·APM 수집
   Logs / Metrics / Transaction / Trace / Profile / Error
                  ↓
3. Operational Ontology / Knowledge Graph
   WebPage → APIEndpoint → Service → Instance/Pod → Host/Node
                              ├→ Database / SQL
                              └→ CPU / Memory / Heap / Error
                  ↓
4. 누락 관계·원인 가설 복원
   결과 + 규칙 + 관측 증거 → 가능한 관계/원인
                  ↓
5. RCA
   Root Candidate → Cause Path → Impact Path → Evidence
                  ↓
6. Runbook + LLM
   검증 절차와 근거가 연결된 원인·영향·조치 설명
```

그러나 실제 운영에서는 Topology와 Telemetry가 모두 불완전하다.

- 웹 크롤러가 동적 API, 비동기 호출, 외부 솔루션, Black-box 구간을 모두 관측하지 못한다.
- APM/Trace sampling, unsupported agent, collector 장애, 보존기간으로 호출 관계가 끊긴다.
- `application_id`, `agent_id`, Pod, URL, DB connection name 등 수집원별 식별자가 일치하지 않는다.
- 신규 배포, feature flag, autoscaling, failover로 관계가 시간에 따라 변경된다.
- 특정 서비스가 telemetry를 남기지 않거나, 장애 직전 데이터가 TTL·보존정책으로 사라진다.
- 높은 CPU, API 오류, DB wait처럼 여러 이상이 동시에 발생하여 원인과 증상을 혼동하기 쉽다.
- 동일 현상을 설명하는 복수의 원인 가설이 존재하며 일부는 현재 데이터만으로 식별할 수 없다.
- 구조 관계 `CALLS`, `USES_DATASOURCE`는 장애 인과관계 `CAUSES`와 동일하지 않다.

따라서 현실적인 문제는 단순한 장애 분류가 아니다.

```text
불완전한 Topology
+ 불완전한 Telemetry
+ 복수의 가능한 원인
+ 알려지지 않은 장애 조합
        ↓
검증 가능한 원인 가설과 전파 경로를 어떻게 만들 것인가?
```

---

## 2. Log RCA가 어려운 핵심 이유

### 2.1 Topology가 충분히 관측된다는 가정

많은 Microservice RCA 방법은 Service Call Graph, Dependency Graph 또는 Trace-derived Graph가 충분히 관측된다고 가정한다. 실제 환경에는 trace blind spot, 미지원 agent, 외부 API, black-box service, stale topology가 존재한다.

TORAI는 call graph가 불완전한 상황에서 multi-source telemetry만으로 RCA를 수행하는 graph-free 접근을 제안한다. 이는 blind spot이 실제 RCA의 핵심 문제임을 보여주지만, **누락된 운영 관계 자체를 복원하지는 않는다.**[R12]

### 2.2 Telemetry가 존재해도 원인을 식별할 수 있다는 보장이 없음

- 원인 노드가 장애 시점에 telemetry를 남기지 않을 수 있다.
- 기준선이 이미 포화되어 변화점이 나타나지 않을 수 있다.
- Logs/Metrics/Traces 중 하나의 modality에만 원인 단서가 존재할 수 있다.
- 시간창이 짧거나 잘못 정렬되면 원인보다 증상이 먼저 관측된다.
- Reference topology가 누락되면 실제 전파가 구조적으로 불가능한 경로처럼 보인다.

즉 `정답 레이블이 존재한다`와 `제공된 관측 데이터로 정답을 식별할 수 있다`는 동일하지 않다. 본 연구는 이를 **Identifiability Gap**으로 분리해 평가한다.

### 2.3 원인 서비스 이름과 인과 경로는 다른 문제

서비스 이름 하나를 맞히는 것은 localization에 가깝다. 조치 가능한 RCA는 다음을 함께 설명해야 한다.

```text
어디에서 시작했는가?
→ 어떤 관계를 따라 전파되었는가?
→ 어떤 증거가 각 단계를 지지하거나 반증하는가?
→ 어떤 서비스와 비즈니스 기능에 영향을 주었는가?
→ 어떤 확인·조치가 재발 방지에 연결되는가?
```

Root label만 맞고 중간 경로가 틀리면 조치 대상, 영향 범위, 재발 방지 판단도 잘못될 수 있다.

### 2.4 다중 원인과 경쟁 가설

실제 장애는 다음 구조를 가질 수 있다.

- 하나의 Root가 여러 경로로 증상을 증폭시키는 fan-out
- 여러 Root가 동시에 발생하는 hybrid/multi-root incident
- 동일 telemetry를 설명하는 복수의 경쟁 가설
- 원인, 기여 요인, 전파 증상, 독립 노이즈가 동시에 존재하는 상황

따라서 하나의 후보를 조기에 확정하기보다 복수 가설을 유지하고, 지지·반증 증거를 비교해야 한다.

### 2.5 LLM 단독 RCA의 한계

LLM은 텍스트 생성과 패턴 결합에는 강하지만 다음 문제가 반복된다.

- 첫 번째 강한 이상을 Root Cause로 확정하는 조기 종료
- 정상 수치를 과부하로 해석하는 의미 왜곡과 환각
- 로그 또는 트레이스를 생략하고 메트릭만 탐색하는 데이터 편식
- 중간 서비스를 건너뛰고 관측되지 않은 직접 causal edge를 생성
- 올바른 서비스를 언급하면서도 검증 가능한 전파 경로를 구성하지 못함

따라서 LLM은 최종 설명기·도구 오케스트레이터로 활용하되, **원시 telemetry 전체에서 자유롭게 원인과 경로를 창작하도록 두지 않는다.**

### 2.6 잘못 복원한 관계가 RCA를 악화시킬 위험

관계가 보이지 않는다는 이유만으로 무조건 edge를 추가하면 false edge와 잘못된 shortcut이 생성된다. 연구 목표는 단순 graph completion이 아니라 다음 조건을 동시에 만족하는 것이다.

- 필요한 누락 관계를 높은 Recall로 후보에 포함
- 근거가 약한 관계를 과도하게 확정하지 않음
- 타입·시간·구조 제약과 모순되는 관계 억제
- 불충분한 증거에서 `unresolved/abstain` 허용
- 추가된 관계가 실제 downstream RCA를 개선하는지 별도 검증

---

## 3. OpenRCA 2.0이 보여준 원인 식별과 경로 추론의 공백

### 3.1 벤치마크 범위

OpenRCA 2.0은 PAVE(Path Annotation via Verified Effects)를 이용하여 알려진 fault intervention에서 downstream effect를 정방향으로 검증하고 단계별 causal propagation path를 구성한다.[R1]

| 항목 | OpenRCA 2.0 |
|---|---:|
| 대상 시스템 | TrainTicket, OpenTelemetry Demo, DeathStarBench Hotel Reservation |
| 평가 인스턴스 | **500** |
| 주요 Fault 종류 | **27** |
| 평가 LLM Agent | **11개 Frontier LLM** |
| 인스턴스당 평균 검증 Causal Edge | **7.5개** |
| 평가 계층 | Outcome + Process |
| Process 지표 | Path Reachability, Node F1, Edge F1 |

PAVE는 후보 경로가 다음 조건을 만족할 때 검증 경로로 인정한다.

- 구조적 전파 규칙과 Topology에 부합
- 주입 전 기준선 대비 통계적으로 유의한 변화
- upstream→downstream 시간 정렬에 부합

평가 Agent는 Logs/Metrics/Traces를 이용해 Root Cause와 CausalGraph를 추론하지만, PAVE Ground Truth 생성은 알려진 intervention과 system dependency graph를 사용한다. 이 정보 비대칭이 실제 운영과 벤치마크 사이의 중요한 차이다.[R1]

### 3.2 11개 LLM 평균 핵심 점수

| 계층 | 지표 | 평균 | 의미 |
|---|---|---:|---|
| Outcome | **AnySvc** | **76.0%** | Fault 종류를 무시하고 정답 Root Service 중 하나 이상 언급 |
| Outcome | **Recall** | **33.2%** | 정답 `(service, fault_kind)` 쌍 회수율 |
| Outcome | **Pair F1** | **34.1%** | `(service, fault_kind)` 집합 F1 |
| Outcome | **Exact Match** | **20.7%** | 예측 Root Cause 집합 전체가 정답과 완전히 일치 |
| Process | **Path Reachability** | **61.5%** | 맞힌 Root Service에서 Alarm Node까지 유효 경로가 하나 이상 존재 |
| Process | **Node F1** | **62.2%** | 전파 경로 참여 노드 집합 정확도 |
| Process | **Edge F1** | **43.4%** | 방향성을 포함한 전파 Edge 정확도 |

핵심 해석은 다음과 같다.

- 서비스 하나 이상을 맞히는 AnySvc는 76.0%지만, 전체 Root Cause 집합 Exact Match는 20.7%다.
- AnySvc와 Path Reachability의 차이는 **14.5%p**다. 전체 사례 기준으로 올바른 서비스를 언급했지만 검증 경로를 만들지 못한 진단 공백이다.
- 서비스 적중 사례를 조건으로 계산하면 `61.5 / 76.0 ≈ 80.9%`가 경로에 연결되며, 약 **19.1%의 서비스 적중 진단은 경로가 근거화되지 않았다.**
- Node F1 62.2%보다 Edge F1 43.4%가 **18.8%p 낮다.** 참여 서비스를 찾는 것보다 서비스 사이의 방향성 전파 관계를 맞히는 것이 더 어렵다.[R1]

> `20.7%`는 “근본 원인 서비스 하나를 맞힌 비율”이 아니라 **전체 Root Cause 집합의 strict Exact Match**다. 서비스 하나 이상 적중률은 AnySvc 76.0%다.

### 3.3 모델별 Grounding 차이

| 모델 | Outcome F1 | AnySvc | Path Reachability | `PR / AnySvc` |
|---|---:|---:|---:|---:|
| Qwen3.6-Max | 35.7% | 79.2% | 71.8% | 약 **90.7%** |
| Claude Sonnet 4.6 | 40.5% | 79.4% | 52.6% | 약 **66.2%** |

두 모델의 AnySvc는 유사하지만 경로 연결 능력은 19.2%p 차이가 난다. 이는 **최종 Root 결과만으로는 추론 과정의 신뢰성을 평가할 수 없다는 근거**다.[R1]

### 3.4 다중 원인과 행동 결함

OpenRCA 2.0 trajectory 분석은 다음 패턴을 보고한다.[R1]

- **Premature commitment**: 하나의 그럴듯한 원인을 찾고 공존 원인·대안 가설 탐색을 중단
- **Presence bias**: telemetry가 사라진 killed service를 정상으로 보고 오류를 출력한 upstream을 원인으로 오인
- **Salience capture**: 가장 큰 오류·지연을 Root로 간주하지만 실제로는 downstream 증폭일 수 있음
- 중간 서비스를 건너뛰고 존재하지 않는 직접 causal edge 생성

Hybrid injection에서는 눈에 띄는 원인은 찾지만 미세한 동시 원인을 놓쳐 Recall과 Exact Match가 하락한다. 이는 본 연구가 **복수 가설 생성→검증→축소**를 채택해야 하는 근거다.

---

## 4. OpenRCA 2.0의 발전과 남은 한계

OpenRCA 2.0은 Root label만 평가하던 방식보다 과정 중심 평가를 크게 강화했다. 다만 다음 가정과 외적 타당성 경계가 남는다.[R1]

| 가정·한계 | 본 연구와의 연결 |
|---|---|
| 알려진 controlled intervention 필요 | 실운영에서는 intervention label이 없으므로 backward hypothesis generation 필요 |
| 실제 전파가 structural pruning에서 살아남을 정도의 dependency graph 필요 | Operational Ontology가 누락되면 실제 경로가 후보에서 제거될 수 있음 |
| Logs/Metrics/Traces에 propagation signature가 존재해야 함 | Evidence Gap에서는 단일 정답 대신 복수 가설과 `unresolved` 필요 |
| pre-injection baseline이 no-intervention 분포를 근사해야 함 | 포화·비정상 시계열·비정상 workload에서는 변화점 검증이 약해짐 |
| PAVE annotation은 full telemetry를 사용 | 누락 telemetry 조건의 관계복원·RCA를 별도 평가해야 함 |
| controlled testbed가 production-scale fidelity를 보장하지 않음 | 대용량 검색, 수집 지연, 보존정책, 변경 이벤트를 별도 운영 지표로 검증 |
| rule vocabulary 밖의 propagation은 pruning될 수 있음 | 미등록 원인·관계를 귀추 가설과 rule 확장 후보로 관리 |

OpenRCA 2.0의 Agent 문제는 다음처럼 정리할 수 있다.

```text
관측 Telemetry만으로
Dependency Structure + Root Cause + Propagation Path 역추론
= 본질적인 Backward Abductive Problem
```

반면 PAVE는 알려진 intervention으로 forward verification을 수행한다. 실제 운영 Agent는 원인이 주어진 상태에서 경로를 확인하는 것이 아니라, **결과에서 출발해 가능한 원인과 누락 관계를 먼저 가설화해야 한다.**

### 4.1 보조 근거의 출처 구분

#### LLM Agent 실패율

다음 수치는 OpenRCA 2.0의 11개 모델 평균이 아니라, OpenRCA 1.0의 335개 사건을 5개 LLM으로 실행한 1,675개 Agent run을 분석한 별도 연구의 pitfall 발생률이다.[R3]

| 실패 유형 | 발생률 | 의미 |
|---|---:|---|
| Hallucination in Interpretation | **71.2%** | 조회 결과를 실제와 다른 의미로 재해석 |
| Incomplete Exploration | **63.9%** | 관련 KPI·컴포넌트·증거 범주를 충분히 탐색하지 않음 |
| Symptom-as-Cause | **39.9%** | downstream 증상을 Root Cause로 조기 확정 |
| Limited Telemetry Coverage | 26.9% | Logs/Metrics/Traces 중 일부 modality에 편중 |
| No Cross-Validation | 18.6% | 단일 발견을 다른 증거로 교차검증하지 않음 |

이 비율은 정확도 지표가 아니라 실패 패턴이 관찰된 run 비율이며, 하나의 run에 여러 실패가 중복될 수 있다.

#### Bank 사례에 대한 산업계 비평

Traversal의 분석은 원본 OpenRCA의 Bank 사례에서 변화점 부재, 다수 컴포넌트 동시 이상, 레이블과 관측 증거의 불일치를 지적한다.[R4]

- OpenRCA 2.0의 공식 TrainTicket/OTel Demo/Hotel Reservation 평가 사례가 아니다.
- Peer-reviewed benchmark 결과가 아니라 산업계 외부 분석이다.
- 따라서 “OpenRCA 2.0의 데이터 오류가 입증됐다”고 사용하지 않는다.
- **Identifiability Gap과 competing hypothesis의 보조 사례**로만 사용한다.

### 4.2 근거 강도

| 주장 | 근거 등급 | 사용 방식 |
|---|---|---|
| OpenRCA 2.0 점수·PAVE 가정 | 높음 — 공식 논문 | 핵심 연구 동기와 과정 지표 설계 |
| 71.2%·63.9%·39.9% 실패 패턴 | 중상 — 별도 연구 | LLM Agent 실패 taxonomy |
| Bank 0/60/47 사례 | 보조 — 산업계 비평 | Identifiability 예시, 단독 결론 금지 |
| Ontology·Runbook이 항상 RCA를 개선 | 미확정 | Ablation과 Oracle Utility로 검증 |

---

## 5. 기존 해결 방향과 연구 공백

### 5.1 Runbook / SOP 주입

Flow-of-Action은 SRE의 SOP를 검색·생성·코드화하여 LLM의 action selection을 제약한다. 보고된 실험에서는 ReAct 정확도 35.50%에서 SOP-enhanced 방식 64.01%로 향상됐다.[R5]

| 장점 | 한계 |
|---|---|
| 검증된 조사 순서 제공 | 미등록 신규 원인을 직접 생성하지 못함 |
| 불필요한 action과 환각 감소 | 오래되거나 환경과 다른 Runbook은 잘못된 경로를 강제 |
| 어떤 telemetry를 조회할지 안내 | 누락 Topology를 자동 복원하지 않음 |
| 조치 절차와 연결하기 쉬움 | 검색 성공률과 문서 품질에 의존 |

Runbook은 **검증·조치 지식**으로 유용하지만 open-world 원인과 누락 관계를 생성하는 전체 해법은 아니다.

### 5.2 Ontology / Knowledge Graph

Ontology/KG는 이질적인 observability schema와 시스템 의미 단절을 줄이는 데 사용된다.

- UModel: telemetry, entity, expert knowledge를 semantic graph로 연결하는 virtual ontological layer[R6]
- KGroot: Log/Metric event graph와 historical fault KG를 비교하여 root ranking[R7]
- MetaRCA: component type, metric semantics, connection pattern을 Meta Causal Graph로 구성하고 evidence로 belief 갱신[R8]
- OntoLogX: 비정형 Log를 Ontology-grounded Event KG로 변환[R11]

| 장점 | 한계 |
|---|---|
| `Web→API→Service→DB→Metric` 의미와 허용 관계 명시 | 실제 ABox가 완전하거나 최신이라는 보장이 없음 |
| 수집원별 ID를 공통 객체로 정규화 | Entity resolution 오류가 graph 전체로 전파 |
| 경로 탐색·영향도·제약 검증에 유리 | 구조 관계와 causal relation 혼동 위험 |
| Runbook·과거 Incident·Owner·Criticality 연결 | 기존 평가는 missingness 효과를 분리하지 않는 경우가 많음 |

Ontology는 정답 제조기가 아니라 **가설이 존재할 수 있는 타입·관계 공간과 모순 제약을 제공하는 도구**다.

### 5.3 Causal Discovery / Graph-based RCA

시간 선후관계, Granger/PC/PCMCI, Bayesian/Structural Causal Model 등은 원인 방향 추정에 유용하다. 그러나 unobserved confounder, non-stationarity, 짧은 incident window, missing node/edge에 민감하다. 입력 graph에 누락된 관계는 원인 경로 후보 자체에서 제외될 수 있다.

### 5.4 Graph-free RCA

Call graph가 불완전한 환경에서 telemetry만으로 root를 찾는 접근은 graph 의존성을 줄인다. 다만 다음 downstream 작업에는 구조가 여전히 필요하다.

- 검증 가능한 propagation path
- 영향 범위 계산
- Owner/Runbook/Action 연결
- 잘못된 direct edge 억제

### 5.5 본 연구의 결합 위치

```text
Runbook/SOP
  = 알려진 조사·검증·조치 절차

Ontology/KG
  = 객체·관계·제약·영향 구조

Abduction
  = 불완전한 지식에서 가능한 누락 관계·원인 생성

Telemetry/NLI/Soft Logic
  = 가설 지지·반증·불확실성 평가

RCA
  = 검증된 Root·Cause Path·Impact Path 구성

LLM
  = 도구 호출과 근거 기반 설명·조치 문장화
```

### 5.6 연구 Gap

```text
Log → KG                     : 기존 연구 존재
Telemetry → Root Ranking     : 기존 연구 다수
Blind Spot → Graph-free RCA  : 기존 연구 존재
Runbook → Action Guidance    : 기존 연구 존재

그러나

불완전 Operational KG
        ↓
누락 관계 + 복수 원인 가설 생성
        ↓
Telemetry·Ontology·NLI 기반 지지/반증
        ↓
Soft Logic + Calibration + Abstention
        ↓
Recovered KG 기반 경로 RCA
        ↓
동일 LLM의 Outcome + Process 개선 검증

을 하나의 leakage-controlled 실험 프로토콜로 검증하는 것은 별도의 연구 문제다.
```

본 연구의 novelty는 DeBERTa나 PSL이라는 개별 모델 자체가 아니라, **현실적인 missingness 문제 정의, bounded abduction, open-world 불확실성 제어, 과정 중심 평가, downstream graph intervention**에 둔다.

---

## 6. 왜 Ontology / Knowledge Graph를 사용하는가

### 6.1 운영 의미 모델

| Subject | Relation | Object | 의미 |
|---|---|---|---|
| WebPage | `ROUTES_TO` | APIEndpoint | 사용자 화면과 API 연결 |
| Service/Application | `EXPOSES` | APIEndpoint | API 제공 주체 |
| Service/Instance | `CALLS` | Service/Instance | Runtime 호출 관계 |
| Instance | `INSTANCE_OF` | Service/Application | 실행 인스턴스 소속 |
| Instance/Pod | `LOCATED_ON` | Host/Node | 물리·논리 실행 위치 |
| Service/Application | `USES_DATASOURCE` | DataSource/Database | DB 의존성 |
| Transaction | `EXECUTES` | SQLPattern | 실행 SQL 관계 |
| Entity | `HAS_METRIC` | Metric | CPU/Memory/Heap/Latency/Error 관측 |

### 6.2 관계 레이어 분리

- `structural`: 배포·소속·정적 구성
- `runtime`: 실제 호출·실행·관측 관계
- `hypothesis`: 아직 검증되지 않은 누락 관계·원인 후보
- `causal`: 장애 전파를 지지하는 검증 경로

중요한 원칙:

- `CALLS(A,B)`를 자동으로 `CAUSES(A,B)`로 간주하지 않는다.
- 모든 inferred edge는 `source`, `evidence_id`, `observed_at`, `valid_from`, `valid_to`, `confidence`, `status`를 보존한다.
- 상태는 `observed / inferred / confirmed / unresolved / rejected`로 구분한다.
- 근거가 부족하면 graph에 확정하지 않고 `unresolved`로 남긴다.

---

## 7. 귀추법과 RCA의 구분 및 결합

### 7.1 비교표

| 구분 | 귀추법(Abduction) | 근본 원인 분석(RCA) |
|---|---|---|
| 핵심 목적 | 관측 결과를 설명하는 **가장 그럴듯한 가설 생성** | 장애의 **근본 원인·전파 경로·재발 방지 조치** 규명 |
| 사고 방향 | 결과 + 규칙/배경지식 → 가능한 원인/누락 관계 | 현상 → Why/인과 추적 → Root Cause → Action |
| 입력 | 관측 이상, 불완전 KG, 운영 규칙, prior | 검증 대상 가설, telemetry, dependency, 시간·인과 증거 |
| 출력 | 복수의 관계·원인 후보와 설명력 점수 | Root ranking, Cause Path, Impact Path, Evidence, Action |
| 확실성 | 잠정적·비단조적·open-world | 가능한 한 검증적·행동 지향적 |
| 실패 위험 | 후보 폭증, 그럴듯하지만 틀린 가설 | 누락 graph에서 경로 단절, 증상을 원인으로 오인 |
| 대표 활용 | 의학 진단, 과학 가설, 탐정 추론 | IT 장애, 제조 불량, 안전 사고, 재발 방지 |
| 본 연구 역할 | **RCA 전단 후보 생성기** | **가설을 경로와 조치로 검증하는 후단 분석기** |

다음 표현은 과도하다.

> “RCA는 완전하게 구현된 Ontology에서만 동작한다.”

RCA는 Ontology 없이도 Rule, 통계, causal discovery, graph-free 방식으로 수행할 수 있다. 더 정확한 표현은 다음과 같다.

> **경로 기반·영향도 기반 RCA는 충분히 정확한 dependency/semantic model이 있을수록 검증 가능성과 설명력이 높아진다. 그러나 실제 Operational Ontology는 불완전하므로, 본 연구는 귀추추론으로 누락 관계와 원인 가설을 먼저 생성하고 이를 증거 기반 RCA가 검증하도록 설계한다.**

귀추법은 RCA의 대체재가 아니라, **완전한 원인과 관계를 사전에 알 수 없다는 현실적 공백을 메우는 가설 생성 계층**이다.

### 7.2 귀추 목적함수

- 관측 데이터: `O`
- 배경 Ontology/Rule: `B`
- 관측 그래프: `G_obs`
- 가능한 누락 관계·원인 가설: `H`

```text
B ∪ G_obs ∪ H가 관측 O를 설명한다.
B ∪ G_obs ∪ H가 타입·시간·구조 제약과 모순되지 않는다.
H의 불필요한 관계가 최소화된다.
근거가 부족하면 unresolved로 남긴다.
```

개념적 목적함수:

```text
H* = argmax_H [
      λ1 · Explanation(H, O)
    + λ2 · TelemetrySupport(H)
    + λ3 · RuleConsistency(H, B)
    + λ4 · TemporalConsistency(H)
    - λ5 · Complexity(H)
    - λ6 · Contradiction(H)
]
```

### 7.3 예시

```text
관측 결과
- payment-api latency 급증
- payment-db wait 증가
- payment-api와 payment-db의 직접 관계는 KG에서 누락

운영 규칙
- Service가 DataSource를 사용하고 DB wait가 증가하면 Service latency가 증가할 수 있음

귀추 가설
- USES_DATASOURCE(payment-api, payment-db)
- EXECUTES(payment-transaction, slow-sql-pattern)

검증
- Trace/Profile의 SQL·connection evidence
- 동일 시간창의 DB wait와 API latency
- Service/Database 타입 제약
- 반대 방향·대체 DB 가설과 비교
```

### 7.4 다중 원인 처리

```text
H1: DB wait → payment latency
H2: connection pool exhaustion → payment latency
H3: upstream retry storm → payment latency
H4: application GC pause → payment latency
```

귀추 단계는 하나를 즉시 확정하지 않는다. 각 가설에 지지·반증 증거를 연결한 뒤 RCA 단계에서 `root / contributing cause / propagated symptom / unresolved`를 구분한다.

---

## 8. 제안 방법론

### 8.1 A0 — Observed Operational KG

```text
G_obs = φ(web.csv, source/deployment, logs, metrics, traces; Ontology)
```

모든 관계에 provenance를 유지한다.

```text
source / evidence_id / observed_at / valid_from / valid_to
confidence / status(observed,inferred,confirmed,unresolved)
```

### 8.2 현실적 Blind Spot 생성

| Mask | 운영 대응 |
|---|---|
| IID 20/40% relation mask | 비교 가능한 기본 control |
| Component blackout | 미계측·신규·Black-box 서비스 |
| Trace parent dropout | sampling/export 단절 |
| Relation-source block | 특정 collector 전체 누락 |
| Modality dropout | Log/Metric/Trace 수집원 장애 |
| Identity-link dropout | Web/APM/CMDB 식별자 연결 실패 |
| Temporal-window dropout | TTL·보존기간·collector 지연 |
| Path-critical mask | Root→Symptom bridge relation 누락 |
| Topology staleness | 배포 후 이전 관계가 남은 상황 |

IID random mask는 기본 비교용이며, 최종 주장은 structured blind spot에서 검증한다.

### 8.3 A1 — Direct / Deterministic Evidence

- parent-child span
- 명시 Endpoint↔Service mapping
- Pod→Service deployment
- SQL/DataSource identifier
- operator-confirmed relation

직접 관측 관계는 귀추 후보 budget 밖에서 보호한다.

### 8.4 A2 — Bounded Abductive Candidate Generation

- 결과와 규칙을 설명하는 누락 관계·원인 후보 생성
- 전체 typed universe를 작은 후보집합으로 압축
- 전체 후보와 source/target별 상한 적용
- 목표는 즉시 높은 Precision을 만드는 것이 아니라 **정답 후보 Recall을 보존하면서 탐색 공간을 제한하는 것**이다.

### 8.5 A3 — Tri-state NLI Evidence

DeBERTa/NLI는 hard veto가 아니다.

```text
corroborates : 가설을 지지하는 언어·operation 증거
contradicts  : 가설 또는 방향과 충돌하는 증거
ambiguous    : 판단 근거 불충분
```

- Forward/Reverse hypothesis를 모두 평가
- A2 prior와 후보 전체를 보존
- NLI는 후보를 삭제하는 증명기가 아니라 재랭킹 evidence로 사용
- Equal-size A2-only control과 비교하여 단순 후보 축소 효과를 분리

현재 첫 A3 구현은 과학적 Gate를 통과하지 못했다. 구체적 결과는 13절에 기록한다.

### 8.6 A4 — Multi-source Hypothesis Validation

```text
S(H) =
  α · S_abduction
+ β · S_trace
+ γ · S_log
+ δ · S_metric
+ ε · S_NLI
+ ζ · S_ontology
+ η · S_temporal
+ θ · S_runbook
```

검증 Evidence:

- Operation/HTTP/SQL/connection evidence
- Runtime role와 neighbor pattern
- Log template, exception, error chain
- Metric change point, lag, baseline deviation
- Ontology domain/range/cardinality/path constraint
- Runbook이 요구하는 점검 항목과 실제 관측 결과
- 경쟁 가설의 반증 증거

### 8.7 A5 — PSL / Calibration / Abstention

가설별 local score만 평가하지 않고 연결 규칙을 soft constraint로 공동 추론한다.

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

최종 관계는 다음 형태로 관리한다.

```text
confirmed / inferred / unresolved
confidence ∈ [0,1]
```

- confidence calibration
- risk-coverage 기반 threshold
- 증거 부족 시 abstention
- 잘못된 관계의 자동 영구 반영 방지

### 8.8 RCA Layer

```json
{
  "root_candidates": [],
  "contributing_causes": [],
  "cause_paths": [],
  "impact_paths": [],
  "supporting_evidence": [],
  "contradicting_evidence": [],
  "relation_hypotheses": [],
  "confidence": 0.0,
  "abstain": false
}
```

RCA는 단순 Root Service 하나가 아니라 다음 구조를 제공한다.

```text
장애 현상
   ↓
경쟁 원인 후보
   ↓
Root / Contributing Cause
   ↓
Cause Path
   ↓
Impact Path
   ↓
Supporting / Contradicting Evidence
   ↓
확인·조치·재발 방지
```

### 8.9 Runbook Retrieval + LLM Layer

LLM 입력:

```text
Incident summary
+ Observed / Recovered KG subgraph
+ Root candidates
+ Cause / Impact paths
+ Supporting / Contradicting evidence IDs
+ Runbook sections
+ Confidence / Unresolved status
```

LLM 출력의 모든 핵심 주장에는 `evidence_id` 또는 `graph_path`를 연결한다. 원시 로그 전체를 그대로 투입하는 대신, 구조화된 RCA와 필요한 원문 근거만 제한적으로 제공한다.

최종 효과:

```text
ΔRCA = Metric(LLM(X, G_recovered))
     - Metric(LLM(X, G_observed))
```

즉 **관계 복원 품질**과 **복원 관계가 실제 RCA에 주는 효용**을 분리한다.

---

## 9. 데이터 전략

단일 공개 데이터셋이 Web topology, multimodal telemetry, 완전 topology, multi-root, causal path, impact path, Runbook을 모두 제공하지 않는다. 데이터셋마다 역할을 분리한다.

| 데이터 | 주 역할 | 한계 |
|---|---|---|
| **RCABench/Aegis + TrainTicket source/K8s** | Main: 현실적 telemetry, fault injection, topology blind spot, Root RCA | 대용량이며 WebPage/SQL/JVM 세부 관계는 보완 필요 |
| **RCAEval RE2/RE3** | 빠른 Task A 개발, IID/structured masking, 재현성 | 완전 causal path GT 부족 |
| **OpenRCA 2.0** | Task B: EM/F1/AnySvc + PR/Node F1/Edge F1 기반 경로 평가 | PAVE 가정과 controlled testbed 범위 내 해석 필요 |
| **DejaVu** | 명시 graph 기반 relation recovery control | Metric 중심, Log/Trace 약함 |
| **OntoLogX** | Raw Log→Ontology/Event KG extraction | Microservice topology/RCA path 데이터가 아님 |
| **LO2v2 / Nezha** | Source·Telemetry 동시 검증, cross-source robustness | 시스템·fault 범위가 제한적 |
| **Runbook/SOP corpus** | 조사·조치 지식 효과 비교 | 알려진 절차 편향과 stale knowledge 관리 필요 |

### 9.1 RCABench/Aegis 주요 데이터 매핑

| Ontology 대상 | 데이터 예 |
|---|---|
| Transaction | `trace_id` |
| Span | `span_id`, `parent_span_id` |
| APIEndpoint | `span_name`, HTTP method/status |
| Service | `service_name`, `attr.k8s.service.name` |
| Instance/Pod | `attr.k8s.pod.name` |
| Node/Server | `attr.k8s.node.name` 등 metric attribute |
| Runtime `CALLS` | parent/child span, source/destination workload |
| Error | trace status, HTTP status, log level/message |
| CPU/Memory | container/Kubernetes metrics |
| Root/Fault GT | fault injection / ground-truth metadata |

활용:

- Logs/Metrics/Traces로 `G_obs` 생성
- Component/Trace/Relation/Identity/Path blind spot 실험
- fault injection으로 Root RCA 검증
- TrainTicket source/K8s 구조를 모델 입력에서 숨긴 independent reference topology로 사용

### 9.2 운영 프로젝트와 연구 데이터 대응

| 운영 LogAgent | 연구 Benchmark |
|---|---|
| `web.csv` Topology | TrainTicket UI/source/deployment + telemetry-derived route |
| Jennifer Domain/Instance | K8s Service / Pod / Container |
| Jennifer Transaction | Trace `trace_id` |
| Jennifer Profile | Span / operation |
| Jennifer PERF | Metrics |
| `log.file` | Application Logs |
| AGENT_ID / Instance ID | K8s Pod/Service identity |
| Endpoint | Span operation + HTTP attribute |
| CPU/Memory/Heap | Resource/JVM metric where available |
| Trigger | Fault injection timestamp |
| Neo4j Semantic Graph | Normalized Operational KG |
| RCA JSON/HTML | Root + Cause/Impact Path + Evidence |
| LLM report | OpenRCA 2.0형 Outcome/Process 평가 |

```text
[운영]
web.csv + Jennifer/log.file
       ↓
Operational KG
       ↓
Relation/Root Hypothesis
       ↓
RCA + Runbook
       ↓
LLM

[연구]
TrainTicket topology + RCA benchmark telemetry
       ↓
Operational KG
       ↓
Masked relation + Abductive hypothesis
       ↓
RCA + Runbook control
       ↓
Fixed LLM evaluation
```

---

## 10. 실험 비교군

### 10.1 Task A — 관계·가설 복원

| ID | 방법 | 목적 |
|---|---|---|
| A0 | Observed KG only | 복원 없음 |
| A1 | Direct deterministic evidence | 명시 관계 baseline |
| A2 | Bounded abduction | 높은 Recall의 후보 생성 |
| A3 | A2 + tri-state NLI reranking | NLI 증거의 추가 판별력 |
| A4 | A3 + Trace/Log/Metric/Ontology/Runbook evidence | Multi-source 가설 검증 |
| A5 | A4 + PSL + calibration/abstention | 최종 복원 방식 |
| KGC | TransE/DistMult/Graph completion | 일반 KG completion 비교 |
| LLM-KG | LLM relation extractor | LLM-only 관계 복원 비교 |
| AO | Oracle graph | 복원 상한선 |
| AW | Proposed graph + controlled false edges | 잘못된 관계의 downstream 위험 |

### 10.2 Task B — 동일 LLM RCA

LLM model, prompt, temperature, tool budget, telemetry window를 고정한다.

| ID | 입력 | 분리하려는 효과 |
|---|---|---|
| B0 | Raw telemetry only | LLM 단독 baseline |
| BR | Raw telemetry + Runbook/SOP | 절차 지식 효과 |
| B1 | Raw telemetry + Observed KG | 불완전 graph 효과 |
| B2 | Raw telemetry + strongest baseline-completed KG | 일반 graph completion 효과 |
| B3 | Raw telemetry + Proposed recovered KG + RCA evidence | 귀추·검증 graph 효과 |
| B4 | B3 + Runbook/SOP | 구조 지식과 절차 지식의 결합 효과 |
| BO | Raw telemetry + Oracle graph/path | 가능한 상한선 |
| BW | Raw telemetry + 동일 개수의 shuffled/false edges | graph token 증가와 정확 관계 효과 분리 |
| BG | Graph-free RCA baseline | topology 복원 없이 telemetry만 사용하는 대안 |

---

## 11. 평가 지표

### 11.1 Task A — 관계복원

| 지표 | 의미 |
|---|---|
| Candidate Recall@K | A2가 정답 관계를 후보에 보존하는가 |
| Accepted Relation Precision/Recall/F1 | 최종 확정 관계 품질 |
| Silver Precision Lower Bound | 불완전 Silver GT에서의 보수적 정밀도 |
| AUPRC | 관계 class imbalance 조건의 품질 |
| MRR / Hits@1/3/10 | 정답 관계 ranking |
| Candidate Count / Compression | 후보 폭증 제어 |
| False-edge Rate | 잘못 추가한 관계 비율 |
| Brier / ECE | confidence calibration |
| Risk-Coverage | abstention 시 오류 감소 |
| Constraint Violation | Ontology domain/range/path 위반 |
| Root→Symptom Reachability | 단절 경로 복원 여부 |
| Path Edge F1 | 정답 경로 edge 복원 |
| Shortest-path Distortion | 잘못된 shortcut·우회 경로 생성 여부 |

### 11.2 Task B — RCA/LLM, OpenRCA 2.0 정렬

| 계층 | 지표 | 의미 |
|---|---|---|
| Outcome | Exact Match | 전체 Root Cause 집합 완전 일치 |
| Outcome | Pair Precision/Recall/F1 | `(service, fault_kind)` 정확도 |
| Outcome | AnySvc | 적어도 하나의 올바른 Root Service |
| Process | Path Reachability | 올바른 Root에서 Alarm까지 유효 경로 존재 |
| Process | Node F1 | 참여·영향 서비스 집합 |
| Process | Edge F1 | 방향성을 포함한 causal propagation edge |
| Grounding | Evidence Citation Precision | 주장과 실제 telemetry 연결 |
| Grounding | Unsupported Claim Rate | 근거 없는 원인·수치·경로 비율 |
| Safety | Abstention Accuracy | 식별 불가능 사례의 올바른 보류 |
| Multi-root | Complete Root Set Recall / Count Error | 다중 원인 누락 여부 |
| Behavior | Symptom-as-Cause Rate | downstream 증상을 root로 오인 |
| Behavior | Exploration Coverage | component/KPI/modality 탐색 범위 |
| Operation | Latency / Tokens / Cost / Memory | 실제 운영 가능성 |

### 11.3 핵심 개선량

```text
ΔRoot    = Metric_root(B3) - Metric_root(B1)
ΔPath    = Metric_path(B3) - Metric_path(B1)
ΔGround  = Grounding(B3) - Grounding(B0)
ΔRunbook = Metric(B4) - Metric(B3)
ΔGraph   = Metric(B3) - Metric(BR)
```

### 11.4 통계 검증

- Incident 단위 paired comparison
- 최소 5개 masking seed
- 동일 campaign/topology는 같은 split에 배치
- Calibration incident와 Held-out incident 분리
- Incident-level paired bootstrap confidence interval
- Binary correctness: McNemar 또는 paired permutation
- Continuous metric: paired bootstrap 또는 Wilcoxon
- Multiple comparison: Holm correction
- IID mask와 structured blind spot을 분리 보고
- Identifiability가 낮은 사례는 별도 stratum으로 보고
- Equal-size shortlist control로 단순 후보 수 감소 효과 분리

---

## 12. 핵심 연구가설

### H1 — 귀추 후보 생성

> 불완전 Operational KG에서 bounded abduction은 전체 typed universe보다 훨씬 작은 후보집합으로 누락 관계를 압축하면서 높은 Candidate Recall을 유지한다.

### H2 — 가설 검증

> Tri-state NLI와 multi-source evidence는 A2 Recall을 크게 훼손하지 않으면서 후보 수, P-LB, MRR, calibration을 개선한다.

### H3 — 경로 RCA

> 복원된 Operational KG는 Observed KG보다 Root→Symptom Path Reachability와 Edge F1을 향상시킨다.

### H4 — LLM Grounding

> 복원 KG와 evidence-bounded RCA를 제공하면 Raw telemetry 또는 Runbook만 제공한 LLM보다 ungrounded diagnosis와 unsupported claim을 줄인다.

### H5 — Runbook과 Ontology의 상보성

> Runbook은 알려진 조사·조치 절차를 개선하고, Ontology+Abduction은 누락 관계와 신규 원인 후보를 보완하므로 두 지식은 결합할 때 가장 효과적이다.

### H6 — 안전성

> Calibration과 abstention은 무조건적인 graph completion보다 false edge가 downstream RCA를 악화시키는 위험을 줄인다.

---

## 13. 연구 Gate와 현재 진행 결과

### 13.1 연구 Gate

| Gate | 질문 | 실패 시 |
|---|---|---|
| D0 Data | provenance/license/schema/GT가 명확한가 | 실험 중단 |
| D1 Leakage | Mask target, intervention, root alias가 입력에 남지 않았는가 | 데이터 분리 수정 |
| D2 Candidate | 귀추 후보가 정답을 충분히 포함하는가 | A2 개선 |
| D3 Multi-case | 여러 Incident·Seed에서도 Recall과 budget을 유지하는가 | 주장 범위 제한 |
| D4 Hypothesis Validation | A3/A4가 Recall을 유지하며 후보·P-LB·MRR을 개선하는가 | NLI/증거 결합 재설계 |
| D5 Oracle Utility | 완전 graph/path가 downstream RCA를 개선하는가 | 관계복원 목적 재검토 |
| D6 Completion | 제안 방식이 KGC/Rule/LLM-only보다 우수한가 | H1/H2 기각 |
| D7 RCA | Recovered KG가 EM/PR/Node F1/Edge F1을 개선하는가 | H3/H4 기각 |
| D8 Generalization | Structured blind spot·held-out system에서도 유지되는가 | 외적 주장 제한 |
| D9 Safety | Wrong-edge control보다 안전하고 abstention이 유효한가 | 자동 반영 금지 |

특히 D5 Oracle Utility가 중요하다. Oracle graph를 제공해도 Path/RCA가 개선되지 않는다면 topology recovery 정확도만 높여서는 최종 연구 목적을 증명할 수 없다.

### 13.2 A2 — 다중 Incident·Seed 검증 결과

| 지표 | 결과 |
|---|---:|
| Incident × Seed × Mask | 6 × 5 × 2 = **60 Cell** |
| Candidate Recall 평균 / 최저 | **1.000 / 1.000** |
| 후보 수 평균 / 최대 | **20.83 / 32** |
| 후보 압축률 평균 | **96.70%** |
| P-LB 평균 / 최저 | **0.7066 / 0.4231** |
| MRR 평균 / 최저 | **0.9872 / 0.8500** |
| 후보 Budget 포화 | **12/60 = 20%** |
| Leakage Check | **60/60 PASS** |

해석:

- A2는 누락 관계를 놓치지 않는 bounded candidate generator로 사용할 수준이다.
- 병목은 후보 Recall이 아니라 **후보의 타당성 검증과 unverified 관계 축소**다.
- 특히 IID40, MEM, DELAY 조건에서 후보 32개 상한이 자주 포화됐다.

상세 결과: `reports/task_a_phase2_results.md`

### 13.3 A3 — Tri-state DeBERTa 검증 결과

A3는 A2 후보를 삭제하지 않고 `corroborates / ambiguous / contradicts` 증거로 재랭킹하도록 구현했다. Calibration 2 Incident·20 Cell과 Held-out 4 Incident·40 Cell을 분리했다.

| Held-out 지표 | A2 전체 후보 | A3 Shortlist | 변화 |
|---|---:|---:|---:|
| Candidate Recall Macro | 1.0000 | **0.9977** | -0.0023 |
| Candidate Recall Minimum | 1.0000 | **0.9545** | 조건 충족 |
| 후보 수 평균 | 19.375 | **17.025** | -2.350 |
| P-LB Macro | 0.7234 | **0.8079** | +0.0845 |
| MRR Macro | 0.9919 | **0.9977** | +0.0058 |

그러나 최종 과학적 판정은 **D4 FAIL**이다.

- Calibration feasible 정책: **0 / 96**
- 동일 후보 수 A2-only control 대비 추가 이득:
  - Recall: `0`
  - P-LB: `0`
  - MRR: `0`
- NLI 상태 분포:
  - `corroborates`: 0
  - `ambiguous`: 7
  - `contradicts`: 1,243
- 미통과 조건:
  - `CALIBRATION_POLICY_FEASIBLE`
  - `MATCHED_BUDGET_ADDITIVE_GAIN`

즉 후보 수 감소로 절대 지표는 개선됐지만, **현재 DeBERTa tri-state score는 A2 기본 순위에 추가적인 판별력을 제공하지 못했다.** 따라서 A3 성공 또는 NLI 효과를 주장하지 않는다.

상세 결과:

- `reports/task_a_phase3_results.md`
- `reports/task_a_phase3_results.json`
- `reports/task_a_phase3_heldout_cells.csv`
- `reports/task_a_phase3_policy_grid.csv`
- `reports/task_a_phase3_artifact_manifest.md`

### 13.4 현재 위치와 다음 연구 결정

```text
A0 Observed KG
      ↓
A1 Direct Evidence
      ↓
A2 Bounded Abduction
  └─ 다중 Incident·Seed D3 PASS
      ↓
A3 Tri-state DeBERTa
  └─ 실행 완료, D4 FAIL
      ↓
A3 Evidence/Premise 재설계 ← 현재
      ↓
A4 Multi-source Validation
      ↓
A5 PSL + Calibration + Abstention
      ↓
D5 Oracle Utility
      ↓
Task B Fixed LLM RCA
```

다음 A3/A4 설계 원칙:

1. Threshold만 사후 조정하지 않는다.
2. A2 후보와 prior는 계속 보존한다.
3. 단일 합성 문장보다 `Operation`, `HTTP`, `Trace direction`, `Log template`, `Metric lag`, `Ontology role`을 독립 Evidence로 분해한다.
4. Relation-specific verbalization과 calibration을 사용한다.
5. Equal-size A2-only control보다 추가 이득이 있어야 NLI 효과를 인정한다.
6. `corroborates`가 0으로 붕괴한 원인을 premise/model mismatch 관점에서 분석한다.
7. D4가 통과하기 전 A3 결과를 확정 graph로 사용하지 않는다.

---

## 14. 타당성·표현 점검

| 주장 | 판정 | 올바른 사용 방식 |
|---|---|---|
| LLM은 원인 서비스는 찾지만 경로 추론에 약함 | 타당 | AnySvc 76.0%, PR 61.5%, Node F1 62.2%, Edge F1 43.4로 근거화 |
| 정확한 근본 원인 파악률이 20.7% | 표현 수정 필요 | 전체 Root Cause 집합 Exact Match 20.7% |
| 14.5%가 근거 없는 진단 | 단위 보완 필요 | 전체 사례 기준 14.5%p, 서비스 적중 조건부 약 19.1% |
| Symptom-as-Cause 39.9%, Hallucination 71.2% | 출처 분리 필요 | OpenRCA 1.0 기반 별도 1,675-run 연구로 명시 |
| Bank 0/60은 OpenRCA 2.0 한계 | 부정확 | 원본 OpenRCA에 대한 산업계 비평으로 제한 |
| Runbook이 전체 문제를 해결 | 부분 타당 | 알려진 절차에는 강하지만 신규 원인·누락 관계에는 제한 |
| Ontology가 RCA를 자동 해결 | 부분 타당 | 의미·제약·경로에는 유용하지만 불완전 ABox는 별도 복원 필요 |
| RCA는 완전한 Ontology에서만 가능 | 과도함 | 정확한 semantic/dependency model이 경로 RCA를 강화한다고 표현 |
| 불완전 Ontology 때문에 귀추법 사용 | 강한 연구 논리 | 귀추를 RCA 이전의 open-world 후보 생성기로 정의 |
| Benchmark가 다중 원인을 온전히 평가하지 못함 | 조건부 타당 | OpenRCA 2.0은 개선했으나 identifiability·graph/telemetry 가정과 production transfer 한계가 남음 |

추가 검증 원칙:

- Identifiability 판정
- Competing Hypothesis coverage와 counter-evidence 평가
- Oracle Utility
- Wrong-edge control
- Runbook vs Recovered KG ablation
- Outcome/Process 이중 평가
- 근거 출처별 신뢰도 분리

---

## 15. 논문에서 주장해야 할 것과 주장하지 말아야 할 것

### 15.1 주장 목표

- 실제와 유사한 topology/telemetry blind spot을 명시적으로 모델링한다.
- 귀추는 누락 관계와 원인 가설의 bounded candidate set을 생성한다.
- 다양한 관측 증거와 Ontology constraint로 가설을 지지·반증한다.
- 불충분한 증거에는 abstention한다.
- 관계복원 정확도뿐 아니라 **실제 downstream RCA/LLM 성능 개선을 intervention으로 검증**한다.
- Root 결과와 Cause Path/Evidence Grounding을 함께 평가한다.

### 15.2 주장하지 않음

- 로그가 자동으로 완전한 TBox/Ontology를 발명한다.
- 구조적 dependency가 곧 causal relation이다.
- DeBERTa가 관계를 증명한다.
- 현재 A3가 성공했다.
- PSL이 형식논리적 proof를 제공한다.
- 관측되지 않은 edge는 모두 false이다.
- graph context가 많을수록 LLM이 항상 좋아진다.
- 하나의 benchmark 결과가 production 일반화를 보장한다.

---

## 16. 최종 연구 흐름

```text
Web Topology + Source/Deployment
                ↓
Logs + Metrics + Transactions/Traces/Profiles
                ↓
Observed Operational Ontology/KG
                ↓
현실적 Blind Spot
관계·컴포넌트·Trace·Identity·시간창 누락
                ↓
Bounded Abduction
결과 + 규칙 → 누락 관계·원인 가설 후보
                ↓
Tri-state NLI + Multi-source Evidence
지지 / 반증 / 모호
                ↓
Ontology Constraint + PSL
Soft inference + calibration + abstention
                ↓
Recovered KG
observed / inferred / confirmed / unresolved
                ↓
RCA
Root + Contributing Cause + Cause Path + Impact Path + Evidence
                ↓
Runbook Retrieval
검증 절차 + 조치안
                ↓
Fixed LLM
근거가 연결된 원인·영향·조치 설명
                ↓
Outcome + Process Evaluation
EM / Pair F1 / AnySvc / PR / Node F1 / Edge F1 / Unsupported Claim
```

---

## 17. 예상 연구 기여

1. **문제 정의**: 완전한 graph를 전제한 RCA가 아니라 불완전 Topology·Telemetry에서의 open-world relation/root hypothesis recovery를 정의한다.
2. **방법론**: bounded abduction, tri-state evidence, Ontology constraint, soft logic, calibration을 누적하는 구조를 제안한다.
3. **안전성**: inferred relation을 관측 사실과 분리하고 contradiction·unresolved·abstention을 지원한다.
4. **과정 중심 평가**: Root label뿐 아니라 검증 가능한 Cause Path와 Evidence Grounding을 평가한다.
5. **Downstream intervention**: 동일 LLM에 Observed KG와 Recovered KG를 각각 제공하여 실제 RCA 개선 효과를 분리 검증한다.
6. **실무 연결**: `web.csv + Jennifer/log.file → Neo4j Ontology → RCA JSON/HTML → Runbook/LLM` 운영 구조와 동일한 연구 pipeline을 사용한다.
7. **Negative result 관리**: A3처럼 절대 지표가 개선돼도 matched-budget control에서 추가 효용이 없으면 성공으로 해석하지 않는 엄격한 Gate를 적용한다.

---

## References

- **[R1]** Fang et al., *OpenRCA 2.0: From Outcome Labels to Causal Process Supervision*, 2026. https://arxiv.org/abs/2606.27154
- **[R2]** Xu et al., *OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures?*, ICLR 2025. https://openreview.net/forum?id=M4qNIzQYpd
- **[R3]** Kim et al., *Why Do AI Agents Systematically Fail at Cloud Root Cause Analysis?*, 2026. https://arxiv.org/abs/2602.09937
- **[R4]** Traversal, *OpenRCA Isn’t Root Cause Analysis — and Why That Matters*, 2026. https://www.traversal.com/blog/openrca-isnt-root-cause-analysis-why-that-matters
- **[R5]** Pei et al., *Flow-of-Action: SOP Enhanced LLM-Based Multi-Agent System for Root Cause Analysis*, WWW Companion 2025. https://arxiv.org/abs/2502.08224
- **[R6]** Pei et al., *UModel: An Agent-Ready Observability Data Modeling Method at Scale*, 2026. https://arxiv.org/abs/2606.04799
- **[R7]** Wang et al., *KGroot: Enhancing Root Cause Analysis through Knowledge Graphs and Graph Convolutional Neural Networks*, 2024. https://arxiv.org/abs/2402.13264
- **[R8]** Liang et al., *MetaRCA: A Generalizable Root Cause Analysis Framework for Cloud-Native Systems Powered by Meta Causal Knowledge*, 2026. https://arxiv.org/abs/2603.02032
- **[R9]** Pham et al., *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data*, 2024/2025. https://arxiv.org/abs/2412.17015
- **[R10]** Fang et al., *Rethinking the Evaluation of Microservice RCA with a Fault Propagation-Aware Benchmark*, 2025. https://arxiv.org/abs/2510.04711
- **[R11]** Cotti et al., *OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models*, 2025. https://arxiv.org/abs/2510.01409
- **[R12]** Pham et al., *TORAI: Unsupervised Fine-grained RCA using Multi-Source Telemetry Data*, 2026. https://arxiv.org/abs/2604.13522
- **[R13]** Ding et al., *TraceDiag: Adaptive, Interpretable, and Efficient Root Cause Analysis on Large-Scale Microservice Systems*, 2023. https://arxiv.org/abs/2310.18740
- **[R14]** Pham et al., *Root Cause Analysis for Microservice System based on Causal Inference: How Far Are We?*, 2024. https://arxiv.org/abs/2408.13729
- **[R15]** AegisLab / RCABench, *Data Formats and Fault Propagation-Aware Benchmark*. https://operationspai.github.io/AegisLab-doc/algorithm-developers/development-guide/data-formats/
- **[R16]** TrainTicket benchmark system. https://github.com/FudanSELab/train-ticket

관련 문서:

- 이전 개요: `docs/research_overview_ko.md`
- 추가 근거 개요: `docs/research_overview_ko2.md`
- 연구 프로토콜: `docs/research_protocol.md`
- 데이터 준비상태: `docs/dataset_readiness.md`
- A2 다중 검증: `reports/task_a_phase2_results.md`
- A3 진단 결과: `reports/task_a_phase3_results.md`
