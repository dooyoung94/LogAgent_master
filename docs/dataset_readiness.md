# Dataset readiness and acquisition matrix

Audit date: **2026-08-27**

## Decision

There is no public dataset that simultaneously provides heterogeneous logs, metrics, traces, an independently verified complete operational graph, realistic missing relationships, gold root cause, and gold cause/impact paths. The benchmark therefore uses several datasets with non-overlapping roles instead of pretending that one source covers the full claim.

Recommended order:

1. **RCABench/Aegis** as the main realistic TrainTicket incident corpus.
2. **RCAEval RE2/RE3** as the reproducible multi-system baseline and standardized root-cause evaluation.
3. **DejaVu** as an exact fault-dependency-graph masking control, while keeping its metrics-only limitation explicit.
4. **OntoLogX** for the raw-log-to-event-KG front end only.
5. **Nezha and LO2v2** for topology-evidence and cross-source robustness.
6. **OpenRCA 1.0** for heavy downstream LLM RCA validation after the core pipeline passes.

This separation is important: an event-triple extraction score from OntoLogX is not a topology-completion score, and a root-service label from RCAEval is not a gold causal path.

## Readiness definition

| Status | Meaning |
|---|---|
| READY | Public bytes or repository were located; version/access, modality, ground truth, and source-specific license were audited sufficiently for acquisition. |
| CONDITIONAL | Data exists, but a license, access, label, or scale gate remains. It is plan-only in the acquisition tool. |
| BLOCKED | The claimed artifact is not publicly available or cannot yet support a reproducible run. |

READY does not mean “perfect for this paper.” It means the bytes can be acquired reproducibly and their limitations are explicit.

## Audited matrix

| Dataset | Actual access and scale | Useful ground truth | Status | Assigned role |
|---|---|---|---|---|
| [RCABench/Aegis](https://zenodo.org/records/17105974) | 1,430 validated TrainTicket cases; 13.4 GB telemetry tar plus small benchmark artifacts | injected service/pod/container/function/metric; logs, metrics, traces, injection metadata | READY, Zenodo API reports CC-BY-4.0 | primary realistic RCA and path-critical masking corpus |
| [RCAEval](https://github.com/phamquiluan/RCAEval) | 735 cases: RE1 375, RE2 270, RE3 90; 3.442 GB case-level HF Parquet or 5.2 GB Zenodo archives | root service, root indicator, fault, injection time | READY; HF/GitHub MIT, Zenodo CC-BY-4.0 | reproducibility baseline, cross-system root RCA, relation masking |
| [OntoLogX data](https://zenodo.org/records/17251494) | 193.9 KB; 70 selected AIT logs with CSV and manually annotated TTL splits | event triples, entity links, relation links | READY | event-KG population baseline only |
| [Nezha](https://github.com/IntelligentDDS/Nezha) | about 343 MB in the repository; 56 Online Boutique and 45 TrainTicket fault cases | service and inner-service RCA labels; trace IDs | READY, MIT | auxiliary relation recovery and RCA baseline |
| [DejaVu](https://zenodo.org/records/6955909) | 17.5 MB smallest processed split; 1.24 GB processed; `graph.yml`, `metrics.csv`, `faults.csv` | explicit fault-dependency graph and fault instances | CONDITIONAL; code is MIT but Zenodo data license is blank | exact topology masking control; metrics-only, so it cannot validate the DeBERTa log claim |
| [LO2v2](https://zenodo.org/records/18937117) | 115 runs × 54 tests; 65.6 GB raw, 70.0 GB all files; 50.7 MB index+source profile | run/test timing, source architecture, logs/metrics/traces | READY, data CC-BY-4.0 and code Apache-2.0 | topology recovery and observability-degradation stress tests |
| [Eadro data](https://zenodo.org/records/7615394) | 127.7 MB; TrainTicket and DeathStarBench SocialNetwork | fault/root annotations with logs, metrics, traces | CONDITIONAL | license metadata gate; useful compact auxiliary after clearance |
| [Multi-Source OpenStack](https://zenodo.org/records/3549604) | 650.5 MB in two checksummed archives | workload/fault scripts and Rally report; synchronized multi-source telemetry | CONDITIONAL | license/schema audit; non-microservice external relation evidence |
| [GAIA](https://github.com/CloudWise-OpenSource/GAIA-DataSet) | about 7.8 GB repository; two weeks, millions of logs and rich traces | anomaly injections and trace parentage | CONDITIONAL | repository LICENSE says GPL-2.0 while README says Apache-2.0 |
| [Loghub OpenStack](https://zenodo.org/records/8196385) | 5.4 MB compressed OpenStack log corpus | no topology or RCA label | READY for the Zenodo CC-BY-4.0 artifact | cheap log-parser/event-KG pretraining only; GitHub repository uses different custom terms |
| [OpenRCA 1.0](https://github.com/microsoft/OpenRCA) | 335 enterprise failures, over 68 GB, distributed through Google Drive | root component, reason, and time | CONDITIONAL; code is MIT but the external data has no separate license declaration | heavy external LLM RCA validation; Telecom lacks logs and topology gold is limited |
| [LEMMA-RCA](https://lemma-rca.github.io/) | 4.74–6.66 GB preprocessed; 33–53.6 GB original | incident/root labels over large log+metric systems | CONDITIONAL | license cards disagree; no trace topology |
| [OpenRCA 2.0](https://arxiv.org/abs/2606.27154) | paper only at audit time | claimed path-level supervision | BLOCKED | do not substitute OpenRCA 1.0 and call it 2.0 |

## Direct byte-level checks

### RCAEval smoke case

The default smoke profile is pinned to Hugging Face commit


`afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`

and downloads `cases.parquet` plus `re2tt_ts-auth-service_cpu_2`.

| File | Bytes | SHA-256 | Parsed shape |
|---|---:|---|---:|
| `cases.parquet` | 29,500 | `c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb` | 735 case index rows |
| `inject_time.txt` | 10 | `ecc8ac8295583809dc546a829b3079120d7762f44cca6a573aa03706f741b8b3` | scalar timestamp |
| `metrics.parquet` | 933,362 | `16597725c18258ce0a3bdedc1833fb52ad6638fdc5068e944dce23c1bbde6d93` | 1,441 × 368 |
| `logs.parquet` | 3,351,283 | `0cc2ed5ff13a20cf776a42d9d4c3914981afe025fbaa69109ac652d9c7502537` | 271,919 × 3 |
| `traces.parquet` | 20,050,343 | `3d704979b684c5450a3ddcd48bb91a71d07c485edc878a2969b04ff152b5857c` | 838,936 × 11 |

The HF repository is public and ungated, contains 2,080 files, and reports 3,441,988,225 bytes used storage. Frozen archive experiments must record that the Zenodo distribution uses different source-specific license metadata.

### RCABench/Aegis payload

The main file is `rcabench-absolute_anomaly.tar.gz`:

- exact size: 13,444,974,981 bytes;
- MD5: `96d32ae290e06be7d4e251be02b36cb8`;
- Zenodo API license: CC-BY-4.0.

A byte-range audit extracted and parsed the first complete case, `ts5-ts-order-service-stress-svfvxk`:

| Split/modality | Rows |
|---|---:|
| normal logs | 86,771 |
| abnormal logs | 89,695 |
| normal metrics | 69,940 |
| abnormal metrics | 69,392 |
| normal traces | 178,626 |
| abnormal traces | 188,533 |

`injection.json` contains service, pod, container, function, and metric fields. Trace parent IDs and Kubernetes service/pod/namespace attributes are present, making this the strongest current source for the proposed log-to-relation-to-RCA pipeline. It does not provide a published gold causal edge sequence, so cause paths still require an independent reference-graph construction and review protocol.

## RCAEval frozen archive checksums

| File | Size | MD5 |
|---|---:|---|
| `RE1-OB.zip` | 31.0 MB | `47cce26ed24140e8974e68f9db2a5e9c` |
| `RE1-SS.zip` | 79.1 MB | `d2b15cbd3bb3cf6ec5f3cc65f7fac225` |
| `RE1-TT.zip` | 279.7 MB | `48a26925ce47fd4bcfbedbae4f31475b` |
| `RE2-OB.zip` | 1.2 GB | `b9e23f8842c404b396ffd2becff15de4` |
| `RE2-SS.zip` | 245.6 MB | `bd747a8fc7c5be00c613e13fbf9dd74b` |
| `RE2-TT.zip` | 2.8 GB | `a7fbcd1ada406067dcc50771ae398408` |
| `RE3-OB.zip` | 190.8 MB | `96947589084348f9d12cba370313458e` |
| `RE3-SS.zip` | 101.5 MB | `467964eddb2a4fef4d6486c4679e749b` |
| `RE3-TT.zip` | 241.7 MB | `ae1eb5906fb4d13a16a6a5e58faaf30b` |

For relation recovery, prioritize Online Boutique and TrainTicket RE2/RE3. Sock Shop remains useful for non-trace baselines but must not be represented as trace-complete.

## Derived benchmark contract

Each normalized incident must produce these versioned artifacts:

```text
dataset_id/system/incident_id/
  incident.json                 # root/fault/time labels and label provenance
  entities.parquet              # canonical IDs, source aliases, entity type
  telemetry/logs.parquet
  telemetry/metrics.parquet
  telemetry/traces.parquet
  graph/reference_edges.parquet # independent held-out reference facts
  graph/observed_edges.parquet  # input evidence only
  masks/<policy>/<seed>.json    # removed edges and removed evidence aliases
  paths/gold.json               # only if independently built and reviewed
  provenance.json               # source URL, revision, hashes, parser version
```

Required edge fields are subject, predicate, object, layer, status, source, evidence IDs, time, extractor, and confidence. A missing `paths/gold.json` is valid and means that the incident participates only in root-level evaluation.

## Acquisition commands

All default operations are read-only:

```bash
python tools/datasets.py catalog
python tools/datasets.py audit
python tools/datasets.py plan rcaeval --profile smoke
python tools/datasets.py plan rcabench_aegis --profile artifact_smoke
```

An actual acquisition is explicit and records source provenance:

```bash
python -m pip install huggingface_hub pyarrow
python tools/datasets.py fetch rcaeval --profile smoke \
  --dest data/raw --accept-license --yes
python tools/datasets.py verify rcaeval --profile smoke \
  --path data/raw/rcaeval/smoke --parse-parquet
```

The full RCABench payload is intentionally not the first download. Begin with its approximately 2 MB artifact profile, then allocate storage and fetch the 13.4 GB telemetry only after the parser contract and split policy are frozen.

## Data gates before any reported result

- **Bytes:** the expected files exist and parse.
- **Identity:** source URL, immutable revision/record, byte size, and checksum are saved.
- **Rights:** the acquisition-source license is recorded; no raw dataset is redistributed from this repository.
- **Labels:** root, fault, time, topology, and path labels are marked independently as present, derived, reviewed, or absent.
- **Leakage:** folder names, injection commands, future events, target aliases, and inverse edges are excluded from model evidence.
- **Split:** repeated campaigns and topology edges stay in one split; leave-one-system-out is separate.
- **Smoke:** one incident passes schema, timestamp, cardinality, and modality checks before a full download.
- **Activation:** ablations must change predictions; zero DeBERTa/PSL flips fail the experiment gate.

## Known benchmark risks

- Root labels are more common than gold cause/impact paths. Never score a model-generated path against a path generated by the same model.
- Complete traces are both powerful evidence and a possible topology answer key. If traces define `G_ref`, the corresponding direct parentage fields must be held out from model input.
- Random edge masking overestimates real performance. Collector, component, identity, and path-critical masks are primary.
- Large public artifacts can silently change or disappear. Pin immutable revisions and retain checksums, but do not commit raw data.
- A permissive code license does not automatically prove the same license for an externally hosted dataset. The registry records source-specific evidence.
