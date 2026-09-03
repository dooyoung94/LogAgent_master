# Task A Phase 3-R3 결과 — Evidence별 DeBERTa Tri-state

- 최종 과학적 Gate: **FAIL**
- Calibration feasible 정책: **0 / 256**
- 선택 정책: `{'retention_fraction': 0.85, 'minimum_keep': 8, 'operational_weight': 0.0, 'nli_weight': 0.1}`
- 미통과 조건: `CALIBRATION_POLICY_FEASIBLE, MRR_NONINFERIOR_TO_FULL_A2`
- 프로토콜: **개발 재검증** — 신규 독립 Incident 확인시험이 별도로 필요함

## Held-out 40 Cell

| 지표 | A2 전체 | A3-R3 | A2 전체 대비 | 동일 크기 A2 대비 | 동일 크기 R2 대비 |
|---|---:|---:|---:|---:|---:|
| Recall Macro | 1.0000 | 0.9989 | -0.0011 | +0.0011 | +0.0011 |
| Recall Minimum | 1.0000 | 0.9545 | - | - | - |
| 후보 수 평균 | 19.375 | 17.025 | -2.350 | +0.000 | +0.000 |
| P-LB Macro | 0.7234 | 0.8090 | +0.0856 | +0.0011 | +0.0011 |
| MRR Macro | 1.0000 | 0.9989 | -0.0011 | +0.0011 | +0.0011 |

## NLI 채널 진단

```json
{
  "backend": {
    "actual_sha256": "03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae",
    "backend": "OnnxDebertaNLIBackend",
    "batch_size": 1,
    "expected_sha256": "03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae",
    "label_to_index": {
      "contradiction": 0,
      "entailment": 1,
      "neutral": 2
    },
    "local_files_only": true,
    "max_length": 512,
    "model_dir": "/home/runner/work/LogAgent_master/LogAgent_master/data/models/nli-deberta-v3-small",
    "onnx_path": "/home/runner/work/LogAgent_master/LogAgent_master/data/models/nli-deberta-v3-small/onnx/model_quint8_avx2.onnx",
    "performance_mode": false,
    "providers": [
      "CPUExecutionProvider"
    ],
    "research_valid": true,
    "revision": "fa2804872c3b4bd748f38c0185cc85775361e735",
    "telemetry_disabled": true,
    "truncation_policy": "reject_over_budget"
  },
  "cache": {
    "hits": 4358,
    "inference_batches": 5642,
    "misses": 5642,
    "size": 5642
  },
  "candidate_count": 1250,
  "candidate_coverage": 1.0,
  "channel_coverage": {
    "http": 1.0,
    "operation": 1.0,
    "role": 1.0,
    "trace": 1.0
  },
  "maximum_pair_tokens": 135,
  "minimum_pair_tokens": 89,
  "nli_score_mean": -0.05074841949689806,
  "nli_score_std": 0.04640515209040336,
  "pair_count": 10000,
  "pairs_per_available_channel": 2,
  "state_counts": {
    "http": {
      "ambiguous": 96,
      "contradicts": 1154,
      "corroborates": 0,
      "unavailable": 0
    },
    "operation": {
      "ambiguous": 1239,
      "contradicts": 11,
      "corroborates": 0,
      "unavailable": 0
    },
    "role": {
      "ambiguous": 428,
      "contradicts": 372,
      "corroborates": 450,
      "unavailable": 0
    },
    "trace": {
      "ambiguous": 500,
      "contradicts": 750,
      "corroborates": 0,
      "unavailable": 0
    }
  },
  "truncation_count": 0
}
```

## 해석 규칙

- Trace, Operation, HTTP, Role은 각각 독립 Premise로 평가한다.
- 채널이 실제 데이터에 없으면 `unavailable`로 남기고 사실을 생성하지 않는다.
- `contradicts`는 순위 Feature일 뿐 후보를 단독 삭제하지 않는다.
- PASS는 동일 후보 수의 A2-only뿐 아니라 **동일 후보 수의 R2 operational control보다도** P-LB 또는 MRR이 실제 개선됐다는 뜻이다.
- `CALLS` 복원은 runtime 구조 관계이며 causal `CAUSES` 복원을 의미하지 않는다.
