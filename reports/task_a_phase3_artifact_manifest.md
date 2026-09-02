# Task A Phase 3 재현 산출물 Manifest

- 실행 브랜치: `research/task-a-phase3-tristate`
- 실행 커밋: `e4634cd78bf11636a23ba99417416762f2f90dd5`
- GitHub Actions Run: `33594080712`
- Artifact ID: `9832926724`
- Artifact 이름: `task-a-phase3-e4634cd78bf11636a23ba99417416762f2f90dd5`
- Artifact 크기: `188,500 bytes`
- Artifact ZIP SHA-256: `1b5e5fbba01702eee359f729de478b19767d596bd88a3b1023f46385b9bdbab6`
- Artifact 만료일: `2026-10-02`
- 과학적 Gate: `D4_A3_HELDOUT_SHORTLIST_UTILITY = FAIL`

## Artifact 내용

```text
phase3/published/task_a_phase3_results.md
phase3/published/task_a_phase3_results.json
phase3/published/task_a_phase3_heldout_cells.csv
phase3/published/task_a_phase3_calibration_cells.csv
phase3/published/task_a_phase3_policy_grid.csv
phase3/published/task_a_phase3_status.txt
phase3/model_output/a3_candidate_evidence.parquet
phase3/evaluator_private/a3_candidate_analysis.parquet
```

GitHub Actions Artifact URL:

`https://github.com/dooyoung94/LogAgent_master/actions/runs/33594080712/artifacts/9832926724`

## 저장소 파일과 ZIP의 차이

- 저장소의 `task_a_phase3_policy_grid.csv`는 96개 정책을 유지하면서 핵심 평가 컬럼만 남긴 compact grid다.
- Artifact ZIP에는 모든 진단 컬럼을 포함한 원본 full grid와 Parquet 증거 파일이 들어 있다.
- 결과는 `PASS`가 아니라 재현 가능한 **negative result**다. Calibration에서 사전 기준을 만족한 정책이 0개였고, 동일 후보 수 A2-only 대조군 대비 DeBERTa의 추가 이득도 0이었다.
