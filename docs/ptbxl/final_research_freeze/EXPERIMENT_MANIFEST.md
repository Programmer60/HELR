# Complete Experiment Manifest & Dataset Split Audit

**Document ID**: `docs/ptbxl/final_research_freeze/EXPERIMENT_MANIFEST.md`  
**Phase**: Phase 4B — Statistical Corrections, Evidence Audit & Research Freeze Finalization  
**Date**: October 31, 2026  

---

## 1. Dataset Integrity & Split Audit

### 1.1 Project-Specific Binary Classification Task Definition

The diagnostic prediction task is framed as a binary abnormality screening problem on the PTB-XL v1.0.3 dataset:
- **Class 0 (Normal)**: All mapped diagnostic statements on the record consist strictly of `NORM`.
- **Class 1 (Abnormal)**: At least one mapped diagnostic statement other than `NORM` is present (including cases where non-NORM statements co-occur with `NORM`).
- **Excluded**: Records lacking mapped diagnostic statements ($411$ records across the dataset) are excluded prior to feature normalization.
- *Scoping Note*: This is a project-specific binary screening formulation for evaluating privacy-preserving logistic regression; it is not an official PhysioNet benchmark competition endpoint.

### 1.2 Record Count Reconciliation

An exhaustive reconciliation between raw PhysioNet PTB-XL v1.0.3 metadata, processed split files, and report documentation was conducted to resolve historical reporting discrepancies:

```
+-----------------------------------+--------------------+--------------------+--------------------+--------------------+
| Data Source / Artifact            | Train (Folds 1-8)  | Val (Fold 9)       | Test (Fold 10)     | Total Records      |
+-----------------------------------+--------------------+--------------------+--------------------+--------------------+
| Raw Database (ptbxl_database.csv) | 17,418             | 2,183              | 2,198              | 21,799             |
| Excluded (Unlabeled / Unmapped)   |    334             |    37              |    40              |    411             |
| Eligible (preprocessing_summary)  | 17,084             | 2,146              | 2,158              | 21,388             |
| Processed Binary (binary_splits)  | 17,084             | 2,146              | 2,158              | 21,388             |
| Median Centered (features_48_med) | 17,084             | 2,146              | 2,158              | 21,388             |
| Phase 3B Training Execution Array | 17,084 (Trained)   | 2,146 (Validated)  | 2,158 (Excluded)   | 19,230 (Used)      |
| Phase 3B Markdown Prose Text      | 17,084             | 2,163 (Typo)       | 2,198 (Raw count)  | N/A                |
+-----------------------------------+--------------------+--------------------+--------------------+--------------------+
```

### 1.3 Resolution of Reporting Discrepancies
1. **Fold 10 Count Discrepancy**:
   - `PTBXL_FINAL_TEST_EVALUATION.md` correctly recorded **$2,158$** eligible test records.
   - The Phase 3B prose text erroneously cited **$2,198$**, which is the *raw* record count in `ptbxl_database.csv` before filtering out the 40 records lacking diagnostic statements.
   - The underlying processed array (`ptbxl_features_48_median_centered.npz`) contains exactly $2,158$ test records.
2. **Fold 9 Count Discrepancy**:
   - The eligible Fold 9 validation split contains **$2,146$** records ($914$ normal, $1,232$ abnormal).
   - In `SCHEDULER_PHASE3B_TRAINER_INTEGRATION.md`, the number "$2,163$" appeared as a typographical error in narrative text.
   - Runtime inspection confirms that `run_scheduler_training.py` executed validation on `strat_fold == 9`, which contains exactly **$2,146$ records**.
3. **Audit Verdict**:
   - All underlying binary `.npz` arrays and runtime evaluation loops are **100% consistent and identical**.
   - No data was leaked, corrupted, or re-split.
   - **Fold 10 was strictly excluded from Phase 3B model fitting, validation, hyperparameter tuning, and performance evaluation.**

---

## 2. Comprehensive Experiment Manifest (Phases 0–3B)

### 2.1 Phase 0: Repository Audit
- **Objective**: Establish baseline integrity, locate scheduling conditions, and verify non-modswitch intervention.
- **Key Artifacts**:
  - [`docs/ptbxl/SCHEDULER_PHASE0_AUDIT.md`](file:///C:/Users/mishr/Desktop/ML/venv/MinorProject/docs/ptbxl/SCHEDULER_PHASE0_AUDIT.md)
  - Identified Condition B intervention as `enc_err * 1.0` in `results/ptbxl/phase4/run_phase4_controlled_experiments.py`.

### 2.2 Phase 1: Microbenchmark Profiling & CostModel_v1
- **Objective**: Profile atomic CKKS operations at all admissible levels; construct interpretable analytical cost model.
- **Key Artifacts**:
  - Profiler: `benchmarks/profile_level_latencies.py`
  - Model: `src/scheduler/cost_model.py` (CostModel_v1)
  - Results: `results/ptbxl/scheduler/microbenchmark_results.json`
  - Validation: 8/8 tests pass; MAPE $< 2\%$ on calibration workload.

### 2.3 Phase 2: Held-Out Workload Grid
- **Objective**: Test CostModel_v1 generalization across an out-of-distribution grid:
  $$N \in \{5000, 10000, 17084\}, \quad D \in \{12, 24, 48\}, \quad B \in \{1, 2, 3\}$$
- **Key Artifacts**:
  - Grid Harness: `benchmarks/run_workload_grid.py`
  - Results: `results/ptbxl/scheduler/phase2/workload_grid_results.json`
  - Report: [`docs/ptbxl/SCHEDULER_PHASE2_HELDOUT_VALIDATION.md`](file:///C:/Users/mishr/Desktop/ML/venv/MinorProject/docs/ptbxl/SCHEDULER_PHASE2_HELDOUT_VALIDATION.md)
  - Finding: Absolute MAPE failed ($63\%$) due to persistent worker calibration vs fresh subprocess execution, but Decision Regret remained $< 3.29\%$.

### 2.4 Phase 3A: Runtime-Context-Aware CostModel_v2
- **Objective**: Characterize persistent vs fresh execution contexts; calibrate operation-specific scaling factors.
- **Key Artifacts**:
  - Context Profiler: `benchmarks/profile_execution_contexts.py`
  - Model: `src/scheduler/cost_model_v2.py` (CostModel_v2)
  - Results: `results/ptbxl/scheduler/phase3a/`
  - Report: [`docs/ptbxl/SCHEDULER_PHASE3A_COSTMODEL_V2.md`](file:///C:/Users/mishr/Desktop/ML/venv/MinorProject/docs/ptbxl/SCHEDULER_PHASE3A_COSTMODEL_V2.md)
  - Finding: Eliminates global factor $\gamma$; achieves $0.00\%$ decision regret across boundary workloads.

### 2.5 Phase 3B: End-to-End Trainer Integration & Matched Evaluation
- **Objective**: Parameterized encrypted training, policy selection, and matched multi-seed comparison.
- **Key Artifacts**:
  - Trainer: `results/ptbxl/scheduler/phase3b/run_scheduler_training.py`
  - Policy: `src/scheduler/workload_policy.py`
  - Test Suite: `tests/test_phase3b_scheduler.py` (26 tests)
  - Results: `results/ptbxl/scheduler/phase3b/phase3b_training_results.json`
  - Report: [`docs/ptbxl/SCHEDULER_PHASE3B_TRAINER_INTEGRATION.md`](file:///C:/Users/mishr/Desktop/ML/venv/MinorProject/docs/ptbxl/SCHEDULER_PHASE3B_TRAINER_INTEGRATION.md)
  - Matrix: 4 schedules $\times$ 3 seeds ($42, 123, 2026$) $\times$ 10 epochs = 12 runs, 120 epochs.
  - Pilot: 1 run (Seed 42, 3 epochs, dynamic `B-cache` plan selection and execution).
  - Finding: $1.589 \pm 0.200\times$ seed-matched speedup (`A-repeat` $\rightarrow$ `B-cache`), agreement with empirical oracle across all 3 seeds, identical validation accuracy ($0.6990 \pm 0.0029$) and ROC-AUC ($0.7842 \pm 0.0041$).
