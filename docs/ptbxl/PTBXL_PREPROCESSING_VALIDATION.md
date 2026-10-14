# PTB-XL Preprocessing Validation Report

## 1. Execution Environment & Provenance

- **Execution Date**: 2026-10-14
- **Operating System**: Windows 11 (AMD64)
- **Python Version**: Python 3.13.5 (MSC v.1943 64-bit)
- **Key Dependencies**:
  - `numpy`: 2.3.4
  - `pandas`: 2.3.3
  - `scikit-learn`: 1.7.2
  - `wfdb`: 4.3.1
  - `joblib`: 1.5.2
- **Dataset Path**: `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/` (resolved to internal extraction directory)
- **Validation Script**: `validate_ptbxl_preprocessing.py`

---

## 2. Execution Log & Status

### A. Phase C: 100-Record Smoke Test
- **Command**:
  ```powershell
  python preprocess_ptbxl.py "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" --max-records 100 --output data/ptbxl_smoke
  ```
- **Execution Result**: Exit code 0 (Completed in 6.2s).
- **Output Artifacts**:
  - `data/ptbxl_smoke/ptbxl_features_48.csv` (45 KB)
  - `data/ptbxl_smoke/ptbxl_binary_splits.npz` (22 KB)
  - `data/ptbxl_smoke/standard_scaler_train_only.joblib` (1.8 KB)
  - `data/ptbxl_smoke/preprocessing_summary.json` (2.7 KB)
- **Validation Command**:
  ```powershell
  python validate_ptbxl_preprocessing.py data/ptbxl_smoke
  ```
- **Validation Outcome**: **PASS** (All 12 critical assertions verified).

### B. Phase D: Full Preprocessing Run
- **Command**:
  ```powershell
  python -u preprocess_ptbxl.py "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" --output data/ptbxl_processed
  ```
- **Execution Result**: Exit code 0 (Completed in 10 minutes 40 seconds across 21,388 waveforms).
- **Output Artifacts**:
  - `data/ptbxl_processed/ptbxl_features_48.csv` (9.63 MB, 21,388 rows)
  - `data/ptbxl_processed/ptbxl_binary_splits.npz` (3.86 MB)
  - `data/ptbxl_processed/standard_scaler_train_only.joblib` (1.77 KB)
  - `data/ptbxl_processed/preprocessing_summary.json` (2.66 KB)
  - `waveform_read_failures.csv`: Not created (0 read failures).
- **Validation Command**:
  ```powershell
  python validate_ptbxl_preprocessing.py data/ptbxl_processed
  ```
- **Validation Outcome**: **PASS** (All 12 critical assertions verified).

---

## 3. Cohort Accounting and Data Breakdown

| Metric | Dataset-Wide Total | Smoke-Test Run (`--max-records 100`) | Full Preprocessing Run |
|:---|:---:|:---:|:---:|
| **Total Metadata Records** | 21,799 | 21,799 | 21,799 |
| **Total Unique Patients** | 18,869 | 18,869 | 18,869 |
| **Excluded (No Mapped Diagnosis)** | 411 | 411 (dataset-wide) | 411 |
| **Eligible Records in Dataset** | 21,388 | 21,388 (dataset-wide) | 21,388 |
| **Subset Selected for Run** | N/A | **100** | **21,388** |
| **Waveform Read Failures** | 0 | 0 | **0** |
| **Successfully Processed Records** | N/A | **100** | **21,388** |

---

## 4. Partitioning & Class Distribution (Official Patient-Aware Folds)

Data partitioning rigorously follows the official PTB-XL stratification (`strat_fold`). All records for any single patient are located exclusively in one fold.

| Split | Folds Included | Total Records | Unique Patients | Normal (Label 0) | Abnormal (Label 1) | % Abnormal |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Training** | Folds 1–8 | **17,084** | 14,823 | 7,243 | 9,841 | 57.60% |
| **Validation** | Fold 9 | **2,146** | 1,917 | 914 | 1,232 | 57.41% |
| **Test** | Fold 10 | **2,158** | 1,877 | 912 | 1,246 | 57.74% |
| **Total Processed** | All Folds 1–10 | **21,388** | **18,617** | **9,069** | **12,319** | **57.60%** |

### Patient Isolation Verification
- $\text{PatientSet}(\text{Train}) \cap \text{PatientSet}(\text{Val}) = \emptyset$ (Overlap = 0)
- $\text{PatientSet}(\text{Train}) \cap \text{PatientSet}(\text{Test}) = \emptyset$ (Overlap = 0)
- $\text{PatientSet}(\text{Val}) \cap \text{PatientSet}(\text{Test}) = \emptyset$ (Overlap = 0)
- **Data Leakage Status**: Zero patient leakage detected across splits.

---

## 5. Feature Matrix and Array Assertions

### Array Shapes and Data Types in `ptbxl_binary_splits.npz`:
- `X_train`: Shape `(17084, 48)`, dtype `float32`
- `y_train`: Shape `(17084,)`, dtype `int64`
- `X_val`: Shape `(2146, 48)`, dtype `float32`
- `y_val`: Shape `(2146,)`, dtype `int64`
- `X_test`: Shape `(2158, 48)`, dtype `float32`
- `y_test`: Shape `(2158,)`, dtype `int64`
- `feature_names`: Array of 48 strings matching `[I_mean, I_std, ..., V6_max]`

### Numerical Integrity:
- **Missing / Infinite Values**: Checked across all $21,388 \times 48 = 1,026,624$ feature elements. Exactly 0 `NaN`, `Inf`, or unphysical values found.
- **Label Integrity**: Checked across all labels. Labels strictly belong to $\{0, 1\}$.
- **Duplicate Records**: Checked across all `ecg_id` values. Exactly 0 duplicate IDs found.

### StandardScaler Verification:
- Scaler was fit on training folds 1–8:
  - Discrepancy between saved scaler mean and training fold mean: $\Delta \mu_{\max} = 7.10 \times 10^{-10}$
  - Discrepancy between saved scaler variance and training fold variance: $\Delta \sigma_{\max}^2 = 2.91 \times 10^{-9}$
- Transformed values: Maximum discrepancy between in-memory `X_train` and recomputed `scaler.transform(X_raw)` is $7.63 \times 10^{-6}$ (floating-point text round-trip tolerance through CSV).

---

## 6. Audit Issues, Diagnoses, and Fixes

During the Phase B code audit, two issues were diagnosed and resolved:

1. **Smoke-Test Fold Truncation Bug**:
   - *Issue*: `eligible.groupby("strat_fold").head(per_fold).head(args.max_records)` took a trailing global slice that cut off records from later folds (e.g. Folds 9 and 10), leaving validation or test splits starved.
   - *Fix*: Replaced with a deterministic, approximately balanced quota allocation loop over all unique folds. For `--max-records 100`, exactly 10 records are deterministically selected per fold.
2. **Ambiguous Smoke-Test Metadata Summary**:
   - *Issue*: In smoke-test runs, dataset-wide metadata counts (`metadata_records = 21799`) were reported alongside subset counts (`eligible = 100`) without clear labels, creating potential confusion.
   - *Fix*: Added explicit `dataset_wide_*` prefixes and `is_smoke_test` / `smoke_test_note` warnings to clearly distinguish full dataset metadata from smoke-test subsets.

---

## 7. Final Validation Verdict

```text
============================================================
VERDICT: PASS (ALL 12 CRITICAL ASSERTIONS SATISFIED)
============================================================
```

- **Output Files Intact**: Yes.
- **No Data Leakage**: Yes ($0$ patient overlap).
- **Exact 48 Features**: Yes.
- **Balanced Class Splits**: Yes (~42.4% normal, ~57.6% abnormal across all splits).
- **Waveform Read Failures**: 0.
- **Ready for Plaintext Baseline**: **YES**.
