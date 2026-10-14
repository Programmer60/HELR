# PTB-XL Preprocessing Pipeline: End-to-End Implementation and Execution Guide

## 1. Pipeline Overview

The preprocessing pipeline converts the raw PTB-XL (v1.0.3) database into a clean, leakage-free, standardized 48-feature binary classification task suitable for plaintext benchmarking and downstream CKKS homomorphic encryption.

```text
Raw PTB-XL (21,799 records)
       │
       ▼ [Step 1-2] Parse SCP-ECG codes & map to 5 diagnostic superclasses
 Diagnostic Mapping
       │
       ▼ [Step 3-4] Exclude 411 unmapped records; apply Binary Policy (NORM vs Abnormal)
 Eligible Records (21,388 records)
       │
       ▼ [Step 5-6] Load 100 Hz waveforms via WFDB (12 leads x 1,000 samples)
 Signal Validation (Verify 12 leads, check finiteness, enforce lead ordering)
       │
       ▼ [Step 7] Extract 4 summary statistics per lead (mean, std, min, max)
 48-Feature Matrix (D = 48)
       │
       ▼ [Step 8] Partition into official patient-aware folds (Train: 1-8, Val: 9, Test: 10)
 Folds 1–8 (Train) │ Fold 9 (Val) │ Fold 10 (Test)
       │                  │               │
       ▼ [Step 9] Fit StandardScaler ONLY on Training Folds 1–8
 Standardized Train│ Standardized Val│ Standardized Test
       │                  │               │
       └──────────────────┼───────────────┘
                          ▼ [Step 10]
           Save Compressed Outputs & Summary:
      - ptbxl_features_48.csv
      - ptbxl_binary_splits.npz
      - standard_scaler_train_only.joblib
      - preprocessing_summary.json
```

---

## 2. Step-by-Step Technical Implementation

### Step 1: Metadata Ingestion
- Ingests `ptbxl_database.csv` (21,799 rows) and `scp_statements.csv` (71 clinical statements).
- Asserts presence of required columns: `ecg_id`, `patient_id`, `strat_fold`, `filename_lr`, `scp_codes`.
- Asserts presence of `diagnostic` and `diagnostic_class` in `scp_statements.csv`.

### Step 2: SCP-ECG Parsing
- Parses the string-serialized Python dictionaries in `scp_codes` using `ast.literal_eval`.
- Filters dictionary keys against statements where `diagnostic == 1` and extracts the corresponding `diagnostic_class`.

### Step 3: Diagnostic Mapping & Binary Target Assignment
- Matches SCP codes against the 5 primary diagnostic superclasses: `NORM`, `MI`, `STTC`, `CD`, `HYP`.
- **Label 0 (Normal)**: Assigned if and only if all mapped diagnostic statements belong exclusively to `NORM`.
- **Label 1 (Abnormal)**: Assigned if at least one mapped diagnostic statement belongs to `MI`, `STTC`, `CD`, or `HYP`.
- **Mixed Cases (`NORM` + non-`NORM`)**: Assigned explicitly to **Label 1 (Abnormal)**. For instance, an ECG displaying baseline normal features alongside a minor non-specific repolarization abnormality (`STTC`) is screened as abnormal.

### Step 4: Record Filtering & Exclusion
- Records containing no statements that map to any of the 5 diagnostic superclasses (e.g., records annotated with pacemaker spikes or rhythm-only codes without morphological diagnoses) return `None`.
- Exactly **411 records** are excluded, leaving **21,388 eligible records**.

### Step 5: Waveform Ingestion (100 Hz Low Resolution)
- Reads low-resolution signals via `wfdb.rdsamp()` from relative paths defined in `filename_lr` (`records100/`).
- Sampling rate: 100 Hz. Signal duration: exactly 10.0 seconds ($1,000$ discrete time steps).

### Step 6: Signal Validation & Lead Ordering
- Validates that each signal has dimension `(1000, 12)`.
- Validates that all signal values are strictly finite (no `NaN`, `Inf`, or saturation errors).
- Asserts that the lead ordering matches across all records:
  `['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']`

### Step 7: Feature Extraction ($D = 48$)
For each lead channel $c \in [1 \dots 12]$, computes four statistical aggregates across the 1,000 time steps:
$$\text{mean}_c = \frac{1}{T} \sum_{t=1}^T s_{t, c}$$
$$\text{std}_c = \sqrt{\frac{1}{T} \sum_{t=1}^T (s_{t, c} - \text{mean}_c)^2} \quad (\text{ddof} = 0)$$
$$\text{min}_c = \min_{t=1 \dots T} s_{t, c}$$
$$\text{max}_c = \max_{t=1 \dots T} s_{t, c}$$

Features are ordered lead-by-lead:
`[I_mean, I_std, I_min, I_max, II_mean, II_std, ..., V6_max]`, yielding exactly **48 features**.

### Step 8: Patient-Aware Partitioning
Preserves the official PTB-XL stratification (`strat_fold`):
- **Training Set (Folds 1–8)**: 17,084 records (7,243 normal, 9,841 abnormal)
- **Validation Set (Fold 9)**: 2,146 records (914 normal, 1,232 abnormal)
- **Test Set (Fold 10)**: 2,158 records (912 normal, 1,246 abnormal)
- All records belonging to any individual patient are strictly contained within a single fold, ensuring $0\%$ patient leakage.

### Step 9: Leakage-Free Standardization
- Initializes `StandardScaler()`.
- Fits the scaler **exclusively on the training fold records (Folds 1–8)**:
  $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} X_i, \quad \sigma_{\text{train}}^2 = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} (X_i - \mu_{\text{train}})^2$$
- Transforms validation and test folds using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ without refitting.
- Persists the fitted scaler to `standard_scaler_train_only.joblib`.

### Step 10: Artifact Generation
Writes out the four standardized files:
1. `ptbxl_features_48.csv`: Raw unscaled features with metadata.
2. `ptbxl_binary_splits.npz`: Standardized feature arrays (`X_train`, `X_val`, `X_test`), integer labels (`y_train`, `y_val`, `y_test`), patient IDs, and feature names.
3. `standard_scaler_train_only.joblib`: Trained scikit-learn scaler.
4. `preprocessing_summary.json`: Comprehensive machine-readable accounting metadata.

---

## 3. The Smoke-Test Logic & Bug Fix

### The Problem in Prior Implementation
The original smoke-test selection logic used:
```python
per_fold = max(1, int(np.ceil(args.max_records / 10)))
eligible = (eligible.groupby("strat_fold", sort=True, group_keys=False)
            .head(per_fold).head(args.max_records).copy())
```
When `per_fold * 10 > args.max_records`, or when rows were sliced globally with `.head(args.max_records)`, the trailing global `head()` truncated records from later folds (e.g., Folds 9 and 10). Under certain subsets, Fold 10 (the test set) received zero records, causing downstream model evaluation to fail.

### The Corrected Implementation
In `preprocess_ptbxl.py`, we replaced this with a balanced, deterministic per-fold quota allocation:
```python
folds = sorted(eligible["strat_fold"].unique())
n_folds = len(folds)
base = args.max_records // n_folds
rem = args.max_records % n_folds
fold_subsets = []
for i, fold in enumerate(folds):
    quota = base + (1 if i < rem else 0)
    if quota > 0:
        fold_df = eligible[eligible["strat_fold"] == fold].sort_values("ecg_id").head(quota)
        fold_subsets.append(fold_df)
eligible = pd.concat(fold_subsets, axis=0).sort_values("ecg_id").reset_index(drop=True)
```
- For a 100-record smoke test, each of the 10 folds deterministically receives exactly 10 records.
- Train (Folds 1–8) receives 80 records, Validation (Fold 9) receives 10 records, and Test (Fold 10) receives 10 records.
- Within each fold, sorting by `ecg_id` guarantees exact reproducibility across runs.

---

## 4. Execution Commands

### A. Verify Dependencies
```powershell
python -c "import numpy, pandas, sklearn, wfdb, joblib; print('All dependencies available')"
```

### B. Run the 100-Record Smoke Test
```powershell
python preprocess_ptbxl.py "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" --max-records 100 --output data/ptbxl_smoke
```

### C. Validate Smoke-Test Outputs
```powershell
python validate_ptbxl_preprocessing.py data/ptbxl_smoke
```

### D. Run Full-Dataset Preprocessing (21,388 Records)
```powershell
python preprocess_ptbxl.py "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" --output data/ptbxl_processed
```

### E. Validate Full-Dataset Preprocessed Artifacts
```powershell
python validate_ptbxl_preprocessing.py data/ptbxl_processed
```
