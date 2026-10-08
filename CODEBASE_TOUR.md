# Codebase & Results Tour

This guide explains the repository structure, what each script does in detail, what computation happens inside it, and how data flows from raw MIMIC-III hospital files all the way to the final Homomorphic Encryption benchmarks.


---

## 1. `preprocessing/` Directory

**Purpose:** Tame the raw, massive, messy MIMIC-III clinical database and produce a single clean CSV that is ready for machine learning.

### `build_mimic3_dataset.py`

MIMIC-III's raw event tables are too large to load into RAM at once (CHARTEVENTS alone is 33 GB). This script solves that using **chunked reading** — processing the file in small pieces, extracting only the rows we need, then discarding the rest.

**Step-by-step what it does:**

1. **Load admissions and patients** (`ADMISSIONS.csv.gz`, `PATIENTS.csv.gz`): Calculates each patient's age at admission using the de-identified MIMIC-III dates. Ages above 89 are clipped to 90 (MIMIC-III privacy convention). Filters out anyone under 18 years old.

2. **Define the target label**: An admission is labelled `target = 1` (Ischemic Heart Disease positive) if any ICD-9 diagnosis code for that stay falls in the range 410–414. Otherwise `target = 0`.

3. **Define the approved ITEMIDs**: These are the specific numeric codes MIMIC-III uses for each clinical measurement. We approved two code sets — one for the older CareVue ICU system and one for the newer MetaVision system — and treat them as the same clinical feature.

4. **Chunk-process CHARTEVENTS** (vitals like Heart Rate, Blood Pressure, Temperature, SpO2): For each chunk, it keeps only rows whose `ITEMID` is in our approved list AND whose `CHARTTIME` falls within the first 24 hours of the patient's admission. It then applies **physiological plausibility bounds** to remove obvious data-entry errors (e.g., a Heart Rate of 0 or 999 is impossible). Temperature values recorded in Fahrenheit are converted to Celsius.

5. **Chunk-process LABEVENTS** (lab results like Sodium, Potassium, Creatinine, Hemoglobin, WBC, Platelets, Glucose): Same chunk-and-filter logic. Lab outliers like extreme WBC or Creatinine are deliberately **kept** for now (they may be clinically real extreme values) and flagged for the post-aggregation audit.

6. **Aggregate per admission**: For each `HADM_ID`, computes the **mean** of all valid measurements for each feature within the 24-hour window. This collapses the time-series event data into a single flat row per admission.

7. **Output**: Saves `mimic3_aggregated_features.csv` — a clean table with one row per admission, 49,303 rows total, containing 16 clinical features plus the target label.

---

## 2. `baseline/` Directory

**Purpose:** All experimental scripts, run sequentially in Phases. This is the heart of the research.

---

### Phase 1: `train_mimic3_lr.py` — Plaintext Baseline

**Goal:** Establish the maximum possible accuracy before any encryption is involved.

**What it does:**
1. Loads `mimic3_aggregated_features.csv`.
2. Performs a **patient-level 80/20 train/test split** with `seed=42`. This means if a patient had multiple admissions, ALL of their admissions land in the same split — guaranteeing zero data leakage between train and test.
3. Applies **median imputation** (fitted on train only) to fill missing values.
4. Applies **StandardScaler** (fitted on train only) to normalize all features to zero mean and unit variance.
5. Trains a **Logistic Regression** model using `sklearn`.
6. Evaluates on the held-out test set.

**Output:** Baseline performance locked as: Accuracy = 0.685, ROC-AUC = 0.7262. All future HE experiments are measured against this.

---

### Phase 2: `feature_selection.py` — Crypto-Aware Feature Trimming

**Goal:** In HE, every feature = one ciphertext = extra cost. Remove features that do not contribute to accuracy.

**What it does:**
1. Trains a **Random Forest** classifier and ranks all 16 features by importance.
2. Separately ranks all 16 features by their **Logistic Regression absolute coefficient magnitude**.
3. Tests the held-out accuracy of the Top-10 and Top-15 subsets for both ranking methods.

**Key finding:** The LR-coefficient Top-10 drops 6 features (WBC, Creatinine, Respiratory Rate, Sodium, Temperature, Platelets) but loses only **−0.0005 AUC**. We select this as our official feature set because it reduces the per-batch ciphertext count from 16 to 10 — a 37.5% reduction in cryptographic workload.

**Selected features:** AGE, Systolic BP, MAP, Diastolic BP, GENDER, Potassium, Heart Rate, Hemoglobin, Glucose, Oxygen Saturation.

---

### Phase 1+2 Export: `phase12_he_export.py` — The HE Airlock

**Goal:** Freeze the preprocessed data and trained model into pure numpy arrays that TenSEAL can work with directly.

**What it does:**
1. Re-applies the exact same imputer and scaler (fitted on training data only) to produce final `X_train`, `X_test` matrices.
2. Trains the final Logistic Regression using only the 10 selected features.
3. Saves everything to `results/he_ready/`:
   - `X_train.npy`, `X_test.npy` — scaled, imputed feature matrices
   - `y_train.npy`, `y_test.npy` — binary labels
   - `weights.npy`, `intercept.npy` — the trained LR model weights
   - `scaler.joblib`, `imputer.joblib` — saved preprocessing objects

This folder is the **bridge** between standard ML and HE. Every subsequent phase loads from here — no raw CSV is ever touched again.

---

### Phase 3: `phase3_poly_fit.py` — Custom Polynomial Sigmoid

**Goal:** HE cannot compute the standard `sigmoid(z) = 1 / (1 + e^-z)` because it requires an infinite Taylor series. We must approximate it with a low-degree polynomial.

**What it does:**
1. Loads `X_train.npy` and `weights.npy`, and computes the actual pre-sigmoid decision scores: `z = X_train @ weights + intercept`.
2. Analyses the distribution of these logit values. Finds that 99% of real MIMIC-III patient scores fall in the range `[-3.25, 1.00]`.
3. Fits a **Degree-3 polynomial** and a **Degree-5 polynomial** over a safe `[-6.0, 6.0]` interval using least-squares optimization.
4. Saves the coefficients (`C0, C1, C3, C5`) to `results/he_ready/mimic_polynomials.json`.

**Why this is novel:** Instead of using polynomial coefficients from a generic maths paper, we tailored the approximation specifically to our clinical data range. This guarantees the model stays accurate even for extreme outlier patients.

---

### Phase 5: `phase5_full_circuit_validation.py` — Encrypted Training Circuit Proof

**Goal:** Prove that the CKKS context `[60, 40, 40, 40, 40, 40, 40, 60]` can host a complete **forward + backward pass** — i.e., the full encrypted training loop from `HELR3.py` — without running out of modulus levels.

**What makes this phase special:** This is the ONLY phase that uses **Ciphertext × Ciphertext (Ct×Ct)** multiplication. This is because we simulate the encrypted training scenario (like `HELR3.py`) where the **weights are also encrypted**. The Cloud Server does not know `w` in this scenario, so it must compute `E(X) × E(W)` — a Ct×Ct operation that costs a full modulus level.

**Step by step:**

1. **Hospital step (simulate):** Encrypts the feature matrix columns AND broadcasts each weight as a repeated ciphertext — `enc_w[j] = E([w_j, w_j, ..., w_j])`.
2. **CSP Forward Pass (Ct×Ct):**
   - `enc_z = enc_X[0] * enc_w[0] + enc_X[1] * enc_w[1] + ...`  → Each `*` here is **Ct×Ct**
   - Apply Degree-5 polynomial to `enc_z` → 4 more Ct×Ct levels consumed
3. **CSP Backward Pass (Ct×Ct):**
   - Compute `enc_err = enc_p - enc_y` (error term)
   - Compute `enc_grad[j] = enc_err * enc_X[j]` → One more Ct×Ct
4. **Track levels at every step** using `ciphertext()[0].coeff_modulus_size()`.

**Result proved:** Starting with 7 usable levels, the full forward + backward circuit consumes exactly 6 levels, leaving **1 level remaining** for decryption. The gradient error vs plaintext is less than 0.000003 — 3 parts per million.

**This is the validation of `HELR3.py`'s architecture on the MIMIC-III parameters.**

---

### Phase 6: `phase6_scalability.py` — First Scalability Benchmark (500–5000)

**Goal:** Measure how HE inference time changes as we increase the number of patients from 500 to 5000.

**Critical shift in approach — Ciphertext × Plaintext (Ct×Pt):**

This is where we switch from **Encrypted Training** (Phase 5) to **Encrypted Inference**. The model is already trained. The Cloud Server knows the weights in plaintext. Therefore:

`enc_z = enc_X[0] * w[0] + enc_X[1] * w[1] + ...`  → Each `*` here is **Ct×Pt**

This costs NO extra modulus level because multiplying a ciphertext by a known plaintext scalar is essentially just rescaling. This gives us the depth budget to run the full Degree-5 polynomial.

**SIMD batching:** With `poly_modulus_degree = 16384`, each ciphertext holds up to 8,192 slots. So 500 patients and 5,000 patients fit inside the same single batch — the HE compute time is virtually identical. This was the key insight that justifies practical deployment.

---

### Phase 7: `phase7_scalability_full.py` — Extended Scalability (500 → 49,303)

**Goal:** Cross the SIMD slot boundary (N = 8,192) and prove that dataset size only adds batches — it does NOT change circuit depth.

**What it does:**
1. Combines `X_test` and `X_train` into a single 49,303-row pool.
2. Runs HE inference at workloads: 500, 1000, 2500, 5000, 8192, 10000, 16384, 20000, 32768, 49303.
3. For N > 8192, slices the data into multiple batches, processes each through the same fixed Degree-5 circuit, and concatenates results.
4. Verifies that `final_levels_remaining = 2` for every single workload size.

**Key result proved:** The SIMD slot boundary at N=8,192 creates a "step" in ciphertext count and runtime, but the circuit depth — the number of modulus levels consumed per batch — remains perfectly constant regardless of patient count.

---

### Phase 8: `phase8_feature_count.py` — Feature-Count Experiment

**Goal:** Prove that adding more clinical features increases ciphertext workload but does NOT increase circuit depth.

**What it does:**
1. Re-trains three separate LR models using 10, 15, and 16 features (using the same patient-level split and same seed).
2. Runs HE inference on a fixed 8,192-admission workload for each.
3. Evaluates all three models on the **exact same held-out 9,943-record test set**.

**Key result proved:**
- 10 features → 10 ciphertexts, 0.663s, 2 levels remaining
- 15 features → 15 ciphertexts, 0.879s, 2 levels remaining
- 16 features → 16 ciphertexts, 0.986s, 2 levels remaining

Feature count increases the number of `Ct×Pt` linear combination steps (cheap) but has zero impact on the polynomial circuit depth (expensive).

---

### Phase 9: `phase9_final_benchmark.py` — Full 49,303-Admission Benchmark

**Goal:** The final proof of concept. Run the complete MIMIC-III cohort through encrypted inference and generate all master result tables.

**What it does:**
1. **Task 1:** Runs Ct×Pt encrypted inference across all 49,303 admissions, using 7 SIMD batches and 70 total feature ciphertexts. Measures timing over 3 repetitions.
2. **Task 2:** Evaluates predictive fidelity on the original 9,943-record held-out test set. Compares Plaintext LR vs HELR side-by-side.
3. **Task 3:** Assembles the Master Research Table consolidating all three experiments.
4. **Task 4:** Generates final plots and saves all CSVs.

**Final result:** 49,303 encrypted patient records evaluated in **4.2 seconds**, with identical ROC-AUC (0.7257) to the plaintext baseline.

---

## 3. `results/he_ready/` Directory

**Purpose:** The cryptographic airlock — pure numpy matrices and model parameters frozen at the exact state needed by TenSEAL.

| File | What it is |
|:---|:---|
| `X_train.npy` | Imputed + scaled training features (39,360 × 10) |
| `X_test.npy` | Imputed + scaled held-out test features (9,943 × 10) |
| `y_train.npy` | IHD labels for training set |
| `y_test.npy` | IHD labels for held-out test set |
| `weights.npy` | Trained LR model weights (10,) |
| `intercept.npy` | Trained LR bias term (scalar) |
| `mimic_polynomials.json` | Degree-3 and Degree-5 polynomial coefficients |
| `scaler.joblib` | Fitted StandardScaler object |
| `imputer.joblib` | Fitted median imputer object |
| `train_ids.csv` | HADM_IDs in training set (for audit) |
| `test_ids.csv` | HADM_IDs in held-out test set (for audit) |

---

## 4. `results/` Directory

**Purpose:** Final outputs — CSVs, graphs, and tables from every experiment.

| File | What it shows |
|:---|:---|
| `master_experiment_table.csv` | All three experiments (degree, size, features) in one table |
| `plaintext_vs_he_metrics.csv` | Side-by-side LR vs HELR accuracy, precision, recall, F1, AUC |
| `mimic3_full_he_results.csv` | Timing for the final 49,303-admission run |
| `feature_selection_comparison.csv` | Why we dropped 6 features (−0.0005 AUC for 37.5% CT savings) |
| `scalability_full.csv` | All N=500 to N=49,303 timing data |
| `feature_count_results.csv` | 10 vs 15 vs 16 feature timing data |
| `final_runtime_scalability.png` | Runtime step-up at the SIMD slot boundary |
| `final_level_consumption.png` | Flat line proving depth is invariant to dataset size and feature count |
| `final_feature_count.png` | Runtime and CT count vs feature count |

---

## 💡 How to Walk Someone Through the Code (Study Order)

```
1. preprocessing/build_mimic3_dataset.py    ← Understand the raw data
2. baseline/train_mimic3_lr.py              ← Understand the plaintext target
3. baseline/feature_selection.py            ← Understand why we use 10 features
4. baseline/phase12_he_export.py            ← Understand the data freeze
5. baseline/phase3_poly_fit.py              ← Understand why we approximate sigmoid
6. HELR3.py                                 ← Understand full Ct×Ct encrypted TRAINING
7. baseline/phase5_full_circuit_validation  ← Prove HELR3 architecture fits our context
8. baseline/phase6_scalability.py           ← Switch to Ct×Pt encrypted INFERENCE
9. baseline/phase7_scalability_full.py      ← Scale across SIMD slot boundary
10. baseline/phase8_feature_count.py        ← Prove feature count ≠ depth change
11. baseline/phase9_final_benchmark.py      ← Final 49,303-patient proof of concept
```
