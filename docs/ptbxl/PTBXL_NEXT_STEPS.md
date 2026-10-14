# PTB-XL Next Steps: Roadmap Toward Privacy-Preserving Cardiac Evaluation

## 1. Overview and Progression

This roadmap outlines the systematic transition from PTB-XL preprocessing (Step 4, completed) to plaintext baseline validation, encrypted gradient validation, and controlled modulus-level-aware CKKS training.

---

## 2. Status Summary: Completed vs. Planned Steps

| Step | Objective | Status | Artifact / Deliverable |
|:---:|:---|:---:|:---|
| **Step 1** | Inspect repository, dataset root, WFDB files, and dependencies | **COMPLETED** | Verified PTB-XL v1.0.3 and Python environment |
| **Step 2** | Audit preprocessing script, diagnose fold-allocation bug, clarify summary counts | **COMPLETED** | `preprocess_ptbxl.py` patched and audited |
| **Step 3** | Execute 100-record smoke test across 10 official folds | **COMPLETED** | `data/ptbxl_smoke/` generated and verified |
| **Step 4** | Execute full-dataset preprocessing (21,388 records) and independent assertions | **COMPLETED** | `data/ptbxl_processed/` (48 features, 0 failures, PASS) |
| **Step 5** | Review class balance and train plaintext Logistic Regression baseline | **PLANNED** | `baseline_ptbxl.py` evaluation (`baseline_metrics.json`) |
| **Step 6** | Establish numerical-gradient and SIMD slot packing tests ($D = 48$) | **PLANNED** | Gradient numerical error validation ($\text{MaxAE} \le 10^{-4}$) |
| **Step 7** | Adapt CKKS encrypted-training pipeline to the 48-feature PTB-XL representation | **PLANNED** | Column-packed CKKS vectors ($D=48$) |
| **Step 8** | Execute Phase 4 controlled modulus-level validation on PTB-XL (Conditions A, B, C) | **PLANNED** | Stage-wise profiling on cardiac ECG data |
| **Step 9** | Evaluate clinical classification metrics, runtime speedups, and generalization | **PLANNED** | Phase 5 research report comparing MHEALTH vs PTB-XL |

---

## 3. Detailed Planned Steps

### Step 5: Plaintext Baseline Training (`baseline_ptbxl.py`)
- **Objective**: Establish the unencrypted empirical performance ceiling on the 48-feature PTB-XL task.
- **Execution**: Run `python baseline_ptbxl.py data/ptbxl_processed` (using scikit-learn's `LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)`).
- **Deliverables**:
  - Training, validation, and test classification metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
  - Confusion matrix and baseline model artifact (`baseline_logistic_regression.joblib`).
- **Safety Rule**: Plaintext baseline must remain frozen as the reference standard before any encrypted training begins.

### Step 6: Numerical Gradient and SIMD Slot Packing Tests
- **Objective**: Validate numerical stability and precision of CKKS encrypted gradients on 48 features.
- **SIMD Layout**: With $N=16384$, TenSEAL provides $8,192$ complex SIMD slots.
  - The PTB-XL training set has $17,084$ records, which exceeds a single ciphertext's slot capacity ($8,192$).
  - Two architectural options must be evaluated:
    1. *Subsampling / Batched Training*: Mini-batch training with batches of $P \le 8,192$ samples.
    2. *Two-Ciphertext Partitioning*: Dividing the 17,084 records into $\lceil 17084 / 8192 \rceil = 3$ ciphertext chunks per feature.
- **Gradient Precision**: Compare encrypted gradients against plaintext analytical gradients across $w=0$, random weights, and intermediate states. Assert $\text{MaxAE} \le 1.0 \times 10^{-4}$.

### Step 7: Adaptation of the CKKS Training Pipeline
- **Objective**: Configure the hospital–cloud encrypted training loop for $D=48$ cardiac features.
- **Parameters**: Retain proven CKKS context:
  - Polynomial modulus degree: $N = 16384$
  - Scale: $2^{40}$
  - Coefficient modulus bit sizes: `[60, 40, 40, 40, 40, 40, 40, 60]`
- **Model Parameters**: 48 encrypted weight vectors $\text{Enc}(w_j)$, 1 encrypted label vector $\text{Enc}(y)$, and 48 feature ciphertexts $\text{Enc}(X_j)$.

### Step 8: Controlled Modulus-Level-Aware Circuit Scheduling on PTB-XL
- **Objective**: Test whether the modulus-level scheduling principle validated on MHEALTH in Phase 4 holds on 48-feature cardiac ECG telemetry.
- **Conditions**:
  - **Condition A (Control — Original Degree 3)**: Gradient computation begins at **Level 3**.
  - **Condition B (Intervention — Degree 3 with Controlled Level Adjustment)**: Prediction error ciphertext is dropped to **Level 2** via a single scalar rescale (`enc_err * 1.0`), and features are aligned to Level 2.
  - **Condition C (Control — Original Degree 5)**: Gradient computation naturally begins at **Level 2**.
- **Measurements**:
  - Measure pure $O(D)$ gradient computation runtime across all 3 conditions.
  - Test hypothesis: Does Condition B achieve gradient runtime parity with Condition C on PTB-XL, saving $\approx 40\%$ gradient time compared to Condition A?

### Step 9: Comparative Analysis and Publication Synthesis
- Compare the MHEALTH wearable telemetry results against the PTB-XL clinical ECG results.
- Synthesize findings into the final research thesis:
  1. *Algorithmic Contribution*: Modulus-level-aware circuit scheduling accelerates encrypted logistic regression training by up to $45\%$ in the gradient stage without modifying model parameters or losing numerical fidelity.
  2. *Domain Contribution*: Validated across both wearable motion kinematics (MHEALTH) and clinical multi-lead cardiac electrophysiology (PTB-XL).
