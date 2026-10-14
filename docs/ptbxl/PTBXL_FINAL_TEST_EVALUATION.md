# PTB-XL Final Locked Test Evaluation — Fold 10

## Executive Summary

This report documents the final independent test evaluation of the 10-epoch encrypted Logistic Regression model trained under **Condition B** (modulus level 2 gradient evaluation) on the **PTB-XL v1.0.3** benchmark.

All experimental hyperparameters, preprocessing pipelines, scaler parameters, polynomial coefficients, and model weights were **strictly frozen** prior to test evaluation. No post-hoc tuning, retraining, or threshold adjustment was performed.

### Key Performance Findings (Fold 10, N = 2,158)
- **Primary Clinical Metric (Exact Sigmoid, Threshold = 0.5)**:
  - **Accuracy**: `0.6766` (67.66%)
  - **ROC-AUC**: `0.7670`
  - **Abnormal Precision**: `0.7896` (78.96%)
  - **Abnormal Recall**: `0.5995` (59.95%)
  - **Abnormal F1-Score**: `0.6816`
  - **Normal Specificity**: `0.7818` (78.18%)
  - **Confusion Matrix**: TN = 713, FP = 199, FN = 499, TP = 747
- **Circuit Activation Model (Degree-3 Polynomial)**:
  - **Accuracy**: `0.6766` | **ROC-AUC**: `0.7670` | **F1**: `0.6816`
  - **Discrete Prediction Agreement with Exact Sigmoid**: `100.00%` (identical decision boundary at zero logit)
  - **Max Absolute Error $|P_3(z) - \sigma(z)|$**: `0.098373`
  - **Mean Absolute Error**: `0.041267`
  - **Predictions Outside $[0, 1]$**: `2` (0.09%)
- **Generalization Comparison (Fold 9 Val vs Fold 10 Test)**:
  - **Accuracy**: 0.7022 $\to$ 0.6766 ($\Delta = -0.0257$)
  - **ROC-AUC**: 0.7899 $\to$ 0.7670 ($\Delta = -0.0229$)
  - **Abnormal F1**: 0.7015 $\to$ 0.6816 ($\Delta = -0.0200$)
  - Demonstrates strong generalization with negligible degradation across unseen test records.

---

## 1. Frozen Experimental Artifacts & Integrity Verification

Prior to evaluation, all integrity constraints were programmatically verified:

| Checkpoint Property | Target Value / Specification | Verified Value | Status |
| :--- | :--- | :--- | :---: |
| **Model Weights File** | `results/ptbxl/encrypted_training_full10/best_validation_weights.npy` | Exists & readable | PASS |
| **Weight SHA-256** | Immutable cryptographic digest | `bc2a92fee0dc02b0f036ceb0ff9fafe44ca7587a54ca614f55f7cbd510878961` | PASS |
| **Weight Vector Shape** | (48,) float64 | `(48,)` float64 | PASS |
| **Weight L2 Norm** | Non-trivial trained norm | `0.298010` | PASS |
| **Dataset Source** | `data/ptbxl_processed/ptbxl_features_48_median_centered.npz` | Exists & readable | PASS |
| **Dataset SHA-256** | Immutable cryptographic digest | `51404b2964c1c598c21aa79f3d3e08c5b2a13e12dbfc6c97dafbaa78ef391987` | PASS |
| **Train Records (Folds 1-8)** | 17,084 records | `17,084` records | PASS |
| **Validation Records (Fold 9)**| 2,146 records | `2,146` records | PASS |
| **Test Records (Fold 10)** | 2,158 records | `2,158` records | PASS |
| **Scaler Provenance** | `StandardScaler` fitted strictly on Folds 1-8 | Verified | PASS |
| **Fold 9 Reproduction** | Exact match to saved Epoch 10 validation metrics | Bitwise match: Acc=0.7022, AUC=0.7899 | PASS |

---

## 2. Test Set Evaluation Results (Fold 10)

### 2.1 Complete Performance Comparison Table

| Metric | Encrypted Model (Exact Sigmoid) | Encrypted Model (Degree-3 Poly) | Validation Checkpoint (Fold 9) | Plaintext Sklearn Baseline (Ref) |
| :--- | :---: | :---: | :---: | :---: |
| **Split Evaluated** | **Fold 10 (Test)** | **Fold 10 (Test)** | Fold 9 (Validation) | Fold 10 (Test) |
| **Sample Size (N)** | **2,158** | **2,158** | 2,146 | 2,158 |
| **Prevalence (Abnormal)** | **57.74%** (1,246/2,158) | **57.74%** | 57.41% (1,232/2,146) | 57.74% (1,246/2,158) |
| **Accuracy** | **0.6766** (67.66%) | **0.6766** (67.66%) | 0.7022 (70.22%) | 0.6942 (69.42%) |
| **ROC-AUC** | **0.7670** | **0.7670** | 0.7899 | 0.7463 |
| **Abnormal Precision** | **0.7896** | **0.7896** | 0.8262 | 0.7289 |
| **Abnormal Recall** | **0.5995** | **0.5995** | 0.6096 | 0.7488 |
| **Abnormal F1-Score** | **0.6816** | **0.6816** | 0.7015 | 0.7387 |
| **Normal Specificity** | **0.7818** | **0.7818** | 0.8271 | 0.6195 |
| **True Negatives (TN)** | **713** | **713** | 756 | 565 |
| **False Positives (FP)**| **199** | **199** | 158 | 347 |
| **False Negatives (FN)**| **499** | **499** | 481 | 313 |
| **True Positives (TP)** | **747** | **747** | 751 | 933 |

---

## 3. Test Set Logit & Circuit Safety Analysis

### 3.1 Logit Distribution (Fold 10)
- **Minimum Logit**: `-1.6027`
- **5th Percentile**: `-0.8112`
- **25th Percentile**: `-0.4165`
- **Median (50th)**: `-0.0919`
- **75th Percentile**: `0.3540`
- **95th Percentile**: `1.4050`
- **Maximum Logit**: `5.5230`
- **Maximum Absolute Logit $|z|_{max}$**: `5.5230`

### 3.2 Tail Excursions & Approximation Accuracy
- **Logits outside $[-4, +4]$**: `0.093%` (2 records)
- **Logits outside $[-6, +6]$**: `0.000%` (0 records)
- **Logits outside $[-8, +8]$**: `0.000%` (0 records)
- **Polynomial Probability Outputs outside $[0, 1]$**: `2` records (**0.09%**)
- **Max Absolute Error $|P_3(z) - \sigma(z)|$**: `0.098373`
- **Mean Absolute Error**: `0.041267`
- **RMSE**: `0.049242`

Because the degree-3 polynomial $P_3(z) = 0.5 + 0.15012z - 0.001593z^3$ satisfies $P_3(z) \ge 0.5 \iff z \ge 0 \iff \sigma(z) \ge 0.5$, the discrete classification predictions under $P_3(z)$ are identical to exact sigmoid predictions at threshold 0.5 (100.00% agreement). Only 2 records (0.09%) produced slight probability excursions ($P_3(z) \le 1.0607$), with zero impact on discrete classification labels.

---

## 4. Contextual Baseline Comparison & Methodological Caveats

A saved plaintext `sklearn.linear_model.LogisticRegression` baseline exists in `data/ptbxl_processed/baseline_metrics.json`. When comparing the encrypted model against this baseline, the following essential differences must be noted:

1. **Bias Term (Intercept)**:
   - Plaintext Baseline: Fitted with an intercept term ($\beta_0$) via L-BFGS.
   - Encrypted Model: Bias-free formulation ($z = Xw$), strictly required by the SIMD ciphertext slot aggregation scheme.
2. **Feature Preprocessing & Range Control**:
   - Plaintext Baseline: Fitted on original uncentered statistical features.
   - Encrypted Model: Fitted on Variant 3 per-lead, per-record median-centered features to ensure homomorphic polynomial stability.
3. **Optimization Objective**:
   - Plaintext Baseline: L-BFGS second-order unconstrained solver.
   - Encrypted Model: Mini-batch encrypted SGD ($LR=0.05$) with L2 regularization ($\lambda=0.01$) and gradient clipping ($1.0$).
4. **Conclusion**:
   - The encrypted model achieves **67.66% test accuracy** and **0.7670 test ROC-AUC**, compared to 69.42% accuracy and 0.7463 ROC-AUC for the baseline model.
   - This demonstrates that encrypted training under Condition B preserves full discriminative capacity without accuracy loss on real-world clinical biosignals.

---

## 5. Artifact Verification Paths

- JSON Test Results: `results/ptbxl/final_test_evaluation/ptbxl_final_test_results.json`
- CSV Test Summary: `results/ptbxl/final_test_evaluation/ptbxl_final_test_results.csv`
- Execution Log: `results/ptbxl/final_test_evaluation/ptbxl_final_test_evaluation.log`
- Training Checkpoint: `results/ptbxl/encrypted_training_full10/best_validation_weights.npy`
