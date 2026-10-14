# PTB-XL Range-Control Validation Report (Reconciled Canonical Record)

**Date**: 2026-10-15 | **Status**: Verified Canonical Record (Synchronized with `range_control_results.json`)

---

## 1. Discrepancy Audit and Reconciled Source of Truth

### Audit of Prior Discrepancy:
An inspection of the previous draft report (`PTBXL_RANGE_CONTROL_VALIDATION.md`) identified a localized inconsistency:
- **Identified Discrepancy**: The narrative summary in Section 1 and Section 4 of the prior draft contained preliminary estimates for Variants 4 and 5 (`[-2.22, +8.54]` and `[-2.22, +7.98], Acc=0.6976`), whereas Table 3 in the same file, `range_control_results.csv`, and `range_control_results.json` contained the verified canonical data.
- **Root Cause**: The narrative paragraphs in the draft were retained from an initial exploratory run prior to running multithreaded extraction on the full 21,388-record dataset. When `extract_median_centered.py` was executed across all official folds, the JSON, CSV, and Table 3 were correctly generated, but the narrative bullet points were not updated to reflect the full-dataset numbers.
- **Reconciliation Action**: Every metric has been independently recomputed directly from `ptbxl_binary_splits.npz` and `ptbxl_features_48_median_centered.npz` using the frozen scaler and exact CKKS objective ($C = 1 / (N \lambda) = 0.005853$). All metrics match `range_control_results.json` with zero error ($< 10^{-6}$). This document represents the reconciled, frozen source of truth.

---

## 2. Reconciled Executive Summary

| Preprocessing Variant | Validation Logit Range | Records $>8$ | Accuracy | ROC-AUC | F1 Score | Full-Val Deg-3 MaxAE | Full-Val MeanAE | Out-of-$[0,1]$ | In-Domain MaxAE ($|z|\le 8$) | Pass $\le 0.10$? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V1_original_unconstrained`** | `[-2.27, 60.12]` | `2` (0.09%) | `0.6752` | `0.7704` | `0.6667` | **`337.6549`** | `0.2127` | `18` | `0.0984` | FAIL |
| **`V2_original_train_minmax`** | `[-2.27, 8.08]` | `1` (0.05%) | `0.6752` | `0.7704` | `0.6667` | **`0.1268`** | `0.0554` | `18` | `0.0984` | FAIL |
| **`V3_median_centered`** | `[-5.49, 7.65]` | `0` (0.00%) | `0.7321` | `0.8247` | `0.7339` | **`0.0984`** | `0.0622` | `29` | `0.0984` | **PASS** |
| **`V4_median_centered_train_minmax`** | `[-4.41, 7.65]` | `0` (0.00%) | `0.7321` | `0.8247` | `0.7339` | **`0.0984`** | `0.0622` | `29` | `0.0984` | **PASS** |
| **`V5_median_centered_clip_pm5`** | `[-4.43, 6.66]` | `0` (0.00%) | `0.7335` | `0.8253` | `0.7349` | **`0.0984`** | `0.0627` | `23` | `0.0984` | **PASS** |

---

## 3. Extreme Logit Analysis (Task 1)

Top 5 Positive Validation Logits under the Saved Baseline ($C=1.0$, with intercept) vs. Bias-Free Reference ($\lambda=0.01$, no intercept):

| Rank | Saved Baseline ECG ID | Baseline Logit | Dominant Feature | Contribution | Bias-Free ECG ID | Bias-Free Logit | Dominant Feature | Contribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `6816` | `112.10` | `V4_mean` | `108.50` | `6816` | `60.12` | `V4_mean` | `57.37` |
| 2 | `13820` | `15.40` | `I_std` | `5.33` | `8081` | `8.08` | `V1_std` | `5.08` |
| 3 | `8081` | `11.92` | `V1_std` | `10.78` | `16359` | `6.32` | `V5_max` | `1.61` |
| 4 | `16359` | `11.27` | `V5_max` | `3.40` | `13820` | `6.15` | `AVL_std` | `1.51` |
| 5 | `6240` | `11.07` | `I_std` | `6.25` | `14887` | `5.76` | `V1_std` | `2.62` |

---

## 4. Frozen Candidate Specification (Part B)

### Primary Candidate: `Variant 3: Median-Centered (Unconstrained)`
- **Exact Centering Operation**: For each record $k$ and each lead $j \in \{1, \dots, 12\}$, subtract the 1000-sample lead median: $S^{\text{centered}}_{t, j} = S_{t, j} - \text{median}_{t=1}^{1000}(S_{t, j})$.
- **Statistical Feature Extraction**: Extract 4 statistics per lead (`mean`, `std` [$ddof=0$], `min`, `max`) producing 48 features in standard lead order: `I_mean, I_std, ..., V6_max`.
- **Scaler**: `StandardScaler` fitted **strictly on folds 1–8** (17,084 records). Saved as `standard_scaler_median_centered_train_only.joblib`.
- **Clamping**: None (unconstrained).
- **Polynomial Sigmoid**: Degree-3 Kim et al. ($c_0 = 0.5, c_1 = 0.15012, c_3 = -0.001593$) on $[-8, +8]$.
- **Validation Logit Range**: **`[-5.49, +7.65]`** (100% within $[-8, +8]$, zero tail violations).
- **Full-Validation MaxAE**: **`0.0984`** (strictly passes $\le 0.10$).
- **Full-Validation MeanAE**: **`0.0622`**.
- **Out-of-$[0, 1]$ Outputs**: Exactly **29 records** (1.35% of validation split). Documented explicitly: polynomial activations are real-valued mathematical approximations, not bounded probabilities.
- **Objective**: Bias-free ($w \in \mathbb{R}^{48}$, no intercept $b=0$), global gradient $g = \frac{1}{N} \sum_{b=1}^3 S_b + \lambda w$, $N=17,084, \lambda = 0.01$.

### Secondary Sensitivity Candidate: `Variant 5: Median-Centered + Clip [-5, +5]`
- Standardized features clamped to $[-5.0, +5.0]$ in plaintext by the Hospital before encryption.
- Validation logit range: **`[-4.43, +6.66]`** (100% within $[-8, +8]$).
- Full-Validation MaxAE: **`0.0984`**, MeanAE: **`0.0627`**, Out-of-$[0, 1]$ outputs: **23 records**.
- Accuracy: **0.7335**, ROC-AUC: **0.8253**, F1: **0.7349**.

---

## 5. CKKS Cryptographic Depth Feasibility

| Stage | Operation | Multiplicative Depth | Usable Budget | Remaining Margin |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-Encryption Centering (Hospital)** | Plaintext Local | 0 levels | 7 levels | 7 levels |
| **Forward Linear Combination ($X w$)** | Ciphertext x Ciphertext | 1 level | 7 levels | 6 levels |
| **Degree-3 Kim et al. Sigmoid ($P_3(z)$)** | Ct x Ct & Ct x Pt | 3 levels | 7 levels | 3 levels |
| **Error Alignment & Backward Gradient ($e \cdot X$)** | Ciphertext x Ciphertext | 1 level | 7 levels | 2 levels |
| **Total Pipeline Depth** | — | **5 levels** | **7 levels** | **+2 levels (FEASIBLE)** |

---

## 6. Certification for Phase 4 Scheduling

**CERTIFIED**: Variant 3 is fully frozen and certified for PTB-XL Phase 4 Scheduling Experiments.
- All validation logits are confined to $[-5.49, +7.65]$.
- Full-validation maximum polynomial error is strictly $\le 0.10$ ($0.0984$).
- Consumes 5 multiplicative levels, providing a verified 2-level depth margin.
- Test fold (Fold 10) was strictly excluded from all calibration and candidate selection decisions.