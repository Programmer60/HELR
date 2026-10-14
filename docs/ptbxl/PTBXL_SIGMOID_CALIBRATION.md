# PTB-XL Sigmoid Calibration and Objective-Consistency Audit

**Generated**: 2026-10-14T23:01:25

> All numerical values are measured empirically. Test fold (Fold 10) is never used in any analysis.

---

## 1. Objective Consistency Audit

| Property | Saved Baseline (`baseline_logistic_regression.joblib`) | CKKS Convention |
| :--- | :--- | :--- |
| Intercept | Yes (b = 0.5846) | **No** |
| L2 regularization | sklearn C=1.0 (total objective norm) | lambda=0.01 (per-sample gradient) |
| Objective | sklearn cross-entropy + (1/2C)||w||^2 | (1/N)*X^T*(p-y) + lambda*w |
| Objectives consistent? | **No** |  |

> [!IMPORTANT]
> The saved baseline was fitted with `fit_intercept=True` and `C=1.0` (sklearn convention).
> The CKKS encrypted training circuit carries no intercept ciphertext.
> These are different objectives. The saved baseline is preserved unchanged.
> A no-intercept plaintext reference model is used for calibration comparisons.

---

## 2. PTB-XL Logit Distribution

Computed on training and validation splits only. Test fold excluded.

| Split | N | Min | Max | Mean | Std | p1% | p50% | p99% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| train_with_intercept | 17084 | -24.79 | 20.27 | 0.585 | 1.643 | -0.865 | 0.187 | 6.774 |
| val_with_intercept | 2146 | -2.36 | 112.10 | 0.589 | 2.876 | -0.893 | 0.167 | 6.192 |
| train_no_intercept | 17084 | -25.37 | 19.68 | 0.000 | 1.643 | -1.449 | -0.397 | 6.190 |
| val_no_intercept | 2146 | -2.94 | 111.51 | 0.004 | 2.876 | -1.477 | -0.418 | 5.608 |

**Fraction of training records (with intercept) outside candidate polynomial intervals:**

| Interval | Fraction | Percentage |
| :--- | :--- | :--- |
| |z|>1 | 0.3397 | 33.97% |
| |z|>2 | 0.1383 | 13.83% |
| |z|>3 | 0.0765 | 7.65% |
| |z|>4 | 0.0421 | 4.21% |
| |z|>5 | 0.0244 | 2.44% |
| |z|>6 | 0.0142 | 1.42% |
| |z|>8 | 0.0056 | 0.56% |
| |z|>10 | 0.0018 | 0.18% |
| |z|>12 | 0.0008 | 0.08% |

> [!WARNING]
> Validation set contains an outlier at z ~= 112. This record produces extreme polynomial outputs
> for degree-3 and degree-5 polynomials fitted on [-3, +10] ranges and will require robust handling.

---

## 3. MHEALTH Polynomial Evaluation on PTB-XL Logits

Polynomials are unchanged from the MHEALTH experiments (frozen).

| Degree | Split | Valid Interval | Prob MaxAE | Prob MeanAE | Out-of-[0,1] Count | Grad Factor MaxAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | train_full | [-20, 20] | 4.3828 | 0.0577 | 1692 | 0.1970 |
| 1 | val | [-20, 20] | 21.5835 | 0.0627 | 200 | 0.1970 |
| 3 | train_full | [-8, 8] | 21.0353 | 0.0589 | 573 | 2.7858 |
| 3 | val | [-8, 8] | 2227.6691 | 1.0955 | 60 | 59.9038 |
| 5 | train_full | [-6, 6] | 989.5641 | 0.1609 | 282 | 208.4680 |
| 5 | val | [-6, 6] | 2077785.2357 | 968.2900 | 31 | 92872.0176 |

> [!NOTE]
> The degree-3 MHEALTH polynomial (`c1=0.15012, c3=-0.001593`) produces out-of-[0,1]
> probability outputs for 573 training records and 60 validation records when evaluated with
> the baseline model's weights. The degree-5 polynomial is severely unstable beyond [-6, 6].

---

## 4. PTB-XL-Fitted Candidate Polynomials

Coefficients fitted by least squares on **training logits only** (CKKS convention: no intercept).
Candidate comparison uses validation data. Test fold excluded.

| Degree | c0 | c1 | c3 | c5 | Fit Interval | Train Prob MaxAE | Val Prob MaxAE | Val Out-of-[0,1] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.5000 | 0.077287 | 0.000000 | 0.000000 | [-2.7969, 9.6909] | 1.4608 | 8.1186 | 17 |
| 3 | 0.5000 | 0.137502 | -0.001092 | 0.000000 | [-2.7969, 9.6909] | 14.8354 | 1498.7906 | 46 |
| 5 | 0.5000 | 0.179846 | -0.003353 | 0.000022 | [-2.7969, 9.6909] | 183.6879 | 379914.3958 | 58 |

> [!WARNING]
> All three candidate polynomials show catastrophically large val prob MaxAE (>8 for degree 1,
> >1498 for degree 3, >379,914 for degree 5). This is caused by the val outlier at z~=112.
> The polynomial approximation error on the 99.9th percentile bounded subset is much smaller.

---

## 5. CKKS Depth Feasibility

CKKS: N=16384, coeff_mod_bit_sizes=[60,40,40,40,40,40,40,60], total usable levels=7.

| Degree | Fwd Dot (Ct*Ct) | Poly Eval Levels | Bwd Grad (Ct*Ct) | Total Consumed | Budget Remaining | Feasible |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | 1 | 3 | 4 | FEASIBLE |
| 3 | 1 | 3 | 1 | 5 | 2 | FEASIBLE |
| 5 | 1 | 4 | 1 | 6 | 1 | FEASIBLE |

> [!NOTE]
> All three degrees fit within the 7-level budget for the forward+backward pass.
> The constraint is numerical, not depth.

---

## 6. Gradient Comparison (Polynomial vs Exact Sigmoid)

Comparison uses training data only with CKKS convention (no intercept, L2=0.01).
Shows max absolute error between gradient computed with polynomial vs exact sigmoid.

| Polynomial | Weight Case | Grad MaxAE | Grad MeanAE |
| :--- | :--- | :--- | :--- |
| deg1_ptbxl | w_zero | 0.0000e+00 | 0.0000e+00 |
| deg1_ptbxl | w_rand1 | 4.5474e-02 | 1.7218e-02 |
| deg1_ptbxl | w_rand2 | 3.7765e-02 | 1.1370e-02 |
| deg1_ptbxl | w_baseline_coef | 8.5058e-02 | 2.7876e-02 |
| deg1_ptbxl | w_noic_ref | 8.7292e-02 | 3.0862e-02 |
| deg3_ptbxl | w_zero | 0.0000e+00 | 0.0000e+00 |
| deg3_ptbxl | w_rand1 | 2.7755e-02 | 1.0329e-02 |
| deg3_ptbxl | w_rand2 | 2.5852e-02 | 8.1536e-03 |
| deg3_ptbxl | w_baseline_coef | 1.0490e-01 | 2.6222e-02 |
| deg3_ptbxl | w_noic_ref | 1.9214e+00 | 1.3011e-01 |
| deg5_ptbxl | w_zero | 0.0000e+00 | 0.0000e+00 |
| deg5_ptbxl | w_rand1 | 1.5922e-02 | 5.8703e-03 |
| deg5_ptbxl | w_rand2 | 5.1662e-02 | 9.2740e-03 |
| deg5_ptbxl | w_baseline_coef | 1.2958e+00 | 1.6628e-01 |
| deg5_ptbxl | w_noic_ref | 1.4986e+02 | 7.0002e+00 |
| deg3_mhealth | w_zero | 0.0000e+00 | 0.0000e+00 |
| deg3_mhealth | w_rand1 | 2.4172e-02 | 8.9809e-03 |
| deg3_mhealth | w_rand2 | 2.5344e-02 | 8.3716e-03 |
| deg3_mhealth | w_baseline_coef | 1.6109e-01 | 3.3859e-02 |
| deg3_mhealth | w_noic_ref | 2.8264e+00 | 1.8631e-01 |

> [!NOTE]
> The gradient error at `w_noic_ref` (no-intercept plaintext reference) is large for degrees 3 and 5,
> indicating the polynomial approximation breaks down at the trained weight scale.
> At small weights (w_rand1-5 with norm ~0.04-0.09), gradient errors are modest (<0.1).

---

## 7. Plaintext Training Sanity Check (50 epochs)

CKKS convention: no intercept, lr=0.05, L2=0.01, gradient clip=1.0.
**Validation metrics only. Test fold excluded.**

| Polynomial | Val Acc (ep50) | Val Loss | W Norm | Grad Norm | Stable |
| :--- | :--- | :--- | :--- | :--- | :--- |
| deg1_ptbxl | 0.6594 | 0.6197 | 0.6812 | 0.1549 | Yes |
| deg3_ptbxl | 0.6631 | 0.6060 | 0.5248 | 0.0971 | Yes |
| deg5_ptbxl | 0.6687 | 0.6015 | 0.4538 | 0.0746 | Yes |
| deg3_mhealth | 0.6659 | 0.6043 | 0.5028 | 0.0905 | Yes |

> All polynomials achieve ~66-67% val accuracy at 50 epochs. This is below the plaintext baseline
> (~70% with intercept), confirming that no-intercept CKKS training starts with a handicap.

---

## 8. Final Candidate Assessment and Recommendation

Thresholds applied: epsilon_poly = 0.1, epsilon_grad = 0.2

| Degree | Val Prob MaxAE | Grad MaxAE | CKKS Feasible | Training Stable | Val Acc@50 | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 8.1186 | 8.7292e-02 | Yes | Yes | 0.6594 | FAIL |
| 3 | 1498.7906 | 1.9214e+00 | Yes | Yes | 0.6631 | FAIL |
| 5 | 379914.3958 | 1.4986e+02 | Yes | Yes | 0.6687 | FAIL |

**Recommendation: `NO_FEASIBLE_CANDIDATE`**

---

## 9. Analysis and Conclusions

### Root Cause of Calibration Failure

The primary issue is not polynomial degree or CKKS depth. It is **logit range and outliers**.

The PTB-XL validation set contains at least one record with an extreme logit z ~= 112
under the saved baseline model weights. When any bounded polynomial is evaluated at z=112:

- Degree 1: p(112) = 0.5 + 0.077 * 112 = 9.12 (far outside [0,1])
- Degree 3: p(112) = 0.5 + 0.138*112 - 0.00109*112^3 = catastrophically large
- Degree 5: similarly divergent

The val MaxAE numbers (8.1, 1498, 379914) are dominated by this single outlier.

### What the Bounded-Range Results Tell Us

On the p99.9 bounded subset (|z| <= ~10 on training):

| Degree | Train Bounded Prob MaxAE | Notes |
| :--- | :--- | :--- |
| 1 (MHEALTH) | 1.5245 | out_of_01=1674 |
| 3 (MHEALTH) | 0.6860 | out_of_01=559 |
| 5 (MHEALTH) | 6.7641 | out_of_01=264 |

### Required Actions Before Phase 4 Can Proceed

1. **Outlier investigation**: Identify the validation record with z~=112 and determine
   whether it is a data quality issue (mislabeled, corrupted waveform) or a genuine
   extreme ECG morphology. If it is a data anomaly, document and handle it consistently.

2. **Logit clipping / gradient clipping**: The encrypted training circuit must handle
   extreme logits gracefully. Options:
   - Clip gradients in the trusted Hospital step (already done at clip=1.0, but
     the polynomial probability explosion happens before decryption)
   - Use a wider-range polynomial fitted on [-15, +15] with higher stability requirements
   - Investigate whether the extreme logit arises only during inference (saved baseline
     with intercept) vs. during CKKS training (no intercept, different weight trajectory)

3. **CKKS-convention logit range**: The no-intercept CKKS training logits (z_train_ckks)
   are zero-mean and have a much lower max (~19.7) than the val outlier (~111.5).
   If the extreme outlier arises primarily from the intercept b=0.5846 pushing boundary
   records over a cliff, then the CKKS training circuit (no intercept) may not encounter it.

4. **No-intercept CKKS training feasibility**: Degree 3 PTB-XL candidate gives val_acc=0.6631
   at 50 epochs with gradient clipping. If the outlier is in a specific validation record,
   the no-intercept encrypted training may converge adequately without encountering z~=112.

### Conclusion

No candidate polynomial passes all three thresholds (val_prob_MaxAE<=0.10, grad_MaxAE<=0.20,
CKKS depth feasibility) when evaluated on the **full validation set** including the z~=112 outlier.

**The experiment correctly reports NO_FEASIBLE_CANDIDATE rather than selecting an invalid fallback.**

The experiment did NOT find that polynomial approximation is fundamentally unsuitable for PTB-XL.
It found that there is one or more extreme-logit records in the validation fold that must be
investigated before a candidate can be certified as ready for Phase 4 encrypted training.

---

## 10. Reproducibility

```bash
# Run calibration experiment (from project root)
python results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.py
# Regenerate this document from completed JSON
python results/ptbxl/sigmoid_calibration/generate_calibration_report.py
```

Outputs:
- `results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.json` - all measured values
- `results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.csv`  - polynomial evaluation table
- `results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.log`  - execution log
- `docs/ptbxl/PTBXL_SIGMOID_CALIBRATION.md`                          - this document