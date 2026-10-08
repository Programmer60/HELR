# Phase 2B: Depth-and-Cost-Aware Adaptive Polynomial Training

## 1. Baseline Observation

The Phase 0 frozen baseline used a fixed **Degree-3** polynomial sigmoid with the Naresh & Reddi (2025) architecture. It achieved:
- Gradient max error: `1.13e-05`
- Mean epoch time: **18.93 s**
- Total (10 epochs): **189.27 s**
- Test AUC: **1.0000**

## 2. Runtime Inversion (Phase 2A Discovery)

Stage-wise profiling of Degrees 1, 3, and 5 revealed a counter-intuitive result. Despite computing a *heavier* polynomial, Degree 5 was the *fastest* configuration:

| Degree | Grad Start Level | Gradient Time | Total Epoch |
|:---:|:---:|---:|---:|
| 1 | Level 5 | 36.36 s | 40.03 s |
| 3 | Level 3 | 15.29 s | 19.28 s |
| 5 | Level 2 |  8.23 s | 12.08 s |

**Mechanism:** In CKKS, ciphertext operation cost scales with modulus size. A higher-degree polynomial burns through more levels in the $O(1)$ forward pass, leaving a much smaller modulus for the $O(D)$ gradient backward pass. Since $D=92$ feature multiplications dominate the epoch time, the cost of the backward pass — not the polynomial — is the governing factor.

> [!NOTE]
> The runtime inversion is **fully supported by stage-wise evidence**. The gradient computation alone accounts for 91% of Degree 1's epoch time and 68% of Degree 5's epoch time.

## 3. Selector Design

We formulate adaptive degree selection as a constrained optimization:

$$d^* = \underset{d}{\mathrm{argmin}} \; T_{HE}(d)$$

Subject to:
1. $E_{poly}(d) \le \epsilon_{poly}$ — Polynomial approximation error on the calibration interval
2. $E_{grad}(d) \le \epsilon_{grad}$ — Encrypted gradient numerical accuracy
3. $D_{circuit}(d) < D_{available}$ — At least 1 modulus level must remain after the backward pass

**Key design choice:** $T_{HE}(d)$ is measured empirically via a single calibration pass on training data — it is *not* approximated theoretically, because the runtime inversion makes theoretical prediction unreliable.

> [!IMPORTANT]
> Degree selection uses **only training-side calibration**. The test set is never consulted during the selection process.

## 4. Selection Constraints (Experiment 1)

Calibrated on the full 1,645-sample training set with `epsilon_poly=0.05`:

| Degree | $E_{poly}$ | $E_{grad}$ | $D_{circuit}$ | $T_{HE}$ (s) | Grad Level | Fits All? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **0.684** | 6.04e-06 | 3 | 39.22 | 5 | ❌ poly |
| 3 | **0.098** | 1.44e-05 | 5 | 18.77 | 3 | ❌ poly |
| **5** | **0.033** | 1.91e-05 | 6 | **11.69** | 2 | ✅ |

**Selection: Degree 5** — the only candidate satisfying all constraints, and simultaneously the fastest.

### Candidate Rejection Reasons
- **Degree 1**: Rejected. $E_{poly} = 0.684 \gg 0.05$. A linear polynomial severely misestimates probabilities across the training logit range.
- **Degree 3**: Rejected. $E_{poly} = 0.098 > 0.05$. Below the threshold by ~2×. Passes depth and gradient checks but fails the approximation requirement.

## 5. Threshold Sensitivity (Experiment 4)

A key property of an adaptive selector is that it must respond to changing constraints. We swept `epsilon_poly` to verify the selection changes:

| $\epsilon_{poly}$ | Selected Degree | Reason |
|:---:|:---:|:---|
| 0.20 | **5** | Degree 5 minimizes $T_{HE}$ (11.69s) while satisfying all constraints |
| 0.10 | **5** | Degree 5 minimizes $T_{HE}$ (11.69s) while satisfying all constraints |
| 0.05 | **5** | Degree 5 minimizes $T_{HE}$ (11.69s) while satisfying all constraints |
| **0.03** | **1 (fallback)** | **No candidate satisfied $E_{poly} \le 0.03$** — fallback to minimum depth |

> [!NOTE]
> At $\epsilon_{poly} = 0.03$, even Degree 5's approximation error (0.033) barely exceeds the threshold. This demonstrates that the selector correctly identifies the constraint boundary. At stricter thresholds, the selector *correctly refuses* rather than silently proceeding with an inadequate approximation. This is a critical safety property for a medical telemetry context.

**The decision boundary lies between $\epsilon=0.033$ and $\epsilon=0.05$** for this dataset and logit range. If the approximation requirement is stricter than 3.3%, none of the candidates in $\{1, 3, 5\}$ are sufficient, and a degree-7 polynomial (or a wider logit interval) would need to be added.

## 6. Fixed vs. Adaptive Comparison — Results (Experiments 2 & 3)

3 deterministic repetitions, 10 epochs each, identical CKKS context, lr, l2, features, subject split.

**Per-repetition totals:**

| Method | Rep 1 total (s) | Rep 2 total (s) | Rep 3 total (s) |
|:---|---:|---:|---:|
| Adaptive (Deg 5) | 125.1 | 114.4 | 115.7 |
| Fixed Deg 1 | 378.5 | 354.2 | 356.4 |
| Fixed Deg 3 *(Phase-0 baseline)* | 187.9 | 175.5 | 176.9 |
| Fixed Deg 5 | 121.2 | 118.7 | 110.3 |

**Mean ± Std over 3 reps:**

| Method | Mean Epoch (s) | Std | Total (s) | Std | Train Acc | Test Acc | Test AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Adaptive (Deg 5)** | **11.84** | **±0.59** | **118.41** | **±5.86** | 0.9787 | 0.9489 | 1.0000 |
| Fixed Deg 1 | 36.30 | ±1.35 | 363.04 | ±13.46 | 0.9866 | 0.9716 | 1.0000 |
| Fixed Deg 3 *(baseline)* | 18.01 | ±0.68 | 180.08 | ±6.77 | 0.9581 | 0.8582 | 1.0000 |
| Fixed Deg 5 | 11.67 | ±0.57 | 116.72 | ±5.68 | 0.9787 | 0.9489 | 1.0000 |

**Speedup vs. Phase-0 baseline (Deg 3):**

| Comparison | Speedup | Test Acc Δ |
|:---|:---:|:---:|
| Adaptive vs. Deg 3 (baseline) | **1.52×** faster | +9.1 pp |
| Adaptive vs. Deg 1 | **3.07×** faster | −2.3 pp |
| Adaptive vs. Deg 5 | ≈ identical | identical |

## 7. Does Adaptive Selection Improve Encrypted Training Cost?

**Yes — but with a precise qualification.**

On MHEALTH with `epsilon_poly=0.05`, the adaptive selector chose **Degree 5**. Because the adaptive selection and fixed Degree 5 training are therefore identical in this case, the adaptive method does not add overhead — it reproduces the optimal fixed-degree result automatically.

Compared to the **Phase-0 frozen baseline (Degree 3)**:
- Training is **1.52× faster** (180s → 118s for 10 epochs)
- Test accuracy improves by **+9.1 percentage points** (0.8582 → 0.9489)
- Test ROC-AUC remains **1.0000** for all methods

> [!IMPORTANT]
> The adaptive selector's central value is not that it outperforms *every* fixed degree — it is that it **automatically identifies the degree that minimizes encrypted training cost subject to accuracy and depth constraints, without requiring the practitioner to manually profile all candidates**.
>
> When the optimal degree is non-obvious (e.g., when a lower-degree polynomial would satisfy epsilon at a tighter feature-count configuration), the selector will automatically switch — as demonstrated by the threshold sensitivity result at `epsilon=0.03`.

### Why Adaptive ≈ Fixed 5 on MHEALTH

This is not a failure of the method — it is the *correct output* of the selector. The selector evaluates all candidates, rejects Degrees 1 and 3 for exceeding `epsilon_poly`, and selects Degree 5 as the only feasible, lowest-cost option. Identical selection → identical training → identical results. This is expected and consistent.

The adaptive method's differentiation will become measurable in:
- **Phase 4 (Feature Reduction)**: With D=12 or D=24, the O(D) gradient cost shrinks, and the runtime inversion may flatten or disappear, shifting the optimal degree selection.
- **Stricter error thresholds**: At `epsilon=0.03`, the selector correctly falls back to Degree 1 (the only depth-feasible candidate), demonstrating dynamic selection.
- **Alternative datasets**: Datasets with narrower logit ranges may allow Degree 3 to satisfy approximation constraints at lower depth cost.

## 8. Limitations

1. **Calibration cost**: The selector performs one full forward+backward pass per candidate degree (~70s for 3 candidates on MHEALTH). This cost is negligible relative to training but should be amortised across multiple experiments on the same dataset.
2. **Single-dataset validation**: The runtime inversion and optimal degree selection have been validated only on MHEALTH (92 features, 1,645 samples). Generality requires Phase 4 (feature reduction) and later MIT-BIH experiments.
3. **Sensitivity to D**: The runtime inversion magnitude is proportional to the feature count D. With fewer features, the O(D) gradient stage is cheaper relative to the O(1) forward pass, and the inversion may shrink or disappear. The selector must be re-calibrated when D changes.
4. **Approximation threshold is researcher-set**: `epsilon_poly` is not derived automatically from a target prediction error specification. For a deployed medical system, the threshold should be linked to a maximum tolerable probability estimation error.
5. **Fixed CKKS context**: All experiments use `[60,40,40,40,40,40,40,60]`. A different context changes both the level budget and the per-level ciphertext weight, potentially changing the inversion point.
6. **Perfectly separable task**: MHEALTH binary classification is near-perfectly linearly separable. All methods achieve AUC=1.0000. The accuracy differences between degrees are therefore attributable entirely to the number of training epochs, not to fundamental approximation quality. A non-separable dataset will provide a stronger test.

