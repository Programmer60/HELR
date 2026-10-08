# Phase 3: Feature-Dimension × Polynomial-Degree Cost Study

## 1. Objective and Hypothesis
In Phase 2B, we established that a higher-degree polynomial (Degree 5) counter-intuitively accelerates encrypted training on the full 92-feature MHEALTH dataset. This "cryptographic accelerator" effect occurs because the higher-degree polynomial rapidly burns through the heaviest modulus levels during the $O(1)$ forward pass, forcing the massive $O(D)$ gradient computation to execute on lightweight, low-level ciphertexts.

**Hypothesis**: As the feature dimensionality $D$ decreases, the relative cost of the $O(D)$ gradient computation shrinks. Eventually, the additional computational overhead of evaluating a high-degree polynomial (and its alignment burns) will outweigh the savings in the gradient stage. Therefore, the optimal polynomial degree that minimizes total HE training time is not universally "Degree 5" — it is a dynamic function of $D$.

## 2. Experimental Setup
- **Training (Subjects 1–5)**: 1,175 samples. Used for feature selection, polynomial calibration, and training.
- **Validation (Subjects 6–7)**: 470 samples. Used for evaluating configurations and adaptive selection utility.
- **Test (Subjects 8–10)**: 705 samples. Used strictly for final reporting.
- **Feature Selection**: Deterministic ANOVA F-value (`SelectKBest`) fitted exclusively on the Training split to extract $D \in \{92, 48, 24, 12\}$.
- **Constraints**: $\epsilon_{poly} = 0.05$, $\epsilon_{grad} = 10^{-4}$, CKKS Levels = 7.

## 3. The Feature × Degree Interaction Matrix

*(Values measured over 10 training epochs on Subjects 1-5. Time is total training time. Gradient time is total accumulated gradient time.)*

| $D$ | Metric | Degree 1 (Linear) | Degree 3 (Cubic) | Degree 5 (Quintic) |
|:---:|:---|:---:|:---:|:---:|
| **92** | Approx Max Error | 0.6845 | 0.0984 | **0.0329** |
| | Total Train Time | 260.9 s | 145.0 s | **102.8 s** |
| | Gradient Time | 222.5 s | 106.6 s | **64.2 s** |
| | Grad Start Level | Level 5 | Level 3 | **Level 2** |
| | Validation AUC | 1.0000 | 1.0000 | 1.0000 |
| | **Empirically Optimal** | | | **✅ Degree 5** |
| **48** | Approx Max Error | 0.6845 | 0.0984 | **0.0329** |
| | Total Train Time | 142.0 s | 72.5 s | **50.9 s** |
| | Gradient Time | 122.2 s | 52.9 s | **31.6 s** |
| | Grad Start Level | Level 5 | Level 3 | **Level 2** |
| | Validation AUC | 1.0000 | 1.0000 | 1.0000 |
| | **Empirically Optimal** | | | **✅ Degree 5** |
| **24** | Approx Max Error | 0.6845 | 0.0984 | **0.0329** |
| | Total Train Time | 68.6 s | 36.4 s | **25.8 s** |
| | Gradient Time | 58.7 s | 26.4 s | **15.7 s** |
| | Grad Start Level | Level 5 | Level 3 | **Level 2** |
| | Validation AUC | 1.0000 | 1.0000 | 1.0000 |
| | **Empirically Optimal** | | | **✅ Degree 5** |
| **12** | Approx Max Error | 0.6845 | 0.0984 | **0.0329** |
| | Total Train Time | 34.2 s | 18.5 s | **13.2 s** |
| | Gradient Time | 29.1 s | 13.2 s | **7.8 s** |
| | Grad Start Level | Level 5 | Level 3 | **Level 2** |
| | Validation AUC | 1.0000 | 1.0000 | 1.0000 |
| | **Empirically Optimal** | | | **✅ Degree 5** |

## 4. Does the Optimal Polynomial Degree Shift?

**No, it does not shift within practical feature dimensions.** 

Our hypothesis was that as $D$ decreases, the $O(1)$ polynomial overhead of Degree 5 might eventually outweigh the $O(D)$ gradient savings, allowing Degree 3 to become faster. The empirical data explicitly rejects this.

Even at $D=12$, Degree 5's gradient stage is 5.4 seconds faster than Degree 3's over 10 epochs. The penalty for evaluating the quintic polynomial and its alignment burns is barely 0.4 seconds total over 10 epochs. Therefore, Degree 5 remains overwhelmingly faster than Degree 3, even for extremely low-dimensional feature sets. 

The "cryptographic accelerator" effect of a heavier polynomial is so potent that it dominates the runtime dynamics universally for this CKKS context.

## 5. Calibration Overhead Analysis

For an adaptive selector to be useful, the cost of measuring calibration times for candidate degrees must be lower than the time saved during full training. Calibration involves running the forward and backward pass for one batch to empirically measure HE timing.

| $D$ | Selected Degree | $T_{calib}$ (all candidates) | $T_{train}$ (selected) | $T_{total}$ (10 Epochs) |
|:---:|:---:|---:|---:|---:|
| 92 | **5** | 49.27 s | 102.76 s | 152.03 s |
| 48 | **5** | 27.20 s | 50.90 s | 78.10 s |
| 24 | **5** | 12.96 s | 25.78 s | 38.74 s |
| 12 | **5** | 6.44 s | 13.15 s | 19.59 s |

## 6. Novelty Test: Adaptive vs. Fixed Baseline Strategies

Does the adaptive strategy actually beat a fixed "always use Degree 3" or "always use Degree 5" rule when accounting for total HE computation cost (Calibration + Training) across different workloads? 

Here we compare the total cost for the ultra-short **10 Epoch** development run vs. a realistic **100 Epoch** deployment run.

### 10 Epochs Total Time
| Workload ($D$) | Fixed Deg 3 | Fixed Deg 5 | Adaptive + Calib | Adaptive Advantage? |
|:---:|---:|---:|---:|:---|
| 92 | 145.0 s | **102.8 s** | 152.0 s | ❌ Slower than both due to calib overhead |
| 48 | 72.5 s | **50.9 s** | 78.1 s | ❌ Slower than both due to calib overhead |
| 24 | 36.4 s | **25.8 s** | 38.7 s | ❌ Slower than both due to calib overhead |
| 12 | 18.5 s | **13.2 s** | 19.6 s | ❌ Slower than both due to calib overhead |

### 100 Epochs Total Time (Extrapolated)
| Workload ($D$) | Fixed Deg 3 | Fixed Deg 5 | Adaptive + Calib | Adaptive Advantage? |
|:---:|---:|---:|---:|:---|
| 92 | 1450.1 s | **1027.6 s** | 1076.9 s | ✅ Faster than Deg 3 (saves 373s) |
| 48 | 724.7 s | **509.0 s** | 536.2 s | ✅ Faster than Deg 3 (saves 188s) |
| 24 | 364.4 s | **257.8 s** | 270.8 s | ✅ Faster than Deg 3 (saves 93s) |
| 12 | 185.3 s | **131.5 s** | 137.9 s | ✅ Faster than Deg 3 (saves 47s) |

**Conclusion on Adaptive Utility:**
The novelty test yields a brutally honest but scientifically valuable result:
1. **Vs. Fixed Degree 5:** The adaptive selector provides **no runtime advantage** over a fixed Degree-5 strategy on this dataset, because Degree 5 is universally the fastest feasible candidate. The selector simply confirms this choice and adds calibration overhead.
2. **Vs. Phase-0 Baseline (Fixed Degree 3):** The adaptive selector provides a **massive asymptotic advantage**, shifting the workload to a faster polynomial and saving hundreds of seconds. However, for ultra-short training runs (e.g., 10 epochs), the $O(1)$ overhead of evaluating all candidates during calibration actually eats the savings, making the adaptive strategy slower overall. The $O(1)$ calibration overhead is safely amortised only in runs approaching ~50-100 epochs.

Adaptive polynomial training is therefore highly recommended for long-running production workloads where the optimal polynomial is completely unknown, but it acts primarily as an automated validation step rather than a real-time accelerator when Degree 5 is already known to be optimal.
