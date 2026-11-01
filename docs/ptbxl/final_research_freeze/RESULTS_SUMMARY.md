# Phase 3B Empirical Results & Statistical Summary

**Document ID**: `docs/ptbxl/final_research_freeze/RESULTS_SUMMARY.md`  
**Phase**: Phase 4B — Statistical Corrections, Evidence Audit & Research Freeze Finalization  
**Audit Status**: Verified by Independent Mathematical Re-computation from Raw Phase 3B Logs  
**Date**: October 31, 2026  

---

## 1. Independent Mathematical Audit of Phase 3B Metrics

All metrics reported below were re-computed directly from the raw individual epoch and batch logs in `results/ptbxl/scheduler/phase3b/phase3b_training_results.json` and `phase3b_timing_breakdown.csv` using the verification script `results/ptbxl/scheduler/phase3b/verify_phase3b_statistics.py`.

### 1.1 Timing Boundary & Setup Cost Accounting Check

For all 12 experimental runs:
1. **Total Cumulative Time**:
   $$\text{Cumulative Training Time} = T_{\text{setup\_alignment}} + \sum_{e=1}^{10} T_{\text{epoch}}(e)$$
   - Re-computation confirmed: For every run, $T_{\text{cumulative}}$ equals the one-time setup cost plus the sum of all 10 individual epoch wall-clock times ($|T_{\text{calc}} - T_{\text{reported}}| < 1.0 \times 10^{-12}\text{ s}$).
2. **Setup Cost Isolation**:
   - For cached runs (`A-cache`, `B-cache`), $T_{\text{setup\_alignment}}$ is counted **exactly once** before Epoch 1 ($1.86 \pm 0.08\text{ s}$ for A-cache, $2.31 \pm 0.28\text{ s}$ for B-cache).
   - In all subsequent epochs ($e \in [1, 10]$), the measured `feature_alignment_s` inside the epoch is **$0.0000\text{ s}$**. Caching overhead is never double-counted.
3. **Repeated Alignment Accounting**:
   - For repeated runs (`A-repeat`, `B-repeat`), $T_{\text{setup\_alignment}} = 0.00\text{ s}$, and `feature_alignment_s` is measured in every batch of every epoch ($\approx 1.5\text{ s}$ per epoch for A, $\approx 1.9\text{ s}$ per epoch for B).

---

## 2. Complete Summary of Matched Experimental Runs

### 2.1 Per-Schedule Aggregates across Seeds 42, 123, 2026

*Convention: Statistics are reported as Mean $\pm$ Population Standard Deviation ($\text{ddof}=0$), with Sample Standard Deviation ($\text{ddof}=1$) provided in brackets for completeness.*

| Schedule ID | Alignment Strategy | Gradient Modulus Level | Cumulative Time (s) [Pop $\sigma$] | Mean Epoch Time (s) | Mean Gradient Time (s) | Model Predicted Time (s) | Model Prediction MAPE | Validation ROC-AUC | Validation Accuracy | Max Gradient AE vs Plaintext |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A-repeat`** | Repeated | Level 3 | $136.07 \pm 11.68\text{ s}$ $[14.30]$ | $13.61\text{ s}$ | $7.07\text{ s}$ | $121.29\text{ s}$ | $10.20 \pm 7.65\%$ | $0.7842 \pm 0.0041$ | $0.6990 \pm 0.0029$ | $5.48 \times 10^{-6}$ |
| **`A-cache`**  | Cached   | Level 3 | $126.29 \pm 6.08\text{ s}$ $[7.44]$   | $12.41\text{ s}$ | $7.58\text{ s}$ | $105.78\text{ s}$ | $16.04 \pm 4.18\%$ | $0.7842 \pm 0.0041$ | $0.6990 \pm 0.0029$ | $5.48 \times 10^{-6}$ |
| **`B-repeat`** | Repeated | Level 2 | $115.10 \pm 14.62\text{ s}$ $[17.91]$ | $11.51\text{ s}$ | $4.12\text{ s}$ | $91.81\text{ s}$  | $19.04 \pm 9.46\%$  | $0.7842 \pm 0.0041$ | $0.6990 \pm 0.0029$ | $7.28 \times 10^{-6}$ |
| **`B-cache`**  | Cached   | Level 2 | **$86.11 \pm 4.09\text{ s}$** $[5.00]$ | **$8.37\text{ s}$** | **$3.85\text{ s}$** | **$74.22\text{ s}$** | **$13.62 \pm 3.97\%$** | **$0.7842 \pm 0.0041$** | **$0.6990 \pm 0.0029$** | **$7.28 \times 10^{-6}$** |

---

## 3. Re-computed Pairwise Speedup Comparisons

### 3.1 Mathematical Formulation
For each pairwise comparison between a baseline schedule and an optimized schedule, two distinct statistical measures are reported:

1. **Seed-Matched Mean Ratio ($\overline{R}$)**:
   $$R_s = \frac{T_{\text{baseline}, s}}{T_{\text{optimized}, s}} \quad \text{for } s \in \{42, 123, 2026\}$$
   $$\overline{R} = \frac{1}{3} \sum_{s=1}^3 R_s, \quad \sigma_R = \sqrt{\frac{1}{3} \sum_{s=1}^3 (R_s - \overline{R})^2} \quad (\text{population std, } \text{ddof}=0)$$

2. **Ratio of Aggregate Means ($R_{\text{agg}}$)**:
   $$R_{\text{agg}} = \frac{\overline{T}_{\text{baseline}}}{\overline{T}_{\text{optimized}}} = \frac{\frac{1}{3}\sum_{s=1}^3 T_{\text{baseline}, s}}{\frac{1}{3}\sum_{s=1}^3 T_{\text{optimized}, s}}$$

### 3.2 Exact Per-Seed and Aggregate Ratios

```
+---------------------------------------------+-------------------+-------------------+-------------------+-----------------------+---------------------+
| Comparison / Factor Effect                  | Seed 42 Ratio     | Seed 123 Ratio    | Seed 2026 Ratio   | Seed-Matched Mean     | Ratio of Agg Means  |
+---------------------------------------------+-------------------+-------------------+-------------------+-----------------------+---------------------+
| Level Effect (repeated) [A-rep / B-rep]     | 0.901x (122.4/135.7s)| 1.424x (150.9/106.0s)| 1.303x (134.9/103.6s)| 1.209 ± 0.223x [0.273] | 1.182x (136.1/115.1s) |
| Level Effect (cached)   [A-cac / B-cac]     | 1.421x (130.6/91.9s) | 1.574x (130.6/83.0s) | 1.410x (117.7/83.5s) | 1.468 ± 0.075x [0.092] | 1.467x (126.3/86.1s)  |
| Cache Effect (Cond A)   [A-rep / A-cac]     | 0.937x (122.4/130.6s)| 1.155x (150.9/130.6s)| 1.147x (134.9/117.7s)| 1.080 ± 0.101x [0.123] | 1.077x (136.1/126.3s) |
| Cache Effect (Cond B)   [B-rep / B-cac]     | 1.477x (135.7/91.9s) | 1.277x (106.0/83.0s) | 1.241x (103.6/83.5s) | 1.332 ± 0.104x [0.127] | 1.337x (115.1/86.1s)  |
| Combined Optimization   [A-rep / B-cac]     | 1.332x (122.4/91.9s) | 1.819x (150.9/83.0s) | 1.617x (134.9/83.5s) | 1.589 ± 0.200x [0.245] | 1.580x (136.1/86.1s)  |
+---------------------------------------------+-------------------+-------------------+-------------------+-----------------------+---------------------+
```

> [!NOTE]
> **Resolution of Condition A Caching Table Entry**:
> In earlier drafts, the per-seed ratios for Condition A caching were inadvertently miscopied as `1.029` and `1.287`, despite the summary text stating `1.080 ± 0.101`. Re-computation from the raw unrounded timings confirms the correct per-seed ratios are **$0.937$** ($122.36\text{s} / 130.55\text{s}$), **$1.155$** ($150.90\text{s} / 130.62\text{s}$), and **$1.147$** ($134.95\text{s} / 117.70\text{s}$), which average exactly to **$1.080 \pm 0.101$** ($\text{ddof}=0$).

---

## 4. Clarification of the 1.71× Gradient Stage Speedup

The paper discusses a gradient-stage speedup of approximately $1.71\times$ to $1.72\times$:
- **Exact Comparison**: This value specifically isolates the **homomorphic gradient dot product stage** under repeated alignment:
  $$R_{\text{grad}} = \frac{\overline{T}_{\text{grad}}(\text{A-repeat})}{\overline{T}_{\text{grad}}(\text{B-repeat})} = \frac{7.069\text{ s}}{4.124\text{ s}} = 1.714\times \approx 1.71\times$$
- Under cached alignment, the isolated gradient dot product ratio is:
  $$\frac{\overline{T}_{\text{grad}}(\text{A-cache})}{\overline{T}_{\text{grad}}(\text{B-cache})} = \frac{7.580\text{ s}}{3.850\text{ s}} = 1.969\times \approx 1.97\times$$
- **Distinction**: The $1.71\times$ gradient dot product speedup applies strictly to the isolated $\mathcal{O}(D)$ ciphertext multiplication and slot summation operations entering at Level 2 vs Level 3. It must not be conflated with the overall cumulative training speedup ($1.589\times$), which accounts for the entire training loop including feature alignment, polynomial evaluation, and client decryption.

---

## 5. Actual Experiments vs. Policy Predictions

It is essential to distinguish between empirical training runs and analytical policy predictions:

1. **Empirically Executed Experiments**:
   - **Full Matched Grid**: 12 full runs ($4\text{ schedules} \times 3\text{ seeds} \times 10\text{ epochs} = 120\text{ epochs}, 360\text{ batches}$).
   - **Pilot Run**: 1 run (Seed 42, 3 epochs, dynamic `B-cache` plan selection and execution).
2. **Analytical Policy Predictions ($E \in \{1, 3, 5, 10\}$)**:
   - Evaluated by querying `WorkloadSchedulerPolicy.select_plan()` across variable epoch horizons:
     - $E=1$: Policy **predicts** `B-repeat` ($9.18\text{ s}$ predicted total) because the caching setup cost ($1.95\text{ s}$) equals the per-epoch alignment cost, saving memory without runtime penalty.
     - $E \in \{3, 5, 10\}$: Policy **predicts** `B-cache` ($23.63\text{ s}$, $38.09\text{ s}$, $74.22\text{ s}$ predicted totals) due to multi-epoch setup amortization.
   - *Status*: These epoch-horizon evaluations represent **model predictions under CostModel_v2**, not separate 10-epoch empirical training experiments.
3. **Constraint Fallbacks Verified by Tests**:
   - Memory Budget Constraint ($M_{\text{budget}} < 100\text{ MB}$): Unit tests in `tests/test_phase3b_scheduler.py` demonstrate that with a $5\text{ MB}$ budget, `B-cache` ($108.0\text{ MB}$) is marked infeasible and the policy selects `B-repeat`.
   - Modulus Constraint (Condition B disabled): Demonstrated by tests to fall back to `A-cache` ($105.78\text{ s}$ predicted).

---

## 6. Decision Regret & Empirical Agreement

For each of the three evaluated seeds:
$$\text{Regret}(s) = T_{\text{measured}}(\mathcal{S}_{\text{policy}}, s) - \min_{\mathcal{S} \in \{A\text{-rep}, A\text{-cac}, B\text{-rep}, B\text{-cac}\}} T_{\text{measured}}(\mathcal{S}, s)$$

```
+------------+------------------+------------------+------------------+------------------+---------------+
| Seed       | Policy Selected  | Policy Time (s)  | Empirical Oracle | Oracle Time (s)  | Regret        |
+------------+------------------+------------------+------------------+------------------+---------------+
| 42         | B-cache          | 91.88 s          | B-cache          | 91.88 s          | 0.00 s (0.0%) |
| 123        | B-cache          | 82.98 s          | B-cache          | 82.98 s          | 0.00 s (0.0%) |
| 2026       | B-cache          | 83.47 s          | B-cache          | 83.47 s          | 0.00 s (0.0%) |
+------------+------------------+------------------+------------------+------------------+---------------+
```

- **Scope of Claim**: The policy recommendation (`B-cache`) achieved **$0.00\%$ decision regret across the three evaluated matched seeds**, agreeing with the empirically fastest configuration in each trial. We do not claim universal optimality across arbitrary unmeasured datasets or hardware targets.

---

## 7. Validation Performance & Numerical Invariance

Across all four schedules and three seeds:
1. **Validation Accuracy**:
   - Seed 42: $0.7022$ ($70.22\%$)
   - Seed 123: $0.6994$ ($69.94\%$)
   - Seed 2026: $0.6952$ ($69.52\%$)
   - **Mean Validation Accuracy across seeds**: **$0.6990 \pm 0.0029$** ($\text{ddof}=0$) [$0.0035$ sample std].
2. **Validation ROC-AUC**:
   - Seed 42: $0.7899$
   - Seed 123: $0.7806$
   - Seed 2026: $0.7821$
   - **Mean Validation ROC-AUC across seeds**: **$0.7842 \pm 0.0041$** ($\text{ddof}=0$) [$0.0050$ sample std].
3. **Concordance**:
   - Within each matched seed, validation metrics exhibit **no observed difference across schedules at the reported precision**.
   - Maximum gradient absolute error vs plaintext reference remains strictly below $7.28 \times 10^{-6}$ (Pass $\le 1.0 \times 10^{-4}$).
