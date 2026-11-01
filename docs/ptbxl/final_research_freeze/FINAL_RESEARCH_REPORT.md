# Workload-Aware Runtime Scheduling for CKKS-Based Privacy-Preserving Logistic Regression

**Document ID**: `docs/ptbxl/final_research_freeze/FINAL_RESEARCH_REPORT.md`  
**Phase**: Phase 4C — Final Technical Corrections & Research Freeze Validation  
**Date**: October 31, 2026  
**Status**: Completed, Mathematically Audited, and Frozen  
**Repository**: `C:\Users\mishr\Desktop\ML\venv\MinorProject`  
**Git Commit**: `5c34664 Worked on PTX-BL dataset`  

---

## Abstract

Homomorphic encryption via the Cheon-Kim-Kim-Song (CKKS) scheme enables privacy-preserving machine learning directly over encrypted clinical data without exposing raw patient records to untrusted cloud service providers. However, homomorphic training incurs substantial computational overhead, dominated by deep RNS ciphertext multiplications and rotations. In this work, we investigate systems-level scheduling of homomorphic training circuits for logistic regression on the standard 12-lead PTB-XL electrocardiography dataset ($N=17,084, D=48$).

We demonstrate that the computational latency of homomorphic gradient descent is highly sensitive to the ciphertext modulus level at which high-fanout gradient dot products enter the circuit, as well as the strategy used to align feature ciphertexts. We formulate a profile-guided, runtime-context-aware analytical cost model (`CostModel_v2`) and a workload-aware policy (`WorkloadSchedulerPolicy`) that evaluates candidate circuit schedules against hardware memory budgets and numerical error thresholds.

Across 12 seed-matched, 10-epoch encrypted training experiments ($120$ full epochs, $360$ batch steps), our optimized execution schedule (`B-cache`) achieved a **$1.589 \pm 0.200\times$ seed-matched cumulative wall-clock training speedup** ($\text{ddof}=0$) over the baseline (`A-repeat`), reducing cumulative training time from $136.07 \pm 11.68\text{ s}$ to $86.11 \pm 4.09\text{ s}$ (a $1.580\times$ ratio of aggregate means). The policy recommendation agreed with the empirically fastest configuration across all evaluated seeds ($0.00\%$ empirical decision regret on this tested workload). We confirm that the speedup is achieved with **no observed difference in validation metrics across schedules at the reported precision**: validation ROC-AUC ($0.7842 \pm 0.0041$) and validation accuracy ($0.6990 \pm 0.0029$) on Fold 9 are identical across schedules for each seed, and gradient absolute errors remain strictly below $7.28 \times 10^{-6}$ (well within the $\le 10^{-4}$ tolerance bound).

---

## 1. Research Motivation & Foundations

Training predictive models on sensitive clinical telemetry, such as 12-lead electrocardiograms (ECGs), across institutional boundaries is constrained by patient confidentiality regulations (e.g., HIPAA, GDPR). While federated learning mitigates raw data centralization, intermediate model updates and gradient vectors remain susceptible to reconstruction and membership inference attacks.

The Residue Number System Cheon-Kim-Kim-Song (RNS-CKKS) scheme (Cheon et al., 2017) provides a rigorous cryptographic foundation by enabling approximate arithmetic over encrypted complex vectors. In a partitioned client-server setting, a trusted Hospital encrypts patient feature matrices and retains the secret key, while an untrusted Cloud Service Provider (CSP) evaluates forward predictions and gradient accumulations homomorphically. We term this framework **CKKS-assisted privacy-preserving training with client-side gradient decryption and plaintext optimizer updates**.

However, encrypted training suffers from two fundamental systems bottlenecks:
1. **Multiplicative Depth Exhaustion**: Non-linear activations require polynomial approximations whose evaluation consumes ciphertext modulus levels (scaling primes).
2. **High-Fanout Gradient Bottleneck**: In feature-packed logistic regression with $D$ features, the outer product of the scalar error vector and the feature columns requires $\mathcal{O}(D)$ homomorphic multiplications and slot summations per batch. The latency of these operations scales super-linearly with the number of active primes in the RNS modulus chain.

---

## 2. Architectural Baseline & Related Work

Our implementation builds upon the column-packed homomorphic logistic regression architecture investigated by Naresh and Reddi (2025) for privacy-preserving medical prediction:
- **Client/Server Division**: The Hospital generates the CKKS context, encrypts data, decrypts batch gradients, and performs regularization and weight updates in plaintext. The CSP executes only ciphertext-ciphertext and ciphertext-plaintext operations without holding secret key material.
- **SIMD Ciphertext Packing**: Features are organized in column-major order. Each ciphertext packs up to $S = 8,192$ sample values across slots, allowing a dataset of size $N$ to be processed in $B = \lceil N / S \rceil$ batches.
- **Activation Approximation**: The non-polynomial sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ is approximated over a bounded clinical range $[-5, 5]$ using a degree-3 odd-symmetric Chebyshev polynomial $P_3(z) = 0.5 + 0.15012z - 0.001593z^3$, following standard polynomial approximation techniques for homomorphic logistic regression (Kim et al., 2018; Han et al., 2019).

---

## 3. PTB-XL Dataset, Preprocessing, and Task Formulation

### 3.1 Project-Specific Binary Classification Task
The diagnostic prediction task is framed as a binary abnormality screening problem on the PTB-XL v1.0.3 clinical electrocardiography benchmark (Wagner et al., 2020; Goldberger et al., 2000):
- **Class 0 (Normal)**: All mapped diagnostic statements on the record consist strictly of `NORM`.
- **Class 1 (Abnormal)**: At least one mapped diagnostic statement other than `NORM` is present (including co-occurring diagnostic abnormalities with `NORM`).
- **Excluded**: Records lacking mapped diagnostic statements ($411$ records) are excluded prior to normalization.
- *Scoping Note*: This is a project-specific binary screening formulation for evaluating privacy-preserving logistic regression; it is not an official PhysioNet benchmark competition endpoint.

### 3.2 Dataset Dimensions & Verified Splits
- **Raw Dataset**: $21,799$ 10-second 12-lead ECG recordings sampled at $100\text{ Hz}$.
- **Eligible Dataset**: $21,388$ records after diagnostic statement filtering.
- **Feature Extraction**: $12\text{ leads} \times 4\text{ summary statistics}$ ($\text{mean}, \text{std}, \text{min}, \text{max}$) = **$48$ features per record**.
- **Standard Splits**:
  - **Training (Folds 1–8)**: $N=17,084$ records ($7,243$ normal, $9,841$ abnormal). Scaler fit strictly on Folds 1–8.
  - **Validation (Fold 9)**: $N=2,146$ records ($914$ normal, $1,232$ abnormal).
  - **Test (Fold 10)**: $N=2,158$ records ($912$ normal, $1,246$ abnormal).
- **Holdout Discipline**: **Fold 10 was strictly excluded from Phase 3B model fitting, validation, hyperparameter tuning, and performance evaluation.**

---

## 4. CKKS Context & Homomorphic Circuit Mechanics

The cryptographic parameters are instantiated in **TenSEAL 0.3.17** (Benaissa et al., 2021), wrapping Microsoft SEAL 3.7–4.0 RNS-CKKS (Microsoft Research, 2021):
- Polynomial modulus degree: $N_{\text{poly}} = 16,384$.
- Configured prime bit sizes: $[60, 40, 40, 40, 40, 40, 40, 60]$ (total configured bit-length = $360\text{ bits}$).
- Scaling factor: $\Delta = 2^{40}$.
- SIMD slot capacity: $S = 8,192$.

### 4.1 Modulus Chain Accounting & Security Limits
Per the *Homomorphic Encryption Standard* (Albrecht et al., 2018), for $N_{\text{poly}} = 16,384$ with discrete Gaussian error ($\sigma = 3.2$) and uniform ternary secret keys ($\{-1, 0, 1\}$):
- **128-bit Classical Security Ceiling**: Maximum total modulus bit-length $\log Q \le 438\text{ bits}$.
- **128-bit Quantum Security Bound**: Under conservative primal BKZ lattice sieving estimators, $\log Q \le 411\text{ bits}$.
- **Modulus Structure**:
  - Configured bit-length: $360\text{ bits}$ (including the final 60-bit prime reserved by Microsoft SEAL as the special key-switching prime $P$).
  - Active data ciphertext modulus in fresh ciphertexts: $q_0 \dots q_6$ ($300\text{ bits}$, `coeff_modulus_size() == 7`).
  - Evaluation: Microsoft SEAL parameter validation compares the total configured modulus against the 128-bit security ceiling:
    $$300\text{ bits} < 360\text{ bits} \le 411\text{ bits} \le 438\text{ bits}$$
  - *Scoping Statement*: The selected parameter set satisfies the stated 128-bit security parameter limit under the cited Homomorphic Encryption Standard (Albrecht et al., 2018) and its documented ternary secret-key and discrete Gaussian error ($\sigma = 3.2$) assumptions.

### 4.2 Circuit Modulus States:
1. **Input**: Fresh features $X$ and weights $w$ enter at **Level 7** ($7$ active primes, $300\text{ bits}$).
2. **Forward**: $z = \sum_{j=1}^D X_j \cdot w_j$ consumes 1 level, dropping to **Level 6** ($6$ primes, $260\text{ bits}$).
3. **Activation**: $P_3(z) = c_0 + c_1 z + c_3 z^3$ evaluates $z^3$ via $(z \cdot z) \cdot z$, consuming 3 levels, dropping to **Level 3** ($4$ primes, $180\text{ bits}$).
4. **Error**: $e = P_3(z) - \text{burn}(y, 4)$ aligns the label to Level 3 via 4 plaintext scale multiplications.

---

## 5. Controlled Scheduling Interventions: Condition A vs Condition B

At the error computation boundary ($e \in \text{Level 3}$), the circuit faces a scheduling branch before computing $\mathcal{O}(D)$ gradient dot products $g_j = \sum_{i=1}^S e_i \cdot X_{i,j}$:

### 5.1 Condition A (Modulus Level 3 Gradient Entry)
- Error ciphertext $e$ enters the gradient calculation directly at Level 3.
- Feature columns $X_j$ must be aligned to Level 3 by applying 4 depth-consuming operations: $\text{burn}(X_j, 4)$.
- Gradient multiplications are executed at Level 3 (4 active primes) and output at Level 2.

### 5.2 Condition B (Controlled Level 2 Gradient Entry)
- Error ciphertext $e$ undergoes a controlled level reduction via:
  $$\text{enc\_err\_adj} = e \cdot 1.0$$
  In TenSEAL, multiplying by scalar $1.0$ performs `multiply_plain` followed immediately by `rescale_to_next`, dropping $e$ to **Level 2** ($3$ active primes, $140\text{ bits}$).
- Feature columns $X_j$ are aligned to Level 2 via $\text{burn}(X_j, 5)$.
- Gradient multiplications are executed at Level 2 (3 active primes) and output at Level 1.
- *Terminology Clarification*: Condition B is **not** a pure SEAL `ModSwitchToNext` call, as TenSEAL does not expose standalone modulus switching.

---

## 6. Feature-Alignment Caching Strategy

In repeated training iterations over static patient datasets, feature inputs $X$ do not change between epochs; only weights $w$ evolve.

- **Repeated Alignment (`-repeat`)**: Feature columns are burned on the fly during every batch step of every epoch ($3 \text{ batches} \times 48 \text{ features} \times 5 \text{ burns} = 720 \text{ burns/epoch}$).
- **Cached Alignment (`-cache`)**: Feature columns are burned once at setup ($t=0$) to the target level (Level 3 for A, Level 2 for B) and retained in memory.
  - Setup cost: $1.86 \pm 0.08\text{ s}$ (A-cache) or $2.31 \pm 0.28\text{ s}$ (B-cache).
  - Per-epoch alignment cost during training: **$0.0000\text{ s}$**.
  - Cache memory footprint: $144.0\text{ MB}$ (A-cache) or $108.0\text{ MB}$ (B-cache).

---

## 7. Mathematical Formulation of CostModel_v2

`CostModel_v2` (implemented in `src/scheduler/cost_model_v2.py`) predicts the latency of homomorphic logistic regression training as a function of dataset dimensions $(N, D)$, batch count $B = \lceil N / S \rceil$, epoch count $E$, modulus levels, alignment strategy, and execution context $\kappa \in \{\text{persistent}, \text{fresh}\}$.

### 7.1 Single-Epoch Stage Components
For an active gradient entry level $L_{\text{grad}} \in \{2, 3\}$, the single-epoch runtime is modeled as:
$$T_{\text{epoch}}(\mathcal{S}, \kappa) = T_{\text{w\_enc}} + T_{\text{fwd}} + T_{\text{poly}} + T_{\text{err}} + T_{\text{adj}}(\mathcal{S}) + T_{\text{align}}(\mathcal{S}) + T_{\text{grad}}(L_{\text{grad}}, \kappa) + T_{\text{dec}}$$

The implementation maps each mathematical term directly to profiled operation benchmarks:

| Model Component | Implementation Function / Key | Operation Modeled | Mathematical Formulation in CostModel_v2 |
| :--- | :--- | :--- | :--- |
| **Weight Encryption** | `weight_encryption_scalar` | Client scalar broadcast encryption | $T_{\text{w\_enc}} = D \cdot t_{\text{wenc\_single}}(\kappa)$ |
| **Forward Linear** | `forward_linear_accumulation_step` | Fused Ct-Ct mul-rescale + Ct add step | $T_{\text{fwd}} = B \cdot D \cdot t_{\text{fwd\_step}}(L_7 \rightarrow L_6, \kappa)$ |
| **Polynomial Eval** | `ct_ct_mul_rescale` + `pt_mul_one` | Chebyshev $P_3(z)$ evaluation per batch | $T_{\text{poly}} = B \cdot \left[ t_{\text{mul}}(L_6) + t_{\text{mul}}(L_5) + t_{\text{pt\_one}}(L_6) + \delta_{\text{poly}} \right]$ |
| **Error Computation**| `feature_alignment_burn4` + sub | Label alignment burn + subtraction | $T_{\text{err}} = B \cdot \left[ t_{\text{burn\_y}}(L_7 \rightarrow L_3) + t_{\text{sub}} \right]$ |
| **Level Adjustment** | `condition_b_level_adjustment` | Plaintext mult by 1.0 + rescale | $T_{\text{adj}}(\mathcal{S}) = \begin{cases} B \cdot t_{\text{pt\_one}}(L_3 \rightarrow L_2) & \text{if Condition B} \\ 0 & \text{if Condition A} \end{cases}$ |
| **Feature Alignment**| `feature_alignment_burn{4,5}` | Column alignment burns per batch | $T_{\text{align}}(\mathcal{S}) = \begin{cases} B \cdot D \cdot t_{\text{burn\_X}}(\mathcal{S}) & \text{if repeated} \\ 0 & \text{if cached and } e > 1 \end{cases}$ |
| **Gradient Dot Prod**| `gradient_dot_product` | $(e \cdot X_j)\text{.sum()}$ composite dot product | $T_{\text{grad}} = B \cdot D \cdot t_{\text{grad\_dot}}(L_{\text{grad}}, \kappa)$ |
| **Decryption & Agg** | `decryption_scalar` | Hospital decryptions ($B \times D$ ciphertexts) | $T_{\text{dec}} = B \cdot D \cdot t_{\text{dec\_single}}(L_{\text{grad}} - 1, \kappa)$ |

Where $\delta_{\text{poly}} \approx 0.012\text{ s}$ captures minor scalar coefficient multiplications and additions, and $t_{\text{sub}} \approx 0.0005\text{ s}$.

### 7.2 Total Training Runtime Formulation
- **Repeated Strategy**:
  $$T_{\text{total}}(\text{repeated}, \kappa) = E \cdot T_{\text{epoch}}(\text{repeated}, \kappa)$$
- **Cached Strategy**:
  $$T_{\text{setup\_cached}} = B \cdot D \cdot t_{\text{burn\_X}}(\mathcal{S})$$
  $$T_{\text{total}}(\text{cached}, \kappa) = T_{\text{setup\_cached}} + E \cdot T_{\text{epoch}}(\text{cached}, \kappa)$$
- **Execution Context Offset**: Under fresh worker execution, fixed process startup overhead is modeled once per process invocation:
  $$T_{\text{fresh\_overhead}} = t_{\text{init}} + t_{\text{keys}} \approx 2.37\text{ s}$$

### 7.3 Cache Memory Model
The RAM required to store cached feature ciphertexts is given by:
$$M_{\text{cached}}(\mathcal{S}) = \frac{B \cdot D \cdot \left( 2 \cdot N_{\text{poly}} \cdot (L_{\text{grad}} + 1) \cdot 8\text{ bytes} \right)}{1024^2}\text{ MB}$$
- For Condition A ($L_{\text{grad}} = 3$): $M = \frac{3 \cdot 48 \cdot (2 \cdot 16384 \cdot 4 \cdot 8)}{1048576} = 144.0\text{ MB}$.
- For Condition B ($L_{\text{grad}} = 2$): $M = \frac{3 \cdot 48 \cdot (2 \cdot 16384 \cdot 3 \cdot 8)}{1048576} = 108.0\text{ MB}$.

### 7.4 Prediction Error & Ranking Analysis
`CostModel_v2` exhibits a non-uniform under-prediction of observed training wall-clock time across configurations:
- `A-repeat`: Predicted $121.29\text{ s}$ vs Observed $136.07\text{ s}$ (MAPE: $10.20\%$)
- `A-cache`: Predicted $105.78\text{ s}$ vs Observed $126.29\text{ s}$ (MAPE: $16.04\%$)
- `B-repeat`: Predicted $91.81\text{ s}$ vs Observed $115.10\text{ s}$ (MAPE: $19.04\%$)
- `B-cache`: Predicted $74.22\text{ s}$ vs Observed $86.11\text{ s}$ (MAPE: $13.62\%$)

Diagnostics indicate that this under-prediction stems from un-modeled host language overhead: specifically Python memory allocations and garbage collection during repeated 144-element vector batch loops, which are absent in isolated single-operation microbenchmarks. 

Across the $N=4$ candidate schedule alternatives, the predicted ranking ($B\text{-cache} < B\text{-repeat} < A\text{-cache} < A\text{-repeat}$) matched the measured aggregate ranking perfectly, yielding a Spearman rank correlation of $\rho = 1.000$. However, because this correlation is evaluated across only four alternatives on a single workload, it indicates ranking concordance for the tested schedule space rather than broad statistical proof of universal ranking generalization across arbitrary hardware or models.

---

## 8. Workload-Aware Policy Logic & Actual vs. Predicted Decisions

The `WorkloadSchedulerPolicy` optimizes the decision rule:
$$\mathcal{S}^* = \arg\min_{\mathcal{S} \in \mathcal{F}} T_{\text{total}}(\mathcal{S}, \kappa)$$
subject to:
1. Valid cryptographic scale and level alignment.
2. Numerical gradient error tolerance: $\max |g_{\text{CKKS}} - g_{\text{pt}}| \le 1.0 \times 10^{-4}$.
3. Memory budget feasibility: $M_{\text{cached}}(\mathcal{S}) \le M_{\text{budget}}$ (default $2,048\text{ MB}$).
4. Caching rationality: $T_{\text{total}}(\text{cached}) < T_{\text{total}}(\text{repeated})$.

### 8.1 Actual Experiments vs Analytical Predictions
- **Actual Empirical Executions**:
  - **Full Matched Grid**: 12 full runs ($4\text{ schedules} \times 3\text{ seeds} \times 10\text{ epochs} = 120\text{ epochs}, 360\text{ batches}$).
  - **Pilot Run**: 1 run (Seed 42, 3 epochs, where policy prospective selected `B-cache` and executed it).
- **Analytical Predictions across Epoch Horizons ($E \in \{1, 3, 5, 10\}$)**:
  - $E=1$: Policy predicts `B-repeat` ($9.18\text{ s}$ predicted total). Setup overhead ($1.95\text{ s}$) equals 1-epoch alignment cost, avoiding $108\text{ MB}$ allocation without runtime penalty.
  - $E \in \{3, 5, 10\}$: Policy predicts `B-cache` ($23.63\text{ s}$, $38.09\text{ s}$, $74.22\text{ s}$ predicted totals) due to multi-epoch setup amortization.
  - *Distinction*: These epoch evaluations represent **analytical policy predictions**, not separate 10-epoch training runs.
- **Constraint Fallbacks Verified by Tests**:
  - Memory-constrained fallback ($M_{\text{budget}} = 5\text{ MB}$): Unit tests verify that caching is rejected and the policy falls back to `B-repeat`.
  - Modulus constraint fallback (Condition B disabled): Unit tests verify fallback to `A-cache`.

---

## 9. End-to-End Matched Empirical Evaluation

We executed full 10-epoch training across 4 configurations $\times$ 3 matched seeds ($42, 123, 2026$) on the PTB-XL production workload ($N=17,084, D=48, B=3$):

| Schedule | Strategy | Gradient Modulus Level | Cumulative Time (s) | Mean Epoch Time (s) | Mean Gradient Dot Time (s) | Validation ROC-AUC | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A-repeat`** | Repeated | Level 3 | $136.07 \pm 11.68\text{ s}$ $[14.30\text{s}]$ | $13.61 \pm 1.17\text{ s}$ $[1.43\text{s}]$ | $7.07 \pm 0.61\text{ s}$ $[0.75\text{s}]$ | $0.7842 \pm 0.0041$ $[0.0050]$ | $0.6990 \pm 0.0029$ $[0.0035]$ |
| **`A-cache`**  | Cached   | Level 3 | $126.29 \pm 6.08\text{ s}$ $[7.44\text{s}]$   | $12.41 \pm 0.59\text{ s}$ $[0.72\text{s}]$ | $7.58 \pm 0.40\text{ s}$ $[0.49\text{s}]$ | $0.7842 \pm 0.0041$ $[0.0050]$ | $0.6990 \pm 0.0029$ $[0.0035]$ |
| **`B-repeat`** | Repeated | Level 2 | $115.10 \pm 14.62\text{ s}$ $[17.91\text{s}]$ | $11.51 \pm 1.46\text{ s}$ $[1.79\text{s}]$ | $4.12 \pm 0.52\text{ s}$ $[0.64\text{s}]$ | $0.7842 \pm 0.0041$ $[0.0050]$ | $0.6990 \pm 0.0029$ $[0.0035]$ |
| **`B-cache`**  | Cached   | Level 2 | **$86.11 \pm 4.09\text{ s}$** $[5.00\text{s}]$ | **$8.37 \pm 0.39\text{ s}$** $[0.48\text{s}]$ | **$3.85 \pm 0.20\text{ s}$** $[0.24\text{s}]$ | **$0.7842 \pm 0.0041$** $[0.0050]$ | **$0.6990 \pm 0.0029$** $[0.0035]$ |

*(Primary values reported as Mean $\pm$ Population Standard Deviation [$\text{ddof}=0$], with Sample Standard Deviation [$\text{ddof}=1$] in brackets).*

### 9.1 Speedup Factor Breakdown (Seed-Matched vs Aggregate Means)
1. **Level Intervention Effect (Cached)**:
   - Seed-matched mean ratio: **$1.468 \pm 0.075\times$** ($1.421\times, 1.574\times, 1.410\times$).
   - Ratio of aggregate means: **$1.467\times$** ($126.29\text{s} / 86.11\text{s}$).
2. **Alignment Caching Effect (Condition B)**:
   - Seed-matched mean ratio: **$1.332 \pm 0.104\times$** ($1.477\times, 1.277\times, 1.241\times$).
   - Ratio of aggregate means: **$1.337\times$** ($115.10\text{s} / 86.11\text{s}$).
3. **Alignment Caching Effect (Condition A)**:
   - Seed-matched mean ratio: **$1.080 \pm 0.101\times$** ($0.937\times, 1.155\times, 1.147\times$).
   - Ratio of aggregate means: **$1.077\times$** ($136.07\text{s} / 126.29\text{s}$).
4. **Combined Optimization**:
   - Seed-matched mean ratio: **$1.589 \pm 0.200\times$** ($1.332\times, 1.819\times, 1.617\times$).
   - Ratio of aggregate means: **$1.580\times$** ($136.07\text{s} / 86.11\text{s}$), saving $\approx 50\text{ s}$ per 10-epoch training run.
5. **Gradient Stage Speedup**:
   - Under repeated alignment, gradient dot products run $1.714\times \approx \mathbf{1.71\times}$ faster ($7.07\text{ s} \rightarrow 4.12\text{ s}$).
   - Under cached alignment, gradient dot products run $\mathbf{1.97\times}$ faster ($7.58\text{ s} \rightarrow 3.85\text{ s}$).

### 9.2 Validation Invariance & Plaintext Gradient Concordance
- Validation Accuracy: Seed 42 = $0.7022$, Seed 123 = $0.6994$, Seed 2026 = $0.6952$ (Mean = $0.6990 \pm 0.0029$).
- Validation ROC-AUC: Seed 42 = $0.7899$, Seed 123 = $0.7806$, Seed 2026 = $0.7821$ (Mean = $0.7842 \pm 0.0041$).
- Within each seed, across all four schedules, there is **no observed difference in validation metrics at the reported precision**.
- Decrypted gradient absolute error vs matched double-precision plaintext reference is strictly $\le 7.28 \times 10^{-6}$ (Pass $\le 1.0 \times 10^{-4}$). Discrepancy between Condition A and Condition B gradient vectors is $\le 1.81 \times 10^{-6}$.

### 9.3 Decision Regret Assessment
- For each seed ($42, 123, 2026$), the policy recommendation (`B-cache`) agreed with the empirical oracle minimum runtime:
  $$\text{Regret}(s) = 0.00\text{ s} \quad (0.00\%)$$
- *Scope of Claim*: The policy recommendation agreed with the empirically fastest measured configuration across the three evaluated matched seeds on the tested workload. We do not claim universal optimality across arbitrary unmeasured datasets or hardware targets.

---

## 10. Quantified Research Contributions

| Category | Item | Specification & Status |
| :--- | :--- | :--- |
| **A. Foundations** | RNS-CKKS Scheme | Cheon et al. (2017), Microsoft SEAL backend |
| | SIMD Slot Packing | Batch capacity $S=8,192$, column-major feature ciphertexts |
| | Chebyshev Polynomial Sigmoid | $P_3(z)$ on range $[-5, 5]$, minimax error minimization (Kim et al., 2018) |
| | Hospital / CSP Boundary | Client key generation & gradient decryption (Naresh & Reddi, 2025) |
| **B. Implemented** | Controlled Gradient Entry | Verified Level 2 entry via `pt_mul_one_rescale_to_next` |
| | Atomic Operation Profiler | Characterized microbenchmark latencies across admissible levels |
| | Execution-Context-Aware Model| Calibrated `CostModel_v2` for persistent vs fresh subprocesses |
| | Workload-Aware Scheduler | `WorkloadSchedulerPolicy` with RAM budgeting and caching logic |
| | Amortized Feature Caching | Precomputation of aligned features for multi-epoch training |
| | Parameterized Trainer | Modular architecture supporting automated plan execution |
| | Matched Multi-Seed Design | Paired seed evaluation ensuring identical initial weights |
| **C. Empirical Results**| Gradient Computation Speedup | $1.71\times$ faster gradient dot products under repeated alignment |
| | Feature Alignment Savings | $1.332\times$ seed-matched speedup from caching in Condition B |
| | End-to-End Cumulative Speedup| **$1.589 \pm 0.200\times$ total training speedup** ($136.07\text{ s} \rightarrow 86.11\text{ s}$) |
| | Decision Regret | **$0.00\%$ regret** across evaluated seeds 42, 123, 2026 |
| | Numerical Concordance | Max gradient AE $7.28 \times 10^{-6} \le 1.0 \times 10^{-4}$; no observed val metric difference |

---

## 11. Limitations & Future Directions

1. **Host Language Overhead**:
   The current Python/TenSEAL implementation incurs memory allocation and garbage collection overhead during large ciphertext array manipulation, explaining the $10–19\%$ model under-prediction. Re-implementing the CSP worker in native C++ using direct Microsoft SEAL bindings would eliminate this runtime overhead.
2. **Fixed Modulus Chain Synthesis**:
   The current scheduler selects modulus levels within a predetermined prime chain. Future extensions could synthesize custom RNS prime bit allocations tailored to specific polynomial degree targets and dataset feature widths.
3. **Ciphertext-Only Updates via Bootstrapping**:
   Extending the system to full multi-party computation or bootstrapping would eliminate the need to decrypt gradients at the client each epoch, albeit at substantial computational cost.

---

## 12. References

1. **Albrecht, M., Chase, M., Chen, H., Ding, J., Goldwasser, S., Gorbunov, S., Halevi, S., Hoffstein, J., Laine, K., Lauter, K., Lokam, S., Micciancio, D., Moody, D., Morrison, T., Sahai, A., & Vaikuntanathan, V.** (2018). *Homomorphic Encryption Standard*. HomomorphicEncryption.org, Toronto, ON.
2. **Benaissa, A., Retiat, B., Cebere, B., & Elsayed, A.** (2021). TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption. *arXiv preprint arXiv:2104.03152*.
3. **Cheon, J. H., Kim, A., Kim, M., & Song, Y.** (2017). Homomorphic Encryption for Arithmetic of Approximate Numbers. In *Advances in Cryptology – ASIACRYPT 2017* (pp. 409–437). Springer, Cham. DOI: 10.1007/978-3-319-70694-8_15.
4. **Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E.** (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation*, 101(23), e215–e220. DOI: 10.1161/01.CIR.101.23.e215.
5. **Han, K., Hong, S., Cheon, J. H., & Park, D.** (2019). Logistic Regression on Homomorphically Encrypted Data at Scale. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 9466–9471. DOI: 10.1609/aaai.v33i01.33019466. URL: https://ojs.aaai.org/index.php/AAAI/article/view/5000.
6. **Kim, A., Song, Y., Kim, M., Lee, K., & Cheon, J. H.** (2018). Logistic regression on homomorphic encrypted data based on flexible affine transformations. *BMC Medical Genomics*, 11(Suppl 2), 81. DOI: 10.1186/s12920-018-0399-2.
7. **Microsoft Research.** (2021). Microsoft SEAL (release 3.7+). https://github.com/microsoft/SEAL.
8. **Naresh, V. S., & Reddi, S.** (2025). Exploring the future of privacy-preserving heart disease prediction: a fully homomorphic encryption-driven logistic regression approach. *Journal of Big Data*, 12, Article 52. DOI: 10.1186/s40537-025-01098-6. URL: https://link.springer.com/article/10.1186/s40537-025-01098-6.
9. **Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T.** (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7(1), 154. DOI: 10.1038/s41597-020-0495-6.

---

## 13. Final Scientific Conclusion & Stopping Condition

All experimental results, cost models, policy implementations, and empirical records across Phases 0–3B have been thoroughly audited, mathematically verified, and preserved.
- **Inconsistencies Resolved**: TenSEAL version confirmed as `0.3.17`; Microsoft SEAL backend documented as `3.7–4.0`; Condition B confirmed as `pt_mul_one_rescale_to_next`; Fold 9 and Fold 10 record count discrepancies traced to raw vs eligible filtering and resolved; Condition A caching table typo corrected to unrounded raw ratios ($0.937, 1.155, 1.147 \rightarrow 1.080 \pm 0.101$); validation accuracy properly aggregated ($0.6990 \pm 0.0029$); bibliography DOIs and venues corrected.
- **Reproducibility**: All 48 test suite cases pass; all speedups reproduce directly from raw data tables via `verify_phase3b_statistics.py`.
- **Scientific Integrity**: Fold 10 was strictly excluded from Phase 3B training and validation, and remains untampered.
- **Readiness**: The research artifacts, statistical evaluations, and documentation are complete and frozen, ready for mentor review and peer-reviewed research paper preparation.
