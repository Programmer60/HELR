# PTB-XL CKKS Gradient Validation Report

## 1. Executive Summary & Experimental Methodology

This report documents the rigorous numerical validation of the three-batch SIMD column-packing and encrypted gradient aggregation harness for the PTB-XL cardiac ECG dataset ($N = 17,084$ training records, $D = 48$ statistical features derived from 12 leads).

### Experimental Context & Constraints:
- **Dataset Split**: Official PTB-XL training folds 1–8 ($17,084$ records, $D=48$). Held-out validation (Fold 9, $2,146$ records) and test (Fold 10, $2,158$ records) were strictly excluded from any gradient calculation or feature scaling.
- **Cryptographic Scheme**: TenSEAL / Microsoft SEAL CKKS homomorphic encryption.
- **Cryptographic Parameters**:
  - Polynomial modulus degree: $N_{\text{poly}} = 16384$
  - RNS coefficient modulus bit sizes: `[60, 40, 40, 40, 40, 40, 40, 60]` ($\sum q_i = 360$ bits, usable multiplicative depth: 7 levels)
  - Global scale factor: $\Delta = 2^{40} \approx 1.0995 \times 10^{12}$
  - SIMD slot capacity: $S = N_{\text{poly}} / 2 = 8,192$ complex slots
- **Sigmoid Approximation**: Degree-3 Kim et al. polynomial approximation:
  $$P_3(z) = 0.5 + 0.15012 z - 0.001593 z^3$$
- **Regularization**: $L_2$ weight decay with $\lambda = 0.01$ applied once globally at the trusted Hospital decryptor.
- **Acceptance Criteria**:
  - Plaintext batching equivalence: $\Delta_{\text{batch}} \le 1.0 \times 10^{-10}$
  - CKKS encryption numerical error: $\Delta_{\text{CKKS}} \le 1.0 \times 10^{-4}$

All nine automated verification checks—spanning zero weights, five deterministic random weight vectors, trained plaintext baseline weights, synthetic batch-edge scaling, and partial-batch label masking—passed without exception.

---

## 2. SIMD Batching Architecture & Mathematical Aggregation

### 2.1 Multi-Batch Partitioning ($N = 17,084, S = 8,192$)

Because the training set exceeds the slot capacity of a single CKKS ciphertext ($17,084 > 8,192$), the dataset is partitioned into $B = \lceil 17,084 / 8,192 \rceil = 3$ column-packed batches:

| Batch | Record Indices | Valid Records | Padded / Unused Slots | Padded Array Shape |
| :--- | :--- | :---: | :---: | :---: |
| **Batch 1** | $0 \le i < 8,192$ | $8,192$ | $0$ | $(8192, 48)$ |
| **Batch 2** | $8,192 \le i < 16,384$ | $8,192$ | $0$ | $(8192, 48)$ |
| **Batch 3** | $16,384 \le i < 17,084$ | $700$ | $7,492$ | $(8192, 48)$ |
| **Total** | Full Training Split | **$17,084$** | **$7,492$** | $3 \times (8192, 48)$ |

### 2.2 Ciphertext Count and Allocation

- **Feature Matrices**: 48 features $\times$ 3 batches = **144** feature ciphertexts.
- **Labels**: 1 label vector $\times$ 3 batches = **3** label ciphertexts.
- **Total Pre-encrypted Data**: **147** ciphertexts (encryption time: 2.32 s).
- **Weight Vectors**: 48 weight scalars broadcast into 8,192 slots = **48** weight ciphertexts.
- **Reusability**: Because Batch 3 is padded to 8,192 slots, the Hospital encrypts the 48 weight ciphertexts **once** and the CSP reuses them across all 3 batches without re-encryption.

### 2.3 Mathematical Proof of Zero-Padding Invariance

In Batch 3, slots $700 \le i < 8,192$ are padded with zero feature values: $X_{3, i, j} = 0 \quad \forall j \in \{1, \dots, 48\}$.

1. **Linear Combination**:
   $$z_{3, i} = \sum_{j=1}^{48} X_{3, i, j} w_j = \sum_{j=1}^{48} 0 \cdot w_j = 0$$
2. **Polynomial Prediction**:
   $$P_3(z_{3, i}) = P_3(0) = 0.5 + 0.15012(0) - 0.001593(0)^3 = 0.5$$
3. **Encrypted Error Vector**:
   $$e_{3, i} = P_3(z_{3, i}) - y_{3, i} = 0.5 - y_{\text{pad}}$$
   Even if $y_{\text{pad}} = 0$ (yielding $e_{3, i} = 0.5$), or any arbitrary finite value:
4. **Gradient Outer Product**:
   $$\text{term}_{3, i, j} = e_{3, i} \cdot X_{3, i, j} = e_{3, i} \cdot 0 = 0$$
5. **Decrypted Slot Summation**:
   $$S_{3, j} = \sum_{i=0}^{8191} e_{3, i} X_{3, i, j} = \sum_{i=0}^{699} e_{3, i} X_{3, i, j} + \sum_{i=700}^{8191} 0 = \sum_{i=0}^{699} e_{3, i} X_{3, i, j}$$

Thus, the padded slots contribute identically zero to the unnormalized sum. The CSP does not need to perform expensive slot-masking ciphertext multiplications, saving both computational runtime and multiplicative depth.

### 2.4 Global Aggregation Formula

The unnormalized gradient sum for feature $j$ across all $B=3$ batches is:
$$S_j = \sum_{b=1}^3 S_{b, j} = \sum_{b=1}^3 \left( \sum_{i \in \text{valid}} e_{b, i} X_{b, i, j} \right)$$
The trusted Hospital decrypts the 48 batch-summed vectors, sums across slots, and applies the exact sample normalization $N = 17,084$ and $L_2$ regularization:
$$g_j = \frac{1}{17,084} S_j + \lambda w_j$$

---

## 3. Three Distinct Comparison Pathways

To prevent conflating polynomial approximation error with CKKS homomorphic arithmetic precision, the harness isolates three orthogonal discrepancies:

```
[Plaintext Exact Sigmoid]
       │
       │ Comparison 2: Mathematical Approximation Error (Δ_approx = |g_poly - g_exact|)
       ▼
[Plaintext Degree-3 Sigmoid (Single Matrix)]
       │
       │ Comparison 1: Plaintext Batch Equivalence (Δ_batch = |g_batched - g_single|)
       ▼
[Plaintext Degree-3 Sigmoid (3-Batch Aggregation)]
       │
       │ Comparison 3: CKKS Encryption Numerical Error (Δ_CKKS = |g_CKKS - g_poly|)
       ▼
[Decrypted CKKS Gradient (3-Batch Aggregation)]
```

1. **Comparison 1: Plaintext Batch Equivalence**
   $$\Delta_{\text{batch}} = \max_j | g_{\text{batched}, j}^{\text{PT}} - g_{\text{single}, j}^{\text{PT}} |$$
   Verifies that multi-batch partitioning, zero-padding, and global normalization introduce zero algorithmic discrepancy beyond 64-bit floating-point roundoff.
2. **Comparison 2: Mathematical Approximation Error**
   $$\Delta_{\text{approx}} = \max_j | g_{\text{poly}, j}^{\text{PT}} - g_{\text{exact}, j}^{\text{PT}} |$$
   Quantifies the intrinsic bias incurred by replacing $\sigma(z) = 1/(1+e^{-z})$ with $P_3(z)$.
3. **Comparison 3: CKKS Encryption Numerical Error**
   $$\Delta_{\text{CKKS}} = \max_j | g_{\text{CKKS}, j} - g_{\text{poly}, j}^{\text{PT}} |$$
   Measures the physical noise floor of CKKS fixed-point rescaling, ciphertext multiplications, and slot decryptions against the matched polynomial reference.

---

## 4. Comprehensive Validation Results

The table below summarizes the empirical validation across all seven weight vector configurations, synthetic batch-edge scaling, and partial-batch label perturbation.

| Case ID | Description | $\|w\|_2$ | Plaintext Batch Equivalence MaxAE | Batching Pass ($\le 10^{-10}$) | Exact vs Poly MaxAE ($\Delta_{\text{approx}}$) | Exact vs Poly MeanAE | CKKS vs Poly MaxAE ($\Delta_{\text{CKKS}}$) | CKKS vs Poly MeanAE | CKKS Mean Relative Error | CKKS Pass ($\le 10^{-4}$) | $\|g_{\text{CKKS}}\|_2$ | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1** | $w = \mathbf{0}$ | $0.0000$ | $0.00 \times 10^0$ | **PASS** | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $4.91 \times 10^{-6}$ | $1.60 \times 10^{-6}$ | $3.05 \times 10^{-5}$ | **PASS** | $0.4823$ | 11.35 s |
| **Case 2** | Random $\mathcal{N}(0, 0.06^2)$ | $0.4066$ | $2.78 \times 10^{-17}$ | **PASS** | $2.68 \times 10^{-2}$ | $9.73 \times 10^{-3}$ | $3.94 \times 10^{-6}$ | $1.49 \times 10^{-6}$ | $3.97 \times 10^{-5}$ | **PASS** | $0.4278$ | 11.47 s |
| **Case 3** | Random $\mathcal{N}(0, 0.06^2)$ | $0.4163$ | $2.78 \times 10^{-17}$ | **PASS** | $5.70 \times 10^{-2}$ | $9.28 \times 10^{-3}$ | $4.95 \times 10^{-6}$ | $1.83 \times 10^{-6}$ | $3.19 \times 10^{-5}$ | **PASS** | $0.5060$ | 12.80 s |
| **Case 4** | Random $\mathcal{N}(0, 0.06^2)$ | $0.4110$ | $2.78 \times 10^{-17}$ | **PASS** | $1.98 \times 10^{-2}$ | $5.36 \times 10^{-3}$ | $5.13 \times 10^{-6}$ | $1.75 \times 10^{-6}$ | $4.50 \times 10^{-5}$ | **PASS** | $0.5174$ | 12.98 s |
| **Case 5** | Random $\mathcal{N}(0, 0.06^2)$ | $0.4002$ | $2.78 \times 10^{-17}$ | **PASS** | $2.48 \times 10^{-2}$ | $1.18 \times 10^{-2}$ | $5.85 \times 10^{-6}$ | $2.13 \times 10^{-6}$ | $3.18 \times 10^{-5}$ | **PASS** | $0.6418$ | 11.75 s |
| **Case 6** | Random $\mathcal{N}(0, 0.06^2)$ | $0.3893$ | $2.78 \times 10^{-17}$ | **PASS** | $1.45 \times 10^{-2}$ | $5.48 \times 10^{-3}$ | $5.53 \times 10^{-6}$ | $1.83 \times 10^{-6}$ | $9.15 \times 10^{-5}$ | **PASS** | $0.5548$ | 13.18 s |
| **Case 7** | Plaintext Baseline $w^*$ | $2.5754$ | $6.94 \times 10^{-18}$ | **PASS** | $1.61 \times 10^{-1}$ | $3.39 \times 10^{-2}$ | $6.06 \times 10^{-6}$ | $1.09 \times 10^{-6}$ | $6.64 \times 10^{-5}$ | **PASS** | $0.3477$ | 12.30 s |

---

## 5. Edge-Case Validation & Padding Robustness

### 5.1 Case 8: Synthetic Batch-Edge Test ($N = 10,000$)

To guarantee that the batching algorithm is fully generalizable to any arbitrary dataset size not divisible by 8,192, we evaluated a synthetic split of $N = 10,000$ records across 2 batches ($8,192$ valid in Batch 1; $1,808$ valid and $6,384$ padded in Batch 2):
- **Plaintext Batch Equivalence MaxAE**: $0.00 \times 10^0$ (exact machine zero)
- **Status**: **PASS**

### 5.2 Case 9: Partial-Batch Masking Invariance Test

To verify that corrupted or non-zero label entries in the padded slots of Batch 3 cannot leak into the gradient, we injected random uniform noise $\mathcal{U}(0, 1)$ into $y_{3, i}$ for all padded slots $700 \le i < 8,192$.
- **Observed Gradient Discrepancy MaxAE**: $3.54 \times 10^{-6}$
- **Analysis**: The difference between the clean zero-padded gradient and the perturbed padded gradient is within the CKKS encryption noise floor ($< 4 \times 10^{-6}$) and well below the $1.0 \times 10^{-4}$ tolerance threshold.
- **Status**: **PASS**

---

## 6. Analytical Discussion & Findings

### 6.1 Cryptographic Precision vs. Weight Norm
Across all test cases, the maximum absolute CKKS encryption error remained tightly bounded between $3.94 \times 10^{-6}$ and $6.06 \times 10^{-6}$, with mean absolute errors between $1.09 \times 10^{-6}$ and $2.13 \times 10^{-6}$. Even when $\|w\|_2$ scaled from $0.0$ to $2.5754$, the CKKS noise increased by less than $1.2 \times 10^{-6}$. This confirms that scale $2^{40}$ with modulus bits `[60, 40, 40, 40, 40, 40, 40, 60]` provides over 17 bits of fractional precision headroom beyond our $10^{-4}$ tolerance.

### 6.2 Mathematical Approximation Error vs. Homomorphic Precision
A critical finding from Table 4 is the clear divergence between mathematical sigmoid approximation error and cryptographic error:
- At $w = \mathbf{0}$, $z = 0$, where $P_3(0) = \sigma(0) = 0.5$ exactly. Here, $\Delta_{\text{approx}} = 0.00 \times 10^0$, while $\Delta_{\text{CKKS}} = 4.91 \times 10^{-6}$.
- At the trained baseline weights ($\|w\|_2 = 2.5754$), linear activations $z = X w$ extend into $|z| > 2$, where $P_3(z)$ deviates from the true sigmoid. Here, $\Delta_{\text{approx}} = 0.1611$, yet $\Delta_{\text{CKKS}} = 6.06 \times 10^{-6}$.

If an evaluation were to compare decrypted CKKS gradients directly against exact sigmoid plaintext gradients without isolating the polynomial step, one would falsely conclude that CKKS has an error of $16\%$, when the cryptographic computation itself has an error of only $0.0006\%$.

### 6.3 Computational Latency
The complete encrypted gradient computation across all 17,084 records (144 feature ciphertexts, 3 label ciphertexts, 48 weight broadcast operations, polynomial evaluation, and dot product aggregation) requires approximately **11.3 to 13.2 seconds** per evaluation on the host CPU.

---

## 7. Deliverables & Acceptance Checklist

| # | Requirement | Result | Verified |
| :---: | :--- | :--- | :---: |
| **1** | Plaintext batching vs. single-matrix equivalence | $\Delta_{\text{batch}} \le 2.78 \times 10^{-17}$ across all cases | **YES** |
| **2** | Maximum absolute CKKS error across all 7 cases | $6.0623 \times 10^{-6}$ (Case 7: baseline weights) | **YES** |
| **3** | Mean absolute CKKS error across all 7 cases | $1.6766 \times 10^{-6}$ | **YES** |
| **4** | Maximum relative CKKS error across all 7 cases | $9.1498 \times 10^{-5}$ | **YES** |
| **5** | Pass/fail against $\epsilon_{\text{CKKS}} \le 1.0 \times 10^{-4}$ threshold | All 7 cases passed ($\Delta_{\text{CKKS}} \le 6.06 \times 10^{-6} \ll 10^{-4}$) | **YES** |
| **6** | Separation of Comp 2 ($\Delta_{\text{approx}}$) and Comp 3 ($\Delta_{\text{CKKS}}$) | Evaluated and documented side-by-side | **YES** |
| **7** | Synthetic batch-edge test ($N = 10,000$ across 2 batches) | MaxAE = $0.00 \times 10^0$ (exact machine zero) | **YES** |
| **8** | Partial-batch masking invariance test | MaxAE = $3.54 \times 10^{-6} \le 1.0 \times 10^{-4}$ | **YES** |

---

## 8. Artifact and File Manifest

- **Harness Script**: [ptbxl_batching_validation.py](file:///c:/Users/mishr/Desktop/ML/venv/MinorProject/results/ptbxl/batching_validation/ptbxl_batching_validation.py)
- **Metrics JSON**: [ptbxl_batching_validation.json](file:///c:/Users/mishr/Desktop/ML/venv/MinorProject/results/ptbxl/batching_validation/ptbxl_batching_validation.json)
- **Metrics CSV**: [ptbxl_batching_validation.csv](file:///c:/Users/mishr/Desktop/ML/venv/MinorProject/results/ptbxl/batching_validation/ptbxl_batching_validation.csv)
- **Execution Log**: [ptbxl_batching_validation.log](file:///c:/Users/mishr/Desktop/ML/venv/MinorProject/results/ptbxl/batching_validation/ptbxl_batching_validation.log)
- **Batching Architectural Design**: [PTBXL_BATCHING_DESIGN.md](file:///c:/Users/mishr/Desktop/ML/venv/MinorProject/docs/ptbxl/PTBXL_BATCHING_DESIGN.md)
