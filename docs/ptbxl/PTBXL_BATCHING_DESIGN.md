# PTB-XL SIMD Batching and Ciphertext Packing Design

## 1. Executive Summary & Design Overview

Extending our privacy-preserving Logistic Regression training pipeline from MHEALTH to the PTB-XL clinical ECG dataset introduces a key architectural constraint:
- **PTB-XL Training Sample Count**: $N_{\text{train}} = 17,084$ records (official Folds 1–8).
- **Feature Dimension**: $D = 48$ statistical features (12 leads $\times$ 4 summary statistics).
- **CKKS Polynomial Modulus Degree**: $N_{\text{poly}} = 16384$.
- **SIMD Slot Capacity per Ciphertext**: $S = N_{\text{poly}} / 2 = 8,192$ complex slots.

Because $17,084 > 8,192$, the entire training set cannot reside in a single set of column ciphertexts. The dataset must be partitioned into:
$$B = \left\lceil \frac{17,084}{8,192} \right\rceil = 3 \text{ batches}$$

This document formalizes the data layout, ciphertext packing strategy, handling of the final partial batch, and the exact mathematical formulation for global gradient aggregation.

---

## 2. Seven Core Architectural Answers (Step A Audit)

Based on our inspection of the reference implementation (`helr_mhealth_train.py`, Phase 0 baseline, and Phase 1–4 ablation suites):

### 1. How samples are packed into ciphertext slots
Samples are packed along the SIMD slot dimension. For a batch of size $P \le 8192$, record $i \in [0 \dots P-1]$ occupies SIMD slot $i$. Operations across slots execute in Single Instruction, Multiple Data (SIMD) parallel fashion via ring automorphisms and dyadic multiplications.

### 2. Layout of ciphertexts and feature columns
The architecture strictly follows the **column-packed representation** (Naresh & Reddi, 2025):
- Each ciphertext encodes **one feature column** across all samples in the batch.
- For $D = 48$ features, each batch $b \in \{1, 2, 3\}$ consists of 48 feature ciphertexts:
  $$\text{enc\_X}_{b} = \left[ \text{Enc}(X_{b}[:, 0]), \text{Enc}(X_{b}[:, 1]), \dots, \text{Enc}(X_{b}[:, 47]) \right]$$
- Labels for the batch are packed into a single ciphertext: $\text{enc\_y}_{b} = \text{Enc}(y_{b})$.
- **Total Ciphertext Footprint for Dataset**:
  - Feature ciphertexts: $48 \times 3 = 144$ ciphertexts.
  - Label ciphertexts: $1 \times 3 = 3$ ciphertexts.
  - Overall dataset ciphertexts: $147$ ciphertexts.

### 3. Representation of weights and intermediate predictions
- **Weights ($\text{Enc}(w)$)**: To preserve privacy of the model during training on an untrusted cloud server (CSP), the Hospital encrypts the weight vector. Each scalar weight $w_j$ is broadcast across all 8,192 SIMD slots:
  $$\text{enc\_w}_j = \text{Enc}([w_j, w_j, \dots, w_j]^T) \in \mathbb{R}^{8192}$$
  Because all three batches share the slot capacity of 8,192, the 48 weight ciphertexts are encrypted **once** by the Hospital and reused across Batches 1, 2, and 3.
- **Intermediate Predictions ($\text{Enc}(z)$ and $\text{Enc}(p)$)**:
  - $\text{enc\_z}_b = \sum_{j=1}^{48} (\text{enc\_X}_{b, j} \cdot \text{enc\_w}_j)$ is evaluated as a ciphertext via $48$ Ct $\times$ Ct multiplications and additions.
  - $\text{enc\_p}_b = P_d(\text{enc\_z}_b)$ is evaluated homomorphically via polynomial evaluation (Ct $\times$ Ct and Ct $\times$ Pt).
  - $\text{enc\_err}_b = \text{enc\_p}_b - \text{burn}(\text{enc\_y}_b)$ is evaluated as a ciphertext subtraction.
- **Decryption Boundary**: Only the final gradient scalars $\text{enc\_grad}_{b, j}$ are decrypted by the Hospital. The CSP never decrypts any intermediate vector, sample, or weight.

### 4. Computation of feature gradients and intercept/bias
- **Feature Gradients**: For feature $j$ in batch $b$, the unnormalized gradient sum is computed by multiplying the error ciphertext by the aligned feature ciphertext, followed by a slot summation:
  $$\text{enc\_S}_{b, j} = \left( \text{enc\_err}_b \cdot \text{enc\_X}_{b, j} \right).\text{sum}()$$
- **Intercept / Bias**: In accordance with the reference implementation (`helr_mhealth_train.py`), **no separate intercept/bias parameter is used** ($w \in \mathbb{R}^{48}$). All 48 features were standardized to zero mean and unit variance ($\mu = 0, \sigma^2 = 1$) during preprocessing on Folds 1–8, centering the data and allowing effective linear separation without an unregularized bias term.

### 5. Application of regularization
Regularization is applied **once per global full-dataset update**, not separately within each batch. Decrypted gradient sums from all three batches are accumulated, normalized by the total valid training records ($N = 17,084$), and then $L_2$ regularization $\lambda w_j$ is added globally:
$$g_j = \frac{1}{N} \sum_{b=1}^3 S_{b, j} + \lambda w_j$$

### 6. Sigmoid function: Exact vs. Polynomial
The encrypted circuit utilizes polynomial sigmoid approximations because the transcendental logistic function $\sigma(z) = 1 / (1 + e^{-z})$ cannot be computed directly under additions and multiplications in Ring-LWE homomorphic encryption:
- **Degree 1 (Linear)**: $P_1(z) = 0.5 + 0.197 z$
- **Degree 3 (Cubic)**: $P_3(z) = 0.5 + 0.15012 z - 0.001593 z^3$
- **Degree 5 (Quintic)**: $P_5(z) = 0.5 + 0.217101 z - 0.007823 z^3 + 0.000118 z^5$

In our numerical validation harness, we distinguish between:
1. *Approximation Error*: Plaintext exact sigmoid vs. Plaintext polynomial sigmoid.
2. *Encryption Error*: Plaintext polynomial sigmoid vs. Decrypted CKKS gradient.

### 7. Handling of the final partial batch
The final partial batch (Batch 3) contains 700 valid records and 7,492 unused slots. Handling is detailed in Section 3 below.

---

## 3. Three-Batch Partitioning and Capacity Allocation

With $N_{\text{train}} = 17,084$ and slot capacity $S = 8,192$:

| Batch Index | Sample Range in Folds 1–8 | Valid Records ($n_b$) | Unused / Padded Slots ($S - n_b$) | Slot Allocation |
|:---:|:---:|---:|---:|:---:|
| **Batch 1** | $0 \dots 8,191$ | 8,192 | 0 | Full Capacity (100%) |
| **Batch 2** | $8,192 \dots 16,383$ | 8,192 | 0 | Full Capacity (100%) |
| **Batch 3** | $16,384 \dots 17,083$ | 700 | 7,492 | Padded with Zeroes |
| **Total** | Full Training Split | **17,084** | **7,492** | $\sum n_b = N_{\text{train}}$ |

### Deterministic Zero-Padding & Masking Mechanism for Batch 3
For Batch 3, feature matrices and labels are padded to exactly 8,192 slots:
$$X_{3, \text{padded}}[i, j] = \begin{cases} X_{16384 + i, j} & \text{if } 0 \le i < 700 \\ 0.0 & \text{if } 700 \le i < 8,192 \end{cases}$$
$$y_{3, \text{padded}}[i] = \begin{cases} y_{16384 + i} & \text{if } 0 \le i < 700 \\ 0.0 & \text{if } 700 \le i < 8,192 \end{cases}$$

#### Proof of Zero Contribution from Padded Slots:
1. In the forward pass, for any padded slot $i \in [700 \dots 8191]$:
   $$z_i = \sum_{j=1}^{48} X_{3, \text{padded}}[i, j] \cdot w_j = \sum_{j=1}^{48} 0.0 \cdot w_j = 0.0$$
2. Under polynomial evaluation, $P_d(0) = c_0 = 0.5$, so:
   $$p_i = 0.5 \quad \text{for } i \in [700 \dots 8191]$$
3. The error in padded slots is:
   $$e_i = p_i - y_{3, \text{padded}}[i] = 0.5 - 0.0 = 0.5$$
4. In the gradient step, each feature gradient accumulates:
   $$S_{3, j} = \sum_{i=0}^{8191} e_i \cdot X_{3, \text{padded}}[i, j]$$
   Decomposing into valid and padded partitions:
   $$S_{3, j} = \sum_{i=0}^{699} e_i X_{3, \text{padded}}[i, j] + \sum_{i=700}^{8191} 0.5 \cdot 0.0 = \sum_{i=0}^{699} e_i X_{i, j} + 0.0$$
5. Thus, the padded slots contribute **identically zero** to the decrypted sum!

---

## 4. Global Gradient Aggregation Formulation

The three batches represent **one single global full-dataset gradient step**, avoiding the gradient noise of stochastic mini-batching.

For feature $j \in \{0 \dots 47\}$:
$$g_j = \frac{1}{N_{\text{train}}} \sum_{b=1}^3 S_{b, j} + \lambda w_j = \frac{1}{17,084} \left( S_{1, j} + S_{2, j} + S_{3, j} \right) + \lambda w_j$$

Where:
- $S_{1, j} = \sum_{i \in \text{Batch 1}} e_i X_{i, j}$ (8,192 valid records)
- $S_{2, j} = \sum_{i \in \text{Batch 2}} e_i X_{i, j}$ (8,192 valid records)
- $S_{3, j} = \sum_{i \in \text{Batch 3}} e_i X_{i, j}$ (700 valid records)

### Weighted Average Equivalence:
If per-batch average gradients $\bar{g}_{b, j} = \frac{1}{n_b} S_{b, j}$ are computed:
$$g_j = \sum_{b=1}^3 \left( \frac{n_b}{N_{\text{train}}} \right) \bar{g}_{b, j} + \lambda w_j = \frac{8192}{17084} \bar{g}_{1, j} + \frac{8192}{17084} \bar{g}_{2, j} + \frac{700}{17084} \bar{g}_{3, j} + \lambda w_j$$

Equal averaging ($\frac{1}{3} \sum \bar{g}_b$) is mathematically invalid because Batch 3 contains only 700 records ($\approx 4.1\%$ of the dataset). The implementation strictly normalizes by valid sample counts.

---

## 5. Verification Protocol Summary

The batching and aggregation design is validated through three distinct comparison pathways:
1. **Plaintext Batch Equivalence**: Asserts that full-matrix plaintext gradient equals three-batch aggregated plaintext gradient to machine precision ($\Delta < 10^{-12}$).
2. **Padding Invariance**: Asserts that changing padded values in Batch 3 produces zero discrepancy in the resulting gradient.
3. **Encrypted Discrepancy**: Asserts that decrypted CKKS aggregated gradients match the plaintext polynomial reference within $\text{MaxAE} \le 1.0 \times 10^{-4}$.
