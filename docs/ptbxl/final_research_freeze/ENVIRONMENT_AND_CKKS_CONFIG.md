# Software Environment & CKKS Parameterization Specification

**Document ID**: `docs/ptbxl/final_research_freeze/ENVIRONMENT_AND_CKKS_CONFIG.md`  
**Phase**: Phase 4C — Final Technical Corrections & Research Freeze Validation  
**Audit Date**: October 31, 2026  
**Auditor**: Independent Research Auditor  

---

## 1. Execution Environment & Hardware Architecture

All microbenchmarks, profile evaluations, and full encrypted training experiments were executed on the following frozen platform:

| Parameter | Specification | Verification Source |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Pro (10.0.26200, SP0) | `platform.platform()` |
| **CPU Architecture** | AMD64 Family 25 Model 80 Stepping 0 (AuthenticAMD, Zen 3) | `platform.processor()` |
| **Python Runtime** | Python 3.13.5 (64-bit, MSC v.1943) | `sys.version` |
| **HE Library** | **TenSEAL 0.3.17** | `tenseal.__version__` |
| **Underlying HE Engine**| **Microsoft SEAL 3.7+ (RNS-CKKS)** | TenSEAL backend |
| **Linear Algebra** | NumPy 2.2.0, scikit-learn 1.6.0 | Package manifests |

> [!IMPORTANT]
> **Resolution of TenSEAL Version Discrepancy**:
> Early documentation in Phase 0 cited "TenSEAL 0.3.14" based on legacy project notes. An audit of the active runtime environment confirms that the installed, imported, and profiled library is **TenSEAL 0.3.17**. All project documentation has been unified to 0.3.17.

---

## 2. CKKS Context & Modulus Chain Specification

The RNS-CKKS context is constructed with the following parameters:

```python
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]
)
ctx.global_scale = 2**40
ctx.generate_galois_keys()
ctx.generate_relin_keys()
```

### 2.1 Distinction of Primes and RNS Representation

A critical scientific distinction must be maintained between *configured* modulus primes, the *special key-switching* prime, and *active data-ciphertext* primes:

1. **Configured Primes ($L+2 = 8$)**:
   - The user supplies eight bit-sizes: $[60, 40, 40, 40, 40, 40, 40, 60]$.
   - Total bit length $\sum \log q_i + \log P = 60 + 6 \times 40 + 60 = 360\text{ bits}$.
2. **Special Key-Switching Prime ($P$)**:
   - In Microsoft SEAL, the final 60-bit prime in the chain is designated as the special prime $P$ (or $q_{\text{special}}$). It is reserved for key-switching during relinearization and Galois automorphism rotations to minimize noise growth without consuming data levels.
3. **Fresh Data Ciphertext Active Primes ($L+1 = 7$)**:
   - Because the 8th prime is reserved for key-switching, a fresh data ciphertext entering the circuit has **`coeff_modulus_size() == 7`** (primes $q_0, \dots, q_6$, representing 300 active bits).
   - *It is technically incorrect to claim that fresh ciphertexts have eight active primes.*
4. **Initial Scale ($\Delta$)**:
   - Configured as $2^{40} \approx 1.0995 \times 10^{12}$. Intermediate multiplications rescale by dividing by the current 40-bit prime $q_k$.

### 2.2 Ciphertext Level Transitions through the Training Circuit

| Circuit Stage | Operation | Condition A Modulus State | Condition B Modulus State |
| :--- | :--- | :--- | :--- |
| **Input Features ($X$)** | Fresh encryption | Level 7 (7 primes, 300 bits) | Level 7 (7 primes, 300 bits) |
| **Input Weights ($w$)** | Fresh broadcast encryption | Level 7 (7 primes, 300 bits) | Level 7 (7 primes, 300 bits) |
| **Forward Linear ($z = Xw$)** | Ct-Ct mul + automatic rescale | Level 6 (6 primes, 260 bits) | Level 6 (6 primes, 260 bits) |
| **Sigmoid Eval ($P_3(z)$)** | Degree-3 Chebyshev poly | Level 3 (4 primes, 180 bits) | Level 3 (4 primes, 180 bits) |
| **Error Computation ($e = \hat{y} - y$)**| Ct-Ct sub (with burned label $y$) | Level 3 (4 primes, 180 bits) | Level 3 (4 primes, 180 bits) |
| **Condition B Intervention** | `enc_err * 1.0` | *None (stays Level 3)* | **Level 2 (3 primes, 140 bits)** |
| **Feature Alignment** | Feature burn | `burn(X, 4)` $\rightarrow$ **Level 3** | `burn(X, 5)` $\rightarrow$ **Level 2** |
| **Gradient Entry** | Operand alignment verified | **Level 3** ($e \in L_3, X \in L_3$) | **Level 2** ($e \in L_2, X \in L_2$) |
| **Gradient Dot Product** | Ct-Ct mul + slot summation | Level 2 (3 primes, 140 bits) | Level 1 (2 primes, 100 bits) |
| **Decryption at Client** | Secret key decrypt | Decrypted at Level 2 | Decrypted at Level 1 |

---

## 3. Cryptographic Security Parameter Conformance

The security of the CKKS scheme rests on the hardness of the Ring Learning With Errors (RLWE) problem.

### 3.1 Standards Compliance (Homomorphic Encryption Standard)
Per the *Homomorphic Encryption Standard* (Albrecht et al., 2018), for a power-of-two polynomial modulus degree $N_{\text{poly}} = 16,384$ with discrete Gaussian error distribution ($\sigma = 3.2$):
- **128-bit Classical Security Ceiling**: Maximum total modulus bit length $\log Q \le 438\text{ bits}$.
- **128-bit Quantum Security Bound**: Under conservative primal BKZ lattice sieving estimators, $\log Q \le 411\text{ bits}$.

### 3.2 Modulus Accounting & Security Assessment
- **Configured Modulus**: Total configured bit-length including the special key-switching prime is $360\text{ bits}$.
- **Data Modulus**: Active data ciphertext modulus in fresh ciphertexts is $300\text{ bits}$.
- **Evaluation**:
  $$300\text{ bits} < 360\text{ bits} \le 411\text{ bits} \le 438\text{ bits}$$
- **Scoping Statement**:
  > *The parameter set is within the stated 128-bit classical security parameter limit under the cited Homomorphic Encryption Standard (Albrecht et al., 2018) and its discrete Gaussian error assumptions ($\sigma = 3.2$).*
- *Negative Constraint*: We do not claim a formal mathematical proof of security beyond checking adherence to standard parameter tables, nor do we assert unqualified "post-quantum security" without citing the underlying lattice reduction estimator assumptions.

---

## 4. Condition B Intervention Semantics

In earlier notes, Condition B was colloquially termed a "modulus switch". This audit formally documents the precise semantics:

1. **Absence of Standalone ModSwitch**:
   TenSEAL 0.3.17 does not wrap SEAL's `Evaluator::mod_switch_to_next` method for CKKS vectors.
2. **Actual Implementation**:
   Condition B is executed via:
   ```python
   enc_err_adj = enc_err * 1.0
   ```
   In TenSEAL, multiplying a `CKKSVector` by a scalar float invokes `Evaluator::multiply_plain`, which produces an unscaled product whose scale becomes $\Delta \times \Delta_0$, followed immediately by `Evaluator::rescale_to_next`, which divides by the current prime modulus and drops the ciphertext to the next lower RNS level.
3. **Correct Scientific Terminology**:
   Condition B must be documented as:
   > *"Plaintext multiplication by 1.0 followed by automatic rescaling (`pt_mul_one_rescale_to_next`)"*.
