# Limitations, Threat Model & Architectural Boundaries

**Document ID**: `docs/ptbxl/final_research_freeze/LIMITATIONS_AND_THREAT_MODEL.md`  
**Phase**: Phase 4B — Statistical Corrections, Evidence Audit & Research Freeze Finalization  
**Date**: October 31, 2026  

---

## 1. System Architecture & Trust Boundary

The privacy-preserving training pipeline operates across a partitioned client-server trust boundary between a **Trusted Hospital (Client)** and an **Untrusted Cloud Service Provider (CSP / Worker)**.

We refer to this workflow as **CKKS-assisted privacy-preserving training with client-side gradient decryption and plaintext optimizer updates**.

```mermaid
flowchart TD
    subgraph Hospital["Trusted Hospital (Client Realm)"]
        H_Keys["Generate & Hold Secret Key<br/>Public, Galois, Relin Keys"]
        H_Data["Clinical ECG Data<br/>PTB-XL Features (Folds 1-8)"]
        H_Enc["Encrypt Batched Features & Labels<br/>Encrypt Weights w_t"]
        H_Dec["Decrypt Aggregated Gradients<br/>sum(g_batch)"]
        H_Opt["Plaintext Optimizer Update<br/>L2 Penalty + Clipping + LR Update"]
        H_Val["Clinical Validation on Fold 9<br/>Exact Sigmoid Evaluation"]
    end

    subgraph CSP["Untrusted Cloud Service Provider (Server Realm)"]
        C_Forward["Homomorphic Forward Linear<br/>z = sum(X_j * w_j) (Level 7 -> 6)"]
        C_Poly["Chebyshev Sigmoid Approx<br/>P_3(z) (Level 6 -> 3)"]
        C_Err["Encrypted Error Computation<br/>e = P_3(z) - y_aligned (Level 3)"]
        C_Sched["Workload-Aware Schedule Execution<br/>Level 2 Intervention & Alignment Caching"]
        C_Grad["Encrypted Gradient Dot Products<br/>g_j = sum(e * X_j) (O(D) Ops)"]
    end

    H_Keys -.->|Public Parameters & Eval Keys| CSP
    H_Enc ==>|Ciphertexts: enc(X), enc(y), enc(w)| CSP
    C_Grad ==>|Encrypted Gradient Sums: enc(g)| H_Dec
    H_Opt -.->|Re-encrypt updated w_t+1| H_Enc
```

### 1.1 Detailed Boundary Responsibilities

1. **Hospital (Trusted Client)**:
   - Holds the CKKS secret key. Secret key material **never leaves the hospital**.
   - Encrypts clinical features $X$ and binary labels $y$ using SIMD slot packing ($8,192$ records per ciphertext).
   - Broadcast-encrypts weight vectors $w$ ($48$ ciphertexts, each slot replicated).
   - Receives encrypted batch gradient sums from the CSP and decrypts them using the secret key.
   - Normalizes by $N=17,084$, adds plaintext $L_2$ regularization ($\lambda = 0.01$), performs gradient clipping ($\text{clip} = 1.0$), and executes the gradient descent step in plaintext.
   - Re-encrypts the updated weights for the subsequent epoch.
   - Conducts validation evaluation on Fold 9 in plaintext using decrypted weights.

2. **Cloud Service Provider (Untrusted Worker)**:
   - Executes homomorphic matrix-vector multiplications, polynomial activations, and gradient dot products exclusively over ciphertexts.
   - Applies the selected execution plan (`A-repeat`, `A-cache`, `B-repeat`, or `B-cache`).
   - Operates entirely under ciphertext arithmetic; possesses zero secret key material and cannot decrypt patient records or intermediate logits.

---

## 2. Information Flow, Leakage Analysis & Scoping

To avoid over-claiming the privacy properties of homomorphic encryption, the information flow and potential leakage channels are explicitly categorized:

| Dimension | Protection Mechanism | Residual Information / Limitations |
| :--- | :--- | :--- |
| **Feature & Label Values** | IND-CPA security under RLWE (CKKS) | Encrypted values are confidential against a passive adversary without the secret key. |
| **Public Metadata & Workload Dimensions** | None (Public by Design) | The CSP intentionally observes: total sample count $N$, feature dimension $D$, batch count $B$, polynomial degree $d=3$, and epoch count $E$. |
| **Execution Schedules & Access Patterns** | None (Public Circuit Structure) | The CSP observes which schedule is being executed, timing of batch steps, and memory allocation for cached ciphertexts. |
| **Timing & Side-Channels** | Out of Scope | Computation latency varies with ciphertext levels. A side-channel adversary observing micro-architectural timing or memory access patterns is not addressed by standard CKKS semantic security. |
| **Malicious Server Deviation** | Semi-Honest Model Only | The CSP is assumed to be honest-but-curious. Active tampering (e.g., injecting malicious ciphertext perturbations into gradient sums) is not detected by basic CKKS without zero-knowledge proofs or verifiable computation. |
| **Decrypted Gradient Model Leakage** | Plaintext Updates at Hospital | Decrypted gradients and final model weights do **not** incorporate differential privacy (DP) noise. Membership inference or reconstruction attacks from the trained weights are outside the cryptographic threat model. |

---

## 3. Negative Constraints and Non-Claims

The following negative constraints are formally recorded:

1. **Not Fully Homomorphic Training**:
   - The architecture is **not** a fully homomorphic end-to-end training system that updates encrypted weights homomorphically without client interaction. Gradients are returned to the client at the end of each epoch, decrypted, and processed in plaintext before re-encryption.
2. **No Claim of Encrypted Inference on Fold 10**:
   - Fold 10 was strictly excluded from Phase 3B training and validation. Any historical evaluation of Fold 10 was performed offline in standard floating-point arithmetic to assess final model generalization, not as an encrypted inference benchmark.
3. **No Claim of Standalone ModSwitch**:
   - Condition B is not executed via SEAL `mod_switch_to_next`, but via `enc_err * 1.0` (plaintext multiplication followed by automatic rescaling).
4. **Systems Contribution vs Cryptographic Primitive**:
   - The research contribution is a **systems-level, runtime-context-aware scheduling contribution** (profile-guided circuit scheduling, alignment caching, and cost modeling), not the invention of a new homomorphic encryption scheme or lattice trapdoor.

---

## 4. Empirical & Cost Modeling Limitations

1. **Systematic Prediction Offset in CostModel_v2**:
   - While `CostModel_v2` achieves perfect monotonic ranking ($\rho = 1.000$) and $0.00\%$ decision regret across the evaluated seeds, it exhibits a systematic $10–19\%$ under-prediction of total training wall-clock time. This offset arises from Python memory management and garbage collection during multi-batch ciphertext rotations that are not captured in isolated microbenchmarks.
2. **Hardware Specificity**:
   - Microbenchmark latencies were calibrated on an AMD Zen 3 architecture running TenSEAL 0.3.17 on Windows 11. Migration to Linux or Intel AVX-512 hardware would shift raw operation latencies, although the relative level-dependent speedup ratios ($\approx 1.7\times$) are structurally governed by SEAL's RNS polynomial arithmetic.
3. **Fixed Modulus Chain**:
   - The current scheduler operates over a fixed 7-level modulus chain. It dynamically selects intermediate modulus levels and caching policies within this chain, but does not synthesize arbitrary prime bit-size configurations.
