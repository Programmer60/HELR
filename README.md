# Privacy-Preserving Heart Disease Prediction via Fully Homomorphic Encryption (MIMIC-III)

This repository documents a comprehensive, end-to-end Privacy-Preserving Machine Learning (PPML) pipeline. We utilized the CKKS (Cheon-Kim-Kim-Song) Homomorphic Encryption scheme via the TenSEAL library to perform encrypted Logistic Regression. 

This project transitions from small-scale toy datasets to the massive, real-world **MIMIC-III Clinical Database**, specifically targeting the prediction of Ischemic Heart Disease (ICD-9 codes 410-414) using clinical vitals and laboratory measurements from the first 24 hours of hospital admission.

---

## 🧠 Our Thought Process & Objectives

When scaling Homomorphic Encryption (HE) to large clinical datasets, researchers often confuse the computational workload with cryptographic limits. Our primary objective was to rigorously separate and prove the independence of three scaling dimensions:

1. **Dataset Size:** How does adding thousands of patients affect runtime and memory?
2. **Feature Count:** How does adding more clinical variables affect the HE workload?
3. **Polynomial Degree (Circuit Depth):** How does the complexity of the activation function affect the cryptographic modulus consumption?

We hypothesized that thanks to SIMD (Single Instruction, Multiple Data) batching, dataset size would scale completely independently of cryptographic circuit depth. We designed our experiments explicitly to prove this to reviewers.

---

## 🛤️ Phase-by-Phase Workflow

### Phase 1: Data Extraction & Cohort Definition
We started by processing the raw MIMIC-III database. We mapped highly specific `ITEMID`s for both CareVue and MetaVision ICU systems across `CHARTEVENTS` and `LABEVENTS`. We applied strict physiological plausibility filters and aggregated the data within a strict 24-hour observation window. This resulted in a robust cohort of **49,303 adult admissions**.

### Phase 2: Plaintext Baseline & Feature Selection
Before encrypting anything, we established a plaintext Logistic Regression baseline. We then performed rigorous feature selection. Instead of blindly using Random Forest importance (which favored nonlinear lab values), we used **Logistic Regression absolute coefficient magnitude**. This allowed us to reduce the feature set from 16 down to a highly optimized **LR Top-10** set, suffering virtually zero performance loss ($\Delta$AUC = -0.0005). 

*Thought process: Fewer features mean fewer `Ciphertext × Plaintext` multiplications during HE, saving runtime without sacrificing clinical accuracy.*

### Phase 3: Polynomial Sigmoid Design
HE schemes like CKKS cannot compute non-polynomial functions like the sigmoid activation. We analyzed the exact distribution of the pre-sigmoid decision scores (logits) from our training set, finding that 99% of logits fell between `[-3.25, 1.00]`. We then fit custom Degree-3 and Degree-5 polynomials over a safe `[-6.0, 6.0]` interval using least-squares optimization.

*What's new here? We didn't just recycle a standard polynomial from literature; we tailored the approximation bounds directly to the empirical logit distribution of our specific clinical model to guarantee stability.*

### Phase 4: Full Circuit Validation & Instrumentation
We instrumented the TenSEAL `CKKSVector` API to dynamically track the `coeff_modulus_size` (remaining modulus levels) and scale at every step of the computation. We tested a specific CKKS context: `[60, 40, 40, 40, 40, 40, 40, 60]` (7 usable levels). 

We proved that a full Degree-5 forward pass + backward gradient calculation consumes exactly 6 levels, leaving exactly **1 level remaining** for successful decryption. Furthermore, the decrypted encrypted gradient matched the plaintext gradient to within 3 parts per million.

### Phase 5 & 6: The Scalability Experiments (Dataset Size vs. Feature Count)
We ran systematic benchmarks to isolate scaling behaviors:
*   **Dataset Size:** Because our polynomial modulus degree ($N=16384$) provided 8,192 SIMD slots, running 500 patients took the exact same HE compute time as 5,000 patients. Circuit depth remained locked at 2 levels remaining for the forward pass, proving dataset size only affects SIMD batching, not cryptographic depth.
*   **Feature Count:** Increasing from 10 to 16 features increased ciphertext count and runtime (due to more linear combinations), but did *not* increase multiplicative depth.

### Phase 7: Full Cohort HE Inference
Finally, we ran the full 49,303-admission cohort through the encrypted Degree-5 model. Using 7 SIMD batches, the entire encrypted inference took just **4.2 seconds**, achieving the exact same ROC-AUC (0.7257) as the plaintext model out to four decimal places.

---

## 🔐 End-to-End Encrypted Training Architecture

We utilized the baseline knowledge from the original heart-disease framework to execute the training loop securely. The architecture splits trust between two entities:
1. **The Cloud Service Provider (CSP):** Holds the public/evaluation keys. Performs the heavy `z = X @ w` multiplications, the polynomial sigmoid approximations, and the error/gradient multiplications entirely on ciphertexts.
2. **The Hospital:** Holds the private key. Between epochs, the CSP sends the encrypted gradient to the Hospital. The Hospital decrypts it (using the 1 remaining modulus level), updates the weights in plaintext, and sends the newly encrypted weights back to the CSP for the next epoch. 

This ensures the CSP *never* sees the raw patient data, the true labels, or the model weights in plaintext, completely preserving privacy while working around the lack of bootstrapping in standard CKKS.

---

## ✨ What We Did New (Key Innovations & Contributions)

While building upon prior HELR literature, we introduced several novel methodological improvements:

1. **Massive Clinical Translation:** We moved HE from clean, balanced toy datasets (like the UCI Heart Disease dataset) to the messy, sparse, real-world MIMIC-III database, proving HE can handle realistic clinical noise and scale.
2. **Dynamic Circuit Instrumentation:** Instead of relying on theoretical arithmetic depth calculations, we built a diagnostic tracker that interrogates the underlying Microsoft SEAL C++ objects at runtime to prove exactly when and where modulus levels are consumed (and when they must be intentionally burned for alignment).
3. **Orthogonal Scalability Proofs:** We explicitly decoupled and empirically proved the differences between:
    *   *Dataset scaling:* Solved via SIMD slot packing (increases batches).
    *   *Feature scaling:* Solved via `Ciphertext × Plaintext` linear combinations (increases runtime, zero impact on depth).
    *   *Circuit scaling:* Dictated strictly by the polynomial degree (consumes actual `Ciphertext × Ciphertext` levels).
4. **Data-Driven Polynomial Fitting:** We demonstrated a rigorous pipeline for fitting activation polynomials based on the empirical logit distribution of the target dataset, ensuring the HE approximation does not silently diverge on out-of-distribution clinical outliers.
