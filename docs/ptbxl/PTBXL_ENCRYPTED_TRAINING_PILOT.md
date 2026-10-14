# PTB-XL End-to-End Encrypted Logistic Regression Training Pilot

**Date**: 2026-10-15 | **Status**: Verified Pilot Execution (3 Epochs, Condition B Scheduling)

---

## 1. Executive Summary & Pilot Verdict

**PILOT VERDICT: PASSED — PROCEED TO FULL TRAINING**

This pilot implements and verifies repeated end-to-end encrypted training of a Logistic Regression model on the PTB-XL ECG dataset ($N = 17,084$ training records, $D = 48$ statistical features, partitioned across three $8,192$-slot SIMD batches) using the **Condition B scheduling strategy**.

### Key Validation Criteria Results:
1. **3-Batch Global Aggregation**: Successfully unified $8,192 + 8,192 + 700$ records into a single global gradient normalized by $N = 17,084$ with zero padding leakage.
2. **Strict Numerical Gradient Tolerance**: Maximum absolute gradient error across all pilot epochs was **`6.87e-06`**, well below the documented threshold of **`1.0e-4`** (mean relative error: `4.06e-05`).
3. **Circuit Modulus State Consistency**: Gradient stage verified at **Level 2** (3 primes, 140 bits), terminating at **Level 1** (2 primes, 100 bits) across all epochs with exact SEAL `parms_id` alignment.
4. **Finite Values & Stability**: All decrypted gradients, weights, logits, and validation metrics remained 100% finite without numerical drift or overflow.
5. **Profiled Epoch Runtime**: Average epoch runtime was **`10.014s +/- 0.307s`** (gradient computation: **`3.555s`**), reproducing the Phase 4 Condition B benchmark.
6. **Validation Dynamics**: Fold 9 accuracy increased from **`0.4436`** to **`0.6752`** and ROC-AUC from **`0.4255`** to **`0.7534`** over 3 pilot epochs.

---

## 2. Documented Trust Boundary & Architecture

```mermaid
sequenceDiagram
    autonumber
    participant H as Trusted Hospital (Private Key Owner)
    participant CSP as Cloud Service Provider (Evaluator)
    
    Note over H,CSP: Pre-computation / One-time Setup
    H->>CSP: Encrypted Batched Features X (3 batches x 48 cols) & Labels y (3 batches)
    
    loop For Each Epoch (1 .. 3)
        Note over H: Step 1: Weight Encryption
        H->>CSP: enc_w_cols (48 ciphertexts, broadcast to 8,192 slots)
        
        Note over CSP: Step 2: Encrypted Forward Linear Combination (3 Batches)<br/>enc_z = sum(X_j * w_j) [Level 7 -> 6]
        Note over CSP: Step 3: Degree-3 Polynomial Sigmoid Evaluation<br/>enc_p = P3(enc_z) [Level 6 -> 3]
        Note over CSP: Step 4: Encrypted Error Computation<br/>enc_err = enc_p - burn(y, 4) [Level 3]
        Note over CSP: Step 5: Condition B Controlled Level Adjustment<br/>enc_err_adj = enc_err * 1.0 [Level 3 -> 2]
        Note over CSP: Step 6: Feature Alignment<br/>enc_X_aligned = burn(X, 5) [Level 7 -> 2]
        Note over CSP: Step 7: Isolated Gradient Computation<br/>enc_sums = (enc_err_adj * enc_X_aligned).sum() [Level 2 -> 1]
        
        CSP->>H: 3 batches x 48 encrypted dot product sums (enc_sums)
        
        Note over H: Step 8: Decryption & Global Aggregation<br/>unreg_sum = sum_batches(dec(enc_sums))<br/>g_unreg = unreg_sum / 17084
        Note over H: Step 9: Regularization & Plaintext Weight Update<br/>g_reg = g_unreg + 0.01 * w<br/>Clip norm if > 1.0; w = w - 0.05 * g_clipped
    end
```

---

## 3. Per-Epoch Execution & Numerical Verification

| Epoch | Total Time (s) | Grad Comp (s) | Dec/Agg (s) | Grad MaxAE vs Plaintext | Rel Error | ||g_raw|| | ||g_clip|| | ||w|| | Val Acc | Val ROC-AUC | Val F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `10.341` | `3.668` | `0.141` | **`6.87e-06`** | `4.05e-05` | `0.6524` | `0.6524` | `0.1935` | `0.5834` | `0.6160` | `0.5991` |
| `2` | `9.967` | `3.521` | `0.135` | **`6.45e-06`** | `4.06e-05` | `0.6091` | `0.6091` | `0.1980` | `0.6566` | `0.7140` | `0.6593` |
| `3` | `9.733` | `3.474` | `0.134` | **`6.07e-06`** | `4.06e-05` | `0.5690` | `0.5690` | `0.2062` | `0.6752` | `0.7534` | `0.6717` |

---

## 4. Validation Split Diagnostics (Fold 9, N = 2,146)

| Metric | Pre-Training (Epoch 0) | Epoch 1 | Epoch 2 | Epoch 3 (Final Pilot) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | `0.4436` | `0.5834` | `0.6566` | **`0.6752`** |
| **ROC-AUC** | `0.4255` | `0.6160` | `0.7140` | **`0.7534`** |
| **Abnormal Precision** | `0.5178` | `0.6693` | `0.7658` | **`0.8002`** |
| **Abnormal Recall** | `0.4481` | `0.5422` | `0.5787` | **`0.5787`** |
| **Abnormal F1** | `0.4804` | `0.5991` | `0.6593` | **`0.6717`** |
| **Logit Range [min, max]** | `[-1.23, 0.97]` | `[-1.02, 1.22]` | `[-0.83, 1.46]` | `[-0.84, 1.69]` |
| **Poly Outputs Outside [0,1]** | `0` | `0` | `0` | **`0`** |

### Final Pilot Confusion Matrix (Fold 9):
```
                 Predicted Normal (0)   Predicted Abnormal (1)
True Normal   (0):       736                    178       
True Abnormal (1):       519                    713       
```

---

## 5. Decision on Longer Training & Stopping Rule

### Safety Assessment:
- **All 6 pilot passing criteria were fully satisfied**.
- Gradient error remained bounded at $\le 6.9 \times 10^{-6}$ (over an order of magnitude cleaner than the $1.0 \times 10^{-4}$ threshold).
- Zero polynomial activations exceeded $[0, 1]$ on Fold 9 during the pilot, with validation logits tightly confined to $[-0.84, +1.69]$ (safely within the $[-8, +8]$ valid domain of the Kim et al. polynomial).

### Proposed Full Training Configuration:
- **Duration**: **10 Epochs** (matching MHEALTH Phase 1/2A benchmark).
- **Estimated Total Runtime**: $10 \times 9.7\text{s} \approx 97\text{s}$ (~1.6 minutes).
- **Stopping Rule**: Early stopping if validation ROC-AUC does not improve by $\ge 0.001$ over 2 consecutive epochs after Epoch 5, or if logit magnitude exceeds $6.0$.

---

## 6. Reproducibility & Saved Artifacts

```powershell
# Run the complete 3-epoch encrypted training pilot
python results/ptbxl/encrypted_training_pilot/run_ptbxl_encrypted_pilot.py --epochs 3 --lr 0.05 --l2 0.01 --grad-clip 1.0
```

### Generated Artifacts:
- Report: `docs/ptbxl/PTBXL_ENCRYPTED_TRAINING_PILOT.md`
- Results JSON: `results/ptbxl/encrypted_training_pilot/ptbxl_encrypted_pilot_results.json`
- Per-Epoch CSV: `results/ptbxl/encrypted_training_pilot/ptbxl_encrypted_pilot_results.csv`
- Execution Log: `results/ptbxl/encrypted_training_pilot/ptbxl_encrypted_pilot.log`
- Initial Weights Checkpoint: `results/ptbxl/encrypted_training_pilot/initial_weights.npy`
- Final Pilot Checkpoint: `results/ptbxl/encrypted_training_pilot/final_pilot_weights.npy`