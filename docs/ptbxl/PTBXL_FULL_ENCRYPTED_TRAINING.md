# PTB-XL Full 10-Epoch Encrypted Logistic Regression Training Report

**Date**: 2026-10-15 | **Status**: Verified Full Training Execution (Condition B Scheduling)

---

## 1. Executive Summary & Verification Verdict

**FINAL TRAINING VERDICT: PASSED — CONVERGED SAFELY**

This report presents the complete end-to-end homomorphic Logistic Regression training experiment on the PTB-XL ECG dataset ($N = 17,084$ training records, $D = 48$ statistical features across three $8,192$-slot SIMD batches) using the **Condition B modulus-level scheduling strategy**.

### Key Experimental Outcomes:
1. **Execution Duration & Stopping Decision**: The run completed **`10`** epochs. **Stopping Reason**: `Completed all 10 planned epochs without early stopping`.
2. **Best Validation Checkpoint**: Identified at **Epoch `10`**, reaching a validation **ROC-AUC of `0.7899`** and **Accuracy of `0.7022`** on Fold 9 ($N = 2,146$).
3. **Strict Cryptographic Noise Bound**: Across all completed epochs, the maximum absolute gradient error between the decrypted CKKS gradient and the matched plaintext reference was **`6.87e-06`** (mean error: `2.51e-06`, relative error: `4.07e-05`), strictly adhering to the **`1.0e-4`** tolerance threshold.
4. **Ciphertext Modulus Level Integrity**: Gradient evaluation was verified to execute at **Level 2** (3 primes, 140 bits) and terminate at **Level 1** (2 primes, 100 bits) across all batches and epochs with exact SEAL `parms_id` and scale alignment.
5. **High-Throughput Profiled Runtime**: Mean epoch runtime was **`9.583s +/- 0.689s`** (gradient computation: **`3.421s`**), yielding a total training wall-clock time of **`95.83s`** (~`1.60` minutes) for 17,084 samples.
6. **Polynomial Safety & Stability**: 100% of validation logits remained within $[-1.02, +3.62]$, strictly inside the $[-8, +8]$ safe interval. Zero polynomial outputs exceeded $[0, 1]$ throughout training.
7. **Test Set Isolation**: Fold 10 ($2,158$ records) remained completely untouched and was neither loaded nor evaluated.

---

## 2. Experimental Configuration

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Dataset** | PTB-XL v1.0.3 | Binary ECG classification (abnormal vs. normal) |
| **Preprocessing** | Frozen Variant 3 | Per-lead, per-record median baseline centering, unconstrained |
| **Scaler** | `StandardScaler` | Fitted strictly on folds 1–8 ($17,084$ records) |
| **Training Records** | $N = 17,084$ | Official folds 1–8 |
| **Validation Records** | $N = 2,146$ | Official fold 9 |
| **Test Records** | $N = 2,158$ | Official fold 10 (**strictly held out**) |
| **Features** | $D = 48$ | 12 leads $\times$ [mean, std, min, max] |
| **Initial Weights** | Seed 42 Uniform | Fresh initialization: `||w_0|| = 0.1939` |
| **CKKS Context** | $N = 16384$, Scale $2^{40}$ | `coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 60]`, 7 usable levels |
| **SIMD Batching** | 3 batches of 8,192 slots | Batch 1: 8,192; Batch 2: 8,192; Batch 3: 700 valid, 7,492 zero-padded |
| **Sigmoid Approximation** | Degree-3 Kim et al. | $P_3(z) = 0.5 + 0.15012z - 0.001593z^3$ |
| **Scheduling Strategy** | Condition B | Controlled level drop ($e \cdot 1.0$) to enter gradient at Level 2 |
| **Optimizer & Objective** | Plaintext Hospital SGD | Learning rate $\eta = 0.05$, $L_2$ decay $\lambda = 0.01$, gradient clip $1.0$ |
| **Early Stopping Rule** | Delta $\ge 0.001$, Patience 2 | Evaluated on Fold 9 after Epoch 5 |

---

## 3. Per-Epoch Execution & Numerical Verification

| Epoch | Total Time (s) | Grad Comp (s) | Dec/Agg (s) | Grad MaxAE vs Plaintext | Rel Error | ||g_raw|| | ||w|| | Val Acc (Exact) | Val ROC-AUC (Exact) | Val F1 (Exact) | Best? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `9.565` | `3.373` | `0.126` | **`6.87e-06`** | `4.05e-05` | `0.6524` | `0.1935` | `0.5834` | `0.6160` | `0.5991` |  |
| `2` | `9.348` | `3.361` | `0.116` | **`6.45e-06`** | `4.06e-05` | `0.6091` | `0.1980` | `0.6566` | `0.7140` | `0.6593` |  |
| `3` | `9.181` | `3.306` | `0.113` | **`6.07e-06`** | `4.06e-05` | `0.5690` | `0.2062` | `0.6752` | `0.7534` | `0.6717` |  |
| `4` | `9.122` | `3.284` | `0.116` | **`5.71e-06`** | `4.06e-05` | `0.5320` | `0.2170` | `0.6803` | `0.7707` | `0.6740` |  |
| `5` | `9.161` | `3.297` | `0.116` | **`5.37e-06`** | `4.07e-05` | `0.4980` | `0.2294` | `0.6883` | `0.7792` | `0.6837` |  |
| `6` | `9.140` | `3.278` | `0.120` | **`5.12e-06`** | `4.07e-05` | `0.4667` | `0.2428` | `0.6925` | `0.7837` | `0.6884` |  |
| `7` | `9.223` | `3.318` | `0.121` | **`4.90e-06`** | `4.07e-05` | `0.4381` | `0.2566` | `0.6957` | `0.7864` | `0.6933` |  |
| `8` | `10.997` | `4.162` | `0.180` | **`4.70e-06`** | `4.08e-05` | `0.4117` | `0.2705` | `0.6999` | `0.7881` | `0.6982` |  |
| `9` | `10.720` | `3.468` | `0.120` | **`4.51e-06`** | `4.09e-05` | `0.3876` | `0.2844` | `0.7022` | `0.7892` | `0.7018` |  |
| `10` | `9.368` | `3.364` | `0.118` | **`4.34e-06`** | `4.09e-05` | `0.3654` | `0.2980` | `0.7022` | `0.7899` | `0.7015` | ⭐ Best |

---

## 4. Separation of Activation Pathways on Fold 9

The table below isolates the clinical exact logistic sigmoid $\sigma(z)$ from the encrypted circuit polynomial $P_3(z)$:

| Epoch | Exact Sigmoid Acc | Exact ROC-AUC | Exact Precision | Exact Recall | Exact F1 | Poly Sigmoid Acc | Poly ROC-AUC | Poly F1 | Poly Outside [0,1] | Val Logit Range [min, max] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `0.5834` | `0.6160` | `0.6693` | `0.5422` | `0.5991` | `0.5834` | `0.6160` | `0.5991` | **`0`** | `[-1.02, 1.22]` |
| `2` | `0.6566` | `0.7140` | `0.7658` | `0.5787` | `0.6593` | `0.6566` | `0.7140` | `0.6593` | **`0`** | `[-0.83, 1.46]` |
| `3` | `0.6752` | `0.7534` | `0.8002` | `0.5787` | `0.6717` | `0.6752` | `0.7534` | `0.6717` | **`0`** | `[-0.84, 1.69]` |
| `4` | `0.6803` | `0.7707` | `0.8131` | `0.5755` | `0.6740` | `0.6803` | `0.7707` | `0.6740` | **`0`** | `[-0.92, 1.90]` |
| `5` | `0.6883` | `0.7792` | `0.8188` | `0.5869` | `0.6837` | `0.6883` | `0.7792` | `0.6837` | **`0`** | `[-0.99, 2.25]` |
| `6` | `0.6925` | `0.7837` | `0.8228` | `0.5917` | `0.6884` | `0.6925` | `0.7837` | `0.6884` | **`0`** | `[-1.06, 2.56]` |
| `7` | `0.6957` | `0.7864` | `0.8227` | `0.5990` | `0.6933` | `0.6957` | `0.7864` | `0.6933` | **`0`** | `[-1.13, 2.86]` |
| `8` | `0.6999` | `0.7881` | `0.8259` | `0.6047` | `0.6982` | `0.6999` | `0.7881` | `0.6982` | **`0`** | `[-1.19, 3.13]` |
| `9` | `0.7022` | `0.7892` | `0.8255` | `0.6104` | `0.7018` | `0.7022` | `0.7892` | `0.7018` | **`0`** | `[-1.29, 3.38]` |
| `10` | `0.7022` | `0.7899` | `0.8262` | `0.6096` | `0.7015` | `0.7022` | `0.7899` | `0.7015` | **`0`** | `[-1.40, 3.62]` |

### Confusion Matrix at Best Validation Checkpoint (Epoch 10):
```
                 Predicted Normal (0)   Predicted Abnormal (1)
True Normal   (0):       756                    158       
True Abnormal (1):       481                    751       
```

---

## 5. Checkpoint Selection & Justification

- **Best Validation Checkpoint (`best_validation_weights.npy`)**: Selected at **Epoch `10`** based strictly on validation ROC-AUC (`0.7899`).
- **Final Checkpoint (`final_weights.npy`)**: Saved at **Epoch `10`** (ROC-AUC = `0.7899`).
- **Selection Policy**: The best checkpoint at Epoch `10` is the **locked candidate** for final independent evaluation on Fold 10.

---

## 6. Reproducibility & Saved Artifacts

```powershell
# Run complete 10-epoch encrypted training experiment
python results/ptbxl/encrypted_training_full10/run_ptbxl_full_training.py --epochs 10 --lr 0.05 --l2 0.01 --grad-clip 1.0
```

### Generated Artifacts:
- Formal Markdown Report: `docs/ptbxl/PTBXL_FULL_ENCRYPTED_TRAINING.md`
- Machine-Readable JSON: `results/ptbxl/encrypted_training_full10/ptbxl_full10_results.json`
- Per-Epoch Metrics CSV: `results/ptbxl/encrypted_training_full10/ptbxl_full10_results.csv`
- Complete Execution Log: `results/ptbxl/encrypted_training_full10/ptbxl_full10.log`
- Best Validation Checkpoint: `results/ptbxl/encrypted_training_full10/best_validation_weights.npy`
- Final Epoch Checkpoint: `results/ptbxl/encrypted_training_full10/final_weights.npy`