# PTB-XL Phase 4: Final Consistency Audit Report

**Date**: 2026-10-15  
**Status**: All four audits completed — PASS / EXPLAINED / CORRECTED / COMPLETE  
**Audited Artifact**: `results/ptbxl/phase4/ptbxl_phase4_results.json` (2026-10-15)

---

## Overview of Audit Tasks

| Audit | Task | Verdict |
| :---: | :--- | :---: |
| **A** | Reproduce all timing statistics from saved per-rep JSON rows | ✅ **PASS** |
| **B** | Root-cause validation-metric inconsistency (`acc=0.4436` vs `0.7321`) | ✅ **EXPLAINED** |
| **C** | Verify SEAL ciphertext modulus chain at each circuit stage | ⚠️ **CORRECTED** |
| **D** | Controlled 10-epoch plaintext model-level validation for A, B, C | ✅ **COMPLETE** |

---

## Audit A — Timing Verification

All timing statistics in the Phase 4 JSON summary (`total_epoch_mean_s`, `total_epoch_std_s`,
`gradient_comp_mean_s`, `gradient_comp_std_s`) were recomputed directly from the 27 individual
per-repetition rows stored in `results/ptbxl/phase4/ptbxl_phase4_results.json`.

### Result: All values reproduce to floating-point precision ($< 10^{-8}$ absolute error).

| Condition | Grad Start Level | Grad Comp Mean (s) | Grad Comp Std (s) | Total Epoch Mean (s) | Total Epoch Std (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`A`** | Level 3 | `6.4832` | `0.1024` | `12.4012` | `0.1940` |
| **`B`** | Level 2 | `3.4790` | `0.0294` | `9.6334` | `0.0839` |
| **`C`** | Level 2 | `3.4864` | `0.0512` | `9.7460` | `0.0666` |

**Scheduling Intervention (Condition A to Condition B):**
- Gradient stage speedup: **`46.34%`** (`6.483s` → `3.479s`)
- Net epoch speedup: **`22.32%`** (`12.401s` → `9.633s`)
- Condition A vs B gradient discrepancy: MaxAE = `1.71e-06` — within CKKS noise floor

**Conclusion**: The Phase 4 timing benchmark is fully reproducible and internally consistent.

---

## Audit B — Validation Metric Root Cause

### Discrepancy

The Phase 4 JSON reports `val_acc = 0.443616` and `val_roc_auc = 0.425541` identically for
all nine repetitions across all three conditions. The Phase 3 / Range-Control Variant-3 results
report `val_acc = 0.7321` and `val_roc_auc = 0.8247`.

### Root Cause

> [!IMPORTANT]
> Phase 4 is a **timing benchmark**. Validation metrics are evaluated at the **initial random weights** (`w_init`, Seed 42, `||w|| = 0.1939`) — before any gradient step is applied. All three conditions share an identical `w_init` and the same `X_val`, so they must produce identical validation metrics.

The code path in `run_ptbxl_phase4.py` (lines ~359–364):
```python
z_val = X_val @ w_init
p_val = true_sigmoid(z_val)
y_pred = (p_val >= 0.5).astype(int)
acc = float(accuracy_score(y_val, y_pred))
```

### Verification

| Evaluation Context | Acc | ROC-AUC | F1 |
| :--- | :---: | :---: | :---: |
| **Phase 4 JSON (initial `w_init`)** | `0.443616` | `0.425541` | `0.480418` |
| Reproduced from `w_init` | `0.443616` | `0.425541` | `0.480418` |
| **Variant-3 plaintext baseline (sklearn, 2000 iter)** | `0.7321` | `0.8247` | `0.7339` |
| After 10 GD epochs Deg-3 (plaintext) | `0.7022` | `0.7899` | `0.7015` |
| After 10 GD epochs Deg-5 (plaintext) | `0.7046` | `0.7909` | `0.7046` |

**Conclusion**: No data error. The Phase 4 validation metrics correctly characterize the initial-weight evaluation. The higher Variant-3 values come from a fully converged sklearn model — a different evaluation context entirely.

---

## Audit C — SEAL Ciphertext Modulus Chain (Correction)

### Key Correction

> [!WARNING]
> The Phase 4 narrative states **"Fresh Ciphertexts: Level 7 (8 primes, 360 bits)"**.
> This is **partially incorrect**.

**Actual SEAL/TenSEAL behaviour** with `coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 60]`:
- The 8 primes in the context definition include one **special prime** (the last 60-bit prime).
  This special prime is used exclusively for key-switching (rotation/relinearisation) and is
  **never** part of a normal data ciphertext's modulus chain.
- **Fresh `ckks_vector`** ciphertexts have `coeff_modulus_size() = 7`
  (7 data primes: 60 + 40 + 40 + 40 + 40 + 40 + 40 = 300 bits of data-chain capacity).
- Each multiplicative operation (ct-ct multiply or `* scalar`) drops one prime,
  reducing `coeff_modulus_size` by 1.

### Corrected Modulus Chain Table

| Circuit Stage | Condition | `coeff_modulus_size()` | Active Data Primes | Active Bit Width |
| :--- | :---: | :---: | :---: | :---: |
| Fresh ciphertexts ($X, y, w$) | All | **7** | 7 | 300 bits |
| Forward linear combination ($Xw$) | All | **6** | 6 | 260 bits |
| Deg-3 polynomial output $P_3(z)$ | A, B | **3** | 3 | 140 bits |
| Deg-5 polynomial output $P_5(z)$ | C | **2** | 2 | 100 bits |
| Encrypted error $e = P - y$ (before adj) | A, B | **3** | 3 | 140 bits |
| Encrypted error $e = P - y$ | C | **2** | 2 | 100 bits |
| Controlled level adjustment ($e \cdot 1.0$) | B only | **2** | 2 | 100 bits |
| Feature ciphertexts aligned for gradient (burn 4) | A | **3** | 3 | 140 bits |
| Feature ciphertexts aligned for gradient (burn 5) | B, C | **2** | 2 | 100 bits |
| Gradient workload **start** | A | **3** | 3 | 140 bits |
| Gradient workload **start** | B, C | **2** | 2 | 100 bits |
| Gradient output (after ct-ct mult) | A | **2** | 2 | 100 bits |
| Gradient output (after ct-ct mult) | B, C | **1** | 1 | 60 bits |

> [!NOTE]
> `coeff_modulus_size()` equals the number of active primes in the ciphertext modulus chain.
> The grad_start_level values (3 for Cond A, 2 for Cond B and C) reported in the JSON are
> verified correct by direct SEAL API inspection. The only error was in the narrative
> description of "8 primes" for fresh ciphertexts — the actual active count is 7.

---

## Audit D — Controlled 10-Epoch Plaintext Model-Level Validation

**Protocol**: All three conditions trained from identical `w_init` (Seed 42), using
gradient descent with `LR=0.05`, `L2=0.01`, for 10 epochs on the bias-free objective.
Evaluation uses `true_sigmoid` on Fold 9 at threshold 0.5. **Fold 10 is not touched.**

| Condition | Deg | Val Acc | Val ROC-AUC | Val Precision | Val Recall | Val F1 | `||w||` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A`** | 3 | `0.7022` | `0.7899` | `0.8262` | `0.6096` | `0.7015` | `0.2980` |
| **`B`** | 3 | `0.7022` | `0.7899` | `0.8262` | `0.6096` | `0.7015` | `0.2980` |
| **`C`** | 5 | `0.7046` | `0.7909` | `0.8271` | `0.6136` | `0.7046` | `0.2804` |

> [!NOTE]
> Conditions A and B produce **mathematically identical** 10-epoch plaintext models because
> they use the same Deg-3 polynomial gradient with the same hyperparameters and initial weights.
> The level-scheduling difference (Level 3 vs Level 2 gradient) is an encrypted-execution
> optimization that does not change the mathematical update rule.

**Are these results trustworthy for proceeding to test evaluation?**

> [!IMPORTANT]
> **YES, with the following conditions and limitations:**
> 
> 1. The 10-epoch plaintext gradient descent (`val_acc ≈ 0.70`, `val_roc_auc ≈ 0.79`) is
>    directionally consistent with the sklearn baseline (`val_acc=0.7321`, `val_roc_auc=0.8247`),
>    confirming that the median-centered Variant-3 preprocessing and bias-free GD objective are sound.
> 2. The **Phase 4 timing result is reproducible**: the +46.3% gradient speedup and +22.3% net
>    epoch speedup from Level-3→Level-2 scheduling are independently verified from saved per-rep rows.
> 3. The model-level validation used a plaintext gradient descent. Full **encrypted training** would
>    introduce additional CKKS noise (empirically $\le 7 \times 10^{-6}$ per gradient element per epoch)
>    and would require a sufficient number of epochs to converge — this has not yet been run.
> 4. **Do not draw causal or statistically significant conclusions from 3 repetitions on one dataset.**
>    The timing results are consistent and reproducible; significance testing requires more runs.

---

## Final Verification Status

| Finding | Status |
| :--- | :---: |
| Phase 4 timing statistics internally consistent with per-rep rows | ✅ Confirmed |
| +46.3% gradient speedup (Level 3→2) reproducible | ✅ Confirmed |
| +22.3% net epoch speedup reproducible | ✅ Confirmed |
| Condition A vs B gradient MaxAE within CKKS noise floor | ✅ Confirmed (`1.71e-06`) |
| `val_acc=0.4436` explanation: initial-weight evaluation, not a trained model | ✅ Explained |
| Phase 4 validation metrics do not reflect any trained model | ✅ Confirmed |
| SEAL `coeff_modulus_size()=7` for fresh ciphertexts (not 8) | ⚠️ Corrected |
| `grad_start_level` values in JSON are correct | ✅ Verified |
| `grad_end_level` values in JSON are correct | ✅ Verified |
| 10-epoch plaintext GD val_acc ≈ 0.70, auc ≈ 0.79 (all conditions) | ✅ Complete |
| Median-centered Variant-3 preprocessing is safe to proceed to test evaluation | ✅ Recommended |

---

## Reproducibility

```powershell
# Re-run the full audit
python results/ptbxl/phase4_audit/run_ptbxl_phase4_audit.py

# Audit deliverables
# results/ptbxl/phase4_audit/ptbxl_phase4_audit_results.json
# results/ptbxl/phase4_audit/ptbxl_phase4_audit_results.csv
# results/ptbxl/phase4_audit/ptbxl_phase4_audit.log
# docs/ptbxl/PTBXL_PHASE4_FINAL_AUDIT.md
```

**Original Phase 4 artifacts preserved unchanged:**
- `results/ptbxl/phase4/ptbxl_phase4_results.json`
- `results/ptbxl/phase4/ptbxl_phase4_results.csv`
- `results/ptbxl/phase4/ptbxl_phase4.log`
- `docs/ptbxl/PTBXL_PHASE4_SCHEDULING.md`