# Privacy-Preserving Machine Learning with CKKS Homomorphic Encryption

**Dataset**: PTB-XL v1.0.3 (12-lead ECG, 21,799 records)  
**Task**: Binary ECG abnormality screening (Normal vs Abnormal) using CKKS-assisted encrypted logistic regression  
**Library**: TenSEAL 0.3.17 · Microsoft SEAL 3.7–4.0 (RNS-CKKS) · Python 3.13.5  
**Platform**: AMD Zen 3 · Windows 11 
**Status**: Research freeze complete — audited and reproducible

---

## What this project actually is

This project investigates whether a cost model can guide the scheduling of homomorphic training circuits to reduce encrypted training time, while preserving model accuracy and numerical gradient fidelity. The encrypted logistic-regression pipeline runs on the PTB-XL ECG dataset. Gradients are computed homomorphically on a simulated cloud server but **decrypted at the hospital client** before plaintext weight updates — this is not fully homomorphic end-to-end training.

We do not claim a new cryptographic primitive, a clinical device, or superiority over plaintext approaches. All claims below are backed by raw experimental logs in `results/ptbxl/scheduler/phase3b/`.

---

## Architecture

```
Hospital (Trusted Client)                 Cloud Server (Untrusted)
─────────────────────────                 ────────────────────────
Hold secret key                           Hold eval/Galois/relin keys
Encrypt features X, labels y  ──enc(X,y,w)──►  Forward pass: z = X·w  (Level 7→6)
                                               Polynomial sigmoid P₃(z) (Level 6→3)
                                               Error e = P₃(z) − y      (Level 3)
                               ◄──enc(grad)──  Gradient sums enc(∇w)
Decrypt gradients                              (B×D homomorphic dot products)
Update weights in plaintext
Re-encrypt updated weights    ──enc(w_new)──►  Next epoch
```

**CKKS context**: `poly_modulus_degree=16384`, `coeff_mod_bit_sizes=[60,40,40,40,40,40,40,60]`, `scale=2^40`  
→ 7 active data levels (300 bits), 1 special key-switching prime (60 bits), total configured modulus 360 bits.  
→ SEAL `MaxBitCount(16384, tc128) = 438 bits`; our 360-bit configuration is within the 128-bit classical ceiling.

---

## Repository layout

```
MinorProject/
├── src/scheduler/
│   ├── cost_model.py          # CostModel_v1 (frozen, calibrated in-process)
│   ├── cost_model_v2.py       # CostModel_v2 (fresh-subprocess-aware)
│   └── workload_policy.py     # Schedule selection policy
├── benchmarks/
│   ├── profile_level_latencies.py       # Phase 1 microbenchmarks
│   ├── profile_execution_contexts.py    # Phase 3A context profiling
│   ├── run_workload_grid.py             # Phase 2 held-out grid
│   └── validate_cost_model_v2.py        # Phase 3A validation
├── results/ptbxl/
│   ├── encrypted_training_full10/       # Full 10-epoch encrypted training
│   ├── final_test_evaluation/           # Locked Fold 10 evaluation
│   └── scheduler/phase3b/               # Phase 3B matched experiment logs
├── tests/                               # 48 pytest tests (all passing)
├── docs/ptbxl/
│   ├── final_research_freeze/           # Frozen audit documents
│   └── SCHEDULER_PHASE*.md              # Per-phase technical reports
└── baseline/                            # Early MIMIC-III / mHealth scripts
```

---

## Results — what we measured

### 1. Encrypted training convergence (Fold 9 validation, Seed 42)

Full 10-epoch CKKS-assisted training on PTB-XL (17,084 train, 2,146 validation):

| Epoch | Val Accuracy | Val ROC-AUC |
|:---:|:---:|:---:|
| 1 (initial) | 0.444 | 0.426 |
| 10 (final) | **0.702** | **0.790** |

Training converged across 10 epochs. Best epoch = Epoch 10.  
Mean epoch wall-clock time: **9.58 s** (std 0.69 s) · Total: **95.8 s**

### 2. Locked Fold 10 test evaluation (one-shot, after training freeze)

Evaluated on 2,158 held-out records (912 normal, 1,246 abnormal). Fold 10 was never accessed during training or schedule selection.

| Metric | Value |
|:---|:---:|
| Accuracy | 0.677 |
| ROC-AUC | 0.767 |
| Precision | 0.790 |
| Recall | 0.600 |
| F1 | 0.682 |

> **Interpretation**: The model learns from the training data and generalises to held-out records. This is a single run with Seed 42. The gap between Fold 9 validation AUC (0.790) and Fold 10 test AUC (0.767) is expected.

### 3. Gradient numerical fidelity

Across all epochs, maximum absolute error between homomorphic gradient and plaintext reference: **6.87 × 10⁻⁶** (tolerance threshold: 1.0 × 10⁻⁴). All epochs pass.

---

## Scheduler experiment — Phase 3B

Four circuit schedule variants were compared across 3 random seeds (42, 123, 2026), 10 epochs each = 12 full training runs:

| Schedule | Strategy | Gradient Entry Level | Cumulative Time (s) ± pop σ | Val ROC-AUC |
|:---|:---:|:---:|:---:|:---:|
| **A-repeat** | Repeated alignment | Level 3 | 136.1 ± 11.7 | 0.784 |
| **A-cache**  | Cached alignment   | Level 3 | 126.3 ± 6.1  | 0.784 |
| **B-repeat** | Repeated alignment | Level 2 | 115.1 ± 14.6 | 0.784 |
| **B-cache**  | Cached alignment   | Level 2 | **86.1 ± 4.1** | 0.784 |

**Key observations**:
- B-cache is the fastest schedule in all 3 seeds.
- Validation accuracy and ROC-AUC are **schedule-invariant**: 0.699 ± 0.003 and 0.784 ± 0.004 across all schedules and seeds. The scheduling decision affects only runtime, not the learned model.
- The combined speedup of B-cache vs A-repeat is **1.589 ± 0.200×** (seed-matched mean ratio, pop σ, N=3 seeds). Range: 1.33× to 1.82× across seeds.
- Gradient absolute error: ≤ 7.28 × 10⁻⁶ for all schedules.

#### What the Level 2 vs Level 3 gradient-entry difference means
"Condition B" applies `enc_err * 1.0` (plaintext multiplication followed by automatic rescaling in TenSEAL), dropping the error ciphertext one RNS level before the gradient dot product. This means the gradient computation operates at lower noise but slightly narrower precision. It is **not** a standalone SEAL `mod_switch_to_next` call.

#### CostModel_v2 prediction vs measurement

| Schedule | Predicted (s) | Measured (s) | MAPE |
|:---|:---:|:---:|:---:|
| A-repeat | 121.3 | 136.1 | 10.2% |
| A-cache  | 105.8 | 126.3 | 16.0% |
| B-repeat |  91.8 | 115.1 | 19.0% |
| B-cache  |  74.2 |  86.1 | 13.6% |

The model **under-predicts** wall-clock time by 10–19%. The bias is non-uniform and comes from Python memory management and garbage collection during 144 ciphertext decryption loops (B×D = 3×48 = 144 decryptions per epoch) that isolated microbenchmarks do not capture. Despite this, the **schedule ranking is preserved exactly**: Spearman ρ = 1.000 over N=4 candidate schedules. Decision regret = 0.00% across all 3 seeds.

---

## Phase progression

| Phase | What we did | Key outcome |
|:---|:---|:---|
| **Baseline** | MIMIC-III cohort extraction, mHealth CKKS baseline | mHealth deg-3 HE: accuracy 0.858, AUC 1.0 on small activity dataset |
| **PTB-XL preprocessing** | 48-feature ECG extraction, median-centering, 10-fold split | 17,084 train / 2,146 val / 2,158 test; 411 records excluded (no label) |
| **Circuit validation** | Level tracking, gradient fidelity, polynomial sigmoid calibration | Degree-3 Chebyshev sigmoid; logit range [-1.40, 3.62] at Epoch 10 |
| **Full encrypted training** | 10-epoch CKKS-assisted LR, Seed 42 | Val AUC 0.790 at Epoch 10; total 95.8 s |
| **Locked test evaluation** | One-shot Fold 10 evaluation | Test AUC 0.767, accuracy 0.677 |
| **Phase 1 — CostModel_v1** | Microbenchmark profiling of atomic CKKS ops | Mean epoch MAPE < 1.3% on calibration workload |
| **Phase 2 — Held-out validation** | 9-workload grid ($N \in \{4096,10000,17084\}$, $D \in \{12,24,48\}$) | Absolute MAPE failed (63%); decision regret passed (max 3.3% < 5%) |
| **Phase 3A — CostModel_v2** | Fresh-subprocess execution-context profiling; per-op scaling factors | Eliminates global correction factor; 0.00% decision regret on boundary workloads |
| **Phase 3B — Trainer integration** | 4 schedules × 3 seeds × 10 epochs matched grid + pilot run | B-cache: 1.589 ± 0.200× speedup; 0.00% decision regret; schedule-invariant accuracy |
| **Phase 4 audit** | Independent statistical re-computation, hash verification, terminology audit | All 48 tests pass; 33 artifacts SHA-256 verified; no Fold 10 leakage |

---

## What we do not claim

- **Not fully homomorphic end-to-end training.** Gradients are decrypted at the hospital client after each epoch. Weights are updated in plaintext.
- **Not a claim of clinical deployment readiness.** This is a research prototype on a public ECG dataset.
- **Not statistically significant speedup.** N=3 seeds is too small for formal hypothesis testing.
- **Not universal schedule optimality.** The B-cache recommendation was evaluated on one production workload (N=17,084, D=48, B=3, E=10). Generalization to other hardware or workloads was not tested.
- **Not post-quantum unconditional security.** The parameter set falls within the 128-bit classical security ceiling under HE Standard tables. Formal proof and quantum-secure hardness reduction are not provided.
- **Not bootstrapping.** The modulus chain is consumed across epochs. Weights are re-encrypted as fresh ciphertexts between epochs.

---

## Reproducing the results

```bash
# 1. Install dependencies
pip install tenseal==0.3.17 numpy pandas scikit-learn pytest

# 2. Run the full test suite (48 tests)
python -m pytest tests/ -v

# 3. Verify all frozen artifact SHA-256 hashes (33 files)
python -c "
import json, hashlib
from pathlib import Path
with open('docs/ptbxl/final_research_freeze/FINAL_ARTIFACT_HASHES.json') as f:
    manifest = json.load(f)
ok = all(
    hashlib.sha256(Path(p).read_bytes()).hexdigest() == m['sha256']
    for p, m in manifest['artifacts'].items() if Path(p).exists()
)
print('VERIFIED' if ok else 'MISMATCH DETECTED')
"

# 4. Re-compute all Phase 3B statistics from raw JSON
python results/ptbxl/scheduler/phase3b/verify_phase3b_statistics.py

# 5. Show policy schedule selection across epoch horizons
python -c "
from src.scheduler.workload_policy import WorkloadSchedulerPolicy
pol = WorkloadSchedulerPolicy()
for epochs in [1, 3, 5, 10]:
    plan, _ = pol.select_plan(N=17084, D=48, epochs=epochs)
    print('E=' + str(epochs), '->', plan.schedule_id, round(plan.predicted_total_training_s, 1), 's')
"
```

Expected verification output:
```
48 passed in 1.18s
VERIFIED
ALL STATISTICAL AND ACCOUNTING INVARIANTS VERIFIED!
E=1  -> B-repeat   9.2 s
E=3  -> B-cache   23.6 s
E=5  -> B-cache   38.1 s
E=10 -> B-cache   74.2 s
```

---

## Frozen audit documents

| Document | Contents |
|:---|:---|
| [`FINAL_RESEARCH_REPORT.md`](docs/ptbxl/final_research_freeze/FINAL_RESEARCH_REPORT.md) | Full technical report, equations, bibliography |
| [`RESULTS_SUMMARY.md`](docs/ptbxl/final_research_freeze/RESULTS_SUMMARY.md) | Per-seed timing tables, speedup ratios, accuracy |
| [`ENVIRONMENT_AND_CKKS_CONFIG.md`](docs/ptbxl/final_research_freeze/ENVIRONMENT_AND_CKKS_CONFIG.md) | Software versions, CKKS parameters, security assessment |
| [`LIMITATIONS_AND_THREAT_MODEL.md`](docs/ptbxl/final_research_freeze/LIMITATIONS_AND_THREAT_MODEL.md) | Trust boundary, leakage model, negative constraints |
| [`EXPERIMENT_MANIFEST.md`](docs/ptbxl/final_research_freeze/EXPERIMENT_MANIFEST.md) | Dataset split reconciliation, full experiment list |
| [`REPRODUCTION_COMMANDS.md`](docs/ptbxl/final_research_freeze/REPRODUCTION_COMMANDS.md) | Step-by-step verification commands |
| [`FINAL_ARTIFACT_HASHES.json`](docs/ptbxl/final_research_freeze/FINAL_ARTIFACT_HASHES.json) | SHA-256 digests for 33 frozen artifacts |

---

## Key references

- Cheon, J.H., Kim, A., Kim, M., Song, Y. (2017). Homomorphic encryption for arithmetic of approximate numbers. *ASIACRYPT 2017*.
- Albrecht, M. et al. (2018). Homomorphic Encryption Standard. [HomomorphicEncryption.org](https://homomorphicencryption.org).
- Han, K., Hong, S., Cheon, J.H., Park, D. (2019). Logistic regression on homomorphically encrypted data at scale. *AAAI 2019*, 33(01):9466–9471. DOI: [10.1609/aaai.v33i01.33019466](https://doi.org/10.1609/aaai.v33i01.33019466).
- Wagner, P. et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7:154.
- Naresh, V.S., Reddi, S. (2025). Privacy-preserving ECG classification using homomorphic encryption. *Journal of Big Data*, 12:Article 52. DOI: [10.1186/s40537-025-01098-6](https://doi.org/10.1186/s40537-025-01098-6).
- Benaissa, A. et al. (2021). TenSEAL: A library for encrypted tensor operations using CKKS. *ICLR 2021 Workshop on PPML*.
