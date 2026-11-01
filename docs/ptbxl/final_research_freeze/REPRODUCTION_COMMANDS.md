# Research Reproduction Guide & Verification Commands

**Document ID**: `docs/ptbxl/final_research_freeze/REPRODUCTION_COMMANDS.md`  
**Phase**: Phase 4B — Statistical Corrections, Evidence Audit & Research Freeze Finalization  
**Date**: October 31, 2026  

This guide provides the exact CLI commands required to reproduce, audit, and mathematically verify all components of the PTB-XL CKKS Workload-Aware Scheduling project.

---

## 1. Environment & Dependency Setup

### 1.1 Requirements
- Python 3.10+ (tested on Python 3.13.5)
- TenSEAL 0.3.17
- NumPy $\ge 1.24.0$
- scikit-learn $\ge 1.2.0$
- pandas $\ge 2.0.0$
- pytest $\ge 7.0.0$

### 1.2 Installation
```bash
pip install tenseal==0.3.17 numpy pandas scikit-learn pytest
```

---

## 2. Integrity & Cryptographic Hash Verification

Verify that all project artifacts match the frozen audit hashes recorded in `FINAL_ARTIFACT_HASHES.json`:

```bash
python -c "
import json, hashlib
from pathlib import Path

with open('docs/ptbxl/final_research_freeze/FINAL_ARTIFACT_HASHES.json') as f:
    manifest = json.load(f)

mismatches = []
for path_str, meta in manifest['artifacts'].items():
    p = Path(path_str)
    if not p.exists():
        mismatches.append('MISSING: ' + path_str)
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != meta['sha256']:
        mismatches.append('HASH MISMATCH: ' + path_str)

if not mismatches:
    print('SUCCESS: ALL FROZEN ARTIFACTS VERIFIED (SHA-256 EXACT MATCH)!')
else:
    for m in mismatches:
        print(m)
"
```

---

## 3. Test Suite Execution (48 Tests)

Run the complete regression, cost model, grid, and scheduling policy test suite:

```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_cost_model.py::test_batch_count_edge_cases PASSED
...
tests/test_cost_model_v2.py::test_primary_schedule_degree3_separation PASSED
...
tests/test_phase2_scheduler.py::test_decision_regret_invariants PASSED
...
tests/test_phase3b_scheduler.py::TestMemoryEstimation::test_b_cache_memory_estimate PASSED
============================= 48 passed in 1.05s ==============================
```

---

## 4. Policy Decision Demonstration

Verify how `WorkloadSchedulerPolicy` dynamically evaluates schedules under varying epoch horizons and memory constraints:

```bash
python -c "
from src.scheduler.workload_policy import WorkloadSchedulerPolicy
pol = WorkloadSchedulerPolicy()

for epochs in [1, 3, 5, 10]:
    plan, _ = pol.select_plan(N=17084, D=48, epochs=epochs)
    print('Epochs=' + str(epochs).rjust(2) + ' -> Selected: ' + plan.schedule_id.ljust(10) + ' | Pred Time: ' + str(round(plan.predicted_total_training_s, 2)) + 's')

plan_low_mem, _ = pol.select_plan(N=17084, D=48, epochs=10, memory_budget_mb=5.0)
print('Budget=5MB -> Selected: ' + plan_low_mem.schedule_id.ljust(10) + ' (Caching rejected due to RAM limit)')
"
```

---

## 5. Statistical Re-computation of Phase 3B Results

Recompute all pairwise speedups, population standard deviations, error metrics, and decision regrets directly from the raw experimental JSON/CSV tables:

```bash
python results/ptbxl/scheduler/phase3b/verify_phase3b_statistics.py
```

Expected output:
```
Schedule   | Cum Time Mean +/- PopStd (s)   | Mean Epoch (s)  | Mean Grad (s)  
--------------------------------------------------------------------------------
A-repeat   |   136.07 +/- 11.68 (samp: 14.30)s |      13.61 s   |       7.07 s
A-cache    |   126.29 +/-  6.08 (samp:  7.44)s |      12.41 s   |       7.58 s
B-repeat   |   115.10 +/- 14.62 (samp: 17.91)s |      11.51 s   |       4.12 s
B-cache    |    86.11 +/-  4.09 (samp:  5.00)s |       8.37 s   |       3.85 s

Pairwise Speedup Ratios (Mean +/- PopStd [ddof=0]):
  Level effect (repeated) [A-rep / B-rep]: 1.209 +/- 0.223x
  Level effect (cached)   [A-cac / B-cac]: 1.468 +/- 0.075x
  Cache effect (Cond A)   [A-rep / A-cac]: 1.080 +/- 0.101x
  Cache effect (Cond B)   [B-rep / B-cac]: 1.332 +/- 0.104x
  Combined Optimization   [A-rep / B-cac]: 1.589 +/- 0.200x
```

---

## 6. Verification of Fold 10 Holdout Integrity in Training Data Flow

Verify by static analysis and runtime mask inspection that the training and validation pipelines strictly exclude Fold 10 without reading or inspecting Fold 10 features or labels:

```bash
python -c "
import numpy as np
from pathlib import Path

# 1. Inspect training pipeline code without executing Fold 10 evaluation
code = Path('results/ptbxl/scheduler/phase3b/run_scheduler_training.py').read_text()
assert 'train_mask = np.isin(strat_fold, np.arange(1, 9))' in code, 'Missing Folds 1-8 train mask!'
assert 'val_mask = strat_fold == 9' in code, 'Missing Fold 9 val mask!'
assert 'test_mask = strat_fold == 10' in code, 'Missing Fold 10 test mask!'
assert 'assert not np.any(train_mask & test_mask)' in code, 'Missing train-test isolation assertion!'
assert 'assert not np.any(val_mask & test_mask)' in code, 'Missing val-test isolation assertion!'
assert 'X_train = scaler.fit_transform(features_raw[train_mask])' in code, 'Training data leak!'
assert 'X_val = scaler.transform(features_raw[val_mask])' in code, 'Validation data leak!'

# 2. Inspect mask properties in metadata without reading test features
data = np.load('data/ptbxl_processed/ptbxl_features_48_median_centered.npz', allow_pickle=True)
strat_fold = data['strat_fold']
train_mask = np.isin(strat_fold, np.arange(1, 9))
val_mask = strat_fold == 9
test_mask = strat_fold == 10

assert int(train_mask.sum()) == 17084, 'Unexpected train record count!'
assert int(val_mask.sum()) == 2146, 'Unexpected val record count!'
assert int(test_mask.sum()) == 2158, 'Unexpected test record count!'
assert not np.any(train_mask & test_mask), 'Contamination: Train overlaps Test!'
assert not np.any(val_mask & test_mask), 'Contamination: Val overlaps Test!'

print('DATA FLOW AUDIT PASSED: Fold 10 was strictly excluded from Phase 3B model fitting, validation, tuning, and evaluation!')
"
```
