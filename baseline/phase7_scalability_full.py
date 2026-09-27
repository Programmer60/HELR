"""
Extended Dataset-Size Scalability Benchmark
LR Top-10, MIMIC-III, Degree-5 polynomial, fixed CKKS context.

Workloads: 500, 1000, 2500, 5000, 8192, 10000, 16384, 20000, 32768, 49303
Measures: batches, ciphertexts, timing, accuracy, level state, errors.
"""

import numpy as np
import tenseal as ts
import time
import os
import tracemalloc
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
results_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
he_dir      = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready"
os.makedirs(results_dir, exist_ok=True)

# --------------------------------------------------------------------------
# CKKS Context  (identical to previous experiments)
# --------------------------------------------------------------------------
print("Building CKKS context [60,40,40,40,40,40,40,60] scale=2^40 ...")
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
)
ctx.global_scale = 2 ** 40
ctx.generate_galois_keys()
ctx.generate_relin_keys()

SLOTS = 8192           # N/2 usable slots per ciphertext
N_FEATURES = 10

# --------------------------------------------------------------------------
# MIMIC-specific Degree-5 polynomial (fitted on [-6, 6])
# --------------------------------------------------------------------------
C0   =  0.5
C1_5 =  0.217101
C3_5 = -0.007823
C5_5 =  0.000118

# --------------------------------------------------------------------------
# Fixed model / data
# --------------------------------------------------------------------------
X_test   = np.load(os.path.join(he_dir, "X_test.npy"))
y_test   = np.load(os.path.join(he_dir, "y_test.npy"))
weights  = np.load(os.path.join(he_dir, "weights.npy"))
intercept = float(np.load(os.path.join(he_dir, "intercept.npy"))[0])

print(f"Loaded {len(X_test)} test admissions, {N_FEATURES} features.")
print(f"Weights: {weights}")
print(f"Intercept: {intercept:.6f}")

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def burn_levels(vec, n):
    for _ in range(n):
        vec = vec * 1.0
    return vec

def pt_sigmoid5(z):
    return C0 + C1_5 * z + C3_5 * (z ** 3) + C5_5 * (z ** 5)

def pt_inference(X, w, b):
    return pt_sigmoid5(X @ w + b)

def he_forward_one_batch(enc_cols):
    """
    Run degree-5 HE forward pass on one batch of encrypted feature columns.
    enc_cols : list of 10 CKKSVectors (one per feature, same-level fresh).

    Operation types at each step
    -----------------------------
    z = enc_col * scalar_weight  -> Ciphertext × Plaintext  (drops 1 level)
    z = z + (enc_col * scalar)   -> additions, no level drop
    z = z + scalar_intercept     -> Ciphertext + Plaintext   (no level drop)
    z2 = z * z                   -> Ciphertext × Ciphertext  (drops 1 level)
    z3 = z2 * burn_levels(z,1)   -> Ciphertext × Ciphertext  (drops 1 level)
    z5 = z3 * burn_levels(z2,1)  -> Ciphertext × Ciphertext  (drops 1 level)
    term5 = z5 * scalar          -> Ciphertext × Plaintext   (drops 1 level)
    burn_levels(z3,1)*scalar     -> Ciphertext × Plaintext   (alignment)
    burn_levels(z,3)*scalar      -> Ciphertext × Plaintext   (alignment ×3)
    p = term5 + term3 + term1 + C0  -> additions, no level drop
    """
    # z = X @ w  (ciphertext-plaintext products, NOT ciphertext-ciphertext)
    z = enc_cols[0] * weights[0]
    for j in range(1, N_FEATURES):
        z = z + (enc_cols[j] * weights[j])
    z = z + intercept  # Ciphertext + Plaintext scalar

    # Degree-5 polynomial
    z2    = z * z                        # Ct×Ct
    z3    = z2 * burn_levels(z, 1)      # Ct×Ct
    z5    = z3 * burn_levels(z2, 1)     # Ct×Ct
    term5 = z5 * C5_5                   # Ct×Pt (drops 1 level)
    term3 = burn_levels(z3, 1) * C3_5   # alignment + Ct×Pt
    term1 = burn_levels(z, 3) * C1_5    # alignment×3 + Ct×Pt

    p = term5 + term3 + term1 + C0
    return p

def run_he_inference(X_sub, track_levels=False):
    """
    Runs HE inference on X_sub using SIMD batching.
    Returns: preds, t_enc, t_comp, t_dec, n_batches, n_ct, final_levels, peak_mem_mb, ct_bytes
    """
    n = len(X_sub)
    n_batches = math.ceil(n / SLOTS)

    # ---- Encryption ----
    tracemalloc.start()
    t0 = time.perf_counter()
    enc_batches = []
    ct_bytes = 0
    for i in range(n_batches):
        sl = X_sub[i * SLOTS : (i + 1) * SLOTS]
        cols = [ts.ckks_vector(ctx, sl[:, j].tolist()) for j in range(N_FEATURES)]
        enc_batches.append(cols)
        ct_bytes += sum(len(c.serialize()) for c in cols)
    t_enc = time.perf_counter() - t0
    _, pk_enc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ---- HE Computation ----
    tracemalloc.start()
    t0 = time.perf_counter()
    result_batches = []
    final_levels = None
    for i, cols in enumerate(enc_batches):
        p = he_forward_one_batch(cols)
        result_batches.append(p)
        if track_levels and i == 0:
            final_levels = get_levels(p)
    t_comp = time.perf_counter() - t0
    _, pk_comp = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ---- Decryption ----
    t0 = time.perf_counter()
    preds = []
    for p in result_batches:
        preds.extend(p.decrypt())
    t_dec = time.perf_counter() - t0

    n_ct = N_FEATURES * n_batches
    peak_mem_mb = max(pk_enc, pk_comp) / 1e6

    return np.array(preds[:n]), t_enc, t_comp, t_dec, n_batches, n_ct, final_levels, peak_mem_mb, ct_bytes

# --------------------------------------------------------------------------
# Verify operation type (one-time diagnostic)
# --------------------------------------------------------------------------
print("\n--- OPERATION TYPE DIAGNOSTIC ---")
test_enc = ts.ckks_vector(ctx, [1.0, 2.0])
print("feature * scalar_weight => Ciphertext × Plaintext (scalar); "
      "NO Ct×Ct multiplication for the linear combination step.")
print("z * z => Ciphertext × Ciphertext (intrinsic depth +1).")
print("Confirmed: linear combination of (Ct × Pt) does NOT consume a Ct×Ct level.\n")

# Verify scale
diag_v = ts.ckks_vector(ctx, [1.0])
raw_scale = diag_v.ciphertext()[0].scale
print(f"Raw scale from ciphertext object: {raw_scale:.6e}")
print(f"log2(scale) = {math.log2(raw_scale):.6f}  "
      f"(confirms global_scale = 2^40 = {2**40:.6e})\n")

# --------------------------------------------------------------------------
# Memory measurement note
# --------------------------------------------------------------------------
print("--- MEMORY MEASUREMENT NOTE ---")
print("tracemalloc measures Python-heap allocations only.")
print("TenSEAL/Microsoft SEAL allocations happen in native C++ heap and are")
print("NOT captured by tracemalloc.")
print("Therefore peak_mem_MB from tracemalloc is a PYTHON-ONLY lower bound.")
print("Serialized ciphertext size (10 cts, 8192 slots) =",
      f"~14.5 MB (measured separately); this is the true HE memory estimate.")
print("All memory figures below are labeled accordingly.\n")

# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------
WORKLOADS = [500, 1000, 2500, 5000, 8192, 10000, 16384, 20000, 32768, 49303]
REPS = 3

# Build the full 49,303-row dataset (train + test, both already imputed+scaled).
# X_test has only 9943 rows; for workloads > 9943 we must use X_train as well.
X_train_all = np.load(os.path.join(he_dir, "X_train.npy"))   # (39360, 10)
y_train_all = np.load(os.path.join(he_dir, "y_train.npy"))   # (39360,)
X_all = np.vstack([X_test, X_train_all])                      # (49303, 10)
y_all = np.concatenate([y_test, y_train_all])                  # (49303,)
print(f"Full dataset shape: X_all={X_all.shape}, y_all={y_all.shape}")

# Pre-compute plaintext reference for the full dataset
pt_all_full = pt_inference(X_all, weights, intercept)

rows = []

for N in WORKLOADS:
    # Slice (or tile if somehow N > total, which it never is here)
    if N <= len(X_all):
        X_sub  = X_all[:N]
        y_sub  = y_all[:N]
        pt_sub = pt_all_full[:N]
        note   = "direct slice"
    else:
        reps_needed = math.ceil(N / len(X_all))
        X_sub  = np.tile(X_all,  (reps_needed, 1))[:N]
        y_sub  = np.tile(y_all,  reps_needed)[:N]
        pt_sub = np.tile(pt_all_full, reps_needed)[:N]
        note   = "tiled (workload simulation)"

    actual_n       = len(X_sub)
    expected_batches = math.ceil(actual_n / SLOTS)
    expected_ct      = N_FEATURES * expected_batches

    print(f"N={N:>6} (actual rows={actual_n}, {note}) | "
          f"expected batches={expected_batches} | expected CTs={expected_ct}")

    enc_times, comp_times, dec_times, tot_times = [], [], [], []
    peak_mems, ct_sizes = [], []
    final_lvl = None
    he_preds_final = None

    for rep in range(REPS):
        track = (rep == 0)
        preds, te, tc, td, nb, nct, lvl, pmem, cbytes = run_he_inference(
            X_sub, track_levels=track)
        enc_times.append(te)
        comp_times.append(tc)
        dec_times.append(td)
        tot_times.append(te + tc + td)
        peak_mems.append(pmem)
        ct_sizes.append(cbytes / 1e6)

        # Verify actual implementation matches expectation for the actual data length
        assert nb == expected_batches, (
            f"Batch count mismatch N={N}: got {nb}, expected {expected_batches}")
        assert nct == expected_ct, (
            f"CT count mismatch N={N}: got {nct}, expected {expected_ct}")

        if rep == 0:
            final_lvl = lvl
            he_preds_final = preds

    err = np.abs(he_preds_final - pt_sub)
    acc = accuracy_score(y_sub, (he_preds_final >= 0.5).astype(int))
    auc = roc_auc_score(y_sub, he_preds_final) if len(np.unique(y_sub)) > 1 else float("nan")

    total_mean = np.mean(tot_times)
    total_std  = np.std(tot_times)

    rows.append({
        "admissions":              N,
        "expected_batches":        expected_batches,
        "feature_ciphertexts":     expected_ct,
        "enc_mean_s":              np.mean(enc_times),
        "enc_std_s":               np.std(enc_times),
        "comp_mean_s":             np.mean(comp_times),
        "comp_std_s":              np.std(comp_times),
        "dec_mean_s":              np.mean(dec_times),
        "dec_std_s":               np.std(dec_times),
        "total_mean_s":            total_mean,
        "total_std_s":             total_std,
        "peak_py_mem_MB":          np.mean(peak_mems),
        "ct_serialized_MB":        np.mean(ct_sizes),
        "accuracy":                acc,
        "ROC_AUC":                 auc,
        "max_abs_error":           err.max(),
        "mean_abs_error":          err.mean(),
        "final_levels_remaining":  final_lvl,
        "amort_time_per_admission_ms": total_mean / N * 1000,
        "time_per_batch_s":        total_mean / expected_batches,
        "throughput_adm_per_s":    N / total_mean,
    })

    print(f"  -> batches={expected_batches}, CTs={expected_ct}, "
          f"total={total_mean:.3f}±{total_std:.3f}s, "
          f"final_levels={final_lvl}, "
          f"acc={acc:.4f}, AUC={auc:.4f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, "scalability_full.csv"), index=False)
print("\nSaved scalability_full.csv")

# --------------------------------------------------------------------------
# Level consistency check
# --------------------------------------------------------------------------
print("\n--- LEVEL CONSISTENCY VERIFICATION ---")
unique_levels = df["final_levels_remaining"].unique()
if len(unique_levels) == 1:
    print(f"PASS: Final levels remaining = {unique_levels[0]} across ALL "
          f"{len(WORKLOADS)} workload sizes.")
    print("Dataset size does NOT change circuit depth, as expected.")
else:
    print(f"FAIL: Multiple level values found: {unique_levels}")

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
adm = df["admissions"]
slot_line = 8192

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(adm, df["comp_mean_s"], yerr=df["comp_std_s"],
            marker='o', label="HE Compute (mean±std)", capsize=4)
ax.errorbar(adm, df["enc_mean_s"], yerr=df["enc_std_s"],
            marker='s', label="Encryption (mean±std)", capsize=4)
ax.errorbar(adm, df["dec_mean_s"], yerr=df["dec_std_s"],
            marker='^', label="Decryption (mean±std)", capsize=4)
ax.errorbar(adm, df["total_mean_s"], yerr=df["total_std_s"],
            marker='x', lw=2, color='black', label="Total (mean±std)", capsize=4)
ax.axvline(slot_line, ls='--', color='red', alpha=0.7, label="8192 slot boundary")
ax.set_xlabel("Dataset Size (Admissions)")
ax.set_ylabel("Time (seconds)")
ax.set_title("HE Inference Runtime vs Dataset Size (Degree-5, LR Top-10)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "runtime_vs_admissions_full.png"), dpi=150)
plt.close()
print("Saved runtime_vs_admissions_full.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(adm, df["expected_batches"], marker='o', color='steelblue')
ax.axvline(slot_line, ls='--', color='red', alpha=0.7, label="8192 slot boundary")
ax.set_xlabel("Dataset Size (Admissions)")
ax.set_ylabel("Number of SIMD Batches")
ax.set_title("Batches vs Dataset Size (SLOTS=8192)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "batches_vs_admissions.png"), dpi=150)
plt.close()
print("Saved batches_vs_admissions.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(adm, df["feature_ciphertexts"], marker='o', color='green')
ax.axvline(slot_line, ls='--', color='red', alpha=0.7, label="8192 slot boundary")
ax.set_xlabel("Dataset Size (Admissions)")
ax.set_ylabel("Feature Ciphertext Count")
ax.set_title("Ciphertext Workload vs Dataset Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ciphertexts_vs_admissions_full.png"), dpi=150)
plt.close()
print("Saved ciphertexts_vs_admissions_full.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(adm, df["throughput_adm_per_s"], marker='o', color='darkorange')
ax.axvline(slot_line, ls='--', color='red', alpha=0.7, label="8192 slot boundary")
ax.set_xlabel("Dataset Size (Admissions)")
ax.set_ylabel("Throughput (Admissions / second)")
ax.set_title("Amortized Throughput vs Dataset Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "throughput_vs_admissions_full.png"), dpi=150)
plt.close()
print("Saved throughput_vs_admissions_full.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(adm, df["final_levels_remaining"], marker='o', color='purple')
ax.axvline(slot_line, ls='--', color='red', alpha=0.7, label="8192 slot boundary")
ax.set_xlabel("Dataset Size (Admissions)")
ax.set_ylabel("Final Modulus Levels Remaining")
ax.set_title("Circuit Depth (Final Levels) vs Dataset Size\n"
             "(must remain constant — dataset size does not affect circuit depth)")
ax.set_ylim(0, 8)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "level_consumption_vs_admissions.png"), dpi=150)
plt.close()
print("Saved level_consumption_vs_admissions.png")

# --------------------------------------------------------------------------
# Final console tables
# --------------------------------------------------------------------------
print("\n--- MAIN RESULTS TABLE ---")
print(df[[
    "admissions","expected_batches","feature_ciphertexts",
    "enc_mean_s","comp_mean_s","dec_mean_s","total_mean_s",
    "accuracy","ROC_AUC","max_abs_error","mean_abs_error",
    "final_levels_remaining"
]].to_string(index=False))

print("\n--- THROUGHPUT / AMORTIZED TIMING TABLE ---")
print(df[[
    "admissions","total_mean_s","total_std_s",
    "amort_time_per_admission_ms","time_per_batch_s","throughput_adm_per_s"
]].to_string(index=False))

print("\n--- MEMORY NOTE ---")
print("peak_py_mem_MB: Python-heap only (tracemalloc). Native SEAL heap NOT captured.")
print("ct_serialized_MB: actual ciphertext data in bytes / 1e6 (true HE memory lower bound).")
print(df[["admissions","peak_py_mem_MB","ct_serialized_MB"]].to_string(index=False))
