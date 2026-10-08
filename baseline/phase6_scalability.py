import numpy as np
import tenseal as ts
import time
import os
import tracemalloc      # This library is used to track memory usage during HE inference
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Context & Config ---
results_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
he_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready"
os.makedirs(results_dir, exist_ok=True)

# MIMIC-specific Degree-5 polynomial (from Phase 3)
c0 = 0.5
c1_5 = 0.217101
c3_5 = -0.007823
c5_5 = 0.000118

print("Initializing CKKS context (practical profile)...")
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60])
ctx.global_scale = 2 ** 40
ctx.generate_galois_keys()
ctx.generate_relin_keys()

MAX_SLOTS = 8192 # N/2 for CKKS

# Load Data
X_test = np.load(os.path.join(he_dir, "X_test.npy"))
y_test = np.load(os.path.join(he_dir, "y_test.npy"))
weights = np.load(os.path.join(he_dir, "weights.npy"))
intercept = np.load(os.path.join(he_dir, "intercept.npy"))[0]

def pt_sigmoid_deg5(z):
    return c0 + c1_5 * z + c3_5 * (z**3) + c5_5 * (z**5)

def pt_inference(X, w, b):
    z = X @ w + b
    return pt_sigmoid_deg5(z)

def burn_levels(vec, n):
    for _ in range(n):
        vec = vec * 1.0
    return vec

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def run_he_inference(X_batch, w, b, track_levels=False):
    t_enc_start = time.time()
    
    n_samples, n_features = X_batch.shape
    # If samples > MAX_SLOTS, we need multiple batches.
    n_batches = int(np.ceil(n_samples / MAX_SLOTS))
    
    enc_batches = []
    for i in range(n_batches):
        start_idx = i * MAX_SLOTS
        end_idx = min((i + 1) * MAX_SLOTS, n_samples)
        batch = X_batch[start_idx:end_idx]
        
        # encrypt each feature column in this batch
        enc_cols = [ts.ckks_vector(ctx, batch[:, j].tolist()) for j in range(n_features)]
        enc_batches.append(enc_cols)
        
    t_enc = time.time() - t_enc_start
    
    t_comp_start = time.time()
    res_batches = []
    final_levels = None
    
    for i, enc_cols in enumerate(enc_batches):
        # 1. z = X @ w + b
        z = enc_cols[0] * w[0]
        for j in range(1, n_features):
            z = z + (enc_cols[j] * w[j])
        z = z + b
        
        # 2. Degree 5 Sigmoid
        z2 = z * z
        z3 = z2 * burn_levels(z, 1)     # Align z for multiplication with z2
        z5 = z3 * burn_levels(z2, 1)
        
        term5 = z5 * c5_5
        term3 = burn_levels(z3, 1) * c3_5
        term1 = burn_levels(z, 3) * c1_5
        
        p = term5 + term3 + term1 + c0
        res_batches.append(p)
        
        if track_levels and i == 0:         # Track levels only for the first batch
            final_levels = get_levels(p)
            
    t_comp = time.time() - t_comp_start
    
    t_dec_start = time.time()
    preds = []          # Decrypt and collect predictions from all batches
    for p in res_batches:
        preds.extend(p.decrypt())           # Decrypt the CKKS vector and extend the list
    t_dec = time.time() - t_dec_start
    
    # Calculate ciphertexts used (features * batches)
    n_ciphertexts = n_features * n_batches
    
    return np.array(preds), t_enc, t_comp, t_dec, n_batches, n_ciphertexts, final_levels

def run_experiment(workload_size, runs=3):
    X_sub = X_test[:workload_size]
    y_sub = y_test[:workload_size]
    
    # Plaintext validation
    pt_preds = pt_inference(X_sub, weights, intercept)
    
    metrics = {"enc": [], "comp": [], "dec": [], "tot": [], "mem": []}
    
    for r in range(runs):
        tracemalloc.start()
        
        he_preds, t_e, t_c, t_d, n_b, n_c, lvl = run_he_inference(X_sub, weights, intercept, track_levels=(r==0))
        
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        mem_mb = peak / (1024 * 1024)
        
        metrics["enc"].append(t_e)
        metrics["comp"].append(t_c)
        metrics["dec"].append(t_d)
        metrics["tot"].append(t_e + t_c + t_d)
        metrics["mem"].append(mem_mb)

        
        # Save exact returns from the first run
        if r == 0:
            final_he_preds = he_preds
            final_batches = n_b
            final_ciphertexts = n_c
            final_levels = lvl
            
    # Errors vs plaintext
    err = np.abs(final_he_preds - pt_preds)
    
    # Classification metrics
    acc = accuracy_score(y_sub, (final_he_preds >= 0.5).astype(int))
    # Some small samples might only have 1 class, avoid roc_auc error
    if len(np.unique(y_sub)) > 1:
        auc = roc_auc_score(y_sub, final_he_preds)
    else:
        auc = np.nan
        
    return {
        "admissions": workload_size,
        "features": 10,
        "ciphertexts": final_ciphertexts,
        "batches": final_batches,
        "encryption_s": np.mean(metrics["enc"]),
        "HE_compute_s": np.mean(metrics["comp"]),
        "decryption_s": np.mean(metrics["dec"]),
        "total_s": np.mean(metrics["tot"]),
        "peak_memory_MB": np.max(metrics["mem"]),
        "accuracy": acc,
        "ROC_AUC": auc,
        "max_error": err.max(),
        "mean_error": err.mean(),
        "final_levels_remaining": final_levels
    }

print("\n--- CONTROL VALIDATION ---")
# Very small control run to verify everything
ctrl = run_experiment(10, runs=1)
print("Control run completed.")
assert np.isnan(ctrl["ROC_AUC"]) or ctrl["ROC_AUC"] >= 0
print(f"Max error against pt: {ctrl['max_error']:.8f}")
print(f"Final modulus level: {ctrl['final_levels_remaining']}")

print("\n--- MAIN SCALABILITY EXPERIMENT ---")
workloads = [500, 1000, 2500, 5000]
results = []
for w in workloads:
    print(f"Running {w} admissions (3 reps)...")
    res = run_experiment(w, runs=3)
    results.append(res)
    
df_res = pd.DataFrame(results)

# Extra derived metrics
df_res["time_per_admission"] = df_res["total_s"] / df_res["admissions"]
df_res["time_per_batch"] = df_res["total_s"] / df_res["batches"]
df_res["throughput"] = df_res["admissions"] / df_res["total_s"]
df_res["ciphertexts_per_admission"] = df_res["ciphertexts"] / df_res["admissions"]

df_res.to_csv(os.path.join(results_dir, "scalability_results.csv"), index=False)

# Plots
import matplotlib.pyplot as plt

# 1. Runtime vs Admissions
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df_res["admissions"], df_res["HE_compute_s"], marker='o', label="HE Compute")
ax.plot(df_res["admissions"], df_res["encryption_s"], marker='s', label="Encryption")
ax.plot(df_res["admissions"], df_res["decryption_s"], marker='^', label="Decryption")
ax.plot(df_res["admissions"], df_res["total_s"], marker='x', label="Total Time", lw=2, color='black')
ax.set_xlabel("Admissions")
ax.set_ylabel("Time (seconds)")
ax.set_title("Runtime vs Dataset Size (SIMD Packed)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "runtime_vs_admissions.png"))
plt.close()

# 2. Peak Memory vs Admissions
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df_res["admissions"], df_res["peak_memory_MB"], marker='o', color='purple')
ax.set_xlabel("Admissions")
ax.set_ylabel("Peak Memory (MB)")
ax.set_title("Memory Usage vs Dataset Size")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "memory_vs_admissions.png"))
plt.close()

# 3. Ciphertexts vs Admissions
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df_res["admissions"], df_res["ciphertexts"], marker='o', color='green')
ax.set_xlabel("Admissions")
ax.set_ylabel("Ciphertext Count")
ax.set_title("Ciphertext Workload vs Dataset Size")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ciphertexts_vs_admissions.png"))
plt.close()

# 4. Throughput vs Admissions
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df_res["admissions"], df_res["throughput"], marker='o', color='darkorange')
ax.set_xlabel("Admissions")
ax.set_ylabel("Throughput (Admissions / Second)")
ax.set_title("Throughput vs Dataset Size (SIMD Batching Benefit)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "throughput_vs_admissions.png"))
plt.close()

# 5. Accuracy vs Admissions
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df_res["admissions"], df_res["accuracy"], marker='o', color='darkblue')
ax.set_xlabel("Admissions")
ax.set_ylabel("Accuracy")
ax.set_title("Inference Accuracy vs Dataset Size")
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_vs_admissions.png"))
plt.close()

print("\n--- RESULTS ---")
print(df_res[["admissions", "features", "ciphertexts", "batches", "encryption_s", "HE_compute_s", "decryption_s", "total_s", "peak_memory_MB", "accuracy", "ROC_AUC", "max_error", "mean_error", "final_levels_remaining"]].to_string(index=False))

print("\n--- THROUGHPUT TABLE ---")
print(df_res[["admissions", "total_s", "time_per_admission", "time_per_batch", "throughput"]].to_string(index=False))

print("\nSaved artifacts and plots to results directory.")
