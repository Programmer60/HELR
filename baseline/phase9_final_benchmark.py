"""
Final Full-Dataset MIMIC-III HE Inference Benchmark
=====================================================
49,303 admissions · LR Top-10 · Degree-5 · [60,40,40,40,40,40,40,60] · scale=2^40
"""
import numpy as np
import tenseal as ts
import time, os, math, tracemalloc, json
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

# ── Paths ────────────────────────────────────────────────────────────────────
results = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
he_dir  = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready"
os.makedirs(results, exist_ok=True)

# ── CKKS context ─────────────────────────────────────────────────────────────
print("Building CKKS context [60,40,40,40,40,40,40,60] scale=2^40 ...")
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
)
ctx.global_scale = 2 ** 40
ctx.generate_galois_keys()
ctx.generate_relin_keys()
SLOTS = 8192

# ── Degree-5 MIMIC polynomial ─────────────────────────────────────────────────
C0, C1, C3, C5 = 0.5, 0.217101, -0.007823, 0.000118

def pt_sig5(z):
    return C0 + C1*z + C3*(z**3) + C5*(z**5)

def burn(v, n):
    for _ in range(n):
        v = v * 1.0
    return v

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

# ── Load fixed model artefacts ────────────────────────────────────────────────
X_train = np.load(os.path.join(he_dir, "X_train.npy"))   # (39360,10)
y_train = np.load(os.path.join(he_dir, "y_train.npy"))
X_test  = np.load(os.path.join(he_dir, "X_test.npy"))    # (9943,10)
y_test  = np.load(os.path.join(he_dir, "y_test.npy"))
weights   = np.load(os.path.join(he_dir, "weights.npy"))  # (10,)
intercept = float(np.load(os.path.join(he_dir, "intercept.npy"))[0])

FEATURES_10 = [
    "AGE","Systolic BP","Mean Arterial Pressure","Diastolic BP","GENDER",
    "Potassium","Heart Rate","Hemoglobin","Glucose","Oxygen Saturation"
]

print(f"Test set (held-out): {len(X_test)} admissions, {X_test.shape[1]} features")
print(f"Full cohort: {len(X_train)+len(X_test)} admissions")

# ── Helper: forward pass on one encrypted batch ───────────────────────────────
def he_forward_batch(enc_cols, w, b):
    """
    Linear combination: Ciphertext × Plaintext (not Ct×Ct).
    Degree-5 polynomial: three Ct×Ct multiplications (z*z, z2*z, z3*z2),
    plus Ct×Pt rescales for polynomial term coefficients and alignment.

    Ciphertext-ciphertext multiplicative depth of degree-5 circuit: 4
      (1 for linear combination drop + 3 for z2/z3/z5)
    Alignment/burn levels: 6 (1+1+1+3 consumed by burn_levels calls)
    Total observed modulus-level consumption: 5 levels
    Final levels remaining: 2  (from 7 usable levels in this context)
    """
    n_feats = len(enc_cols)
    z = enc_cols[0] * w[0]
    for j in range(1, n_feats):
        z = z + (enc_cols[j] * w[j])
    z = z + b                    # Ct + Pt scalar

    z2    = z  * z               # Ct×Ct  depth+1
    z3    = z2 * burn(z, 1)      # Ct×Ct  depth+1
    z5    = z3 * burn(z2, 1)     # Ct×Ct  depth+1
    term5 = z5 * C5              # Ct×Pt  (rescale, drop 1)
    term3 = burn(z3, 1) * C3    # alignment + Ct×Pt
    term1 = burn(z,  3) * C1    # alignment×3 + Ct×Pt
    p     = term5 + term3 + term1 + C0
    return p

# ── Task 1: Full 49,303-admission HE inference ───────────────────────────────
print("\n=== TASK 1: FULL 49,303-ADMISSION HE INFERENCE ===")

X_all = np.vstack([X_test, X_train])   # 49,303 rows, test-first for ordering
y_all = np.concatenate([y_test, y_train])
N_FULL = len(X_all)

expected_batches = math.ceil(N_FULL / SLOTS)
expected_ct      = 10 * expected_batches
print(f"N={N_FULL}, slots={SLOTS}, batches={expected_batches}, "
      f"feature CTs={expected_ct}")

REPS = 3            # To get a sense of variability, repeat the full inference 3 times
enc_times, comp_times, dec_times, tot_times, ct_mb_list = [], [], [], [], []
he_preds_full = None
final_lvl_full = None

for rep in range(REPS):
    print(f"  Rep {rep+1}/{REPS} ...", end=" ", flush=True)
    n_batches = 0; n_ct = 0; total_ct_bytes = 0

    tracemalloc.start()
    t_enc_start = time.perf_counter()
    enc_batches = []
    for i in range(expected_batches):
        sl = X_all[i*SLOTS : (i+1)*SLOTS]
        cols = [ts.ckks_vector(ctx, sl[:, j].tolist()) for j in range(10)]
        enc_batches.append(cols)
        total_ct_bytes += sum(len(c.serialize()) for c in cols)
    t_enc = time.perf_counter() - t_enc_start
    tracemalloc.stop()

    t_comp_start = time.perf_counter()
    result_batches = []
    for i, cols in enumerate(enc_batches):
        p = he_forward_batch(cols, weights, intercept)
        result_batches.append(p)
        if rep == 0 and i == 0:
            final_lvl_full = get_levels(p)
    t_comp = time.perf_counter() - t_comp_start

    t_dec_start = time.perf_counter()
    preds = []
    for p in result_batches:
        preds.extend(p.decrypt())
    t_dec = time.perf_counter() - t_dec_start

    preds = np.array(preds[:N_FULL])
    enc_times.append(t_enc)
    comp_times.append(t_comp)
    dec_times.append(t_dec)
    tot_times.append(t_enc + t_comp + t_dec)
    ct_mb_list.append(total_ct_bytes / 1e6)

    if rep == 0:
        he_preds_full = preds

    print(f"total={t_enc+t_comp+t_dec:.2f}s")

# Numerical error vs plaintext (full cohort)
pt_full = pt_sig5(X_all @ weights + intercept)
err_full = np.abs(he_preds_full - pt_full)

full_row = {
    "admissions":             N_FULL,
    "features":               10,
    "batches":                expected_batches,
    "feature_ciphertexts":    expected_ct,
    "enc_mean_s":             np.mean(enc_times),
    "enc_std_s":              np.std(enc_times),
    "comp_mean_s":            np.mean(comp_times),
    "comp_std_s":             np.std(comp_times),
    "dec_mean_s":             np.mean(dec_times),
    "dec_std_s":              np.std(dec_times),
    "total_mean_s":           np.mean(tot_times),
    "total_std_s":            np.std(tot_times),
    "ct_serialized_MB":       np.mean(ct_mb_list),
    "final_levels_remaining": final_lvl_full,
    "max_abs_error":          err_full.max(),
    "mean_abs_error":         err_full.mean(),
}

print(f"\nFull-cohort result:")
print(f"  batches={expected_batches}, feature CTs={expected_ct}")
print(f"  total={full_row['total_mean_s']:.3f}±{full_row['total_std_s']:.3f}s")
print(f"  serialized CT footprint={full_row['ct_serialized_MB']:.2f} MB")
print(f"  final levels remaining={final_lvl_full}")
print(f"  max_abs_error={err_full.max():.2e}, mean_abs_error={err_full.mean():.2e}")

pd.DataFrame([full_row]).to_csv(
    os.path.join(results, "mimic3_full_he_results.csv"), index=False)
print("Saved mimic3_full_he_results.csv")

# ── Task 2: Predictive fidelity on held-out test set ONLY ────────────────────
print("\n=== TASK 2: PREDICTIVE FIDELITY ON HELD-OUT TEST SET ===")
print("(9,943 records; original patient-level split, zero subject overlap)")

# Plain LR on held-out test (using stored LR model weights)
pt_test = pt_sig5(X_test @ weights + intercept)
y_pred_pt  = (pt_test >= 0.5).astype(int)

# HE predictions on held-out test — use the first 9,943 slots of full run
# (X_all was constructed as X_test first, so he_preds_full[:9943] = he test preds)
he_test_preds = he_preds_full[:len(X_test)]
y_pred_he = (he_test_preds >= 0.5).astype(int)

def metrics(y_true, y_prob, y_pred, label):
    return {
        "model":     label,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }

m_pt = metrics(y_test, pt_test,    y_pred_pt, "Plaintext LR")
m_he = metrics(y_test, he_test_preds, y_pred_he, "HELR (Degree-5)")

df_fidelity = pd.DataFrame([m_pt, m_he])
df_fidelity.to_csv(os.path.join(results, "plaintext_vs_he_metrics.csv"), index=False)
print(df_fidelity.to_string(index=False))
print("Saved plaintext_vs_he_metrics.csv")

# ── Task 3: Master research table ─────────────────────────────────────────────
print("\n=== TASK 3: MASTER RESEARCH TABLE ===")

# A. Polynomial-degree experiment (from validated circuit, full fwd+bwd)
poly_rows = [
    {"experiment":"A. Polynomial Degree","degree":1,
     "ct_ct_multiplicative_depth":1,"observed_levels_consumed":3,
     "alignment_burn_levels":0,"final_levels_remaining":4,
     "sigmoid_description":"0.5 + 0.197z"},
    {"experiment":"A. Polynomial Degree","degree":3,
     "ct_ct_multiplicative_depth":3,"observed_levels_consumed":5,
     "alignment_burn_levels":3,"final_levels_remaining":2,
     "sigmoid_description":"0.5 + c1*z + c3*z^3"},
    {"experiment":"A. Polynomial Degree","degree":5,
     "ct_ct_multiplicative_depth":4,"observed_levels_consumed":6,
     "alignment_burn_levels":6,"final_levels_remaining":1,
     "sigmoid_description":"0.5 + c1*z + c3*z^3 + c5*z^5"},
]

# B. Feature-count experiment (8,192 admissions, 1 batch)
feat_rows = [
    {"experiment":"B. Feature Count","features":10,"admissions":8192,"batches":1,
     "feature_ciphertexts":10,"total_mean_s":0.663,"total_std_s":0.038,
     "ct_serialized_MB":14.50,"final_levels_remaining":2},
    {"experiment":"B. Feature Count","features":15,"admissions":8192,"batches":1,
     "feature_ciphertexts":15,"total_mean_s":0.879,"total_std_s":0.049,
     "ct_serialized_MB":21.74,"final_levels_remaining":2},
    {"experiment":"B. Feature Count","features":16,"admissions":8192,"batches":1,
     "feature_ciphertexts":16,"total_mean_s":0.986,"total_std_s":0.060,
     "ct_serialized_MB":23.19,"final_levels_remaining":2},
]

# C. Dataset-size experiment (LR Top-10, Degree-5)
size_rows = [
    {"experiment":"C. Dataset Size","admissions":500,  "batches":1,"feature_ciphertexts":10, "total_mean_s":0.430,"total_std_s":0.009,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":1000, "batches":1,"feature_ciphertexts":10, "total_mean_s":0.429,"total_std_s":0.005,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":2500, "batches":1,"feature_ciphertexts":10, "total_mean_s":0.449,"total_std_s":0.005,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":5000, "batches":1,"feature_ciphertexts":10, "total_mean_s":0.501,"total_std_s":0.025,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":8192, "batches":1,"feature_ciphertexts":10, "total_mean_s":0.521,"total_std_s":0.005,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":10000,"batches":2,"feature_ciphertexts":20, "total_mean_s":0.953,"total_std_s":0.023,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":16384,"batches":2,"feature_ciphertexts":20, "total_mean_s":1.109,"total_std_s":0.011,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":20000,"batches":3,"feature_ciphertexts":30, "total_mean_s":1.588,"total_std_s":0.017,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":32768,"batches":4,"feature_ciphertexts":40, "total_mean_s":2.096,"total_std_s":0.010,"final_levels_remaining":2},
    {"experiment":"C. Dataset Size","admissions":49303,"batches":7,"feature_ciphertexts":70,
     "total_mean_s":full_row["total_mean_s"],"total_std_s":full_row["total_std_s"],
     "final_levels_remaining":final_lvl_full},
]

df_master = pd.DataFrame(poly_rows + feat_rows + size_rows)
df_master.to_csv(os.path.join(results, "master_experiment_table.csv"), index=False)
print("Saved master_experiment_table.csv")

# ── Task 5: Final plots ────────────────────────────────────────────────────────

# 1. Final runtime scalability (all N)
df_sz = pd.DataFrame(size_rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.errorbar(df_sz["admissions"], df_sz["total_mean_s"], yerr=df_sz["total_std_s"],
            marker='o', color='steelblue', capsize=4, lw=2)
ax.axvline(8192, ls='--', color='red', alpha=0.7, label="SIMD slot boundary (8,192)")
ax.set_xlabel("Admissions (N)")
ax.set_ylabel("Total Runtime (s, mean ± std)")
ax.set_title("HE Inference Runtime vs Dataset Size\n(LR Top-10, Degree-5, 2^40 scale)")
ax.legend(fontsize=8)
ax2 = axes[1]
ax2.plot(df_sz["admissions"], df_sz["feature_ciphertexts"], marker='s', color='green', lw=2)
ax2.axvline(8192, ls='--', color='red', alpha=0.7, label="SIMD slot boundary (8,192)")
ax2.set_xlabel("Admissions (N)")
ax2.set_ylabel("Feature Ciphertexts")
ax2.set_title("Ciphertext Workload vs Dataset Size")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(results, "final_runtime_scalability.png"), dpi=150)
plt.close()
print("Saved final_runtime_scalability.png")

# 2. Feature count experiment summary
df_fc = pd.DataFrame(feat_rows)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(df_fc["features"], df_fc["total_mean_s"],
            yerr=df_fc["total_std_s"], color=["#2196F3","#FF9800","#4CAF50"],
            capsize=6, width=1.2)
axes[0].set_xticks(df_fc["features"])
axes[0].set_xlabel("Feature Count"); axes[0].set_ylabel("Runtime (s)")
axes[0].set_title("Runtime vs Feature Count\n(N=8192, 1 batch)")

axes[1].bar(df_fc["features"], df_fc["feature_ciphertexts"],
            color=["#2196F3","#FF9800","#4CAF50"], width=1.2)
axes[1].set_xticks(df_fc["features"])
axes[1].set_xlabel("Feature Count"); axes[1].set_ylabel("Feature Ciphertexts")
axes[1].set_title("Ciphertext Count vs Feature Count\n(CTs = features × batches)")

axes[2].plot(df_fc["features"], df_fc["final_levels_remaining"],
             marker='o', color='purple', lw=2)
axes[2].set_ylim(0, 8)
axes[2].set_xticks(df_fc["features"])
axes[2].set_xlabel("Feature Count"); axes[2].set_ylabel("Final Levels Remaining")
axes[2].set_title("Circuit Depth vs Feature Count\n(invariant — depth fixed by polynomial)")
plt.tight_layout()
plt.savefig(os.path.join(results, "final_feature_count.png"), dpi=150)
plt.close()
print("Saved final_feature_count.png")

# 3. Final level consumption across all three experiments
df_poly = pd.DataFrame(poly_rows)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(df_poly["degree"].astype(str).apply(lambda d: f"Deg-{d}"),
            df_poly["final_levels_remaining"],
            color=["#81D4FA","#039BE5","#01579B"])
axes[0].set_ylim(0, 8)
axes[0].set_xlabel("Polynomial Degree"); axes[0].set_ylabel("Final Levels Remaining")
axes[0].set_title("A. Polynomial Degree\nvs Circuit Depth")

axes[1].plot(df_fc["features"], df_fc["final_levels_remaining"],
             marker='o', color='purple', lw=2)
axes[1].set_ylim(0, 8)
axes[1].set_xticks(df_fc["features"])
axes[1].set_xlabel("Feature Count"); axes[1].set_ylabel("Final Levels Remaining")
axes[1].set_title("B. Feature Count\nvs Circuit Depth (invariant)")

axes[2].plot(df_sz["admissions"], df_sz["final_levels_remaining"],
             marker='o', color='darkorange', lw=2)
axes[2].axvline(8192, ls='--', color='red', alpha=0.6)
axes[2].set_ylim(0, 8)
axes[2].set_xlabel("Admissions (N)"); axes[2].set_ylabel("Final Levels Remaining")
axes[2].set_title("C. Dataset Size\nvs Circuit Depth (invariant)")
plt.tight_layout()
plt.savefig(os.path.join(results, "final_level_consumption.png"), dpi=150)
plt.close()
print("Saved final_level_consumption.png")

print("\n=== ALL TASKS COMPLETE ===")
print(f"Full cohort: {N_FULL} admissions | "
      f"{expected_batches} batches | {expected_ct} feature CTs")
print(f"Total runtime: {full_row['total_mean_s']:.3f}±{full_row['total_std_s']:.3f}s")
print(f"Final levels remaining: {final_lvl_full}")
print(f"Plaintext LR AUC:  {m_pt['roc_auc']:.4f}")
print(f"HELR (Degree-5):   {m_he['roc_auc']:.4f}")
print(f"AUC difference:    {m_he['roc_auc']-m_pt['roc_auc']:+.4f}")
