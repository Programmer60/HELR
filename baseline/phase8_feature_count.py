"""
Feature-Count Experiment
========================
Fixed workload: 8,192 admissions (one SIMD batch for all configurations).
Fixed CKKS context: [60,40,40,40,40,40,40,60], scale=2^40, Degree-5 polynomial.
Three configurations:
  A. 10 features  — LR-Coefficient Top-10 (HE export)
  B. 15 features  — LR-Coefficient Top-15
  C. 16 features  — Full baseline

For predictive evaluation: the SAME original held-out patient-level
test split (9,943 records) is used for all three models.
The 8,192-admission workload is a separate HE runtime workload.
"""
import numpy as np
import tenseal as ts
import time, os, math, tracemalloc
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
csv_path   = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\preprocessing\mimic3_aggregated_features.csv"
results    = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
he_dir     = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready"
os.makedirs(results, exist_ok=True)

# --------------------------------------------------------------------------
# CKKS context  (identical to all prior experiments)
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
SLOTS = 8192

# --------------------------------------------------------------------------
# Degree-5 polynomial coefficients (MIMIC-fitted, [-6, 6])
# --------------------------------------------------------------------------
C0   =  0.5
C1_5 =  0.217101
C3_5 = -0.007823
C5_5 =  0.000118

def pt_sig5(z):
    return C0 + C1_5*z + C3_5*(z**3) + C5_5*(z**5)

def burn(v, n):
    for _ in range(n): v = v * 1.0
    return v

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

# --------------------------------------------------------------------------
# Feature-set definitions
# --------------------------------------------------------------------------
FEATURES_16 = [
    "AGE", "Systolic BP", "Mean Arterial Pressure", "Diastolic BP",
    "GENDER", "Potassium", "Heart Rate", "Hemoglobin", "Glucose",
    "Oxygen Saturation",
    "WBC", "Creatinine", "Respiratory Rate", "Sodium", "Temperature",
    "Platelets"
]
FEATURES_15 = FEATURES_16[:15]          # drop Platelets (LR-coef rank 16)
FEATURES_10 = FEATURES_16[:10]          # LR-Coef Top-10
TARGET = "is_target"
SEED   = 42
WORKLOAD_N = 8192   # exactly one SIMD batch

# --------------------------------------------------------------------------
# Reproduce the SAME patient-level 80/20 split
# --------------------------------------------------------------------------
print("Loading dataset and reproducing original patient-level split ...")
df = pd.read_csv(csv_path, low_memory=False)
rng = np.random.default_rng(SEED)
unique_subjects = df["SUBJECT_ID"].unique()
rng.shuffle(unique_subjects)
n_train = int(len(unique_subjects) * 0.80)
train_subjects = set(unique_subjects[:n_train])
test_subjects  = set(unique_subjects[n_train:])

train_df = df[df["SUBJECT_ID"].isin(train_subjects)].copy()
test_df  = df[df["SUBJECT_ID"].isin(test_subjects)].copy()
y_train_orig = train_df[TARGET].values
y_test_orig  = test_df[TARGET].values

print(f"Train: {len(train_df)} admissions | Test (held-out): {len(test_df)} admissions")
print(f"Overlap subjects: {len(train_subjects & test_subjects)}")

# --------------------------------------------------------------------------
# For each feature set: fit imputer+scaler+LR on training, evaluate on
# held-out test set, then run HE inference on the 8,192 workload.
# --------------------------------------------------------------------------
CONFIGS = [
    ("10-feature (LR Top-10)",  FEATURES_10),
    ("15-feature (LR Top-15)",  FEATURES_15),
    ("16-feature (full baseline)", FEATURES_16),
]

rows = []
REPS = 3

for config_name, feats in CONFIGS:
    n_feats = len(feats)
    print(f"\n{'='*60}")
    print(f"Config: {config_name}  ({n_feats} features)")
    print(f"{'='*60}")

    # --- Fit preprocessing on training data ---
    X_train_raw = train_df[feats].copy()
    X_test_raw  = test_df[feats].copy()

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train_raw)
    Xte = imp.transform(X_test_raw)

    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xte_sc = sc.transform(Xte)

    # --- Train LR on training data ---
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=SEED)
    lr.fit(Xtr_sc, y_train_orig)
    w = lr.coef_[0]
    b = float(lr.intercept_[0])

    # --- Predictive evaluation on ORIGINAL held-out test set ---
    y_prob_test = lr.predict_proba(Xte_sc)[:, 1]
    acc_ht  = accuracy_score(y_test_orig, (y_prob_test >= 0.5).astype(int))
    auc_ht  = roc_auc_score(y_test_orig, y_prob_test)
    print(f"Held-out test accuracy: {acc_ht:.4f}  ROC-AUC: {auc_ht:.4f}")

    # --- Build 8,192-admission workload (first 8,192 rows of the combined dataset) ---
    # Use all 49,303 combined rows (same ordering as scalability experiment)
    X_all = np.vstack([Xte_sc, Xtr_sc])   # test-then-train for consistent ordering
    X_workload = X_all[:WORKLOAD_N]        # exactly 8,192 rows → 1 SIMD batch
    assert len(X_workload) == WORKLOAD_N

    # Plaintext reference for the workload (for numerical error comparison)
    z_pt = X_workload @ w + b
    pt_preds = pt_sig5(z_pt)

    # --- HE inference (3 reps) ---
    enc_times, comp_times, dec_times, tot_times = [], [], [], []
    ct_serialized_mb_list = []
    final_lvl = None
    he_preds_final = None

    for rep in range(REPS):
        # Encryption
        tracemalloc.start()
        t0 = time.perf_counter()
        enc_cols = [ts.ckks_vector(ctx, X_workload[:, j].tolist()) for j in range(n_feats)]
        t_enc = time.perf_counter() - t0
        ct_bytes = sum(len(c.serialize()) for c in enc_cols)
        tracemalloc.stop()

        # HE Computation
        t0 = time.perf_counter()
        z = enc_cols[0] * w[0]
        for j in range(1, n_feats):
            z = z + (enc_cols[j] * w[j])
        z = z + b

        z2    = z * z
        z3    = z2 * burn(z, 1)
        z5    = z3 * burn(z2, 1)
        term5 = z5 * C5_5
        term3 = burn(z3, 1) * C3_5
        term1 = burn(z,  3) * C1_5
        p     = term5 + term3 + term1 + C0
        t_comp = time.perf_counter() - t0

        if rep == 0:
            final_lvl = get_levels(p)

        # Decryption
        t0 = time.perf_counter()
        he_preds = np.array(p.decrypt()[:WORKLOAD_N])
        t_dec = time.perf_counter() - t0

        enc_times.append(t_enc)
        comp_times.append(t_comp)
        dec_times.append(t_dec)
        tot_times.append(t_enc + t_comp + t_dec)
        ct_serialized_mb_list.append(ct_bytes / 1e6)

        if rep == 0:
            he_preds_final = he_preds

    err = np.abs(he_preds_final - pt_preds)

    row = {
        "config":                  config_name,
        "features":                n_feats,
        "workload_admissions":     WORKLOAD_N,
        "batches":                 1,
        "feature_ciphertexts":     n_feats * 1,
        "enc_mean_s":              np.mean(enc_times),
        "enc_std_s":               np.std(enc_times),
        "comp_mean_s":             np.mean(comp_times),
        "comp_std_s":              np.std(comp_times),
        "dec_mean_s":              np.mean(dec_times),
        "dec_std_s":               np.std(dec_times),
        "total_mean_s":            np.mean(tot_times),
        "total_std_s":             np.std(tot_times),
        "ct_serialized_MB":        np.mean(ct_serialized_mb_list),
        "final_levels_remaining":  final_lvl,
        "max_abs_error":           err.max(),
        "mean_abs_error":          err.mean(),
        # Predictive metrics on original held-out test set
        "held_out_accuracy":       acc_ht,
        "held_out_ROC_AUC":        auc_ht,
    }
    rows.append(row)
    print(f"  -> CTs={n_feats}, total={np.mean(tot_times):.3f}±{np.std(tot_times):.3f}s, "
          f"levels={final_lvl}, max_err={err.max():.2e}")

df_res = pd.DataFrame(rows)
df_res.to_csv(os.path.join(results, "feature_count_results.csv"), index=False)
print("\nSaved feature_count_results.csv")

# --------------------------------------------------------------------------
# Console output
# --------------------------------------------------------------------------
print("\n--- FEATURE-COUNT RESULTS ---")
print(df_res[[
    "features","feature_ciphertexts","enc_mean_s","comp_mean_s","dec_mean_s",
    "total_mean_s","ct_serialized_MB","final_levels_remaining",
    "max_abs_error","mean_abs_error",
    "held_out_accuracy","held_out_ROC_AUC"
]].to_string(index=False))

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
feat_counts = df_res["features"].values
colors = ["#2196F3","#FF9800","#4CAF50"]

# 1. Runtime vs Features
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(feat_counts, df_res["total_mean_s"], yerr=df_res["total_std_s"],
              color=colors, capsize=6, width=1.5)
ax.bar(feat_counts, df_res["enc_mean_s"],  color=colors, alpha=0.5, width=1.5, label="Encryption")
ax.bar(feat_counts, df_res["comp_mean_s"], bottom=df_res["enc_mean_s"],
       color=colors, alpha=0.8, width=1.5, label="HE Compute")
ax.set_xticks(feat_counts)
ax.set_xticklabels([f"{f} feats" for f in feat_counts])
ax.set_xlabel("Feature Count")
ax.set_ylabel("Time (s)")
ax.set_title("HE Inference Runtime vs Feature Count\n(8,192 admissions, 1 SIMD batch, Degree-5)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results, "runtime_vs_features.png"), dpi=150)
plt.close()
print("Saved runtime_vs_features.png")

# 2. Ciphertexts vs Features
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(feat_counts, df_res["feature_ciphertexts"], color=colors, width=1.5)
ax.set_xticks(feat_counts)
ax.set_xticklabels([f"{f} feats" for f in feat_counts])
ax.set_xlabel("Feature Count")
ax.set_ylabel("Feature Ciphertext Count")
ax.set_title("Feature Ciphertext Workload vs Feature Count\n(1 batch; CTs = features × batches)")
plt.tight_layout()
plt.savefig(os.path.join(results, "ciphertexts_vs_features.png"), dpi=150)
plt.close()
print("Saved ciphertexts_vs_features.png")

# 3. Final levels vs Features
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(feat_counts, df_res["final_levels_remaining"], marker='o', color="purple", lw=2)
ax.set_xticks(feat_counts)
ax.set_xticklabels([f"{f} feats" for f in feat_counts])
ax.set_ylim(0, 8)
ax.set_xlabel("Feature Count")
ax.set_ylabel("Final Modulus Levels Remaining")
ax.set_title("Circuit Depth (Final Levels) vs Feature Count\n"
             "(flat — feature count alone does not change multiplicative depth)")
ax.annotate("Flat line expected:\nFeature count affects workload, NOT depth",
            xy=(13, 2.1), fontsize=9, color="purple")
plt.tight_layout()
plt.savefig(os.path.join(results, "levels_vs_features.png"), dpi=150)
plt.close()
print("Saved levels_vs_features.png")

# 4. Accuracy vs Features (held-out)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(feat_counts, df_res["held_out_accuracy"], marker='o', color="darkblue", lw=2)
ax.set_xticks(feat_counts)
ax.set_xticklabels([f"{f} feats" for f in feat_counts])
ax.set_ylim(0.6, 0.75)
ax.set_xlabel("Feature Count")
ax.set_ylabel("Accuracy")
ax.set_title("Held-Out Test Accuracy vs Feature Count\n(same 9,943-record patient-level test set)")
plt.tight_layout()
plt.savefig(os.path.join(results, "accuracy_vs_features.png"), dpi=150)
plt.close()
print("Saved accuracy_vs_features.png")

# 5. ROC-AUC vs Features (held-out)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(feat_counts, df_res["held_out_ROC_AUC"], marker='o', color="darkorange", lw=2)
ax.set_xticks(feat_counts)
ax.set_xticklabels([f"{f} feats" for f in feat_counts])
ax.set_ylim(0.70, 0.74)
ax.set_xlabel("Feature Count")
ax.set_ylabel("ROC-AUC")
ax.set_title("Held-Out Test ROC-AUC vs Feature Count\n(same 9,943-record patient-level test set)")
plt.tight_layout()
plt.savefig(os.path.join(results, "auc_vs_features.png"), dpi=150)
plt.close()
print("Saved auc_vs_features.png")

print("\nAll artefacts saved.")
