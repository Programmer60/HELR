"""
Feature Selection + CKKS Preparation
Phases 1, 2, 3 — using training data only for all selection decisions.
"""
import pandas as pd
import numpy as np
import os, joblib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ---- Paths ----------------------------------------------------------
csv_path    = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\preprocessing\mimic3_aggregated_features.csv"
results_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
he_dir      = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready"
os.makedirs(he_dir, exist_ok=True)

FEATURES = [
    "AGE", "GENDER",
    "Heart Rate", "Systolic BP", "Diastolic BP", "Mean Arterial Pressure",
    "Respiratory Rate", "Temperature", "Oxygen Saturation",
    "Glucose", "Creatinine", "Sodium", "Potassium",
    "Hemoglobin", "WBC", "Platelets"
]
TARGET = "is_target"
SEED   = 42

BASELINE = {
    "accuracy": 0.6850, "precision": 0.5535,
    "recall":   0.3194, "f1": 0.4050, "roc_auc": 0.7262
}

# =====================================================================
# Reproducible split (identical to baseline script)
# =====================================================================
print("Loading dataset and reproducing split...")
df  = pd.read_csv(csv_path, low_memory=False)
rng = np.random.default_rng(SEED)
unique_subjects = df["SUBJECT_ID"].unique()
rng.shuffle(unique_subjects)
n_train = int(len(unique_subjects) * 0.80)
train_subjects = set(unique_subjects[:n_train])
test_subjects  = set(unique_subjects[n_train:])

train_df = df[df["SUBJECT_ID"].isin(train_subjects)].copy()
test_df  = df[df["SUBJECT_ID"].isin(test_subjects)].copy()

X_train_raw = train_df[FEATURES].copy()
y_train     = train_df[TARGET].values
X_test_raw  = test_df[FEATURES].copy()
y_test      = test_df[TARGET].values

assert len(train_subjects & test_subjects) == 0, "Subject overlap!"
print(f"  Train: {len(train_df)} admissions  |  Test: {len(test_df)} admissions  |  Overlap: 0")

# =====================================================================
# Helper: fit imputer+scaler on train, transform both, return arrays
# =====================================================================
def preprocess(X_tr, X_te, feature_list):
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_tr[feature_list]))
    Xte = sc.transform(imp.transform(X_te[feature_list]))
    return Xtr, Xte, imp, sc

# Helper: train LR and score
def eval_lr(Xtr, Xte, ytr, yte):
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0,
                            random_state=SEED)
    lr.fit(Xtr, ytr)
    yp   = lr.predict(Xte)
    yprb = lr.predict_proba(Xte)[:, 1]
    return {
        "accuracy" : accuracy_score(yte, yp),
        "precision": precision_score(yte, yp),
        "recall"   : recall_score(yte, yp),
        "f1"       : f1_score(yte, yp),
        "roc_auc"  : roc_auc_score(yte, yprb)
    }, lr

# =====================================================================
# PHASE 1A — Random Forest feature importance (training only)
# =====================================================================
print("\n=== PHASE 1A: Random Forest Feature Importance ===")
imp16  = SimpleImputer(strategy="median")
X_tr16 = imp16.fit_transform(X_train_raw)   # impute on training only

rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                             n_jobs=-1, random_state=SEED)
rf.fit(X_tr16, y_train)

rf_imp = pd.DataFrame({
    "feature":    FEATURES,
    "rf_importance": rf.feature_importances_
}).sort_values("rf_importance", ascending=False).reset_index(drop=True)
rf_imp["rf_rank"] = range(1, len(rf_imp) + 1)
print(rf_imp[["rf_rank","feature","rf_importance"]].to_string(index=False))
rf_imp.to_csv(os.path.join(results_dir, "rf_feature_importance.csv"), index=False)

# =====================================================================
# PHASE 1B — LR absolute coefficient (baseline 16-feature model)
# =====================================================================
print("\n=== PHASE 1B: LR Coefficient Magnitude ===")
Xtr16_sc, Xte16_sc, _, _ = preprocess(X_train_raw, X_test_raw, FEATURES)
lr16 = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=SEED)
lr16.fit(Xtr16_sc, y_train)

lr_imp = pd.DataFrame({
    "feature":  FEATURES,
    "lr_coef":  lr16.coef_[0],
    "lr_abs":   np.abs(lr16.coef_[0])
}).sort_values("lr_abs", ascending=False).reset_index(drop=True)
lr_imp["lr_rank"] = range(1, len(lr_imp) + 1)
print(lr_imp[["lr_rank","feature","lr_coef","lr_abs"]].to_string(index=False))
lr_imp.to_csv(os.path.join(results_dir, "lr_feature_importance.csv"), index=False)

# =====================================================================
# PHASE 1 PLOTS
# =====================================================================
# RF importance bar chart
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(rf_imp["feature"][::-1], rf_imp["rf_importance"][::-1], color="steelblue")
ax.set_xlabel("RF Mean Decrease in Impurity")
ax.set_title("Random Forest Feature Importance (Training Set Only)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "rf_feature_importance.png"), dpi=150)
plt.close()

# LR coefficient bar chart
sorted_lr = lr_imp.sort_values("lr_coef")
colors = ["tomato" if c < 0 else "steelblue" for c in sorted_lr["lr_coef"]]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(sorted_lr["feature"], sorted_lr["lr_coef"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("LR Coefficient (standardized features)")
ax.set_title("LR Coefficient Magnitude (Training Set Only)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "lr_feature_coef.png"), dpi=150)
plt.close()

# Side-by-side rank comparison
merged = rf_imp[["feature","rf_rank"]].merge(lr_imp[["feature","lr_rank"]], on="feature")
merged = merged.sort_values("rf_rank")
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(merged))
w = 0.35
ax.bar(x - w/2, merged["rf_rank"], w, label="RF Rank",  color="steelblue")
ax.bar(x + w/2, merged["lr_rank"], w, label="LR Rank",  color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(merged["feature"], rotation=45, ha="right")
ax.set_ylabel("Rank (1 = most important)")
ax.set_title("Feature Rank Comparison: RF vs LR (lower = more important)")
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "rank_comparison.png"), dpi=150)
plt.close()
print("\nPlots saved: rf_feature_importance.png, lr_feature_coef.png, rank_comparison.png")

# =====================================================================
# PHASE 1C — Evaluate top-10, top-15, top-16 for each method
# =====================================================================
print("\n=== PHASE 1C: Performance vs Feature Count ===")

rows = []
# Baseline
rows.append({"n_features":16, "selected_by":"baseline (all)",
             "features": FEATURES, **BASELINE,
             "auc_delta": 0.0, "acc_delta": 0.0, "features_removed": 0})

for method_label, ranked_df, rank_col in [
        ("Random Forest", rf_imp,  "rf_rank"),
        ("LR Coefficient", lr_imp, "lr_rank")]:

    for k in [10, 15]:
        top_k = ranked_df.sort_values(rank_col)["feature"].head(k).tolist()
        Xtr_k, Xte_k, _, _ = preprocess(X_train_raw, X_test_raw, top_k)
        metrics_k, _ = eval_lr(Xtr_k, Xte_k, y_train, y_test)
        rows.append({
            "n_features": k,
            "selected_by": method_label,
            "features": top_k,
            "auc_delta": round(metrics_k["roc_auc"] - BASELINE["roc_auc"], 5),
            "acc_delta": round(metrics_k["accuracy"] - BASELINE["accuracy"], 5),
            "features_removed": 16 - k,
            **metrics_k
        })
        print(f"  {method_label} top-{k}: AUC={metrics_k['roc_auc']:.4f}  "
              f"Acc={metrics_k['accuracy']:.4f}  F1={metrics_k['f1']:.4f}  "
              f"delta_AUC={metrics_k['roc_auc']-BASELINE['roc_auc']:+.4f}")

# Build comparison dataframe
comp = pd.DataFrame(rows)
display_cols = ["n_features","selected_by","accuracy","precision","recall","f1","roc_auc",
                "auc_delta","acc_delta","features_removed"]
print("\n--- COMPARISON TABLE ---")
print(comp[display_cols].to_string(index=False))
comp[display_cols].to_csv(os.path.join(results_dir, "feature_selection_comparison.csv"), index=False)

# Comparison AUC line plot
fig, ax = plt.subplots(figsize=(8, 5))
for method in ["baseline (all)", "Random Forest", "LR Coefficient"]:
    sub = comp[comp["selected_by"] == method].sort_values("n_features")
    ax.plot(sub["n_features"], sub["roc_auc"], marker="o", label=method)
ax.set_xlabel("Number of Features")
ax.set_ylabel("ROC-AUC (Test Set)")
ax.set_title("ROC-AUC vs Feature Count (Patient-Level Split)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "auc_vs_features.png"), dpi=150)
plt.close()
print("Saved: auc_vs_features.png")

# =====================================================================
# PHASE 2 — Feature-set decision
# =====================================================================
# Pick the top-10 RF set if AUC drop < 0.010, else top-15
rf_top10_row = comp[(comp["selected_by"]=="Random Forest") & (comp["n_features"]==10)].iloc[0]
rf_top15_row = comp[(comp["selected_by"]=="Random Forest") & (comp["n_features"]==15)].iloc[0]

print("\n=== PHASE 2: Feature Set Decision ===")
if abs(rf_top10_row["auc_delta"]) <= 0.010:
    chosen_method = "Random Forest"
    chosen_k      = 10
    chosen_row    = rf_top10_row
    print("  Decision: Top-10 (RF) retains AUC within 0.010 of baseline -> CHOSEN")
else:
    chosen_method = "Random Forest"
    chosen_k      = 15
    chosen_row    = rf_top15_row
    print("  Decision: Top-10 AUC drop > 0.010 -> using Top-15 (RF) instead")

print(f"  AUC delta : {chosen_row['auc_delta']:+.5f}")
print(f"  Acc delta : {chosen_row['acc_delta']:+.5f}")

ranked_chosen = rf_imp.sort_values("rf_rank")["feature"].head(chosen_k).tolist()
print(f"  Chosen features ({chosen_k}): {ranked_chosen}")

with open(os.path.join(results_dir, "selected_features.json"), "w") as f:
    json.dump({"method": chosen_method, "n_features": chosen_k,
               "features": ranked_chosen}, f, indent=4)

# =====================================================================
# PHASE 3 — CKKS Preparation using CHOSEN feature set
# =====================================================================
print(f"\n=== PHASE 3: CKKS Preparation ({chosen_k} features) ===")

Xtr_he, Xte_he, imp_he, sc_he = preprocess(X_train_raw, X_test_raw, ranked_chosen)
metrics_he, lr_he = eval_lr(Xtr_he, Xte_he, y_train, y_test)
print(f"  Retrained LR ({chosen_k} feats) -> AUC={metrics_he['roc_auc']:.4f}  Acc={metrics_he['accuracy']:.4f}")

# 1. Verify no NaN/inf
assert not np.isnan(Xtr_he).any(), "NaN in Xtr_he!"
assert not np.isinf(Xtr_he).any(), "Inf in Xtr_he!"
assert not np.isnan(Xte_he).any(), "NaN in Xte_he!"
assert not np.isinf(Xte_he).any(), "Inf in Xte_he!"
print("  NaN/Inf check: PASSED")

# 2. Feature stats after scaling
print("\n  Scaled Feature Statistics (Training Set):")
print(f"  {'Feature':<30s} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
for i, feat in enumerate(ranked_chosen):
    col = Xtr_he[:, i]
    print(f"  {feat:<30s} {col.min():8.4f} {col.max():8.4f} {col.mean():8.4f} {col.std():8.4f}")

# 3. Logit distribution on TEST set
logits = Xte_he @ lr_he.coef_[0] + lr_he.intercept_[0]
pctiles = np.percentile(logits, [1, 5, 50, 95, 99])
print("\n  Logit Distribution on Test Set:")
print(f"    Min    : {logits.min():.4f}")
print(f"    Max    : {logits.max():.4f}")
print(f"    Mean   : {logits.mean():.4f}")
print(f"    Std    : {logits.std():.4f}")
print(f"    P1     : {pctiles[0]:.4f}")
print(f"    P5     : {pctiles[1]:.4f}")
print(f"    P50    : {pctiles[2]:.4f}")
print(f"    P95    : {pctiles[3]:.4f}")
print(f"    P99    : {pctiles[4]:.4f}")

# Logit histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(logits, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(logits.mean(), color="red", lw=1.5, label=f"Mean={logits.mean():.2f}")
ax.set_xlabel("Logit (raw decision score)")
ax.set_ylabel("Count")
ax.set_title(f"Test Logit Distribution ({chosen_k} features, before sigmoid)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "logit_distribution.png"), dpi=150)
plt.close()
print("  Saved: logit_distribution.png")

# =====================================================================
# Save all HE-ready artefacts
# =====================================================================
np.save(os.path.join(he_dir, "X_train.npy"), Xtr_he)
np.save(os.path.join(he_dir, "X_test.npy"),  Xte_he)
np.save(os.path.join(he_dir, "y_train.npy"), y_train)
np.save(os.path.join(he_dir, "y_test.npy"),  y_test)
np.save(os.path.join(he_dir, "weights.npy"), lr_he.coef_[0])
np.save(os.path.join(he_dir, "intercept.npy"), lr_he.intercept_)
np.save(os.path.join(he_dir, "logits_test.npy"), logits)

y_proba_he = lr_he.predict_proba(Xte_he)[:, 1]
np.save(os.path.join(he_dir, "proba_test.npy"), y_proba_he)

joblib.dump(imp_he, os.path.join(he_dir, "imputer_he.joblib"))
joblib.dump(sc_he,  os.path.join(he_dir, "scaler_he.joblib"))
joblib.dump(lr_he,  os.path.join(he_dir, "logreg_he.joblib"))

with open(os.path.join(he_dir, "he_feature_names.json"), "w") as f:
    json.dump(ranked_chosen, f, indent=4)

he_meta = {
    "n_features":    chosen_k,
    "n_train":       len(Xtr_he),
    "n_test":        len(Xte_he),
    "weights_shape": list(lr_he.coef_.shape),
    "accuracy":      metrics_he["accuracy"],
    "roc_auc":       metrics_he["roc_auc"],
    "logit_min":     float(logits.min()),
    "logit_max":     float(logits.max()),
    "logit_mean":    float(logits.mean()),
    "logit_std":     float(logits.std()),
    "logit_p1":      float(pctiles[0]),
    "logit_p5":      float(pctiles[1]),
    "logit_p50":     float(pctiles[2]),
    "logit_p95":     float(pctiles[3]),
    "logit_p99":     float(pctiles[4]),
}
with open(os.path.join(he_dir, "he_meta.json"), "w") as f:
    json.dump(he_meta, f, indent=4)

print("\n  All HE-ready artefacts saved to results/he_ready/")
print("  X_train.npy  X_test.npy  y_train.npy  y_test.npy")
print("  weights.npy  intercept.npy  logits_test.npy  proba_test.npy")
print("  imputer_he.joblib  scaler_he.joblib  logreg_he.joblib")
print("  he_feature_names.json  he_meta.json")
print("\nALL PHASES COMPLETE.")
