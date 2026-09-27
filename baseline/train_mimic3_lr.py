"""
PHASE 2 — Plaintext Logistic Regression Baseline
Patient-level train/test split, training-set only imputation + scaling,
Logistic Regression evaluation, saved artefacts and plots.
"""
import pandas as pd
import numpy as np
import os, joblib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# ---- Paths ----------------------------------------------------------
csv_path = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\preprocessing\mimic3_aggregated_features.csv"
results_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
os.makedirs(results_dir, exist_ok=True)

# ---- Feature list (all extracted clinical variables) ----------------
FEATURES = [
    "AGE", "GENDER",
    "Heart Rate", "Systolic BP", "Diastolic BP", "Mean Arterial Pressure",
    "Respiratory Rate", "Temperature", "Oxygen Saturation",
    "Glucose", "Creatinine", "Sodium", "Potassium",
    "Hemoglobin", "WBC", "Platelets"
]
TARGET = "is_target"

# ---- Load dataset ---------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Shape: {df.shape}")

# ---- Patient-level train/test split (80/20) -------------------------
print("\nBuilding patient-level train/test split (80/20)...")
rng = np.random.default_rng(42)
unique_subjects = df["SUBJECT_ID"].unique()
rng.shuffle(unique_subjects)
n_train = int(len(unique_subjects) * 0.80)
train_subjects = set(unique_subjects[:n_train])
test_subjects  = set(unique_subjects[n_train:])

train_df = df[df["SUBJECT_ID"].isin(train_subjects)].copy()
test_df  = df[df["SUBJECT_ID"].isin(test_subjects)].copy()

print(f"  Train subjects  : {len(train_subjects)}")
print(f"  Test  subjects  : {len(test_subjects)}")
print(f"  Train admissions: {len(train_df)}")
print(f"  Test  admissions: {len(test_df)}")
print(f"  Train pos/neg   : {train_df[TARGET].sum()} / {(train_df[TARGET]==0).sum()}")
print(f"  Test  pos/neg   : {test_df[TARGET].sum()}  / {(test_df[TARGET]==0).sum()}")

# Subject overlap sanity check
overlap = train_subjects & test_subjects
print(f"  Subject overlap : {len(overlap)}  (must be 0)")

# ---- Split X/y ------------------------------------------------------
X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].values
X_test  = test_df[FEATURES].copy()
y_test  = test_df[TARGET].values

# ---- Imputation (training-set median only) --------------------------
print("\nFitting median imputer on training set only...")
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

# ---- Scaling (training-set only) ------------------------------------
print("Fitting StandardScaler on training set only...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc  = scaler.transform(X_test_imp)

# ---- Logistic Regression --------------------------------------------
print("Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    C=1.0,
    class_weight=None,
    random_state=42
)
lr.fit(X_train_sc, y_train)

# ---- Evaluation on test set -----------------------------------------
y_pred  = lr.predict(X_test_sc)
y_proba = lr.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

print("\n========== PLAINTEXT LOGISTIC REGRESSION RESULTS ==========")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print("=" * 60)

# ---- PLOTS ----------------------------------------------------------
# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — MIMIC-III IHD Classifier")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=150)
plt.close()
print("Saved: roc_curve.png")

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No IHD (0)", "IHD (1)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — MIMIC-III IHD Classifier")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# 3. Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (split_label, y_arr) in zip(axes, [("Train", y_train), ("Test", y_test)]):
    unique, counts = np.unique(y_arr, return_counts=True)
    ax.bar(["No IHD (0)", "IHD (1)"], counts, color=["steelblue", "tomato"])
    ax.set_title(f"Class Distribution — {split_label}")
    ax.set_ylabel("Count")
    for i, c in enumerate(counts):
        ax.text(i, c + 50, str(c), ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "class_distribution.png"), dpi=150)
plt.close()
print("Saved: class_distribution.png")

# 4. Coefficient magnitude plot (feature importance proxy)
coef_abs = np.abs(lr.coef_[0])
coef_df  = pd.DataFrame({"feature": FEATURES, "coef": lr.coef_[0], "abs_coef": coef_abs})
coef_df  = coef_df.sort_values("abs_coef", ascending=True)
fig, ax  = plt.subplots(figsize=(8, 6))
colors   = ["tomato" if c < 0 else "steelblue" for c in coef_df["coef"]]
ax.barh(coef_df["feature"], coef_df["coef"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("LR Coefficient (standardized features)")
ax.set_title("LR Coefficient Magnitude — MIMIC-III IHD")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "lr_coefficients.png"), dpi=150)
plt.close()
print("Saved: lr_coefficients.png")

# ---- Missingness plot (top 15 features) ----------------------------
miss_pct = (test_df[FEATURES].isna().mean() * 100).sort_values(ascending=True)
fig, ax  = plt.subplots(figsize=(8, 6))
ax.barh(miss_pct.index, miss_pct.values, color="steelblue")
ax.set_xlabel("% Missing (Test Set)")
ax.set_title("Feature Missingness — MIMIC-III (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "missingness.png"), dpi=150)
plt.close()
print("Saved: missingness.png")

# ---- Save artefacts --------------------------------------------------
joblib.dump(imputer, os.path.join(results_dir, "imputer.joblib"))
joblib.dump(scaler,  os.path.join(results_dir, "scaler.joblib"))
joblib.dump(lr,      os.path.join(results_dir, "logreg_model.joblib"))

with open(os.path.join(results_dir, "feature_names.json"), "w") as f:
    json.dump(FEATURES, f)

# Test predictions
pred_df = test_df[["SUBJECT_ID", "HADM_ID", TARGET]].copy()
pred_df["y_pred"]  = y_pred
pred_df["y_proba"] = y_proba
pred_df.to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)

# Train/test split IDs
train_df[["SUBJECT_ID","HADM_ID"]].to_csv(os.path.join(results_dir, "train_ids.csv"), index=False)
test_df[["SUBJECT_ID","HADM_ID"]].to_csv(os.path.join(results_dir, "test_ids.csv"), index=False)

# Coefficient table
coef_df_save = pd.DataFrame({
    "feature"     : FEATURES,
    "coefficient" : lr.coef_[0],
    "abs_coef"    : np.abs(lr.coef_[0])
}).sort_values("abs_coef", ascending=False)
coef_df_save.to_csv(os.path.join(results_dir, "lr_coefficients.csv"), index=False)

# Metrics summary
metrics = {
    "accuracy" : acc, "precision": prec,
    "recall"   : rec, "f1"       : f1,
    "roc_auc"  : auc,
    "train_subjects": len(train_subjects), "test_subjects": len(test_subjects),
    "train_admissions": len(train_df), "test_admissions": len(test_df),
    "train_pos": int(y_train.sum()), "train_neg": int((y_train==0).sum()),
    "test_pos" : int(y_test.sum()),  "test_neg" : int((y_test==0).sum())
}
with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\nAll artefacts saved to results/")
print("\nCoefficient Table (ranked by absolute value):")
print(coef_df_save.to_string(index=False))
print("\nPHASE 2 BASELINE COMPLETE.")
