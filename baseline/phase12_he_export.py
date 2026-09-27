import pandas as pd
import numpy as np
import os, joblib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

csv_path = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\preprocessing\mimic3_aggregated_features.csv"
results_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
he_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready"
os.makedirs(he_dir, exist_ok=True)

# Final 10 Features (LR Top 10)
FEATURES = [
    "AGE", "Systolic BP", "Mean Arterial Pressure", "Diastolic BP", 
    "GENDER", "Potassium", "Heart Rate", "Hemoglobin", "Glucose", "Oxygen Saturation"
]
TARGET = "is_target"
SEED = 42

print("Loading dataset and reproducing split...")
df = pd.read_csv(csv_path, low_memory=False)
rng = np.random.default_rng(SEED)
unique_subjects = df["SUBJECT_ID"].unique()
rng.shuffle(unique_subjects)
n_train = int(len(unique_subjects) * 0.80)
train_subjects = set(unique_subjects[:n_train])
test_subjects = set(unique_subjects[n_train:])

train_df = df[df["SUBJECT_ID"].isin(train_subjects)].copy()
test_df = df[df["SUBJECT_ID"].isin(test_subjects)].copy()

X_train_raw = train_df[FEATURES].copy()
y_train = train_df[TARGET].values
X_test_raw = test_df[FEATURES].copy()
y_test = test_df[TARGET].values

print("Fitting imputer and scaler (training set only)...")
imp = SimpleImputer(strategy="median")
Xtr = imp.fit_transform(X_train_raw)
Xte = imp.transform(X_test_raw)

sc = StandardScaler()
Xtr_sc = sc.fit_transform(Xtr)
Xte_sc = sc.transform(Xte)

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=SEED)
lr.fit(Xtr_sc, y_train)

y_pred = lr.predict(Xte_sc)
y_proba = lr.predict_proba(Xte_sc)[:, 1]
logits = Xte_sc @ lr.coef_[0] + lr.intercept_[0]

print("\n--- Model Verification ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
assert not np.isnan(Xtr_sc).any(), "NaN in X_train"
assert not np.isnan(Xte_sc).any(), "NaN in X_test"
assert not np.isinf(Xtr_sc).any(), "Inf in X_train"
assert not np.isinf(Xte_sc).any(), "Inf in X_test"
print("Shape X_train:", Xtr_sc.shape)
print("Shape X_test:", Xte_sc.shape)
print("Shape weights:", lr.coef_[0].shape)

print("\n--- Saving HE Export ---")
np.save(os.path.join(he_dir, "X_train.npy"), Xtr_sc)
np.save(os.path.join(he_dir, "X_test.npy"), Xte_sc)
np.save(os.path.join(he_dir, "y_train.npy"), y_train)
np.save(os.path.join(he_dir, "y_test.npy"), y_test)
np.save(os.path.join(he_dir, "weights.npy"), lr.coef_[0])
np.save(os.path.join(he_dir, "intercept.npy"), lr.intercept_)
np.save(os.path.join(he_dir, "logits_test.npy"), logits)
np.save(os.path.join(he_dir, "proba_test.npy"), y_proba)

joblib.dump(imp, os.path.join(he_dir, "imputer_he.joblib"))
joblib.dump(sc, os.path.join(he_dir, "scaler_he.joblib"))
joblib.dump(lr, os.path.join(he_dir, "logreg_he.joblib"))

train_df[["SUBJECT_ID", "HADM_ID"]].to_csv(os.path.join(he_dir, "train_ids.csv"), index=False)
test_df[["SUBJECT_ID", "HADM_ID"]].to_csv(os.path.join(he_dir, "test_ids.csv"), index=False)

with open(os.path.join(he_dir, "feature_names.json"), "w") as f:
    json.dump(FEATURES, f, indent=4)

he_meta = {
    "n_features": len(FEATURES),
    "features": FEATURES,
    "n_train": len(Xtr_sc),
    "n_test": len(Xte_sc),
    "accuracy": accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba)
}
with open(os.path.join(he_dir, "metadata.json"), "w") as f:
    json.dump(he_meta, f, indent=4)

print("\n--- Logit Analysis ---")
pctiles = np.percentile(logits, [1, 5, 50, 95, 99])
print(f"Min:  {logits.min():.4f}")
print(f"Max:  {logits.max():.4f}")
print(f"Mean: {logits.mean():.4f}")
print(f"Std:  {logits.std():.4f}")
print(f"P1:   {pctiles[0]:.4f}")
print(f"P5:   {pctiles[1]:.4f}")
print(f"P50:  {pctiles[2]:.4f}")
print(f"P95:  {pctiles[3]:.4f}")
print(f"P99:  {pctiles[4]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(logits, bins=80, color="purple", edgecolor="white", alpha=0.8)
ax.axvline(logits.mean(), color="red", lw=2, label=f"Mean={logits.mean():.2f}")
ax.axvline(pctiles[0], color="black", ls="--", lw=1, label=f"P1={pctiles[0]:.2f}")
ax.axvline(pctiles[4], color="black", ls="--", lw=1, label=f"P99={pctiles[4]:.2f}")
ax.set_xlabel("Logit (raw decision score)")
ax.set_ylabel("Count")
ax.set_title("Test Logit Distribution (LR Top-10 features)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "logit_histogram_top10.png"), dpi=150)
plt.close()
print("Saved: logit_histogram_top10.png")
