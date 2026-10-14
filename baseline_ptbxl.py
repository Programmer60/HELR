#!/usr/bin/env python3
"""Train/evaluate a plaintext Logistic Regression baseline on preprocessed PTB-XL."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def metrics(model, X, y):
    pred = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_abnormal": float(precision_score(y, pred, zero_division=0)),
        "recall_abnormal": float(recall_score(y, pred, zero_division=0)),
        "f1_abnormal": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix_labels_0_normal_1_abnormal": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path, help="Directory containing ptbxl_binary_splits.npz")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON/model path prefix (default: processed_dir/baseline)")
    args = parser.parse_args()
    data_path = args.processed_dir / "ptbxl_binary_splits.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run preprocess_ptbxl.py on the full dataset first.")
    d = np.load(data_path, allow_pickle=False)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training split does not contain both classes. Do not use a small smoke-test subset for model evaluation.")

    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(X_train, y_train)
    result = {
        "task": "binary normal-vs-abnormal ECG classification",
        "positive_class": "abnormal (label 1)",
        "feature_count": int(X_train.shape[1]),
        "model": "sklearn LogisticRegression(max_iter=2000, solver='lbfgs', class_weight=None)",
        "threshold": 0.5,
        "metrics": {
            "train": metrics(model, X_train, y_train),
            "validation": metrics(model, X_val, y_val),
            "test": metrics(model, X_test, y_test),
        },
    }
    out_prefix = args.output or (args.processed_dir / "baseline")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(out_prefix) + "_logistic_regression.joblib")
    with open(str(out_prefix) + "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved model:   {str(out_prefix)}_logistic_regression.joblib")
    print(f"Saved metrics: {str(out_prefix)}_metrics.json")


if __name__ == "__main__":
    main()
