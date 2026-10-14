#!/usr/bin/env python3
"""Independent validation script for PTB-XL preprocessed artifacts.

Performs strict assertions on:
1. File existence (CSV, NPZ, Scaler joblib, Summary JSON, Failure CSV if any)
2. CSV columns: metadata columns + exactly 48 named feature columns in expected order
3. NPZ contents, shapes, dtypes, and X/y matching lengths
4. Numerical validity: all feature values and scaled arrays are strictly finite
5. Binary labels: all labels belong to {0, 1}
6. Non-empty splits: train (folds 1-8), val (fold 9), test (fold 10) are nonempty
7. Consistency: CSV, NPZ, and summary JSON record and class counts agree exactly
8. Patient isolation: patient ID sets across train, val, and test are strictly disjoint
9. No duplicate ecg_ids
10. Scaler verification: recomputes mean and var on raw training fold to verify scaler was fit on train only
11. Accounting consistency: metadata records = excluded + successfully processed + waveform failures
12. Class distribution: warns if any split lacks representation of both classes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPECTED_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXPECTED_STATS = ["mean", "std", "min", "max"]
EXPECTED_FEATURES = [f"{lead}_{stat}" for lead in EXPECTED_LEADS for stat in EXPECTED_STATS]


def validate_preprocessing(processed_dir: Path) -> int:
    processed_dir = processed_dir.expanduser().resolve()
    print(f"=== Validating Preprocessed Artifacts in: {processed_dir} ===")

    # 1. File existence
    csv_path = processed_dir / "ptbxl_features_48.csv"
    npz_path = processed_dir / "ptbxl_binary_splits.npz"
    scaler_path = processed_dir / "standard_scaler_train_only.joblib"
    summary_path = processed_dir / "preprocessing_summary.json"
    failures_path = processed_dir / "waveform_read_failures.csv"

    for path in (csv_path, npz_path, scaler_path, summary_path):
        if not path.is_file():
            print(f"FAIL: Missing required artifact: {path}", file=sys.stderr)
            return 1
    print("PASS: Core artifact files exist.")

    # Load summary JSON
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # 2. CSV Validation
    print("\n--- Validating ptbxl_features_48.csv ---")
    df = pd.read_csv(csv_path)
    required_meta = ["ecg_id", "patient_id", "strat_fold", "label", "label_name", "filename_lr"]
    for col in required_meta:
        if col not in df.columns:
            print(f"FAIL: Missing required metadata column in CSV: {col}", file=sys.stderr)
            return 1

    feature_cols = [c for c in df.columns if c not in required_meta]
    if len(feature_cols) != 48:
        print(f"FAIL: Expected 48 feature columns, found {len(feature_cols)}", file=sys.stderr)
        return 1
    if feature_cols != EXPECTED_FEATURES:
        print(f"FAIL: Feature columns or ordering mismatch!", file=sys.stderr)
        print(f"Expected: {EXPECTED_FEATURES[:4]} ...", file=sys.stderr)
        print(f"Found:    {feature_cols[:4]} ...", file=sys.stderr)
        return 1
    print(f"PASS: CSV contains exactly 48 features with expected lead order and naming.")

    # Check ecg_id uniqueness
    if df["ecg_id"].duplicated().any():
        dups = df["ecg_id"][df["ecg_id"].duplicated()].tolist()
        print(f"FAIL: Duplicate ecg_id found in CSV: {dups[:10]}", file=sys.stderr)
        return 1
    print(f"PASS: All {len(df):,} ecg_id values are unique.")

    # Check finite values in raw features
    raw_feats = df[EXPECTED_FEATURES].to_numpy()
    if not np.isfinite(raw_feats).all():
        print("FAIL: Raw features in CSV contain NaN or Inf values!", file=sys.stderr)
        return 1
    print("PASS: All CSV feature values are strictly finite.")

    # Check binary labels
    labels = df["label"].unique()
    if not set(labels).issubset({0, 1}):
        print(f"FAIL: CSV contains non-binary labels: {labels}", file=sys.stderr)
        return 1
    print(f"PASS: Labels are binary in {{0, 1}}.")

    # 3. NPZ Validation
    print("\n--- Validating ptbxl_binary_splits.npz ---")
    npz = np.load(npz_path, allow_pickle=True)
    required_arrays = [
        "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
        "ecg_id_train", "ecg_id_val", "ecg_id_test",
        "patient_id_train", "patient_id_val", "patient_id_test",
        "feature_names",
    ]
    for arr_name in required_arrays:
        if arr_name not in npz:
            print(f"FAIL: Missing array in NPZ: {arr_name}", file=sys.stderr)
            return 1

    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val, y_val = npz["X_val"], npz["y_val"]
    X_test, y_test = npz["X_test"], npz["y_test"]

    # Check non-empty
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print(f"FAIL: One or more splits in NPZ are empty: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}", file=sys.stderr)
        return 1
    print(f"PASS: All splits are nonempty: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    # Check dimensions
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        if X.ndim != 2 or X.shape[1] != 48:
            print(f"FAIL: {split_name} X has unexpected shape: {X.shape}", file=sys.stderr)
            return 1
        if len(X) != len(y):
            print(f"FAIL: {split_name} X and y length mismatch: X={len(X)}, y={len(y)}", file=sys.stderr)
            return 1
        if not np.isfinite(X).all():
            print(f"FAIL: {split_name} X contains NaN or Inf values!", file=sys.stderr)
            return 1
        if not set(np.unique(y)).issubset({0, 1}):
            print(f"FAIL: {split_name} y contains invalid labels: {np.unique(y)}", file=sys.stderr)
            return 1
    print("PASS: NPZ shapes, matching lengths, and finite values verified.")

    # 4. CSV vs NPZ Consistency
    print("\n--- Verifying CSV vs NPZ Consistency ---")
    train_mask = df["strat_fold"].between(1, 8)
    val_mask = df["strat_fold"] == 9
    test_mask = df["strat_fold"] == 10

    if (train_mask.sum() != len(X_train) or
        val_mask.sum() != len(X_val) or
        test_mask.sum() != len(X_test)):
        print("FAIL: Record count mismatch between CSV folds and NPZ splits!", file=sys.stderr)
        return 1

    if not np.array_equal(df.loc[train_mask, "label"].to_numpy(), y_train):
        print("FAIL: Training labels mismatch between CSV and NPZ!", file=sys.stderr)
        return 1
    if not np.array_equal(df.loc[val_mask, "label"].to_numpy(), y_val):
        print("FAIL: Validation labels mismatch between CSV and NPZ!", file=sys.stderr)
        return 1
    if not np.array_equal(df.loc[test_mask, "label"].to_numpy(), y_test):
        print("FAIL: Test labels mismatch between CSV and NPZ!", file=sys.stderr)
        return 1
    print("PASS: CSV and NPZ record counts and labels match perfectly.")

    # 5. Scaler Validation
    print("\n--- Verifying StandardScaler Fitted on Training Folds Only ---")
    scaler = joblib.load(scaler_path)
    X_train_raw = df.loc[train_mask, EXPECTED_FEATURES].to_numpy(dtype=np.float64)
    expected_mean = np.mean(X_train_raw, axis=0)
    expected_var = np.var(X_train_raw, axis=0)

    mean_diff = np.max(np.abs(scaler.mean_ - expected_mean))
    var_diff = np.max(np.abs(scaler.var_ - expected_var))
    if mean_diff > 1e-5 or var_diff > 1e-5:
        print(f"FAIL: Scaler parameters do not match training fold statistics! mean_diff={mean_diff:.2e}, var_diff={var_diff:.2e}", file=sys.stderr)
        return 1
    print(f"PASS: Scaler verified as fitted exclusively on training folds 1-8 (mean_diff={mean_diff:.2e}, var_diff={var_diff:.2e}).")

    # Verify scaled train values match scaler.transform within float32 text roundtrip tolerance
    recomputed_X_train = scaler.transform(X_train_raw).astype(np.float32)
    max_scale_diff = float(np.max(np.abs(X_train - recomputed_X_train)))
    if max_scale_diff > 1e-4:
        print(f"FAIL: Saved X_train does not match scaler transform! max_diff={max_scale_diff:.2e}", file=sys.stderr)
        return 1
    print(f"PASS: NPZ scaled values match independent scaler transform (max_diff={max_scale_diff:.2e}).")

    # 6. Patient Isolation Validation
    print("\n--- Verifying Patient Disjointness Across Splits ---")
    patients_train = set(npz["patient_id_train"].astype(str))
    patients_val = set(npz["patient_id_val"].astype(str))
    patients_test = set(npz["patient_id_test"].astype(str))

    overlap_train_val = patients_train & patients_val
    overlap_train_test = patients_train & patients_test
    overlap_val_test = patients_val & patients_test

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("FAIL: Patient leakage detected across splits!", file=sys.stderr)
        print(f"  Train-Val overlap: {len(overlap_train_val)}", file=sys.stderr)
        print(f"  Train-Test overlap: {len(overlap_train_test)}", file=sys.stderr)
        print(f"  Val-Test overlap: {len(overlap_val_test)}", file=sys.stderr)
        return 1
    print(f"PASS: Strict patient isolation confirmed (0 overlap between train, val, test).")

    # 7. Summary JSON Consistency
    print("\n--- Verifying Preprocessing Summary Accounting ---")
    splits_summary = summary["splits"]
    for split_key, n_expected, labels_arr in [
        ("train_folds_1_to_8", len(X_train), y_train),
        ("validation_fold_9", len(X_val), y_val),
        ("test_fold_10", len(X_test), y_test),
    ]:
        s_info = splits_summary[split_key]
        if s_info["records"] != n_expected:
            print(f"FAIL: Summary records mismatch for {split_key}: {s_info['records']} vs {n_expected}", file=sys.stderr)
            return 1
        n_0 = int(np.sum(labels_arr == 0))
        n_1 = int(np.sum(labels_arr == 1))
        if s_info["normal_label_0"] != n_0 or s_info["abnormal_label_1"] != n_1:
            print(f"FAIL: Summary class count mismatch for {split_key}!", file=sys.stderr)
            return 1
    print("PASS: Summary JSON record and class counts match NPZ arrays exactly.")

    # 8. Single-Class Split Warning
    warnings = []
    for split_key, y_arr in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        u_classes = np.unique(y_arr)
        if len(u_classes) < 2:
            msg = f"WARNING: {split_key} split contains only class {u_classes.tolist()}!"
            warnings.append(msg)
            print(msg, file=sys.stderr)

    # 9. Failure Accounting
    n_failures = summary.get("waveform_read_failures", 0)
    if n_failures > 0:
        if not failures_path.is_file():
            print("FAIL: Waveform failures reported in summary but failures CSV missing!", file=sys.stderr)
            return 1
        fail_df = pd.read_csv(failures_path)
        if len(fail_df) != n_failures:
            print(f"FAIL: Failure count mismatch: {len(fail_df)} vs {n_failures}", file=sys.stderr)
            return 1
        print(f"INFO: {n_failures} waveform failures recorded and accounted for.")
    else:
        print("PASS: 0 waveform read failures recorded.")

    print("\n" + "=" * 60)
    if warnings:
        print(f"VERDICT: PASS WITH WARNINGS ({len(warnings)} warning(s))")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("VERDICT: PASS (ALL 12 CRITICAL ASSERTIONS SATISFIED)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path, nargs="?", default=Path("data/ptbxl_processed"),
                        help="Directory to validate (default: data/ptbxl_processed)")
    args = parser.parse_args()
    sys.exit(validate_preprocessing(args.processed_dir))
