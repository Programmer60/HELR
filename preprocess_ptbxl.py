#!/usr/bin/env python3
"""Preprocess PTB-XL v1.0.3 into a simple 48-feature binary ECG task.

Feature design: 12 leads x [mean, standard deviation, minimum, maximum].
Target policy:
  - label 0 (normal): mapped diagnostic classes are NORM only.
  - label 1 (abnormal): at least one mapped diagnostic class other than NORM.
  - exclude records with no mapped diagnostic class.
If a record has both NORM and a non-NORM diagnostic class, this script labels it
abnormal. This is an explicit project baseline policy, not the only possible
clinical label definition.

Uses official PTB-XL strat_fold split (patient-wise): folds 1-8 train, 9 validation,
10 test. StandardScaler is fitted on training features only.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import StandardScaler
import joblib


def locate_dataset_root(path: Path) -> Path:
    """Find the PTB-XL root, allowing one or two nested extraction folders."""
    path = path.expanduser().resolve()
    candidates = [path]
    if path.exists() and path.is_dir():
        candidates.extend(p for p in path.iterdir() if p.is_dir())
        for p in list(candidates[1:]):
            try:
                candidates.extend(q for q in p.iterdir() if q.is_dir())
            except OSError:
                pass
    for candidate in candidates:
        if ((candidate / "ptbxl_database.csv").is_file()
                and (candidate / "scp_statements.csv").is_file()
                and (candidate / "records100").is_dir()):
            return candidate
    raise FileNotFoundError(
        f"Could not locate PTB-XL root under {path}. Expected ptbxl_database.csv, "
        "scp_statements.csv, and records100/ in the same directory."
    )


def parse_scp_codes(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def map_binary_label(codes: dict[str, float], diagnostic_class_by_code: dict[str, str]) -> int | None:
    classes = {
        diagnostic_class_by_code[code]
        for code in codes
        if code in diagnostic_class_by_code and diagnostic_class_by_code[code]
    }
    if not classes:
        return None
    # Explicit binary baseline: any non-NORM diagnostic class -> abnormal.
    if any(class_name != "NORM" for class_name in classes):
        return 1
    if "NORM" in classes:
        return 0
    return None


def make_feature_names(lead_names: list[str]) -> list[str]:
    cleaned = [str(name).strip().replace(" ", "_").replace("/", "_") for name in lead_names]
    return [f"{lead}_{stat}" for lead in cleaned for stat in ("mean", "std", "min", "max")]


def extract_stats(signal: np.ndarray) -> np.ndarray:
    """Return 4 stats/channel from a (time_samples, channels) signal."""
    return np.column_stack([
        np.mean(signal, axis=0),
        np.std(signal, axis=0, ddof=0),
        np.min(signal, axis=0),
        np.max(signal, axis=0),
    ]).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path, help="PTB-XL root or parent extraction folder")
    parser.add_argument("--output", type=Path, default=Path("data/ptbxl_processed"),
                        help="Output directory (default: data/ptbxl_processed)")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optional small smoke test; use e.g. 100. Do not use for final results.")
    args = parser.parse_args()

    root = locate_dataset_root(args.dataset_path)
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = root / "ptbxl_database.csv"
    statements_path = root / "scp_statements.csv"
    metadata = pd.read_csv(metadata_path)
    statements = pd.read_csv(statements_path, index_col=0)

    required_meta = {"ecg_id", "patient_id", "strat_fold", "filename_lr", "scp_codes"}
    missing = required_meta - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required columns in ptbxl_database.csv: {sorted(missing)}")
    if "diagnostic" not in statements.columns or "diagnostic_class" not in statements.columns:
        raise ValueError("scp_statements.csv must contain diagnostic and diagnostic_class columns")

    diagnostic_statements = statements[statements["diagnostic"].fillna(0).astype(int) == 1]
    diagnostic_class_by_code = {}
    for code, row in diagnostic_statements.iterrows():
        value = row.get("diagnostic_class")
        if pd.notna(value) and str(value).strip():
            diagnostic_class_by_code[str(code)] = str(value).strip()

    metadata["_parsed_scp"] = metadata["scp_codes"].apply(parse_scp_codes)
    metadata["label"] = metadata["_parsed_scp"].apply(
        lambda codes: map_binary_label(codes, diagnostic_class_by_code)
    )
    n_unlabelled = int(metadata["label"].isna().sum())
    eligible = metadata[metadata["label"].notna()].copy()
    eligible["label"] = eligible["label"].astype(np.int64)
    if args.max_records is not None:
        if args.max_records < 10:
            raise ValueError("--max-records must be at least 10 so all official folds can be represented")
        # Deterministically select an approximately balanced subset across all official folds
        # without using a trailing global head() that could truncate or starve later folds.
        folds = sorted(eligible["strat_fold"].unique())
        n_folds = len(folds)
        base = args.max_records // n_folds
        rem = args.max_records % n_folds
        fold_subsets = []
        for i, fold in enumerate(folds):
            quota = base + (1 if i < rem else 0)
            if quota > 0:
                fold_df = eligible[eligible["strat_fold"] == fold].sort_values("ecg_id").head(quota)
                fold_subsets.append(fold_df)
        eligible = pd.concat(fold_subsets, axis=0).sort_values("ecg_id").reset_index(drop=True)

    feature_rows: list[np.ndarray] = []
    kept_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    lead_names_reference: list[str] | None = None

    total = len(eligible)
    for position, (_, row) in enumerate(eligible.iterrows(), start=1):
        rel_path = str(row["filename_lr"])
        record_path = root / rel_path
        try:
            signal, fields = wfdb.rdsamp(str(record_path))
            signal = np.asarray(signal, dtype=np.float64)
            lead_names = list(fields.get("sig_name", []))
            if signal.ndim != 2 or signal.shape[1] != 12:
                raise ValueError(f"Expected 12-lead signal; got shape {signal.shape}")
            if not np.isfinite(signal).all():
                raise ValueError("Signal contains NaN or infinite values")
            if lead_names_reference is None:
                lead_names_reference = lead_names if len(lead_names) == 12 else [f"lead{i+1:02d}" for i in range(12)]
            elif len(lead_names) == 12 and lead_names != lead_names_reference:
                raise ValueError(f"Lead ordering/names changed: {lead_names} vs {lead_names_reference}")

            features = extract_stats(signal)
            if features.size != 48 or not np.isfinite(features).all():
                raise ValueError("Feature extraction produced invalid values")

            feature_rows.append(features.astype(np.float32))
            kept_records.append({
                "ecg_id": int(row["ecg_id"]),
                "patient_id": row["patient_id"],
                "strat_fold": int(row["strat_fold"]),
                "label": int(row["label"]),
                "label_name": "abnormal" if int(row["label"]) == 1 else "normal",
                "filename_lr": rel_path,
            })
        except Exception as exc:  # record failures explicitly instead of silently losing files
            failures.append({"ecg_id": row.get("ecg_id"), "filename_lr": rel_path,
                             "error_type": type(exc).__name__, "error": str(exc)})
        if position % 1000 == 0 or position == total:
            print(f"Processed {position:,}/{total:,} eligible records; kept={len(kept_records):,}; failures={len(failures):,}")

    if not feature_rows:
        raise RuntimeError("No ECG records were successfully processed. Check the dataset path and WFDB files.")

    features_np = np.vstack(feature_rows)
    records_df = pd.DataFrame(kept_records)
    feature_names = make_feature_names(lead_names_reference or [f"lead{i+1:02d}" for i in range(12)])
    feature_df = pd.DataFrame(features_np, columns=feature_names)
    all_df = pd.concat([records_df.reset_index(drop=True), feature_df], axis=1)

    # Preserve the official PTB-XL folds: folds 1-8 train, 9 validation, 10 test.
    train_mask = all_df["strat_fold"].between(1, 8)
    val_mask = all_df["strat_fold"] == 9
    test_mask = all_df["strat_fold"] == 10
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise RuntimeError("One or more official folds are empty after preprocessing; inspect records and label policy.")

    X_raw = all_df[feature_names].to_numpy(dtype=np.float64)
    y = all_df["label"].to_numpy(dtype=np.int64)
    train_idx = np.flatnonzero(train_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
    X_val = scaler.transform(X_raw[val_idx]).astype(np.float32)
    X_test = scaler.transform(X_raw[test_idx]).astype(np.float32)

    all_df.to_csv(out_dir / "ptbxl_features_48.csv", index=False)
    joblib.dump(scaler, out_dir / "standard_scaler_train_only.joblib")
    np.savez_compressed(
        out_dir / "ptbxl_binary_splits.npz",
        X_train=X_train, y_train=y[train_idx],
        X_val=X_val, y_val=y[val_idx],
        X_test=X_test, y_test=y[test_idx],
        ecg_id_train=all_df.iloc[train_idx]["ecg_id"].to_numpy(),
        ecg_id_val=all_df.iloc[val_idx]["ecg_id"].to_numpy(),
        ecg_id_test=all_df.iloc[test_idx]["ecg_id"].to_numpy(),
        patient_id_train=all_df.iloc[train_idx]["patient_id"].astype(str).to_numpy(),
        patient_id_val=all_df.iloc[val_idx]["patient_id"].astype(str).to_numpy(),
        patient_id_test=all_df.iloc[test_idx]["patient_id"].astype(str).to_numpy(),
        feature_names=np.asarray(feature_names),
    )

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "waveform_read_failures.csv", index=False)

    # Patient overlap sanity check; official folds are patient-wise by design.
    patient_sets = {
        "train": set(all_df.iloc[train_idx]["patient_id"].astype(str)),
        "validation": set(all_df.iloc[val_idx]["patient_id"].astype(str)),
        "test": set(all_df.iloc[test_idx]["patient_id"].astype(str)),
    }
    overlap = {
        "train_validation": len(patient_sets["train"] & patient_sets["validation"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "validation_test": len(patient_sets["validation"] & patient_sets["test"]),
    }

    def split_summary(indices: np.ndarray) -> dict[str, Any]:
        labels = y[indices]
        return {
            "records": int(len(indices)),
            "unique_patients": int(all_df.iloc[indices]["patient_id"].nunique()),
            "normal_label_0": int(np.sum(labels == 0)),
            "abnormal_label_1": int(np.sum(labels == 1)),
        }

    summary = {
        "dataset_root": str(root),
        "target_definition": {
            "0": "normal: mapped diagnostic classes are NORM only",
            "1": "abnormal: at least one mapped diagnostic class other than NORM",
            "mixed_NORM_and_nonNORM": "labelled abnormal",
            "unmapped_or_no_diagnostic_class": "excluded",
        },
        "sampling_rate_hz": 100,
        "records_are_10_seconds": True,
        "feature_definition": "12 leads x mean/std/min/max = 48 features",
        "feature_names": feature_names,
        "is_smoke_test": args.max_records is not None,
        "smoke_test_note": (
            "Smoke-test subset is for pipeline verification only; unsuitable for final model evaluation."
            if args.max_records is not None else None
        ),
        "dataset_wide_metadata_records": int(len(metadata)),
        "dataset_wide_records_without_mapped_diagnostic_label_excluded": n_unlabelled,
        "dataset_wide_eligible_records": int(len(metadata[metadata["label"].notna()])),
        "subset_selected_records": int(len(eligible)),
        "metadata_records": int(len(metadata)),
        "records_without_mapped_diagnostic_label_excluded": n_unlabelled,
        "eligible_records_before_waveform_read": int(len(eligible)),
        "successfully_processed_records": int(len(all_df)),
        "waveform_read_failures": int(len(failures)),
        "splits": {
            "train_folds_1_to_8": split_summary(train_idx),
            "validation_fold_9": split_summary(val_idx),
            "test_fold_10": split_summary(test_idx),
        },
        "patient_overlap_counts": overlap,
        "scaler_fit_on": "training folds 1-8 only",
        "max_records_smoke_test": args.max_records,
        "outputs": [
            "ptbxl_features_48.csv",
            "ptbxl_binary_splits.npz",
            "standard_scaler_train_only.joblib",
        ] + (["waveform_read_failures.csv"] if failures else []),
    }
    with (out_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== PTB-XL preprocessing complete ===")
    if args.max_records is not None:
        print(f"[NOTE] SMOKE-TEST SUBSET: {len(all_df)} records processed across official folds.")
        print("[NOTE] This smoke-test artifact is for pipeline validation only; unsuitable for final model evaluation.")
    print(f"Dataset root: {root}")
    print(f"Output:       {out_dir}")
    print(f"Records kept: {len(all_df):,}; waveform failures: {len(failures):,}; unlabeled excluded: {n_unlabelled:,}")
    print("Split summary:")
    print(json.dumps(summary["splits"], indent=2))
    print("Patient-overlap counts (expected all zero):")
    print(json.dumps(overlap, indent=2))
    print("Label rule: any non-NORM diagnostic class => abnormal; NORM-only => normal; no mapped diagnosis => excluded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
