#!/usr/bin/env python3
"""
Extract per-lead, per-record median-centered statistical features from PTB-XL raw waveforms.
Saves to data/ptbxl_processed/ptbxl_features_48_median_centered.npz without modifying any original files.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "ptbxl_processed"
CSV_PATH = DATA_DIR / "ptbxl_features_48.csv"
DATASET_ROOT = (
    PROJECT_ROOT
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
OUT_PATH = DATA_DIR / "ptbxl_features_48_median_centered.npz"


def extract_median_centered_stats(rel_path: str) -> np.ndarray:
    rec_path = DATASET_ROOT / rel_path
    signal, _ = wfdb.rdsamp(str(rec_path))
    signal = np.asarray(signal, dtype=np.float64)
    # Per-lead median centering
    med = np.median(signal, axis=0, keepdims=True)
    sig_c = signal - med
    return np.column_stack([
        np.mean(sig_c, axis=0),
        np.std(sig_c, axis=0, ddof=0),
        np.min(sig_c, axis=0),
        np.max(sig_c, axis=0),
    ]).reshape(-1).astype(np.float32)


def main():
    print("=" * 80)
    print(" EXTRACTING MEDIAN-CENTERED PTB-XL FEATURES")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)
    n_total = len(df)
    print(f"Total records to process: {n_total:,}")
    paths = df["filename_lr"].tolist()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as executor:
        features_list = list(executor.map(extract_median_centered_stats, paths))
    t1 = time.perf_counter()

    features_np = np.vstack(features_list)
    print(f"Extraction completed in {t1 - t0:.2f}s ({n_total / (t1 - t0):.1f} rec/s)")
    print(f"Features shape: {features_np.shape}")

    # Preserve all splits and patient metadata
    train_mask = df["strat_fold"].between(1, 8).to_numpy()
    val_mask = (df["strat_fold"] == 9).to_numpy()
    test_mask = (df["strat_fold"] == 10).to_numpy()

    # Feature names
    npz_orig = np.load(DATA_DIR / "ptbxl_binary_splits.npz", allow_pickle=True)
    feature_names = npz_orig["feature_names"]

    np.savez_compressed(
        OUT_PATH,
        features=features_np,
        ecg_id=df["ecg_id"].to_numpy(),
        patient_id=df["patient_id"].astype(str).to_numpy(),
        strat_fold=df["strat_fold"].to_numpy(),
        label=df["label"].to_numpy(),
        feature_names=feature_names,
    )
    print(f"Saved: {OUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
