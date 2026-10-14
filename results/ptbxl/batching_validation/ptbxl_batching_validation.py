#!/usr/bin/env python3
"""
PTB-XL SIMD Batching and CKKS Gradient Validation Harness
=========================================================
Implements and rigorously validates:
1. Three-batch SIMD column-packing for PTB-XL (N=17,084, D=48, S=8,192).
2. Correct global gradient aggregation across batches without inter-batch updates.
3. Three distinct comparison pathways:
   - Comparison 1: Exact sigmoid vs. Polynomial sigmoid in plaintext (approximation error).
   - Comparison 2: Plaintext polynomial gradient vs. Decrypted CKKS gradient (encryption error).
   - Comparison 3: Three-batch aggregation vs. Full-matrix reference in plaintext (batching error).
4. Deterministic test cases:
   - Zero weights (w = 0)
   - 5 deterministic random weight vectors
   - Checkpoint weights from the trained plaintext baseline model
   - Synthetic batch-edge test (non-power-of-2 subset)
   - Partial-batch masking test (verifying padding invariance)
"""

import os
import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tenseal as ts

# ── Paths & Configuration ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "ptbxl_processed" / "ptbxl_binary_splits.npz"
BASELINE_MODEL_PATH = PROJECT_ROOT / "data" / "ptbxl_processed" / "baseline_logistic_regression.joblib"
OUT_DIR = Path(__file__).resolve().parent

SEED = 42
L2_REG = 0.01
TOLERANCE_CKKS = 1e-4       # Threshold for encryption discrepancy
TOLERANCE_BATCHING = 1e-10   # Threshold for plaintext batch equivalence

# Polynomial Sigmoid Coefficients (Degree 3, Kim et al.)
POLY3_C0 = 0.5
POLY3_C1 = 0.15012
POLY3_C3 = -0.001593

def build_ckks_context():
    """Initializes standard CKKS context matching MHEALTH configuration."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

def burn(v, n):
    """Consumes n modulus levels via scalar multiplication by 1.0."""
    for _ in range(n):
        v = v * 1.0
    return v

# ── Plaintext Mathematical References ────────────────────────────────────────
def true_sigmoid(z):
    """Exact transcendental logistic sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def poly_sigmoid(z):
    """Degree-3 polynomial sigmoid approximation."""
    return POLY3_C0 + POLY3_C1 * z + POLY3_C3 * (z**3)

def compute_plaintext_gradient_full(X, y, w, l2=L2_REG, use_poly=True):
    """
    Computes reference gradient over the entire matrix in one global operation:
    g = (1/N) * X^T (sigma(Xw) - y) + l2 * w
    """
    N, D = X.shape
    z = X @ w
    p = poly_sigmoid(z) if use_poly else true_sigmoid(z)
    err = p - y
    grad_unreg = (X.T @ err) / N
    return grad_unreg + l2 * w

def compute_plaintext_gradient_batched(batches, w, N_total, l2=L2_REG, use_poly=True):
    """
    Computes reference gradient by accumulating unregularized sums across batches,
    normalizing by N_total, and adding L2 regularization once globally.
    """
    D = len(w)
    unreg_sum = np.zeros(D, dtype=np.float64)
    
    for b in batches:
        X_b = b["X_valid"]
        y_b = b["y_valid"]
        z_b = X_b @ w
        p_b = poly_sigmoid(z_b) if use_poly else true_sigmoid(z_b)
        err_b = p_b - y_b
        # Unnormalized sum of gradients for this batch
        unreg_sum += X_b.T @ err_b
        
    global_grad = (unreg_sum / N_total) + l2 * w
    return global_grad

# ── Encrypted Computation Pipeline (CSP Side) ─────────────────────────────────
def enc_sigmoid_deg3(enc_z):
    """Homomorphic evaluation of degree-3 polynomial sigmoid."""
    z2 = enc_z * enc_z
    z3 = z2 * burn(enc_z, 1)
    term3 = z3 * POLY3_C3
    term1 = burn(enc_z, 2) * POLY3_C1
    return term3 + term1 + POLY3_C0

def csp_batch_gradient_sums(enc_X_cols, enc_y, enc_w_cols, D):
    """
    Computes unnormalized encrypted gradient sums for one batch:
    enc_S_j = sum_{i=1}^S (err_i * X_{i, j})
    Depth: 1 (dot) + 3 (poly) + 1 (grad) = 5 levels consumed.
    """
    # 1. Linear combination: enc_z = sum_j (X_j * w_j)
    enc_z = enc_X_cols[0] * enc_w_cols[0]
    for j in range(1, D):
        enc_z = enc_z + (enc_X_cols[j] * enc_w_cols[j])

    # 2. Polynomial sigmoid (Degree 3 consumes 3 levels)
    enc_p = enc_sigmoid_deg3(enc_z)

    # 3. Align label ciphertext (burn 4 levels) and subtract error
    enc_y_aligned = burn(enc_y, 4)
    enc_err = enc_p - enc_y_aligned

    # 4. Multiply by aligned feature ciphertexts (burn 4 levels) and sum slots
    enc_sums = []
    for j in range(D):
        enc_Xj_aligned = burn(enc_X_cols[j], 4)
        g_j = (enc_err * enc_Xj_aligned).sum()
        enc_sums.append(g_j)

    return enc_sums

# ── Batching Preparation ─────────────────────────────────────────────────────
def partition_training_data(X_train, y_train, slot_capacity=8192):
    """
    Partitions N training samples into batches of maximum size slot_capacity.
    For the final partial batch, creates zero-padded arrays of shape (slot_capacity, D).
    """
    N, D = X_train.shape
    num_batches = int(np.ceil(N / slot_capacity))
    batches = []

    for b_idx in range(num_batches):
        start = b_idx * slot_capacity
        end = min(start + slot_capacity, N)
        n_valid = end - start
        n_unused = slot_capacity - n_valid

        X_valid = X_train[start:end]
        y_valid = y_train[start:end]

        # Pad to slot_capacity if partial
        if n_unused > 0:
            X_padded = np.pad(X_valid, ((0, n_unused), (0, 0)), mode="constant", constant_values=0.0)
            y_padded = np.pad(y_valid, (0, n_unused), mode="constant", constant_values=0.0)
        else:
            X_padded = X_valid.copy()
            y_padded = y_valid.copy()

        batches.append({
            "batch_idx": b_idx + 1,
            "start_idx": start,
            "end_idx": end,
            "n_valid": n_valid,
            "n_unused": n_unused,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_padded": X_padded,
            "y_padded": y_padded,
        })

    return batches

# ── Main Validation Harness ──────────────────────────────────────────────────
def run_validation():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 80)
    log(" PTB-XL SIMD BATCHING & CKKS GRADIENT VALIDATION HARNESS")
    log("=" * 80)
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")

    # 1. Load Data
    log("\n[1] Loading PTB-XL Preprocessed Training Split...")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run preprocess_ptbxl.py first.")

    npz = np.load(DATA_PATH, allow_pickle=True)
    X_train = npz["X_train"]
    y_train = npz["y_train"].astype(np.float64)
    ecg_id_train = npz["ecg_id_train"]
    patient_id_train = npz["patient_id_train"]
    feature_names = list(npz["feature_names"])

    N, D = X_train.shape
    log(f"  Training matrix shape: N = {N:,}, D = {D}")
    log(f"  Labels shape: {y_train.shape} (Positive={int(np.sum(y_train==1)):,}, Negative={int(np.sum(y_train==0)):,})")

    # Assert correct dimensions
    assert N == 17084, f"Expected 17,084 training records, found {N}"
    assert D == 48, f"Expected 48 features, found {D}"
    assert len(np.unique(ecg_id_train)) == N, "Duplicate ECG IDs detected in training split!"

    # 2. Partition into Three Batches
    log("\n[2] Partitioning into 8,192-Slot Batches...")
    SLOT_CAP = 8192
    batches = partition_training_data(X_train, y_train, slot_capacity=SLOT_CAP)
    log(f"  Total Batches: {len(batches)}")
    
    total_valid = sum(b["n_valid"] for b in batches)
    total_unused = sum(b["n_unused"] for b in batches)
    assert total_valid == N, f"Valid count mismatch: {total_valid} vs {N}"
    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

    for b in batches:
        log(f"  Batch {b['batch_idx']}: records [{b['start_idx']:,} : {b['end_idx']:,}], "
            f"valid={b['n_valid']:,}, padded_unused={b['n_unused']:,}, padded_shape={b['X_padded'].shape}")

    # 3. Initialize CKKS Context
    log("\n[3] Initializing CKKS Context...")
    ctx = build_ckks_context()
    log("  Context: N=16384, coeff_mod_bit_sizes=[60,40,40,40,40,40,40,60], scale=2^40, slots=8192")

    # 4. Encrypt Batched Datasets
    log("\n[4] Encrypting Feature Matrices and Labels (Column-Packed)...")
    t0 = time.perf_counter()
    enc_batches = []
    total_cts = 0
    for b in batches:
        X_pad = b["X_padded"]
        y_pad = b["y_padded"]
        enc_X_cols = [ts.ckks_vector(ctx, X_pad[:, j].tolist()) for j in range(D)]
        enc_y = ts.ckks_vector(ctx, y_pad.tolist())
        enc_batches.append({"enc_X": enc_X_cols, "enc_y": enc_y})
        total_cts += (D + 1)
    t_enc = time.perf_counter() - t0
    log(f"  Encrypted {total_cts} ciphertexts (48 features x 3 + 3 labels) in {t_enc:.2f}s")

    # 5. Load Trained Baseline Checkpoint Weights
    log("\n[5] Loading Plaintext Baseline Checkpoint Weights...")
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    w_baseline = baseline_model.coef_[0].copy()
    log(f"  Loaded baseline model with 48 coefficients (||w|| = {np.linalg.norm(w_baseline):.4f})")

    # 6. Define Test Weight Vectors
    np.random.seed(SEED)
    test_cases = [
        ("case_1_w_zero", np.zeros(D)),
        ("case_2_w_rand1", np.random.uniform(-0.1, 0.1, D)),
        ("case_3_w_rand2", np.random.uniform(-0.1, 0.1, D)),
        ("case_4_w_rand3", np.random.uniform(-0.1, 0.1, D)),
        ("case_5_w_rand4", np.random.uniform(-0.1, 0.1, D)),
        ("case_6_w_rand5", np.random.uniform(-0.1, 0.1, D)),
        ("case_7_w_baseline", w_baseline),
    ]

    # 7. Execute Gradient Validation Suite
    log("\n" + "=" * 80)
    log(" [6] RUNNING NUMERICAL GRADIENT VALIDATION SUITE")
    log("=" * 80)

    csv_records = []
    assertion_results = {
        "plaintext_batch_equivalence_all_cases": True,
        "ckks_encryption_error_within_tolerance": True,
        "partial_batch_masking_invariance_passed": True,
        "synthetic_batch_edge_test_passed": True,
    }

    for tag, w in test_cases:
        log(f"\n>>> Evaluating {tag} (||w|| = {np.linalg.norm(w):.4f})...")

        # Step 1: Plaintext References
        # Comparison 3: Full-matrix plaintext vs. Three-batch plaintext
        g_pt_exact_full = compute_plaintext_gradient_full(X_train, y_train, w, l2=L2_REG, use_poly=False)
        g_pt_poly_full  = compute_plaintext_gradient_full(X_train, y_train, w, l2=L2_REG, use_poly=True)
        g_pt_poly_batch = compute_plaintext_gradient_batched(batches, w, N_total=N, l2=L2_REG, use_poly=True)

        diff_batch_vs_full = np.abs(g_pt_poly_batch - g_pt_poly_full)
        max_batch_diff = float(diff_batch_vs_full.max())
        mean_batch_diff = float(diff_batch_vs_full.mean())
        batching_ok = max_batch_diff <= TOLERANCE_BATCHING
        if not batching_ok:
            assertion_results["plaintext_batch_equivalence_all_cases"] = False

        # Comparison 1: Exact sigmoid vs. Polynomial sigmoid
        diff_poly_vs_exact = np.abs(g_pt_poly_full - g_pt_exact_full)
        max_approx_err = float(diff_poly_vs_exact.max())
        mean_approx_err = float(diff_poly_vs_exact.mean())

        # Step 2: Encrypted Gradient Computation (CSP)
        # Encrypt weights once (broadcast to 8,192 slots)
        enc_w_cols = [ts.ckks_vector(ctx, [float(w[j])] * SLOT_CAP) for j in range(D)]

        # CSP evaluates each batch and Hospital decrypts gradient sums
        t_csp_0 = time.perf_counter()
        unreg_enc_sum = np.zeros(D, dtype=np.float64)

        for b_idx in range(len(batches)):
            enc_b = enc_batches[b_idx]
            enc_sums = csp_batch_gradient_sums(enc_b["enc_X"], enc_b["enc_y"], enc_w_cols, D)
            # Decrypt batch sums
            dec_b_sums = np.array([g.decrypt()[0] for g in enc_sums])
            unreg_enc_sum += dec_b_sums

        t_csp = time.perf_counter() - t_csp_0

        # Step 3: Global Aggregation & Regularization
        # g = (1/N) * sum_b(S_b) + l2 * w
        g_ckks = (unreg_enc_sum / N) + L2_REG * w

        # Comparison 2: Plaintext polynomial gradient vs. Decrypted CKKS gradient
        diff_ckks_vs_poly = np.abs(g_ckks - g_pt_poly_full)
        max_ckks_err = float(diff_ckks_vs_poly.max())
        mean_ckks_err = float(diff_ckks_vs_poly.mean())

        # Relative error where reference is significant
        mask = np.abs(g_pt_poly_full) > 1e-6
        mean_re = float(np.mean(diff_ckks_vs_poly[mask] / np.abs(g_pt_poly_full[mask]))) if np.any(mask) else 0.0

        ckks_ok = max_ckks_err <= TOLERANCE_CKKS
        if not ckks_ok:
            assertion_results["ckks_encryption_error_within_tolerance"] = False

        is_finite = bool(np.all(np.isfinite(g_ckks)))

        log(f"  Plaintext Batch Equivalence Diff : MaxAE = {max_batch_diff:.2e} (Pass={batching_ok})")
        log(f"  Exact vs. Poly Approx Diff       : MaxAE = {max_approx_err:.2e}, MeanAE = {mean_approx_err:.2e}")
        log(f"  CKKS Encryption Error vs. Poly   : MaxAE = {max_ckks_err:.2e}, MeanAE = {mean_ckks_err:.2e}, MeanRE = {mean_re:.2e} (Pass={ckks_ok})")
        log(f"  ||g_CKKS|| = {np.linalg.norm(g_ckks):.4f} | Compute Time = {t_csp:.2f}s | Finite = {is_finite}")

        csv_records.append({
            "test_case": tag,
            "weight_norm": float(np.linalg.norm(w)),
            "batch_diff_max_ae": max_batch_diff,
            "batch_diff_mean_ae": mean_batch_diff,
            "approx_err_max_ae": max_approx_err,
            "approx_err_mean_ae": mean_approx_err,
            "ckks_err_max_ae": max_ckks_err,
            "ckks_err_mean_ae": mean_ckks_err,
            "ckks_err_mean_re": mean_re,
            "ckks_grad_norm": float(np.linalg.norm(g_ckks)),
            "compute_time_s": t_csp,
            "is_finite": is_finite,
            "batching_pass": batching_ok,
            "ckks_pass": ckks_ok,
        })

    # 8. Test Case 8: Synthetic Batch-Edge Test
    log("\n" + "=" * 80)
    log(" [7] SYNTHETIC BATCH-EDGE TEST (Arbitrary N not divisible by 8192)")
    log("=" * 80)
    N_edge = 10000  # 8192 + 1808 records
    log(f"  Testing subset of N = {N_edge:,} records across 2 batches (8192 + 1808)...")
    X_edge = X_train[:N_edge]
    y_edge = y_train[:N_edge]
    w_edge = test_cases[1][1] # use rand1 weights

    edge_batches = partition_training_data(X_edge, y_edge, slot_capacity=SLOT_CAP)
    g_edge_full = compute_plaintext_gradient_full(X_edge, y_edge, w_edge, l2=L2_REG, use_poly=True)
    g_edge_batch = compute_plaintext_gradient_batched(edge_batches, w_edge, N_total=N_edge, l2=L2_REG, use_poly=True)

    diff_edge = float(np.max(np.abs(g_edge_batch - g_edge_full)))
    edge_ok = diff_edge <= TOLERANCE_BATCHING
    assertion_results["synthetic_batch_edge_test_passed"] = edge_ok
    log(f"  Batch-Edge Equivalence: MaxAE = {diff_edge:.2e} (Pass={edge_ok})")

    # 9. Test Case 9: Partial-Batch Masking Test
    log("\n" + "=" * 80)
    log(" [8] PARTIAL-BATCH MASKING INVARIANCE TEST")
    log("=" * 80)
    log("  Modifying padded slots (slots 700..8191) with random non-zero label values...")
    b3 = batches[2]
    # Copy padded Batch 3
    y_corrupted = b3["y_padded"].copy()
    y_corrupted[700:] = np.random.uniform(0.0, 1.0, 7492) # inject random noise in padded slots

    # Encrypt corrupted batch 3
    enc_y_corrupted = ts.ckks_vector(ctx, y_corrupted.tolist())
    enc_w_test = [ts.ckks_vector(ctx, [float(test_cases[1][1][j])] * SLOT_CAP) for j in range(D)]
    enc_b3_original = enc_batches[2]

    # Compute gradient on original vs corrupted padded batch 3
    g_clean_sums = csp_batch_gradient_sums(enc_b3_original["enc_X"], enc_b3_original["enc_y"], enc_w_test, D)
    g_corrupted_sums = csp_batch_gradient_sums(enc_b3_original["enc_X"], enc_y_corrupted, enc_w_test, D)

    dec_clean = np.array([g.decrypt()[0] for g in g_clean_sums])
    dec_corrupted = np.array([g.decrypt()[0] for g in g_corrupted_sums])

    masking_diff = float(np.max(np.abs(dec_clean - dec_corrupted)))
    # Because X[700:, j] == 0, the product err * X is identically 0 in all padded slots!
    masking_ok = masking_diff <= 1e-4
    assertion_results["partial_batch_masking_invariance_passed"] = masking_ok
    log(f"  Masking Invariance Discrepancy: MaxAE = {masking_diff:.2e} (Pass={masking_ok})")

    # 10. Summary & Save Outputs
    log("\n" + "=" * 80)
    log(" [9] FINAL ASSERTION SUMMARY")
    log("=" * 80)
    all_passed = all(assertion_results.values())
    for k, v in assertion_results.items():
        log(f"  - {k:45s}: {'PASS' if v else 'FAIL'}")
    log(f"\nOVERALL STATUS: {'PASS' if all_passed else 'FAIL'}")

    # Save CSV
    df_results = pd.DataFrame(csv_records)
    csv_file = OUT_DIR / "ptbxl_batching_validation.csv"
    df_results.to_csv(csv_file, index=False)
    log(f"\nSaved CSV: {csv_file}")

    # Save JSON summary
    summary = {
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "tenseal_version": ts.__version__,
            "ckks_poly_modulus_degree": 16384,
            "ckks_coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60],
            "ckks_scale": "2^40",
            "simd_slot_capacity": SLOT_CAP,
        },
        "dataset": {
            "name": "PTB-XL v1.0.3",
            "split": "Folds 1-8 (Training only)",
            "total_records": N,
            "feature_dimension": D,
            "batch_count": len(batches),
            "batches": [
                {"batch": b["batch_idx"], "valid_records": b["n_valid"], "unused_slots": b["n_unused"]}
                for b in batches
            ],
            "total_ciphertexts": total_cts,
        },
        "assertions": assertion_results,
        "metrics_summary": {
            "max_batching_error_ae": float(df_results["batch_diff_max_ae"].max()),
            "max_ckks_encryption_error_ae": float(df_results["ckks_err_max_ae"].max()),
            "mean_ckks_encryption_error_ae": float(df_results["ckks_err_mean_ae"].mean()),
            "tolerance_batching": TOLERANCE_BATCHING,
            "tolerance_ckks": TOLERANCE_CKKS,
            "all_tests_passed": all_passed,
        },
        "test_cases": csv_records,
    }

    json_file = OUT_DIR / "ptbxl_batching_validation.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved JSON: {json_file}")

    # Save execution log
    log_file = OUT_DIR / "ptbxl_batching_validation.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"Saved Log:  {log_file}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(run_validation())
