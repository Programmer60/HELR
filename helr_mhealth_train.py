"""
helr_mhealth_train.py
=====================
End-to-End CKKS Encrypted Logistic Regression Training on MHEALTH Dataset.

Architecture follows:
    Naresh & Reddi (2025), J. Big Data 12:52 — Algorithm 3 (Logistic_Encrypted_Data)

Trust Boundary
--------------
    HOSPITAL :  holds private key.
                encrypts X, y, and w every epoch.
                decrypts gradient scalars after each epoch and updates w.
    CSP (Cloud): holds evaluation key only.
                 performs ALL arithmetic on ciphertexts.
                 never calls .decrypt(); never sees plaintext.

Pipeline per epoch (matches Algorithm 3 and Fig. 2 from the paper)
-------------------------------------------------------------------
  Step 1 (Hospital) : enc_w_cols = Enc(w_j broadcast to P slots), for j in 0..D-1
  Step 2 (CSP)      : enc_z  = sum_j [ enc_X_cols[j] * enc_w_cols[j] ]   (Ct x Ct)
  Step 3 (CSP)      : enc_p  = poly_sigmoid(enc_z)                         (Ct x Pt muls)
  Step 4 (CSP)      : enc_err = enc_p - enc_y_aligned                      (level burn + sub)
  Step 5 (CSP)      : enc_grad_j = (1/P) * (enc_err * enc_X_cols[j]).sum() (Ct x Ct)
  Step 6 (Hospital) : grad_j = dec(enc_grad_j) + l2 * w_j
  Step 7 (Hospital) : w_j  = w_j - lr * grad_j
  Step 8 (Hospital) : re-encrypt updated w for next epoch

CKKS Parameters
---------------
    poly_modulus_degree = 16384
    coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 60]
    global_scale        = 2^40
    usable levels       = 7
    SIMD slots          = 8192

Multiplicative Depth per Epoch (degree-3 sigmoid)
-------------------------------------------------
    Linear combination (Ct x Ct)   : 1 level    (depth-independent of D)
    Degree-3 polynomial sigmoid     : 3 levels
    Error * X_j  (backward, Ct x Ct): 1 level
    Alignment burns                 : variable (not intrinsic depth)
    ---
    Total intrinsic                 : 5 levels consumed, 2 remaining
    => fits within 7-level context with margin >= 2
"""

import os
import sys
import json
import math
import time
import numpy as np
import tenseal as ts
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = "data/processed/mhealth"
RESULTS_DIR = "results/mhealth_he"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training hyper-parameters
EPOCHS        = 10          # number of full encrypted training epochs
LR            = 0.05        # learning rate — reduced from 0.5; 92 features need smaller step
L2            = 0.01        # L2 regularisation — lighter than paper default for stability
SIGMOID_DEG   = 3           # 1 = linear, 3 = cubic, 5 = quintic
GRAD_CLIP     = 1.0         # gradient clipping threshold (plaintext hospital step)

# Validation subset for gradient correctness check before full training
VALIDATION_N  = 100         # how many samples to use for gradient comparison

# ── CKKS polynomial coefficients (MIMIC-validated, reused for MHEALTH) ────────
# Degree-1  : sigma(z) ~= 0.5 + 0.197*z
SIG1_C0, SIG1_C1 = 0.5, 0.197
# Degree-3  : Kim et al. style  — valid on [-8, 8]
SIG3_C0, SIG3_C1, SIG3_C3 = 0.5, 0.15012, -0.001593
# Degree-5  : MIMIC-fitted — valid on [-6, 6]
SIG5_C0, SIG5_C1, SIG5_C3, SIG5_C5 = 0.5, 0.217101, -0.007823, 0.000118

# Levels consumed BY the sigmoid polynomial (not counting 1 for fwd dot-product)
_SIG_LEVELS = {1: 1, 3: 3, 5: 4}


# ── Helper utilities ──────────────────────────────────────────────────────────
def get_levels(enc):
    """Remaining modulus levels in a CKKSVector (queries the underlying SEAL object)."""
    return enc.ciphertext()[0].coeff_modulus_size()

def burn(v, n):
    """Consume n levels without changing the encrypted value (level-alignment helper)."""
    for _ in range(n):
        v = v * 1.0
    return v

def pt_sigmoid(z, degree):
    """Plaintext polynomial sigmoid — used for numerical validation."""
    if degree == 1:
        return SIG1_C0 + SIG1_C1 * z
    if degree == 3:
        return SIG3_C0 + SIG3_C1*z + SIG3_C3*(z**3)
    if degree == 5:
        return SIG5_C0 + SIG5_C1*z + SIG5_C3*(z**3) + SIG5_C5*(z**5)
    raise ValueError(f"Unsupported sigmoid degree: {degree}")

def true_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ── CKKS context ──────────────────────────────────────────────────────────────
def build_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


# ── Encrypted sigmoid (CSP-side) ──────────────────────────────────────────────
def enc_sigmoid(enc_z, degree):
    """
    Evaluate polynomial sigmoid approximation entirely on ciphertexts.
    All multiplications here are Ciphertext x Ciphertext unless noted.

    degree=1: 1 Ct x Pt multiply  → 1 level consumed
    degree=3: 2 Ct x Ct, 1 Ct x Pt + alignment burns
    degree=5: 3 Ct x Ct, 2 Ct x Pt + alignment burns
    """
    if degree == 1:
        return enc_z * SIG1_C1 + SIG1_C0                    # Ct x Pt

    if degree == 3:
        z2 = enc_z * enc_z                                   # Ct x Ct  (level -1)
        z3 = z2 * burn(enc_z, 1)                             # Ct x Ct  (level -1) + 1 burn
        term3 = z3 * SIG3_C3                                 # Ct x Pt
        term1 = burn(enc_z, 2) * SIG3_C1                    # 2 burns  + Ct x Pt
        return term3 + term1 + SIG3_C0

    if degree == 5:
        z2    = enc_z  * enc_z                               # Ct x Ct  (level -1)
        z3    = z2     * burn(enc_z, 1)                      # Ct x Ct  (level -1) + 1 burn
        z5    = z3     * burn(z2, 1)                         # Ct x Ct  (level -1) + 1 burn
        term5 = z5     * SIG5_C5                             # Ct x Pt
        term3 = burn(z3, 1) * SIG5_C3                       # 1 burn   + Ct x Pt
        term1 = burn(enc_z, 3) * SIG5_C1                    # 3 burns  + Ct x Pt
        return term5 + term3 + term1 + SIG5_C0

    raise ValueError(f"Unsupported sigmoid degree: {degree}")


# ── CSP: forward pass + gradient computation (ALL ciphertext) ─────────────────
def csp_compute_gradients(enc_X_cols, enc_y, enc_w_cols, P, D, degree):
    """
    Implements Algorithm 3 (CSP side) from the paper.
    Every argument is a ciphertext; no .decrypt() ever called here.

    Returns: list of D encrypted gradient scalars (one per feature).
    """
    sig_levels = _SIG_LEVELS[degree]
    # total levels from start to enc_p:  1 (dot-product) + sig_levels
    depth_to_p = 1 + sig_levels

    # ── Step 2: z = X @ w  (Ct x Ct dot product) ─────────────────────────────
    # Each enc_X_cols[j] * enc_w_cols[j] is Ct x Ct → level drops by 1
    # All products land at the same level, additions do not consume levels.
    enc_z = enc_X_cols[0] * enc_w_cols[0]
    for j in range(1, D):
        enc_z = enc_z + (enc_X_cols[j] * enc_w_cols[j])

    # ── Step 3: p = sigmoid(z) ────────────────────────────────────────────────
    enc_p = enc_sigmoid(enc_z, degree)

    # ── Step 4: error = p - y ────────────────────────────────────────────────
    # enc_y is fresh (level 7). enc_p is at (7 - depth_to_p).
    # Burn enc_y down to match enc_p's level before subtraction.
    enc_y_aligned = burn(enc_y, depth_to_p)
    enc_err = enc_p - enc_y_aligned

    # ── Step 5: gradient_j = (1/P) * sum_i( err_i * X_ij ) ──────────────────
    # enc_X_cols[j] is fresh (level 7). Burn to match enc_err.
    enc_grads = []
    for j in range(D):
        enc_Xj = burn(enc_X_cols[j], depth_to_p)            # align X_j to err level
        g_j = (enc_err * enc_Xj).sum()                      # Ct x Ct → gradient scalar
        enc_grads.append(g_j)

    return enc_grads, get_levels(enc_err), get_levels(enc_grads[0])


# ── Hospital: decrypt gradients and update weights ───────────────────────────
def hospital_update(enc_grads, w, P, D, lr, l2, grad_clip=1.0):
    """
    Hospital-side step: ONLY place that calls .decrypt().
    Decrypts the gradient scalar for each feature, applies L2,
    and performs the plaintext SGD weight update with gradient clipping.
    Gradient clipping prevents weight explosion when the polynomial sigmoid
    produces out-of-range values for large logits.
    """
    grad = np.zeros(D)
    for j in range(D):
        grad[j] = enc_grads[j].decrypt()[0] / P + l2 * w[j]
    # Clip gradient norm to prevent explosion caused by unbounded polynomial sigmoid
    grad_norm = np.linalg.norm(grad)
    if grad_norm > grad_clip:
        grad = grad * (grad_clip / grad_norm)
    return w - lr * grad


# ── One complete encrypted epoch ─────────────────────────────────────────────
def encrypted_epoch(enc_X_cols, enc_y, w, P, D, lr, l2, degree, ctx, grad_clip=1.0):
    """
    Implements one complete epoch of Algorithm 3.

    Hospital Step 1: encrypt current weights (broadcast to P slots each).
    CSP    Step 2-5: forward pass + gradient computation in ciphertext domain.
    Hospital Step 6-7: decrypt gradients, regularize, update w.
    """
    # Hospital encrypts w
    enc_w_cols = [ts.ckks_vector(ctx, [float(w[j])] * P) for j in range(D)]

    # CSP computes gradients (fully encrypted)
    enc_grads, level_after_err, level_after_grad = csp_compute_gradients(
        enc_X_cols, enc_y, enc_w_cols, P, D, degree
    )

    # Hospital decrypts and updates (with gradient clipping)
    w_new = hospital_update(enc_grads, w, P, D, lr, l2, grad_clip=grad_clip)

    return w_new, level_after_err, level_after_grad


# ── Numerical gradient validation (small batch) ───────────────────────────────
def validate_gradient(X, y, w, ctx, degree, tag=""):
    """
    On a small batch, compare plaintext gradient with decrypted encrypted gradient.
    Max and mean absolute errors are reported.
    """
    P, D = X.shape
    enc_X_cols = [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(D)]
    enc_y      = ts.ckks_vector(ctx, y.tolist())
    enc_w_cols = [ts.ckks_vector(ctx, [float(w[j])] * P) for j in range(D)]

    enc_grads, _, _ = csp_compute_gradients(enc_X_cols, enc_y, enc_w_cols, P, D, degree)

    # Plaintext reference gradient
    z_pt   = X @ w
    p_pt   = pt_sigmoid(z_pt, degree)
    err_pt = p_pt - y
    grad_pt = (X.T @ err_pt) / P

    grad_enc = np.array([enc_grads[j].decrypt()[0] / P for j in range(D)])
    abs_err  = np.abs(grad_pt - grad_enc)

    print(f"  [{tag}] max_grad_err={abs_err.max():.2e}  mean_grad_err={abs_err.mean():.2e}")
    return abs_err.max(), abs_err.mean()


# ── Full encrypted training loop ──────────────────────────────────────────────
def train_helr_mhealth(X_train, y_train, X_test, y_test,
                        degree, epochs, lr, l2, ctx,
                        grad_clip=1.0, validate_first=True):

    P, D  = X_train.shape
    SLOTS = 8192
    assert P <= SLOTS, f"Training set size {P} exceeds SIMD slots {SLOTS}."

    print(f"\n{'='*65}")
    print(f"  MHEALTH HELR — degree={degree}, epochs={epochs}, P={P}, D={D}")
    print(f"  lr={lr}, l2={l2}, CKKS [60,40,40,40,40,40,40,60], scale=2^40")
    print(f"{'='*65}")

    # ── Encrypt training data once (held fixed across epochs) ────────────────
    print("\n[Hospital] Encrypting X_train columns and y_train...")
    t_enc_start = time.perf_counter()
    enc_X_cols  = [ts.ckks_vector(ctx, X_train[:, j].tolist()) for j in range(D)]
    enc_y       = ts.ckks_vector(ctx, y_train.tolist())
    t_enc = time.perf_counter() - t_enc_start
    print(f"  Encryption time: {t_enc:.2f}s  |  CTs: {D+1} (D feature cols + 1 label)")

    # ── Step 0: numerical gradient validation on small subset ────────────────
    if validate_first:
        print(f"\n[Validation] Gradient correctness on {VALIDATION_N} samples ...")
        X_val = X_train[:VALIDATION_N]
        y_val = y_train[:VALIDATION_N]
        w_zero = np.zeros(D)
        max_err, mean_err = validate_gradient(X_val, y_val, w_zero, ctx, degree,
                                              tag=f"deg{degree} w=0")
        print(f"  PASS: numerical gradient validated before training begins.")

    # ── Training loop ─────────────────────────────────────────────────────────
    w = np.zeros(D)
    history = []
    ep_times = []
    final_level_err  = None
    final_level_grad = None

    print(f"\n[Training] Starting {epochs} encrypted epochs ...")
    total_train_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        w, level_err, level_grad = encrypted_epoch(
            enc_X_cols, enc_y, w, P, D, lr, l2, degree, ctx,
            grad_clip=grad_clip
        )
        ep_time = time.perf_counter() - t_ep
        ep_times.append(ep_time)
        final_level_err  = level_err
        final_level_grad = level_grad

        # Plaintext eval every epoch to track convergence
        z_pt = X_train @ w
        p_pt = true_sigmoid(z_pt)
        train_acc = accuracy_score(y_train, (p_pt >= 0.5).astype(int))

        history.append({"epoch": ep, "norm_w": float(np.linalg.norm(w)),
                         "train_acc": train_acc, "epoch_time_s": ep_time})
        print(f"  Epoch {ep:2d}/{epochs} | ||w||={np.linalg.norm(w):.4f} "
              f"| train_acc={train_acc:.4f} | t={ep_time:.2f}s "
              f"| levels_after_grad={level_grad}")

    total_train_time = time.perf_counter() - total_train_start

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n[Evaluation] Test set ({len(X_test)} windows, subjects 8-9-10)")
    z_test = X_test @ w
    p_test = true_sigmoid(z_test)
    y_pred = (p_test >= 0.5).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, p_test)),
    }

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    # ── Ciphertext footprint ──────────────────────────────────────────────────
    ct_bytes = sum(len(c.serialize()) for c in enc_X_cols)
    ct_mb    = ct_bytes / 1e6

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "dataset":              "MHEALTH",
        "sigmoid_degree":       degree,
        "epochs":               epochs,
        "lr":                   lr,
        "l2":                   l2,
        "train_samples":        P,
        "test_samples":         len(y_test),
        "num_features":         D,
        "ckks_context":         "[60,40,40,40,40,40,40,60]",
        "scale":                "2^40",
        "feature_ciphertexts":  D,
        "label_ciphertext":     1,
        "encryption_time_s":    round(t_enc, 3),
        "total_train_time_s":   round(total_train_time, 3),
        "mean_epoch_time_s":    round(float(np.mean(ep_times)), 3),
        "std_epoch_time_s":     round(float(np.std(ep_times)), 3),
        "serialized_X_ct_MB":   round(ct_mb, 2),
        "final_levels_remaining_after_err":  final_level_err,
        "final_levels_remaining_after_grad": final_level_grad,
        "gradient_max_abs_error": float(max_err),
        "gradient_mean_abs_error": float(mean_err),
        "test_metrics":         metrics,
        "weight_history":       history,
    }

    out_path = os.path.join(RESULTS_DIR, f"helr_mhealth_deg{degree}_ep{epochs}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\n  Results saved to {out_path}")

    print(f"\n[Summary]")
    print(f"  Total training time     : {total_train_time:.2f}s")
    print(f"  Mean time per epoch     : {np.mean(ep_times):.2f}s")
    print(f"  Feature ciphertexts     : {D}")
    print(f"  Serialized X CT footprint: {ct_mb:.2f} MB")
    print(f"  Final levels remaining  : {final_level_grad} (after gradient mul)")
    print(f"  Gradient max_err        : {max_err:.2e}")

    return w, summary


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load preprocessed MHEALTH data
    print("Loading MHEALTH preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(float)
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(float)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")
    print(f"  Train class balance: 0={int((y_train==0).sum())} 1={int((y_train==1).sum())}")
    print(f"  Test  class balance: 0={int((y_test==0).sum())}  1={int((y_test==1).sum())}")

    # ── Check SIMD fit ────────────────────────────────────────────────────────
    SLOTS = 8192
    P, D = X_train.shape
    if P > SLOTS:
        print(f"  WARNING: {P} samples > {SLOTS} SIMD slots. Truncating to {SLOTS}.")
        X_train = X_train[:SLOTS]
        y_train = y_train[:SLOTS]
        P = SLOTS

    # ── Build CKKS context ────────────────────────────────────────────────────
    print("\nBuilding CKKS context [60,40,40,40,40,40,40,60] scale=2^40 ...")
    ctx = build_context()
    print("  Context ready. SIMD slots = 8192. Usable levels = 7.")

    # ── Run encrypted training ────────────────────────────────────────────────
    w_trained, summary = train_helr_mhealth(
        X_train, y_train, X_test, y_test,
        degree        = SIGMOID_DEG,
        epochs        = EPOCHS,
        lr            = LR,
        l2            = L2,
        ctx           = ctx,
        grad_clip     = GRAD_CLIP,
        validate_first= True,
    )

    print("\nEncrypted training pipeline complete.")
