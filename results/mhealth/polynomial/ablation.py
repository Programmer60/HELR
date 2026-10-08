"""
Phase 1 — Polynomial Ablation
==============================
Degree 1 / 3 / 5 encrypted sigmoid comparison on MHEALTH.

Rules:
- EVERYTHING is identical except sigmoid degree.
- Same dataset, split, features, initialisation, optimiser, epochs, context.
- Baseline (degree-3) results are NOT overwritten; this script writes to
  results/mhealth/polynomial/ only.
- No test data used for polynomial fitting, feature selection, or tuning.
- Failures (scale / modulus exhaustion) are recorded explicitly, not silently patched.

CKKS depth budget per degree (7 usable levels):
  Degree 1 : dot(1) + sigmoid(1) + backward(1) = 3 consumed, 4 remaining
  Degree 3 : dot(1) + sigmoid(3) + backward(1) = 5 consumed, 2 remaining
  Degree 5 : dot(1) + sigmoid(4) + backward(1) = 6 consumed, 1 remaining
  All three fit within [60,40,40,40,40,40,40,60].
"""

import os, json, time, math
import numpy as np
import pandas as pd
import tenseal as ts
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data/processed/mhealth"
OUT_DIR   = "results/mhealth/polynomial"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Frozen hyperparameters (identical across all degrees) ────────────────────
LR         = 0.05
L2         = 0.01
GRAD_CLIP  = 1.0
EPOCHS     = 10
SEED       = 42
VAL_N      = 100          # gradient validation batch size
APPROX_INT = (-6.0, 6.0)  # polynomial fitting / error evaluation interval
N_EVAL_PTS = 2000         # points for approximation-error computation

# ── Polynomial coefficients (fixed; not fitted on test data) ─────────────────
POLY = {
    1: {"c0": 0.5,  "c1": 0.197,     "c3": 0.0,       "c5": 0.0,
        "name": "linear",  "valid_interval": "(-inf, +inf)"},
    3: {"c0": 0.5,  "c1": 0.15012,   "c3": -0.001593,  "c5": 0.0,
        "name": "cubic",   "valid_interval": "[-8, 8]"},
    5: {"c0": 0.5,  "c1": 0.217101,  "c3": -0.007823,  "c5": 0.000118,
        "name": "quintic", "valid_interval": "[-6, 6]"},
}

# Levels the SIGMOID portion of the circuit consumes (not including 1 for
# the initial Ct×Ct dot product or 1 for the backward Ct×Ct multiplication).
_SIG_LEVELS = {1: 1, 3: 3, 5: 4}
_TOTAL_CONSUMED = {d: 1 + _SIG_LEVELS[d] + 1 for d in [1, 3, 5]}
_LEVELS_REMAINING = {d: 7 - _TOTAL_CONSUMED[d] for d in [1, 3, 5]}


# ── Utilities ────────────────────────────────────────────────────────────────
def true_sigmoid(z):
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))   # numerically stable

def poly_sigmoid(z, degree):
    p = POLY[degree]
    out = p["c0"] + p["c1"]*z
    if degree >= 3:
        out = out + p["c3"]*(z**3)
    if degree == 5:
        out = out + p["c5"]*(z**5)
    return out

def approx_error(degree, interval=APPROX_INT, n=N_EVAL_PTS):
    z = np.linspace(interval[0], interval[1], n)
    err = np.abs(true_sigmoid(z) - poly_sigmoid(z, degree))
    return {"max": float(err.max()), "mean": float(err.mean()),
            "interval": list(interval)}

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def burn(v, n):
    for _ in range(n):
        v = v * 1.0
    return v

def build_context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384,
                     coeff_mod_bit_sizes=[60,40,40,40,40,40,40,60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


# ── Encrypted sigmoid ────────────────────────────────────────────────────────
def enc_sigmoid(enc_z, degree):
    p = POLY[degree]
    if degree == 1:
        return enc_z * p["c1"] + p["c0"]
    if degree == 3:
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        t3 = z3 * p["c3"]
        t1 = burn(enc_z, 2) * p["c1"]
        return t3 + t1 + p["c0"]
    if degree == 5:
        z2 = enc_z  * enc_z
        z3 = z2     * burn(enc_z, 1)
        z5 = z3     * burn(z2, 1)
        t5 = z5     * p["c5"]
        t3 = burn(z3, 1) * p["c3"]
        t1 = burn(enc_z, 3) * p["c1"]
        return t5 + t3 + t1 + p["c0"]
    raise ValueError(degree)


# ── CSP: encrypted gradient (ciphertext-only) ────────────────────────────────
def csp_gradients(enc_X, enc_y, enc_w, P, D, degree):
    depth_p = 1 + _SIG_LEVELS[degree]
    enc_z = enc_X[0] * enc_w[0]
    for j in range(1, D):
        enc_z = enc_z + enc_X[j] * enc_w[j]
    enc_p = enc_sigmoid(enc_z, degree)
    enc_err = enc_p - burn(enc_y, depth_p)
    level_err  = get_levels(enc_err)
    enc_grads  = [(enc_err * burn(enc_X[j], depth_p)).sum() for j in range(D)]
    level_grad = get_levels(enc_grads[0])
    return enc_grads, level_err, level_grad


# ── Hospital: decrypt + update ────────────────────────────────────────────────
def hospital_update(enc_grads, w, P, D, lr, l2, grad_clip):
    grad = np.array([enc_grads[j].decrypt()[0] / P for j in range(D)]) + l2 * w
    n = np.linalg.norm(grad)
    if n > grad_clip:
        grad *= grad_clip / n
    return w - lr * grad


# ── Gradient numerical validation ────────────────────────────────────────────
def validate_gradient(X, y, w, ctx, degree):
    P, D = X.shape
    enc_X = [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y.tolist())
    enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
    enc_g, _, _ = csp_gradients(enc_X, enc_y, enc_w, P, D, degree)
    grad_enc = np.array([enc_g[j].decrypt()[0] / P for j in range(D)])
    z   = X @ w
    p   = poly_sigmoid(z, degree)
    err = p - y
    grad_pt = (X.T @ err) / P
    ae = np.abs(grad_pt - grad_enc)
    return float(ae.max()), float(ae.mean())


# ── Full ablation run for one degree ─────────────────────────────────────────
def run_degree(X_train, y_train, X_test, y_test, degree, ctx):
    P, D = X_train.shape
    print(f"\n{'='*60}")
    print(f"  DEGREE-{degree} ({POLY[degree]['name']})  "
          f"expected_levels_remaining={_LEVELS_REMAINING[degree]}")
    print(f"{'='*60}")

    row = {
        "degree": degree,
        "name": POLY[degree]["name"],
        "poly_coeff": POLY[degree],
        "expected_levels_consumed": _TOTAL_CONSUMED[degree],
        "expected_levels_remaining": _LEVELS_REMAINING[degree],
    }

    # 1. Approximation error (offline, no HE)
    ae = approx_error(degree)
    row["approx_max_err"]  = ae["max"]
    row["approx_mean_err"] = ae["mean"]
    row["approx_interval"] = str(ae["interval"])
    print(f"  Polynomial approx error over {ae['interval']}:"
          f"  max={ae['max']:.4f}  mean={ae['mean']:.4f}")

    # 2. Encrypt training data (once, held fixed)
    t0 = time.perf_counter()
    enc_X = [ts.ckks_vector(ctx, X_train[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y_train.tolist())
    enc_time = time.perf_counter() - t0
    row["encryption_time_s"] = round(enc_time, 3)
    ct_bytes = sum(len(c.serialize()) for c in enc_X)
    row["serialized_X_ct_MB"] = round(ct_bytes / 1e6, 2)
    print(f"  Encryption time: {enc_time:.2f}s  |  X CT footprint: {ct_bytes/1e6:.2f} MB")

    # 3. Gradient validation at w=0
    w_zero = np.zeros(D)
    gmax, gmean = validate_gradient(X_train[:VAL_N], y_train[:VAL_N],
                                    w_zero, ctx, degree)
    row["grad_max_err"]  = gmax
    row["grad_mean_err"] = gmean
    print(f"  Gradient error (w=0, n={VAL_N}): max={gmax:.2e}  mean={gmean:.2e}")

    # 4. Encrypted training loop
    w = np.zeros(D)
    ep_times, train_accs = [], []
    final_level_err = final_level_grad = None
    ckks_failure = None

    t_total = time.perf_counter()
    for ep in range(1, EPOCHS+1):
        t_ep = time.perf_counter()
        try:
            enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
            enc_g, lev_err, lev_grad = csp_gradients(enc_X, enc_y, enc_w, P, D, degree)
            w = hospital_update(enc_g, w, P, D, LR, L2, GRAD_CLIP)
            final_level_err  = lev_err
            final_level_grad = lev_grad
        except Exception as e:
            ckks_failure = f"Epoch {ep}: {type(e).__name__}: {e}"
            print(f"  [CKKS FAILURE] {ckks_failure}")
            break

        ep_time = time.perf_counter() - t_ep
        ep_times.append(ep_time)
        z_tr = X_train @ w
        acc  = accuracy_score(y_train, (true_sigmoid(z_tr) >= 0.5).astype(int))
        train_accs.append(acc)
        print(f"  Epoch {ep:2d}/{EPOCHS}  ||w||={np.linalg.norm(w):.4f}  "
              f"train_acc={acc:.4f}  t={ep_time:.2f}s  lev={lev_grad}")

    total_time = time.perf_counter() - t_total

    row["ckks_failure"]         = ckks_failure
    row["total_train_time_s"]   = round(total_time, 3)
    row["mean_epoch_time_s"]    = round(float(np.mean(ep_times)), 3) if ep_times else None
    row["std_epoch_time_s"]     = round(float(np.std(ep_times)), 3)  if ep_times else None
    row["train_acc_epoch10"]    = float(train_accs[-1]) if train_accs else None
    row["final_levels_after_err"]  = final_level_err
    row["final_levels_after_grad"] = final_level_grad
    row["observed_levels_consumed"]= (7 - final_level_grad) if final_level_grad else None

    # 5. Test evaluation
    if ckks_failure is None:
        z_te  = X_test @ w
        p_te  = true_sigmoid(z_te)
        y_hat = (p_te >= 0.5).astype(int)
        row["test_accuracy"]  = float(accuracy_score(y_test, y_hat))
        row["test_precision"] = float(precision_score(y_test, y_hat, zero_division=0))
        row["test_recall"]    = float(recall_score(y_test, y_hat, zero_division=0))
        row["test_f1"]        = float(f1_score(y_test, y_hat, zero_division=0))
        row["test_roc_auc"]   = float(roc_auc_score(y_test, p_te))
        # Numerical prediction error vs true sigmoid
        z_val = np.linspace(-4, 4, 5000)
        pred_err = np.abs(true_sigmoid(z_val) - poly_sigmoid(z_val, degree))
        row["pred_max_err_on_4_4"] = float(pred_err.max())
        print(f"\n  Test: acc={row['test_accuracy']:.4f}  "
              f"AUC={row['test_roc_auc']:.4f}  "
              f"F1={row['test_f1']:.4f}")
    else:
        for k in ["test_accuracy","test_precision","test_recall",
                  "test_f1","test_roc_auc","pred_max_err_on_4_4"]:
            row[k] = None

    # Save per-degree JSON
    out = os.path.join(OUT_DIR, f"deg{degree}_result.json")
    with open(out, "w") as f:
        json.dump(row, f, indent=4)
    print(f"  Saved: {out}")
    return row


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(SEED)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(float)
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(float)

    print(f"MHEALTH Polynomial Ablation — Phase 1")
    print(f"Data: train={X_train.shape} test={X_test.shape}")
    print(f"Fixed: lr={LR} l2={L2} grad_clip={GRAD_CLIP} epochs={EPOCHS}")
    print(f"CKKS: [60,40,40,40,40,40,40,60] scale=2^40  usable_levels=7")

    # Compute actual training logit range (using degree-3 weights from baseline)
    # to ground the approximation error analysis in reality
    baseline_w = np.zeros(X_train.shape[1])  # start fresh, same as training
    z_sample = X_train @ baseline_w
    print(f"\nLogit range at w=0: [{z_sample.min():.3f}, {z_sample.max():.3f}]")

    print("\nBuilding CKKS context (shared across all degrees)...")
    ctx = build_context()
    print("Context ready.")

    all_rows = []
    for deg in [1, 3, 5]:
        row = run_degree(X_train, y_train, X_test, y_test, deg, ctx)
        all_rows.append(row)

    # Save master CSV
    csv_cols = [
        "degree", "name",
        "approx_max_err", "approx_mean_err", "approx_interval",
        "grad_max_err", "grad_mean_err",
        "expected_levels_consumed", "expected_levels_remaining",
        "observed_levels_consumed", "final_levels_after_grad",
        "encryption_time_s", "serialized_X_ct_MB",
        "mean_epoch_time_s", "std_epoch_time_s",
        "total_train_time_s",
        "train_acc_epoch10",
        "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_roc_auc",
        "pred_max_err_on_4_4", "ckks_failure"
    ]
    df = pd.DataFrame(all_rows)[csv_cols]
    csv_path = "results/mhealth/polynomial_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMaster CSV saved: {csv_path}")
    print("\n" + df.to_string(index=False))
