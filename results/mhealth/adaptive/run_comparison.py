"""
Phase 2B — Continuation: Fixed vs Adaptive Training Comparison (Experiments 2 & 3)
Experiments 1 and 4 already completed and are saved. This script ONLY runs the
training comparison loop with 3 deterministic repetitions.
Writes to results/mhealth/adaptive/adaptive_comparison.csv
Does NOT overwrite selection_decision.json or threshold_sensitivity.csv.
"""

import os, json, time
import numpy as np
import pandas as pd
import tenseal as ts
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/processed/mhealth"
OUT_DIR  = "results/mhealth/adaptive"

# ── Hyperparameters (frozen) ─────────────────────────────────────────────────
LR        = 0.05
L2        = 0.01
GRAD_CLIP = 1.0
EPOCHS    = 10
SEED      = 42
ADAPTIVE_DEGREE = 5   # from selection_decision.json

POLY = {
    1: {"c0": 0.5,  "c1": 0.197,     "c3": 0.0,      "c5": 0.0},
    3: {"c0": 0.5,  "c1": 0.15012,   "c3": -0.001593, "c5": 0.0},
    5: {"c0": 0.5,  "c1": 0.217101,  "c3": -0.007823, "c5": 0.000118},
}
_SIG_LEVELS = {1: 1, 3: 3, 5: 4}

# ── Helpers ──────────────────────────────────────────────────────────────────
def true_sigmoid(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))

def poly_sigmoid(z, d):
    p = POLY[d]
    out = p["c0"] + p["c1"]*z
    if d >= 3: out += p["c3"]*(z**3)
    if d == 5: out += p["c5"]*(z**5)
    return out

def burn(v, n):
    for _ in range(n): v = v * 1.0
    return v

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def enc_sigmoid(enc_z, d):
    p = POLY[d]
    if d == 1:
        return enc_z * p["c1"] + p["c0"]
    if d == 3:
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        return z3*p["c3"] + burn(enc_z, 2)*p["c1"] + p["c0"]
    if d == 5:
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        z5 = z3 * burn(z2, 1)
        return z5*p["c5"] + burn(z3, 1)*p["c3"] + burn(enc_z, 3)*p["c1"] + p["c0"]

def csp_gradients(enc_X, enc_y, enc_w, P, D, d):
    dp = 1 + _SIG_LEVELS[d]
    enc_z = enc_X[0] * enc_w[0]
    for j in range(1, D):
        enc_z = enc_z + enc_X[j] * enc_w[j]
    enc_p = enc_sigmoid(enc_z, d)
    enc_err = enc_p - burn(enc_y, dp)
    enc_grads = [(enc_err * burn(enc_X[j], dp)).sum() for j in range(D)]
    return enc_grads, get_levels(enc_grads[0])

def hospital_update(enc_grads, w, P, D):
    grad = np.array([g.decrypt()[0]/P for g in enc_grads]) + L2*w
    n = np.linalg.norm(grad)
    if n > GRAD_CLIP: grad *= GRAD_CLIP/n
    return w - LR*grad

def build_context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384,
                     coeff_mod_bit_sizes=[60,40,40,40,40,40,40,60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

def run_training(X_tr, y_tr, X_te, y_te, degree, ctx):
    P, D = X_tr.shape
    enc_X = [ts.ckks_vector(ctx, X_tr[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y_tr.tolist())
    w = np.zeros(D)
    ep_times = []
    t_start = time.perf_counter()

    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
        enc_g, l_grad = csp_gradients(enc_X, enc_y, enc_w, P, D, degree)
        w = hospital_update(enc_g, w, P, D)
        ep_times.append(time.perf_counter() - t0)
        print(f"    ep {ep+1}/{EPOCHS} acc={accuracy_score(y_tr,(true_sigmoid(X_tr@w)>=0.5).astype(int)):.4f} t={ep_times[-1]:.1f}s lev={l_grad}")

    total = time.perf_counter() - t_start
    z_te = X_te @ w
    p_te = true_sigmoid(z_te)
    acc_te = accuracy_score(y_te, (p_te>=0.5).astype(int))
    auc_te = roc_auc_score(y_te, p_te)
    acc_tr = accuracy_score(y_tr, (true_sigmoid(X_tr@w)>=0.5).astype(int))

    # gradient error at final w
    enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
    enc_g, _ = csp_gradients(enc_X, enc_y, enc_w, P, D, degree)
    grad_enc = np.array([g.decrypt()[0]/P for g in enc_g])
    grad_pt  = (X_tr.T @ (poly_sigmoid(X_tr@w, degree) - y_tr)) / P
    g_err = float(np.abs(grad_pt - grad_enc).max())

    z_val = np.linspace(-4, 4, 1000)
    pred_err = float(np.abs(true_sigmoid(z_val) - poly_sigmoid(z_val, degree)).max())

    return {
        "degree": degree,
        "mean_epoch_time": float(np.mean(ep_times)),
        "std_epoch_time":  float(np.std(ep_times)),
        "total_time":      float(total),
        "train_acc":       float(acc_tr),
        "test_acc":        float(acc_te),
        "test_auc":        float(auc_te),
        "final_level":     int(l_grad),
        "pred_max_err":    float(pred_err),
        "grad_max_err":    float(g_err),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(float)
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(float)
    P, D = X_train.shape
    if P > 8192: X_train, y_train, P = X_train[:8192], y_train[:8192], 8192

    print(f"Data: train={X_train.shape}  test={X_test.shape}")
    print(f"Building CKKS context...")
    ctx = build_context()
    print("Context ready.\n")

    reps   = 3
    degrees_to_test = [1, 3, 5]
    all_res = []

    for r in range(reps):
        np.random.seed(SEED + r)
        print(f"\n{'='*60}")
        print(f"  REPETITION {r+1}/{reps}")
        print(f"{'='*60}")

        # Adaptive first (selected degree from Experiment 1)
        print(f"\n  [Adaptive — Degree {ADAPTIVE_DEGREE}]")
        res = run_training(X_train, y_train, X_test, y_test, ADAPTIVE_DEGREE, ctx)
        res["method"] = "adaptive"
        res["rep"] = r + 1
        all_res.append(res)
        print(f"  -> test_acc={res['test_acc']:.4f}  auc={res['test_auc']:.4f}  total={res['total_time']:.1f}s")

        # Fixed degrees
        for d in degrees_to_test:
            print(f"\n  [Fixed Degree {d}]")
            res = run_training(X_train, y_train, X_test, y_test, d, ctx)
            res["method"] = f"fixed_{d}"
            res["rep"] = r + 1
            all_res.append(res)
            print(f"  -> test_acc={res['test_acc']:.4f}  auc={res['test_auc']:.4f}  total={res['total_time']:.1f}s")

    df = pd.DataFrame(all_res)
    out_csv = os.path.join(OUT_DIR, "adaptive_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\n\n=== SUMMARY (Mean ± Std over 3 reps) ===")
    mean_df = df.groupby("method")[["mean_epoch_time","total_time","train_acc","test_acc","test_auc"]].agg(["mean","std"])
    mean_df.columns = ["_".join(c) for c in mean_df.columns]
    print(mean_df.to_string())
