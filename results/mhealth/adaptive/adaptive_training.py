import os, json, time, math
import numpy as np
import pandas as pd
import tenseal as ts
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data/processed/mhealth"
OUT_DIR   = "results/mhealth/adaptive"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
LR         = 0.05
L2         = 0.01
GRAD_CLIP  = 1.0
EPOCHS     = 10
SEED       = 42

POLY = {
    1: {"c0": 0.5,  "c1": 0.197,     "c3": 0.0,       "c5": 0.0},
    3: {"c0": 0.5,  "c1": 0.15012,   "c3": -0.001593,  "c5": 0.0},
    5: {"c0": 0.5,  "c1": 0.217101,  "c3": -0.007823,  "c5": 0.000118},
}
_SIG_LEVELS = {1: 1, 3: 3, 5: 4}

# ── HE Functions ─────────────────────────────────────────────────────────────
def true_sigmoid(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))

def poly_sigmoid(z, degree):
    p = POLY[degree]
    out = p["c0"] + p["c1"]*z
    if degree >= 3: out += p["c3"]*(z**3)
    if degree == 5: out += p["c5"]*(z**5)
    return out

def burn(v, n):
    for _ in range(n): v = v * 1.0
    return v

def enc_sigmoid(enc_z, degree):
    p = POLY[degree]
    if degree == 1:
        return enc_z * p["c1"] + p["c0"]
    if degree == 3:
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        return z3*p["c3"] + burn(enc_z, 2)*p["c1"] + p["c0"]
    if degree == 5:
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        z5 = z3 * burn(z2, 1)
        return z5*p["c5"] + burn(z3, 1)*p["c3"] + burn(enc_z, 3)*p["c1"] + p["c0"]

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def csp_gradients(enc_X, enc_y, enc_w, P, D, degree):
    depth_p = 1 + _SIG_LEVELS[degree]
    enc_z = enc_X[0] * enc_w[0]
    for j in range(1, D):
        enc_z = enc_z + enc_X[j] * enc_w[j]
    enc_p = enc_sigmoid(enc_z, degree)
    enc_err = enc_p - burn(enc_y, depth_p)
    enc_grads = [(enc_err * burn(enc_X[j], depth_p)).sum() for j in range(D)]
    return enc_grads, get_levels(enc_err), get_levels(enc_grads[0])

def hospital_update(enc_grads, w, P, D, lr, l2, grad_clip):
    grad = np.array([enc_g.decrypt()[0] / P for enc_g in enc_grads]) + l2 * w
    n = np.linalg.norm(grad)
    if n > grad_clip: grad *= grad_clip / n
    return w - lr * grad

def build_context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384,
                     coeff_mod_bit_sizes=[60,40,40,40,40,40,40,60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

# ── Adaptive Selector ────────────────────────────────────────────────────────
# Cache for calibration results to save time across multiple calls
_CALIB_CACHE = {}

def select_polynomial_degree(
    X_calib, y_calib, ctx,
    logit_range=(-6.0, 6.0),
    candidate_degrees=[1, 3, 5],
    epsilon_poly=0.05,
    epsilon_grad=1e-4,
    available_depth=7
):
    """
    Depth-and-Cost-Aware Adaptive Polynomial Training Selector.
    Minimizes expected HE time subject to polynomial error, gradient error, and depth constraints.
    """
    P, D = X_calib.shape
    z_eval = np.linspace(logit_range[0], logit_range[1], 2000)
    true_sig = true_sigmoid(z_eval)
    
    print(f"\n--- Running Selector (epsilon_poly={epsilon_poly}) ---")
    enc_X = [ts.ckks_vector(ctx, X_calib[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y_calib.tolist())
    w_zero = np.zeros(D)
    enc_w = [ts.ckks_vector(ctx, [0.0]*P) for j in range(D)]
    
    candidates_info = {}
    feasible = []
    
    for deg in sorted(candidate_degrees):
        print(f"  Calibrating Degree {deg}...")
        
        # 1. Polynomial approx error
        poly_sig = poly_sigmoid(z_eval, deg)
        e_poly = float(np.abs(true_sig - poly_sig).max())
        
        # 2. Depth required
        d_circuit = 1 + _SIG_LEVELS[deg] + 1
        
        # 3. Empirical Gradient Error & Time (Cached)
        if deg not in _CALIB_CACHE:
            t0 = time.perf_counter()
            enc_g, l_err, l_grad = csp_gradients(enc_X, enc_y, enc_w, P, D, deg)
            t_he = time.perf_counter() - t0
            
            grad_enc = np.array([g.decrypt()[0] / P for g in enc_g])
            grad_pt = (X_calib.T @ (poly_sigmoid(X_calib @ w_zero, deg) - y_calib)) / P
            e_grad = float(np.abs(grad_pt - grad_enc).max())
            
            _CALIB_CACHE[deg] = {"t_he": t_he, "e_grad": e_grad, "l_err": l_err}
            
        t_he = _CALIB_CACHE[deg]["t_he"]
        e_grad = _CALIB_CACHE[deg]["e_grad"]
        l_err = _CALIB_CACHE[deg]["l_err"]
        
        info = {
            "degree": deg,
            "e_poly": e_poly,
            "e_grad": e_grad,
            "d_circuit": d_circuit,
            "t_he": t_he,
            "grad_start_level": l_err,
            "fits_poly": bool(e_poly <= epsilon_poly),
            "fits_grad": bool(e_grad <= epsilon_grad),
            "fits_depth": bool(d_circuit < available_depth) # strict inequality to leave 1 level
        }
        candidates_info[deg] = info
        
        if info["fits_poly"] and info["fits_grad"] and info["fits_depth"]:
            feasible.append(deg)
            
    # Selection: argmin T_HE among feasible
    if feasible:
        selected = min(feasible, key=lambda d: candidates_info[d]["t_he"])
        reason = f"Degree {selected} minimizes T_HE ({candidates_info[selected]['t_he']:.2f}s) while satisfying all constraints."
    else:
        # Fallback if constraints too strict
        print("  WARNING: No candidate satisfied all constraints. Falling back to minimum depth.")
        selected = min(candidate_degrees, key=lambda d: candidates_info[d]["d_circuit"])
        reason = f"Fallback to degree {selected} due to constraint failure."
        
    decision = {
        "selected_degree": selected,
        "reason": reason,
        "candidates": candidates_info,
        "constraints": {
            "epsilon_poly": epsilon_poly,
            "epsilon_grad": epsilon_grad,
            "available_depth": available_depth
        }
    }
    return decision

# ── Experiment Runner ────────────────────────────────────────────────────────
def run_training(X_tr, y_tr, X_te, y_te, degree, ctx, epochs):
    P, D = X_tr.shape
    enc_X = [ts.ckks_vector(ctx, X_tr[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y_tr.tolist())
    
    w = np.zeros(D)
    ep_times = []
    t_start = time.perf_counter()
    
    for ep in range(epochs):
        t0 = time.perf_counter()
        enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
        enc_g, _, l_grad = csp_gradients(enc_X, enc_y, enc_w, P, D, degree)
        w = hospital_update(enc_g, w, P, D, LR, L2, GRAD_CLIP)
        ep_times.append(time.perf_counter() - t0)
        
    t_total = time.perf_counter() - t_start
    
    z_tr = X_tr @ w
    acc_tr = accuracy_score(y_tr, (true_sigmoid(z_tr) >= 0.5).astype(int))
    
    z_te = X_te @ w
    p_te = true_sigmoid(z_te)
    y_hat = (p_te >= 0.5).astype(int)
    acc_te = accuracy_score(y_te, y_hat)
    auc_te = roc_auc_score(y_te, p_te)
    
    z_val = np.linspace(-4, 4, 1000)
    pred_err = float(np.abs(true_sigmoid(z_val) - poly_sigmoid(z_val, degree)).max())
    
    grad_pt = (X_tr.T @ (poly_sigmoid(X_tr @ w, degree) - y_tr)) / P
    enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
    enc_g, _, _ = csp_gradients(enc_X, enc_y, enc_w, P, D, degree)
    grad_enc = np.array([g.decrypt()[0] / P for g in enc_g])
    g_err = float(np.abs(grad_pt - grad_enc).max())
    
    return {
        "degree": degree,
        "mean_epoch_time": np.mean(ep_times),
        "total_time": t_total,
        "train_acc": acc_tr,
        "test_acc": acc_te,
        "test_auc": auc_te,
        "final_level": l_grad,
        "pred_max_err": pred_err,
        "grad_max_err": g_err
    }

if __name__ == "__main__":
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(float)
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(float)

    P, D = X_train.shape
    if P > 8192: X_train, y_train, P = X_train[:8192], y_train[:8192], 8192
    
    ctx = build_context()
    
    # --- Experiment 1: Selector Behavior ---
    print("\n[EXPERIMENT 1: SELECTOR BEHAVIOR]")
    decision = select_polynomial_degree(X_train, y_train, ctx, epsilon_poly=0.05)
    with open(os.path.join(OUT_DIR, "selection_decision.json"), "w") as f:
        json.dump(decision, f, indent=4)
    print(json.dumps(decision, indent=2))
    
    # --- Experiment 4: Threshold Sensitivity ---
    print("\n[EXPERIMENT 4: THRESHOLD SENSITIVITY]")
    sens_results = []
    for eps in [0.20, 0.10, 0.05, 0.03]:
        dec = select_polynomial_degree(X_train, y_train, ctx, epsilon_poly=eps)
        sens_results.append({
            "epsilon_poly": eps,
            "selected_degree": dec["selected_degree"],
            "reason": dec["reason"]
        })
    pd.DataFrame(sens_results).to_csv(os.path.join(OUT_DIR, "threshold_sensitivity.csv"), index=False)
    
    # --- Experiment 2 & 3: Fixed vs Adaptive (3 Reps) ---
    print("\n[EXPERIMENT 2 & 3: FIXED VS ADAPTIVE (3 Reps)]")
    reps = 3
    all_res = []
    
    for r in range(reps):
        np.random.seed(SEED + r)
        print(f"\n--- Repetition {r+1}/{reps} ---")
        
        # Adaptive run
        adapt_deg = decision["selected_degree"]
        print(f"  Running Adaptive (Degree {adapt_deg})...")
        res_ad = run_training(X_train, y_train, X_test, y_test, adapt_deg, ctx, EPOCHS)
        res_ad["method"] = "adaptive"
        res_ad["rep"] = r
        all_res.append(res_ad)
        
        # Fixed runs
        for d in [1, 3, 5]:
            print(f"  Running Fixed Degree {d}...")
            res_fix = run_training(X_train, y_train, X_test, y_test, d, ctx, EPOCHS)
            res_fix["method"] = f"fixed_{d}"
            res_fix["rep"] = r
            all_res.append(res_fix)
            
    df = pd.DataFrame(all_res)
    df.to_csv(os.path.join(OUT_DIR, "adaptive_comparison.csv"), index=False)
    
    print("\n--- Summary (Mean over 3 reps) ---")
    summary = df.groupby("method").mean(numeric_only=True)[
        ["degree", "mean_epoch_time", "total_time", "train_acc", "test_acc", "test_auc"]
    ]
    print(summary.to_string())
    print("\nAdaptive Training Experiments Complete!")
