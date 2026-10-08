import os, json, time, math
import numpy as np
import pandas as pd
import tenseal as ts
from contextlib import contextmanager

# ── Paths & Hyperparameters ──────────────────────────────────────────────────
DATA_DIR  = "data/processed/mhealth"
OUT_DIR   = "results/mhealth/polynomial"
os.makedirs(OUT_DIR, exist_ok=True)

LR         = 0.05
L2         = 0.01
GRAD_CLIP  = 1.0
SEED       = 42
VAL_N      = 100

POLY = {
    1: {"c0": 0.5,  "c1": 0.197,     "c3": 0.0,       "c5": 0.0},
    3: {"c0": 0.5,  "c1": 0.15012,   "c3": -0.001593,  "c5": 0.0},
    5: {"c0": 0.5,  "c1": 0.217101,  "c3": -0.007823,  "c5": 0.000118},
}

_SIG_LEVELS = {1: 1, 3: 3, 5: 4}

class Profiler:
    def __init__(self):
        self.timers = {}
    
    @contextmanager
    def measure(self, name):
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        if name not in self.timers:
            self.timers[name] = 0.0
        self.timers[name] += (t1 - t0)
        
    def get_and_reset(self, name):
        val = self.timers.get(name, 0.0)
        self.timers[name] = 0.0
        return val

prof = Profiler()

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def burn(v, n):
    with prof.measure("alignment_burns"):
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

def poly_sigmoid(z, degree):
    p = POLY[degree]
    out = p["c0"] + p["c1"]*z
    if degree >= 3: out += p["c3"]*(z**3)
    if degree == 5: out += p["c5"]*(z**5)
    return out

def enc_sigmoid(enc_z, degree):
    p = POLY[degree]
    with prof.measure("poly_eval"):
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

def csp_gradients_profiled(enc_X, enc_y, enc_w, P, D, degree):
    depth_p = 1 + _SIG_LEVELS[degree]
    
    # 1. Forward linear combination
    with prof.measure("forward_linear"):
        enc_z = enc_X[0] * enc_w[0]
        for j in range(1, D):
            enc_z = enc_z + (enc_X[j] * enc_w[j])
            
    # 2. Polynomial eval
    enc_p = enc_sigmoid(enc_z, degree)
    
    # 3. Align & subtract error
    with prof.measure("error_computation"):
        enc_y_aligned = burn(enc_y, depth_p)
        enc_err = enc_p - enc_y_aligned
        level_err = get_levels(enc_err)
        
    grad_start_level = get_levels(enc_err)
        
    # 4. Gradient computation
    enc_grads = []
    with prof.measure("gradient_computation"):
        for j in range(D):
            enc_Xj = burn(enc_X[j], depth_p)
            enc_grads.append((enc_err * enc_Xj).sum())
            
    grad_end_level = get_levels(enc_grads[0])
            
    return enc_grads, level_err, grad_start_level, grad_end_level

def hospital_update_profiled(enc_grads, w, P, D, lr, l2, grad_clip):
    with prof.measure("decryption_update"):
        grad = np.array([enc_grads[j].decrypt()[0] / P for j in range(D)]) + l2 * w
        n = np.linalg.norm(grad)
        if n > grad_clip:
            grad *= grad_clip / n
        return w - lr * grad

def validate_gradient_detailed(X, y, w, ctx, degree):
    P, D = X.shape
    enc_X = [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y.tolist())
    enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
    
    enc_g, _, _, _ = csp_gradients_profiled(enc_X, enc_y, enc_w, P, D, degree)
    grad_enc = np.array([enc_g[j].decrypt()[0] / P for j in range(D)])
    
    z   = X @ w
    p   = poly_sigmoid(z, degree)
    err = p - y
    grad_pt = (X.T @ err) / P
    
    ae = np.abs(grad_pt - grad_enc)
    
    # Relative error where grad_pt is large enough
    mask = np.abs(grad_pt) > 1e-7
    re = np.abs((grad_pt[mask] - grad_enc[mask]) / grad_pt[mask])
    mean_re = float(re.mean()) if len(re) > 0 else 0.0
    
    return float(ae.max()), float(ae.mean()), mean_re

def run_profiling(X_train, y_train, ctx):
    P, D = X_train.shape
    enc_X = [ts.ckks_vector(ctx, X_train[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y_train.tolist())
    
    records = []
    
    # Gradient Validation points
    val_points = []
    val_points.append(("w_zero", np.zeros(D)))
    
    np.random.seed(SEED)
    for i in range(5):
        val_points.append((f"w_rand_{i+1}", np.random.uniform(-0.1, 0.1, D)))
        
    print("\n--- Generating intermediate weights via standard training (deg=3) ---")
    w_inter = np.zeros(D)
    for ep in range(1, 9):
        # fast plaintext simulation for intermediate weights
        grad_pt = (X_train.T @ (poly_sigmoid(X_train @ w_inter, 3) - y_train)) / P + L2 * w_inter
        n = np.linalg.norm(grad_pt)
        if n > GRAD_CLIP: grad_pt *= GRAD_CLIP / n
        w_inter = w_inter - LR * grad_pt
        if ep == 2: val_points.append(("w_epoch_2", w_inter.copy()))
        if ep == 8: val_points.append(("w_epoch_8", w_inter.copy()))

    print("\n================== GRADIENT VALIDATION ==================")
    for deg in [1, 3, 5]:
        print(f"\nDegree {deg}:")
        for tag, w_val in val_points:
            gmax, gmean, g_re = validate_gradient_detailed(X_train[:VAL_N], y_train[:VAL_N], w_val, ctx, deg)
            print(f"  {tag:10s} : MaxAE={gmax:.2e}  MeanAE={gmean:.2e}  MeanRE={g_re:.2e}")
            
    print("\n================== STAGE-WISE PROFILING ==================")
    for deg in [1, 3, 5]:
        print(f"\nProfiling Degree {deg} ...")
        
        reps = 3
        totals = {"forward_linear": 0, "poly_eval": 0, "alignment_burns": 0,
                  "error_computation": 0, "gradient_computation": 0, "decryption_update": 0}
        
        grad_start_lvl = None
        grad_end_lvl = None
        
        w = np.zeros(D)
        for r in range(reps):
            prof.timers.clear()
            enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
            
            enc_g, lev_err, gsl, gel = csp_gradients_profiled(enc_X, enc_y, enc_w, P, D, deg)
            w = hospital_update_profiled(enc_g, w, P, D, LR, L2, GRAD_CLIP)
            
            grad_start_lvl = gsl
            grad_end_lvl = gel
            
            for k in totals:
                totals[k] += prof.timers.get(k, 0.0)
                
        # Average
        for k in totals:
            totals[k] /= reps
            
        total_time = sum(totals.values())
        
        row = {
            "degree": deg,
            "forward_time": round(totals["forward_linear"], 3),
            "polynomial_time": round(totals["poly_eval"], 3),
            "alignment_time": round(totals["alignment_burns"], 3),
            "error_time": round(totals["error_computation"], 3),
            "gradient_time": round(totals["gradient_computation"], 3),
            "decrypt_time": round(totals["decryption_update"], 3),
            "total_time": round(total_time, 3),
            "gradient_start_level": grad_start_lvl,
            "gradient_end_level": grad_end_lvl,
            "final_level": grad_end_lvl
        }
        records.append(row)
        
        print(f"  Forward XxW   : {row['forward_time']:.3f} s")
        print(f"  Polynomial    : {row['polynomial_time']:.3f} s")
        print(f"  Align (burns) : {row['alignment_time']:.3f} s")
        print(f"  Error (p-y)   : {row['error_time']:.3f} s")
        print(f"  Gradient      : {row['gradient_time']:.3f} s")
        print(f"  Decrypt       : {row['decrypt_time']:.3f} s")
        print(f"  TOTAL         : {row['total_time']:.3f} s")
        print(f"  Grad lvl      : {grad_start_lvl} -> {grad_end_lvl}")
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "runtime_profile.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved profile to {csv_path}")

if __name__ == "__main__":
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(float)
    
    # ── Check SIMD fit ──
    SLOTS = 8192
    P, D = X_train.shape
    if P > SLOTS:
        X_train = X_train[:SLOTS]
        y_train = y_train[:SLOTS]
        P = SLOTS
        
    ctx = build_context()
    run_profiling(X_train, y_train, ctx)
