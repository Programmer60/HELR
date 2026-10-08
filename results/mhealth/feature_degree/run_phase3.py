import os, json, time
import numpy as np
import pandas as pd
import tenseal as ts
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from contextlib import contextmanager

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/processed/mhealth"
OUT_DIR  = "results/mhealth/feature_degree"

# ── Hyperparameters ─────────────────────────────────────────────────────────
LR        = 0.05
L2        = 0.01
GRAD_CLIP = 1.0
EPOCHS    = 10
SEED      = 42

POLY = {
    1: {"c0": 0.5,  "c1": 0.197,     "c3": 0.0,      "c5": 0.0},
    3: {"c0": 0.5,  "c1": 0.15012,   "c3": -0.001593, "c5": 0.0},
    5: {"c0": 0.5,  "c1": 0.217101,  "c3": -0.007823, "c5": 0.000118},
}
_SIG_LEVELS = {1: 1, 3: 3, 5: 4}

# ── Profiler ─────────────────────────────────────────────────────────────────
class Profiler:
    def __init__(self):
        self.timers = {}
    @contextmanager
    def measure(self, name):
        t0 = time.perf_counter()
        yield
        self.timers[name] = self.timers.get(name, 0.0) + (time.perf_counter() - t0)

prof = Profiler()

# ── HE Helpers ───────────────────────────────────────────────────────────────
def true_sigmoid(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))

def poly_sigmoid(z, d):
    p = POLY[d]
    out = p["c0"] + p["c1"]*z
    if d >= 3: out += p["c3"]*(z**3)
    if d == 5: out += p["c5"]*(z**5)
    return out

def burn(v, n):
    with prof.measure("alignment"):
        for _ in range(n): v = v * 1.0
    return v

def enc_sigmoid(enc_z, d):
    p = POLY[d]
    with prof.measure("poly_eval"):
        if d == 1: return enc_z * p["c1"] + p["c0"]
        if d == 3:
            z2 = enc_z * enc_z
            z3 = z2 * burn(enc_z, 1)
            t3 = z3*p["c3"]
            t1 = burn(enc_z, 2)*p["c1"]
            return t3 + t1 + p["c0"]
        if d == 5:
            z2 = enc_z * enc_z
            z3 = z2 * burn(enc_z, 1)
            z5 = z3 * burn(z2, 1)
            t5 = z5*p["c5"]
            t3 = burn(z3, 1)*p["c3"]
            t1 = burn(enc_z, 3)*p["c1"]
            return t5 + t3 + t1 + p["c0"]

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def csp_gradients_profiled(enc_X, enc_y, enc_w, P, D, d):
    dp = 1 + _SIG_LEVELS[d]
    with prof.measure("forward"):
        enc_z = enc_X[0] * enc_w[0]
        for j in range(1, D):
            enc_z = enc_z + (enc_X[j] * enc_w[j])
            
    enc_p = enc_sigmoid(enc_z, d)
    
    with prof.measure("error"):
        enc_y_aligned = burn(enc_y, dp)
        enc_err = enc_p - enc_y_aligned
        l_err = get_levels(enc_err)
        
    enc_grads = []
    with prof.measure("gradient"):
        for j in range(D):
            enc_Xj = burn(enc_X[j], dp)
            enc_grads.append((enc_err * enc_Xj).sum())
            
    return enc_grads, l_err, get_levels(enc_grads[0])

def hospital_update_profiled(enc_grads, w, P, D):
    with prof.measure("decrypt_update"):
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

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(SEED)
    
    print("Loading data...")
    X_train_full = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_full = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(float)
    X_test_full  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_full  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(float)
    subj_train   = np.load(os.path.join(DATA_DIR, "subject_train.npy"))
    
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    feature_names = np.array(meta["feature_names"])
    
    # Internal split: Train=1-5, Val=6-7
    train_mask = (subj_train >= 1) & (subj_train <= 5)
    val_mask   = (subj_train >= 6) & (subj_train <= 7)
    
    X_tr = X_train_full[train_mask]
    y_tr = y_train_full[train_mask]
    X_va = X_train_full[val_mask]
    y_va = y_train_full[val_mask]
    
    print(f"Split sizes -> Train (1-5): {X_tr.shape}, Val (6-7): {X_va.shape}, Test (8-10): {X_test_full.shape}")
    
    # ── 1. Feature Configurations ──
    print("\n[PERFORMING FEATURE SELECTION]")
    configs = [92, 48, 24, 12]
    feature_sets = {}
    
    for D in configs:
        if D == 92:
            idx = np.arange(92)
        else:
            selector = SelectKBest(f_classif, k=D)
            selector.fit(X_tr, y_tr)
            idx = selector.get_support(indices=True)
            
        feature_sets[str(D)] = feature_names[idx].tolist()
        
    with open(os.path.join(OUT_DIR, "feature_sets.json"), "w") as f:
        json.dump(feature_sets, f, indent=4)
        
    # ── 2. Ablation and Profiling ──
    print("\n[BUILDING CKKS CONTEXT]")
    ctx = build_context()
    
    results = []
    adaptive_results = []
    calib_overheads = []
    
    EPSILON_POLY = 0.05
    EPSILON_GRAD = 1e-4
    DEPTH_BUDGET = 7
    
    for D in configs:
        print(f"\n{'='*60}")
        print(f"  FEATURE DIMENSION: {D}")
        print(f"{'='*60}")
        
        # Get subset data
        idx = [feature_names.tolist().index(f) for f in feature_sets[str(D)]]
        X_tr_D = X_tr[:, idx]
        X_va_D = X_va[:, idx]
        X_te_D = X_test_full[:, idx]
        P = X_tr_D.shape[0]
        
        # Encrypt datasets once per D
        t0 = time.perf_counter()
        enc_X = [ts.ckks_vector(ctx, X_tr_D[:, j].tolist()) for j in range(D)]
        enc_y = ts.ckks_vector(ctx, y_tr.tolist())
        enc_time = time.perf_counter() - t0
        
        calib_stats = {}
        
        for deg in [1, 3, 5]:
            print(f"\n  [D={D} | Degree={deg}]")
            prof.timers.clear()
            
            # --- CALIBRATION STAGE ---
            # Measure time and accuracy of 1 gradient pass at w=0
            t_calib_start = time.perf_counter()
            w_zero = np.zeros(D)
            enc_w_zero = [ts.ckks_vector(ctx, [0.0]*P) for _ in range(D)]
            enc_g, l_err, l_grad = csp_gradients_profiled(enc_X, enc_y, enc_w_zero, P, D, deg)
            t_calib = time.perf_counter() - t_calib_start
            
            # Validate numerical gradient
            grad_enc = np.array([g.decrypt()[0]/P for g in enc_g])
            grad_pt  = (X_tr_D.T @ (poly_sigmoid(X_tr_D @ w_zero, deg) - y_tr)) / P
            e_grad = float(np.abs(grad_pt - grad_enc).max())
            
            z_eval = np.linspace(-6.0, 6.0, 2000)
            e_poly = float(np.abs(true_sigmoid(z_eval) - poly_sigmoid(z_eval, deg)).max())
            d_circuit = 1 + _SIG_LEVELS[deg] + 1
            
            calib_stats[deg] = {
                "t_calib": t_calib,
                "e_grad": e_grad,
                "e_poly": e_poly,
                "d_circuit": d_circuit,
                "l_err": l_err,
                "l_grad": l_grad,
                "fits": (e_poly <= EPSILON_POLY and e_grad <= EPSILON_GRAD and d_circuit < DEPTH_BUDGET)
            }
            
            # --- TRAINING STAGE ---
            t_train_start = time.perf_counter()
            w = np.zeros(D)
            for ep in range(EPOCHS):
                enc_w = [ts.ckks_vector(ctx, [float(w[j])]*P) for j in range(D)]
                enc_g, _, _ = csp_gradients_profiled(enc_X, enc_y, enc_w, P, D, deg)
                w = hospital_update_profiled(enc_g, w, P, D)
            t_train = time.perf_counter() - t_train_start
            
            # --- EVALUATION ---
            acc_tr = accuracy_score(y_tr, (true_sigmoid(X_tr_D@w)>=0.5).astype(int))
            p_va = true_sigmoid(X_va_D@w)
            acc_va = accuracy_score(y_va, (p_va>=0.5).astype(int))
            auc_va = roc_auc_score(y_va, p_va)
            
            p_te = true_sigmoid(X_te_D@w)
            acc_te = accuracy_score(y_test_full, (p_te>=0.5).astype(int))
            auc_te = roc_auc_score(y_test_full, p_te)
            
            avg_prof = {k: v/(EPOCHS+1) for k,v in prof.timers.items()} # +1 for calib pass
            
            res_row = {
                "D": D,
                "degree": deg,
                "approx_max_err": e_poly,
                "grad_max_err": e_grad,
                "forward_time": avg_prof.get("forward", 0),
                "poly_time": avg_prof.get("poly_eval", 0),
                "align_time": avg_prof.get("alignment", 0),
                "error_time": avg_prof.get("error", 0),
                "gradient_time": avg_prof.get("gradient", 0),
                "decrypt_time": avg_prof.get("decrypt_update", 0),
                "epoch_runtime": t_train / EPOCHS,
                "total_train_runtime": t_train,
                "total_calib_runtime": t_calib,
                "circuit_levels_consumed": d_circuit,
                "grad_start_level": l_err,
                "final_levels_remaining": l_grad,
                "train_acc": acc_tr,
                "val_acc": acc_va,
                "val_auc": auc_va,
                "test_acc": acc_te,
                "test_auc": auc_te
            }
            results.append(res_row)
            print(f"    Train: {t_train:.1f}s | Val AUC: {auc_va:.4f} | Grad Time: {res_row['gradient_time']:.1f}s")
            
        # --- ADAPTIVE SELECTION ---
        feasible = [d for d, s in calib_stats.items() if s["fits"]]
        if feasible:
            selected = min(feasible, key=lambda d: calib_stats[d]["t_calib"])
            status = "SUCCESS"
        else:
            selected = -1
            status = "INFEASIBLE"
            
        print(f"\n  [Adaptive Selection for D={D}] -> Degree {selected} ({status})")
        
        if status == "SUCCESS":
            sel_row = next(r for r in results if r["D"] == D and r["degree"] == selected)
            calib_overheads.append({
                "D": D,
                "selected_degree": selected,
                "T_calibration": sum(s["t_calib"] for s in calib_stats.values()),
                "T_training": sel_row["total_train_runtime"],
                "T_total": sum(s["t_calib"] for s in calib_stats.values()) + sel_row["total_train_runtime"]
            })
            adaptive_results.append({
                "D": D,
                "method": "adaptive",
                "selected_degree": selected,
                "val_auc": sel_row["val_auc"],
                "val_acc": sel_row["val_acc"],
                "test_auc": sel_row["test_auc"],
                "test_acc": sel_row["test_acc"],
                "total_HE_cost": sel_row["total_train_runtime"] + sum(s["t_calib"] for s in calib_stats.values())
            })
            
    # --- SAVE OUTPUTS ---
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "feature_degree_results.csv"), index=False)
    
    stage_cols = ["D", "degree", "forward_time", "poly_time", "align_time", "error_time", "gradient_time", "decrypt_time"]
    pd.DataFrame(results)[stage_cols].to_csv(os.path.join(OUT_DIR, "stage_profile.csv"), index=False)
    
    pd.DataFrame(adaptive_results).to_csv(os.path.join(OUT_DIR, "adaptive_feature_degree.csv"), index=False)
    pd.DataFrame(calib_overheads).to_csv(os.path.join(OUT_DIR, "calibration_overhead.csv"), index=False)
    
    print("\nDone! Results saved.")
