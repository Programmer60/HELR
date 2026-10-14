"""
Phase 4: Controlled Validation of Modulus-Level-Aware Circuit Scheduling in CKKS
================================================================================
Investigates whether shifting a Degree-3 computation to enter the gradient stage
at CKKS Level 2 (the same modulus level as Degree 5) causally eliminates the
runtime gap between Degree 3 and Degree 5 gradient evaluation.

Three conditions:
  Condition A: Original Degree-3 circuit (gradient at Level 3) [Control]
  Condition B: Degree-3 with controlled level adjustment to Level 2 [Intervention]
  Condition C: Original Degree-5 circuit (gradient at Level 2) [Control]
"""

import os
import json
import time
import math
import numpy as np
import pandas as pd
import tenseal as ts
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ── Paths & Configuration ────────────────────────────────────────────────────
DATA_DIR = "data/processed/mhealth"
OUT_DIR = "results/mhealth/phase4"
os.makedirs(OUT_DIR, exist_ok=True)

LR = 0.05
L2 = 0.01
GRAD_CLIP = 1.0
EPOCHS = 10
SEED = 42
VAL_N = 100  # sample count for detailed numerical validation batches

POLY = {
    1: {"c0": 0.5, "c1": 0.197, "c3": 0.0, "c5": 0.0},
    3: {"c0": 0.5, "c1": 0.15012, "c3": -0.001593, "c5": 0.0},
    5: {"c0": 0.5, "c1": 0.217101, "c3": -0.007823, "c5": 0.000118},
}

_SIG_LEVELS = {1: 1, 3: 3, 5: 4}

# ── High-Precision Profiler ──────────────────────────────────────────────────
class Profiler:
    def __init__(self):
        self.timers = {}

    @contextmanager
    def measure(self, name):
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        self.timers[name] = self.timers.get(name, 0.0) + (t1 - t0)

    def get(self, name):
        return self.timers.get(name, 0.0)

    def clear(self):
        self.timers.clear()

prof = Profiler()

# ── CKKS Utilities ───────────────────────────────────────────────────────────
def build_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

def get_levels(enc):
    """Directly returns SEAL Ciphertext::coeff_modulus_size() via pybind11."""
    return enc.ciphertext()[0].coeff_modulus_size()

def get_parms_id(enc):
    """Directly returns SEAL parms_id (128-bit hash) for modulus level verification."""
    return list(enc.ciphertext()[0].parms_id())

def get_scale(enc):
    """Directly returns SEAL ciphertext scale."""
    return float(enc.ciphertext()[0].scale)

def burn(v, n):
    """Performs n plaintext multiplications by 1.0 to drop n modulus levels."""
    for _ in range(n):
        v = v * 1.0
    return v

# ── Mathematical Functions ───────────────────────────────────────────────────
def true_sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def poly_sigmoid(z, degree):
    p = POLY[degree]
    out = p["c0"] + p["c1"] * z
    if degree >= 3:
        out = out + p["c3"] * (z**3)
    if degree == 5:
        out = out + p["c5"] * (z**5)
    return out

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
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        z5 = z3 * burn(z2, 1)
        t5 = z5 * p["c5"]
        t3 = burn(z3, 1) * p["c3"]
        t1 = burn(enc_z, 3) * p["c1"]
        return t5 + t3 + t1 + p["c0"]
    raise ValueError(f"Unsupported degree: {degree}")

# ── Clean Condition-Specific CSP Pipeline ─────────────────────────────────────
def csp_step_profiled(enc_X, enc_y, enc_w, P, D, condition):
    """
    Executes one CSP training step with clean, non-overlapping stage profiling.
    Conditions:
      'A': Degree 3, standard gradient at Level 3
      'B': Degree 3, controlled level adjustment to Level 2
      'C': Degree 5, standard gradient at Level 2
    """
    meta = {}
    
    # 1. Forward linear combination
    with prof.measure("forward_linear"):
        enc_z = enc_X[0] * enc_w[0]
        for j in range(1, D):
            enc_z = enc_z + (enc_X[j] * enc_w[j])
    meta["forward_level"] = get_levels(enc_z)
    meta["forward_scale"] = get_scale(enc_z)

    # 2. Polynomial evaluation
    deg = 3 if condition in ("A", "B") else 5
    with prof.measure("poly_eval"):
        enc_p = enc_sigmoid(enc_z, deg)
    meta["poly_level"] = get_levels(enc_p)
    meta["poly_scale"] = get_scale(enc_p)

    # 3. Error computation
    with prof.measure("error_computation"):
        burn_y = 4 if deg == 3 else 5
        enc_y_aligned = burn(enc_y, burn_y)
        enc_err = enc_p - enc_y_aligned
    meta["err_before_adj_level"] = get_levels(enc_err)
    meta["err_before_adj_scale"] = get_scale(enc_err)
    meta["err_before_adj_parms_id"] = get_parms_id(enc_err)

    # 4. Controlled Level Adjustment (Condition B only)
    with prof.measure("level_adjustment"):
        if condition == "B":
            # Explicit, valid CKKS level adjustment:
            # Drop 1 modulus level via scalar multiplication & automatic rescaling
            enc_err_adj = enc_err * 1.0
            num_adj_ops = 1
            num_adj_cts = 1
        else:
            enc_err_adj = enc_err
            num_adj_ops = 0
            num_adj_cts = 0
    meta["err_after_adj_level"] = get_levels(enc_err_adj)
    meta["err_after_adj_scale"] = get_scale(enc_err_adj)
    meta["err_after_adj_parms_id"] = get_parms_id(enc_err_adj)
    meta["num_adj_ops"] = num_adj_ops
    meta["num_adj_cts"] = num_adj_cts

    # 5. Feature alignment for gradient stage
    # Condition A: burn 4 levels (Level 7 -> Level 3)
    # Condition B: burn 5 levels (Level 7 -> Level 2)
    # Condition C: burn 5 levels (Level 7 -> Level 2)
    burn_X = 4 if condition == "A" else 5
    with prof.measure("feature_alignment"):
        enc_X_aligned = [burn(enc_X[j], burn_X) for j in range(D)]
    meta["feature_aligned_level"] = get_levels(enc_X_aligned[0])
    meta["feature_aligned_scale"] = get_scale(enc_X_aligned[0])
    meta["feature_aligned_parms_id"] = get_parms_id(enc_X_aligned[0])

    # Operand compatibility check immediately before gradient
    meta["grad_operand_err_level"] = get_levels(enc_err_adj)
    meta["grad_operand_X_level"] = get_levels(enc_X_aligned[0])
    meta["grad_operands_compatible"] = (
        meta["grad_operand_err_level"] == meta["grad_operand_X_level"] and
        meta["err_after_adj_parms_id"] == meta["feature_aligned_parms_id"]
    )

    # 6. Isolated Gradient computation: pure O(D) dot products
    enc_grads = []
    with prof.measure("gradient_computation"):
        for j in range(D):
            enc_grads.append((enc_err_adj * enc_X_aligned[j]).sum())
    meta["grad_end_level"] = get_levels(enc_grads[0])
    meta["grad_end_scale"] = get_scale(enc_grads[0])

    return enc_grads, meta, enc_err, enc_err_adj

def hospital_update_profiled(enc_grads, w, P, D, lr=LR, l2=L2, grad_clip=GRAD_CLIP):
    with prof.measure("decryption_update"):
        grad = np.array([enc_grads[j].decrypt()[0] / P for j in range(D)]) + l2 * w
        n = np.linalg.norm(grad)
        if n > grad_clip:
            grad *= grad_clip / n
        return w - lr * grad

# ── Main Experiment Runner ───────────────────────────────────────────────────
def run_all_experiments():
    print("=" * 80)
    print(" PHASE 4: CONTROLLED VALIDATION OF MODULUS-LEVEL-AWARE SCHEDULING")
    print("=" * 80)

    # 1. Load MHEALTH Dataset (exact Phase 2A split: 1645 train, 705 test, 92 features)
    print("\n[1] Loading MHEALTH dataset...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(float)
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(float)
    
    P, D = X_train.shape
    print(f"  Train: {X_train.shape} (Subjects 1–7)")
    print(f"  Test : {X_test.shape}  (Subjects 8–10)")

    # 2. Build CKKS Context
    print("\n[2] Initializing CKKS Context...")
    ctx = build_context()
    print("  N = 16384, scale = 2^40, coeff_mod_bit_sizes = [60,40,40,40,40,40,40,60]")

    # 3. Encrypt Training Features and Labels
    print("\n[3] Encrypting training data (column-packed)...")
    t0 = time.perf_counter()
    enc_X = [ts.ckks_vector(ctx, X_train[:, j].tolist()) for j in range(D)]
    enc_y = ts.ckks_vector(ctx, y_train.tolist())
    t_enc = time.perf_counter() - t0
    print(f"  Encryption complete in {t_enc:.2f}s (92 feature CTs, 1 label CT)")

    # 4. Warm-up Pass (explicitly excluded from reported timings)
    print("\n[4] Performing system warm-up (excluded from reported measurements)...")
    w_warmup = np.zeros(D)
    enc_w_warmup = [ts.ckks_vector(ctx, [0.0] * P) for _ in range(D)]
    prof.clear()
    _ = csp_step_profiled(enc_X[:4], enc_y, enc_w_warmup[:4], P, 4, "A")
    prof.clear()
    print("  Warm-up completed.")

    # ── PART 1: STAGE-WISE PROFILING (3 Repetitions per Condition) ───────────
    print("\n" + "=" * 80)
    print(" PART 1: STAGE-WISE PROFILING (3 REPETITIONS PER CONDITION)")
    print("=" * 80)

    conditions = ["A", "B", "C"]
    condition_names = {
        "A": "Condition A (Deg 3 Baseline, Grad Lvl 3)",
        "B": "Condition B (Deg 3 Adjusted to Lvl 2)",
        "C": "Condition C (Deg 5 Control, Grad Lvl 2)",
    }

    profiling_records = []
    metadata_records = {}

    for cond in conditions:
        print(f"\n>>> Running {condition_names[cond]} (3 repetitions)...")
        w_test = np.zeros(D)
        
        rep_timings = []
        for rep in range(1, 4):
            prof.clear()
            enc_w = [ts.ckks_vector(ctx, [float(w_test[j])] * P) for j in range(D)]
            
            t_epoch_start = time.perf_counter()
            enc_grads, meta, _, _ = csp_step_profiled(enc_X, enc_y, enc_w, P, D, cond)
            w_updated = hospital_update_profiled(enc_grads, w_test, P, D)
            t_epoch_total = time.perf_counter() - t_epoch_start
            
            metadata_records[cond] = meta

            row = {
                "condition": cond,
                "repetition": rep,
                "forward_linear": prof.get("forward_linear"),
                "poly_eval": prof.get("poly_eval"),
                "error_computation": prof.get("error_computation"),
                "level_adjustment": prof.get("level_adjustment"),
                "feature_alignment": prof.get("feature_alignment"),
                "gradient_computation": prof.get("gradient_computation"),
                "decryption_update": prof.get("decryption_update"),
                "total_epoch_time": t_epoch_total,
                "grad_start_level": meta["grad_operand_err_level"],
                "grad_end_level": meta["grad_end_level"],
            }
            profiling_records.append(row)
            rep_timings.append(row)
            print(f"  Rep {rep}: Grad={row['gradient_computation']:.3f}s | "
                  f"Align={row['feature_alignment']:.3f}s | "
                  f"Adj={row['level_adjustment']:.4f}s | "
                  f"Epoch={row['total_epoch_time']:.3f}s | "
                  f"Lvl={meta['grad_operand_err_level']}->{meta['grad_end_level']}")

    df_prof = pd.DataFrame(profiling_records)
    df_prof.to_csv(os.path.join(OUT_DIR, "timing_profile_reps.csv"), index=False)

    # Compute Summary Stats (Mean ± Std)
    summary_rows = []
    stages = [
        "forward_linear", "poly_eval", "error_computation",
        "level_adjustment", "feature_alignment",
        "gradient_computation", "decryption_update", "total_epoch_time"
    ]

    for cond in conditions:
        sub = df_prof[df_prof["condition"] == cond]
        s_row = {"condition": cond}
        for stg in stages:
            s_row[f"{stg}_mean"] = sub[stg].mean()
            s_row[f"{stg}_std"] = sub[stg].std()
        s_row["grad_start_level"] = sub["grad_start_level"].iloc[0]
        s_row["grad_end_level"] = sub["grad_end_level"].iloc[0]
        summary_rows.append(s_row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(OUT_DIR, "timing_profile_summary.csv"), index=False)

    # ── PART 2: NUMERICAL CORRECTNESS & ERROR VALIDATION ──────────────────────
    print("\n" + "=" * 80)
    print(" PART 2: NUMERICAL CORRECTNESS & ERROR VALIDATION")
    print("=" * 80)

    # Test weight vectors:
    # 1. w_zero
    # 2. 5 deterministic small random weights
    # 3. 2 intermediate weights from training epochs 2 & 8
    np.random.seed(SEED)
    eval_weights = [("w_zero", np.zeros(D))]
    for i in range(1, 6):
        eval_weights.append((f"w_rand_{i}", np.random.uniform(-0.1, 0.1, D)))

    # Generate intermediate weights
    print("  Generating intermediate training weights (epoch 2 and 8)...")
    w_sim = np.zeros(D)
    for ep in range(1, 9):
        z_sim = X_train @ w_sim
        g_sim = (X_train.T @ (poly_sigmoid(z_sim, 3) - y_train)) / P + L2 * w_sim
        n = np.linalg.norm(g_sim)
        if n > GRAD_CLIP:
            g_sim *= GRAD_CLIP / n
        w_sim = w_sim - LR * g_sim
        if ep == 2:
            eval_weights.append(("w_epoch_2", w_sim.copy()))
        if ep == 8:
            eval_weights.append(("w_epoch_8", w_sim.copy()))

    numerical_records = []

    # Prepare small validation subset
    X_val = X_train[:VAL_N]
    y_val = y_train[:VAL_N]
    P_val = VAL_N
    enc_X_val = [ts.ckks_vector(ctx, X_val[:, j].tolist()) for j in range(D)]
    enc_y_val = ts.ckks_vector(ctx, y_val.tolist())

    for w_name, w_vec in eval_weights:
        enc_w_val = [ts.ckks_vector(ctx, [float(w_vec[j])] * P_val) for j in range(D)]
        
        # Plaintext references
        z_pt = X_val @ w_vec
        p_pt_deg3 = poly_sigmoid(z_pt, 3)
        p_pt_deg5 = poly_sigmoid(z_pt, 5)
        grad_pt_deg3 = (X_val.T @ (p_pt_deg3 - y_val)) / P_val
        grad_pt_deg5 = (X_val.T @ (p_pt_deg5 - y_val)) / P_val

        # Execute Conditions A, B, C on validation batch
        g_A, _, err_A, _ = csp_step_profiled(enc_X_val, enc_y_val, enc_w_val, P_val, D, "A")
        g_B, _, err_B_raw, err_B_adj = csp_step_profiled(enc_X_val, enc_y_val, enc_w_val, P_val, D, "B")
        g_C, _, err_C, _ = csp_step_profiled(enc_X_val, enc_y_val, enc_w_val, P_val, D, "C")

        # Decrypt gradients
        g_A_dec = np.array([g.decrypt()[0] / P_val for g in g_A])
        g_B_dec = np.array([g.decrypt()[0] / P_val for g in g_B])
        g_C_dec = np.array([g.decrypt()[0] / P_val for g in g_C])

        # Decrypt errors for Condition B adjustment validation
        dec_err_B_raw = np.array(err_B_raw.decrypt())
        dec_err_B_adj = np.array(err_B_adj.decrypt())
        err_adj_diff = np.abs(dec_err_B_raw - dec_err_B_adj)

        # Gradient errors vs PT references
        err_A_vs_PT = np.abs(g_A_dec - grad_pt_deg3)
        err_B_vs_PT = np.abs(g_B_dec - grad_pt_deg3)
        err_C_vs_PT = np.abs(g_C_dec - grad_pt_deg5)
        diff_B_vs_A = np.abs(g_B_dec - g_A_dec)

        # Relative errors
        mask_deg3 = np.abs(grad_pt_deg3) > 1e-7
        rel_err_A = np.abs((g_A_dec[mask_deg3] - grad_pt_deg3[mask_deg3]) / grad_pt_deg3[mask_deg3])
        rel_err_B = np.abs((g_B_dec[mask_deg3] - grad_pt_deg3[mask_deg3]) / grad_pt_deg3[mask_deg3])

        row_num = {
            "weight_vector": w_name,
            "err_adj_max_ae": float(err_adj_diff.max()),
            "err_adj_mean_ae": float(err_adj_diff.mean()),
            "grad_A_max_ae": float(err_A_vs_PT.max()),
            "grad_A_mean_ae": float(err_A_vs_PT.mean()),
            "grad_A_mean_re": float(rel_err_A.mean()) if len(rel_err_A) > 0 else 0.0,
            "grad_B_max_ae": float(err_B_vs_PT.max()),
            "grad_B_mean_ae": float(err_B_vs_PT.mean()),
            "grad_B_mean_re": float(rel_err_B.mean()) if len(rel_err_B) > 0 else 0.0,
            "diff_B_vs_A_max_ae": float(diff_B_vs_A.max()),
            "diff_B_vs_A_mean_ae": float(diff_B_vs_A.mean()),
            "grad_C_max_ae": float(err_C_vs_PT.max()),
            "grad_C_mean_ae": float(err_C_vs_PT.mean()),
            "tolerance_check_pass": bool(err_B_vs_PT.max() <= 1e-4),
        }
        numerical_records.append(row_num)
        print(f"  {w_name:10s} | Adj Error: MaxAE={row_num['err_adj_max_ae']:.2e} | "
              f"Grad B vs PT: MaxAE={row_num['grad_B_max_ae']:.2e} | "
              f"Grad B vs A: MaxAE={row_num['diff_B_vs_A_max_ae']:.2e} | "
              f"Pass={row_num['tolerance_check_pass']}")

    df_num = pd.DataFrame(numerical_records)
    df_num.to_csv(os.path.join(OUT_DIR, "numerical_validation.csv"), index=False)

    # ── PART 3: MODEL-LEVEL SANITY CHECK (10 EPOCHS TRAINING) ────────────────
    print("\n" + "=" * 80)
    print(" PART 3: MODEL-LEVEL SANITY CHECK (10 EPOCHS TRAINING)")
    print("=" * 80)

    model_records = []

    for cond in conditions:
        print(f"\n>>> Running 10-Epoch Training for {condition_names[cond]}...")
        np.random.seed(SEED)
        w = np.zeros(D)
        
        epoch_times = []
        t_train_start = time.perf_counter()
        
        for ep in range(1, EPOCHS + 1):
            t_ep_0 = time.perf_counter()
            enc_w = [ts.ckks_vector(ctx, [float(w[j])] * P) for j in range(D)]
            enc_grads, _, _, _ = csp_step_profiled(enc_X, enc_y, enc_w, P, D, cond)
            w = hospital_update_profiled(enc_grads, w, P, D)
            t_ep = time.perf_counter() - t_ep_0
            epoch_times.append(t_ep)
            print(f"    Epoch {ep:2d}/{EPOCHS}: Time={t_ep:.2f}s | Norm(w)={np.linalg.norm(w):.4f}")

        total_train_time = time.perf_counter() - t_train_start

        # Inference on Train & Test
        deg = 3 if cond in ("A", "B") else 5
        p_train = true_sigmoid(X_train @ w)
        p_test  = true_sigmoid(X_test @ w)
        y_pred_train = (p_train >= 0.5).astype(int)
        y_pred_test  = (p_test >= 0.5).astype(int)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, zero_division=0)
        test_rec  = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1   = f1_score(y_test, y_pred_test, zero_division=0)
        test_auc  = roc_auc_score(y_test, p_test)

        is_finite = bool(np.all(np.isfinite(w)))
        is_stable = bool(np.linalg.norm(w) < 10.0 and not np.isnan(w).any())

        row_model = {
            "condition": cond,
            "description": condition_names[cond],
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "total_train_time_s": total_train_time,
            "mean_epoch_time_s": float(np.mean(epoch_times)),
            "std_epoch_time_s": float(np.std(epoch_times)),
            "final_weight_norm": float(np.linalg.norm(w)),
            "weights_finite": is_finite,
            "numerical_stable": is_stable,
        }
        model_records.append(row_model)
        print(f"  Result -> Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f} | "
              f"Total Time: {total_train_time:.1f}s | Stable: {is_stable}")

    df_model = pd.DataFrame(model_records)
    df_model.to_csv(os.path.join(OUT_DIR, "model_sanity_check.csv"), index=False)

    # ── PART 4: CONSOLIDATED JSON MASTER RESULTS ──────────────────────────────
    master_results = {
        "metadata": {
            "dataset": "MHEALTH",
            "task": "binary_activity_classification",
            "train_samples": int(P),
            "test_samples": int(X_test.shape[0]),
            "num_features": int(D),
            "poly_modulus_degree": 16384,
            "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60],
            "global_scale": "2^40",
            "epochs": EPOCHS,
            "lr": LR,
            "l2": L2,
            "grad_clip": GRAD_CLIP,
            "seed": SEED,
        },
        "stage_timings_summary": summary_rows,
        "metadata_records": metadata_records,
        "numerical_validation_summary": {
            "max_err_adj_ae": float(df_num["err_adj_max_ae"].max()),
            "max_grad_B_vs_PT_ae": float(df_num["grad_B_max_ae"].max()),
            "max_diff_B_vs_A_ae": float(df_num["diff_B_vs_A_max_ae"].max()),
            "all_passed_tolerance": bool(df_num["tolerance_check_pass"].all()),
        },
        "model_sanity_summary": model_records,
    }

    with open(os.path.join(OUT_DIR, "phase4_results.json"), "w") as f:
        json.dump(master_results, f, indent=4)

    print("\n" + "=" * 80)
    print(" ALL EXPERIMENTS COMPLETE. Output saved to results/mhealth/phase4/")
    print("=" * 80)

if __name__ == "__main__":
    run_all_experiments()
