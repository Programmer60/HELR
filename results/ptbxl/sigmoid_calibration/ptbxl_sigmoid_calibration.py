#!/usr/bin/env python3
"""
PTB-XL Sigmoid Calibration and Objective-Consistency Audit
===========================================================
Implements Tasks 1?10 of the sigmoid calibration experiment spec.

Sections:
  1. Load data and audit baseline model
  2. Establish consistent LR objective (bias-free CKKS convention)
  3. Logit distribution analysis (train + val only)
  4. Evaluate degree 1/3/5 MHEALTH polynomials on PTB-XL logits
  5. Fit candidate PTB-XL-specific polynomials on training logits,
     select best with validation only.  No test data used.
  6. CKKS depth & numerical feasibility check
  7. Gradient comparison (polynomial vs. exact sigmoid)
  8. Short plaintext polynomial-training sanity check (50 epochs, val metrics only)
  9. Write docs/ptbxl/PTBXL_SIGMOID_CALIBRATION.md, JSON, CSV

Does NOT start full encrypted training. Does NOT touch MHEALTH code.
Does NOT overwrite existing baselines.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ??? Paths ???????????????????????????????????????????????????????????????????
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_NPZ = PROJECT_ROOT / "data" / "ptbxl_processed" / "ptbxl_binary_splits.npz"
BASELINE_MODEL = PROJECT_ROOT / "data" / "ptbxl_processed" / "baseline_logistic_regression.joblib"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = PROJECT_ROOT / "docs" / "ptbxl"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

# ??? CKKS budget from existing configuration ?????????????????????????????????
CKKS_TOTAL_LEVELS = 7          # [60,40,40,40,40,40,40,60] ? 6 inner primes ? 7 usable
LEVELS_FWD_DOT   = 1           # Ct?Ct for linear combination
LEVELS_BACKWARD  = 1           # Ct?Ct for err * X_j in gradient
LEVELS_BUDGET    = CKKS_TOTAL_LEVELS  # total available from fresh ciphertext

# ??? MHEALTH polynomial coefficients (frozen) ????????????????????????????????
MHEALTH_POLYS = {
    1: {"c0": 0.5, "c1": 0.197,    "c3": 0.0,       "c5": 0.0,
        "levels": 1, "valid_interval": (-20, 20), "name": "deg1_MHEALTH"},
    3: {"c0": 0.5, "c1": 0.15012,  "c3": -0.001593, "c5": 0.0,
        "levels": 3, "valid_interval": (-8, 8),   "name": "deg3_MHEALTH"},
    5: {"c0": 0.5, "c1": 0.217101, "c3": -0.007823, "c5": 0.000118,
        "levels": 4, "valid_interval": (-6, 6),   "name": "deg5_MHEALTH"},
}

# ??? Numerics ?????????????????????????????????????????????????????????????????
def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def poly_eval(z: np.ndarray, c0, c1, c3=0.0, c5=0.0) -> np.ndarray:
    return c0 + c1 * z + c3 * z**3 + c5 * z**5

def poly_gradient_factor(z: np.ndarray, c1, c3=0.0, c5=0.0) -> np.ndarray:
    """Derivative dP/dz  ? the gradient multiplier in backprop."""
    return c1 + 3 * c3 * z**2 + 5 * c5 * z**4

def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def levels_for_polynomial(degree: int) -> int:
    """
    Levels consumed inside the polynomial evaluation only
    (not including the preceding fwd dot-product).
    Degree 1  ? 1 (one Ct?Pt)
    Degree 3  ? 3 (two Ct?Ct, one Ct?Pt)
    Degree 5  ? 4 (three Ct?Ct)
    """
    return {1: 1, 3: 3, 5: 4}.get(degree, degree)


def depth_for_forward_pass(degree: int) -> int:
    return LEVELS_FWD_DOT + levels_for_polynomial(degree)


def check_depth_feasibility(degree: int) -> dict:
    fwd = depth_for_forward_pass(degree)
    remaining_for_grad = CKKS_TOTAL_LEVELS - fwd
    # gradient step needs 1 level (err?X_j), labels must be aligned to err level
    grad_ok = remaining_for_grad >= 1
    return {
        "degree": degree,
        "forward_levels": fwd,
        "remaining_after_forward": remaining_for_grad,
        "gradient_feasible": grad_ok,
        "feasible": grad_ok,
    }


# ??? Polynomial fitting ???????????????????????????????????????????????????????
def fit_minimax_odd_poly(z_support: np.ndarray, degree: int,
                         n_grid: int = 2000) -> dict:
    """
    Least-squares fit of an odd polynomial (odd powers only, plus c0=0.5 fixed)
    on a dense grid over the given support interval.
    Returns coefficients and the fitting interval used.
    """
    assert degree in (1, 3, 5), "Only degrees 1, 3, 5 supported"
    z_min, z_max = float(z_support.min()), float(z_support.max())
    # Clip extreme outliers to 99.9th percentile of support for fitting stability
    p001, p999 = np.percentile(z_support, [0.1, 99.9])
    fit_lo = max(z_min, p001)
    fit_hi = min(z_max, p999)

    z_grid = np.linspace(fit_lo, fit_hi, n_grid)
    y_target = sigmoid(z_grid) - 0.5  # remove constant, fit the odd part

    if degree == 1:
        # P(z) = 0.5 + c1*z
        A = z_grid[:, None]
        coeffs, _, _, _ = np.linalg.lstsq(A, y_target, rcond=None)
        c1, c3, c5 = float(coeffs[0]), 0.0, 0.0
    elif degree == 3:
        # P(z) = 0.5 + c1*z + c3*z^3
        A = np.column_stack([z_grid, z_grid**3])
        coeffs, _, _, _ = np.linalg.lstsq(A, y_target, rcond=None)
        c1, c3, c5 = float(coeffs[0]), float(coeffs[1]), 0.0
    else:
        # P(z) = 0.5 + c1*z + c3*z^3 + c5*z^5
        A = np.column_stack([z_grid, z_grid**3, z_grid**5])
        coeffs, _, _, _ = np.linalg.lstsq(A, y_target, rcond=None)
        c1, c3, c5 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    return {"c0": 0.5, "c1": c1, "c3": c3, "c5": c5,
            "fit_interval": [round(fit_lo, 4), round(fit_hi, 4)],
            "fit_n_grid": n_grid}


def evaluate_poly_vs_sigmoid(z: np.ndarray, c0, c1, c3, c5, label: str) -> dict:
    """Evaluate polynomial vs exact sigmoid on given logit array."""
    p_poly = poly_eval(z, c0, c1, c3, c5)
    p_sig = sigmoid(z)
    prob_err = np.abs(p_poly - p_sig)
    out_of_01 = int(np.sum((p_poly < 0) | (p_poly > 1)))
    # Gradient factor error: |dP/dz - sigma'(z)|
    dp_poly = poly_gradient_factor(z, c1, c3, c5)
    dp_sig = sigmoid_deriv(z)
    grad_err = np.abs(dp_poly - dp_sig)
    return {
        "label": label,
        "prob_max_ae": float(prob_err.max()),
        "prob_mean_ae": float(prob_err.mean()),
        "prob_out_of_01": out_of_01,
        "prob_out_of_01_frac": float(out_of_01) / len(z),
        "grad_factor_max_ae": float(grad_err.max()),
        "grad_factor_mean_ae": float(grad_err.mean()),
    }


def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                     c0, c1, c3, c5, l2: float = 0.0,
                     exact_sigmoid: bool = False) -> np.ndarray:
    """
    CKKS-consistent gradient: no intercept.
    g = (1/N) * X^T (p(Xw) - y) + l2 * w
    """
    z = X @ w
    p = sigmoid(z) if exact_sigmoid else poly_eval(z, c0, c1, c3, c5)
    err = p - y
    return (X.T @ err) / len(y) + l2 * w


# ??? Short plaintext training loop ????????????????????????????????????????????
def plaintext_train(X_train, y_train, X_val, y_val,
                    c0, c1, c3, c5,
                    lr: float = 0.05, l2: float = 0.01,
                    epochs: int = 50, clip: float = 1.0,
                    w_init: np.ndarray | None = None) -> dict:
    D = X_train.shape[1]
    w = np.zeros(D) if w_init is None else w_init.copy()
    history = []
    for ep in range(1, epochs + 1):
        g = compute_gradient(X_train, y_train, w, c0, c1, c3, c5, l2=l2)
        g = np.clip(g, -clip, clip)
        if not np.all(np.isfinite(g)):
            history.append({"epoch": ep, "status": "nan_grad"})
            break
        w = w - lr * g
        if ep % 10 == 0 or ep == epochs:
            z_v = X_val @ w
            p_v = poly_eval(z_v, c0, c1, c3, c5)
            p_v_clipped = np.clip(p_v, 1e-12, 1.0 - 1e-12)
            val_loss = float(-np.mean(y_val * np.log(p_v_clipped) +
                                      (1 - y_val) * np.log(1 - p_v_clipped)))
            pred = (p_v >= 0.5).astype(int)
            val_acc = float(np.mean(pred == y_val))
            history.append({
                "epoch": ep,
                "w_norm": float(np.linalg.norm(w)),
                "w_finite": bool(np.all(np.isfinite(w))),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "grad_norm": float(np.linalg.norm(g)),
            })
    return {"w_final": w, "history": history}


# ??? Main ?????????????????????????????????????????????????????????????????????
def main():
    log_lines: list[str] = []
    def log(msg: str = ""):
        print(msg, flush=True)
        log_lines.append(str(msg))

    log("=" * 72)
    log(" PTB-XL SIGMOID CALIBRATION & OBJECTIVE-CONSISTENCY AUDIT")
    log("=" * 72)
    log(f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Python    : {sys.version.split()[0]}")

    # ?? 1. Load data & audit baseline model ??????????????????????????????????
    log("\n[1] Loading data and auditing baseline model...")
    if not DATA_NPZ.exists():
        raise FileNotFoundError(f"Missing {DATA_NPZ}. Run preprocess_ptbxl.py first.")
    npz = np.load(DATA_NPZ, allow_pickle=False)
    X_train = npz["X_train"].astype(np.float64)
    y_train = npz["y_train"].astype(np.float64)
    X_val   = npz["X_val"].astype(np.float64)
    y_val   = npz["y_val"].astype(np.float64)

    N_train, D = X_train.shape
    N_val = len(y_val)
    log(f"  X_train : {X_train.shape}  pos={int(y_train.sum())}/{N_train}")
    log(f"  X_val   : {X_val.shape}   pos={int(y_val.sum())}/{N_val}")

    baseline = joblib.load(BASELINE_MODEL)
    w_base_raw = baseline.coef_[0].copy()     # shape (48,)
    b_base     = float(baseline.intercept_[0])

    audit = {
        "fit_intercept": bool(baseline.fit_intercept),
        "intercept_value": b_base,
        "C": float(baseline.C),
        "penalty": baseline.penalty,
        "solver": baseline.solver,
        "coef_norm": float(np.linalg.norm(w_base_raw)),
        "sklearn_effective_L2": float(1.0 / baseline.C),
    }
    log(f"  Baseline fit_intercept={audit['fit_intercept']}, "
        f"intercept={b_base:.4f}, C={audit['C']}, "
        f"||w||={audit['coef_norm']:.4f}")

    # ?? 2. Establish consistent CKKS-convention objective ?????????????????????
    log("\n[2] Establishing CKKS-consistent objective...")
    # CKKS circuit convention: NO intercept, z = X@w
    # Regularisation: L2 with lambda=0.01 (matching MHEALTH CKKS setup)
    # Normalization:  (1/N) per batch, summed over batches
    #
    # The saved baseline used fit_intercept=True, C=1.0 (sklearn convention).
    # We do NOT modify the saved baseline. Instead we define a CKKS reference
    # model with no-intercept, L2=0.01, and refit a plaintext reference in Sec 8.
    #
    # For gradient comparisons in this script we use w_base_raw (no intercept)
    # but we note the effective logit shift of b_base = 0.5846.
    CKKS_L2 = 0.01
    log(f"  CKKS convention : no intercept, L2={CKKS_L2}")
    log(f"  Baseline convention : intercept={b_base:.4f}, "
        f"effective L2 ? {1.0/baseline.C:.1f} (sklearn total, not per-sample)")
    log(f"  Decision: use CKKS convention for calibration. "
        f"Do NOT overwrite saved baseline.")

    objective_audit = {
        "ckks_has_intercept": False,
        "ckks_l2_lambda": CKKS_L2,
        "ckks_l2_convention": "(1/N) * X^T(p-y) + lambda*w  [no bias]",
        "baseline_has_intercept": True,
        "baseline_intercept": b_base,
        "baseline_sklearn_objective": "sum(log_loss) + (1/(2*C)) * ||w||^2",
        "baseline_C": float(baseline.C),
        "consistent": False,
        "notes": (
            "The saved baseline was fitted WITH intercept=True and C=1.0. "
            "The CKKS harness uses no intercept and L2=0.01. "
            "These are different objectives. "
            "For logit analysis, we use logits WITH intercept for the baseline model "
            "and WITHOUT intercept for the CKKS-convention model. "
            "A no-intercept plaintext reference is fitted in Section 8 for a fair comparison."
        ),
    }

    # ?? 3. Logit distribution analysis ????????????????????????????????????????
    log("\n[3] Logit distribution analysis...")

    # Logits under saved baseline (WITH intercept) ? full picture
    z_train_full = X_train @ w_base_raw + b_base
    z_val_full   = X_val   @ w_base_raw + b_base

    # Logits under CKKS convention (NO intercept, weights from baseline coef_)
    z_train_ckks = X_train @ w_base_raw
    z_val_ckks   = X_val   @ w_base_raw

    quantile_levels = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5,
                       0.75, 0.9, 0.95, 0.99, 0.995, 0.999]

    def logit_stats(z: np.ndarray, name: str) -> dict:
        q = np.quantile(z, quantile_levels).tolist()
        half_widths = [1, 2, 3, 4, 5, 6, 8, 10, 12]
        outside = {f"|z|>{h}": float(np.mean(np.abs(z) > h)) for h in half_widths}
        return {
            "name": name,
            "n": len(z),
            "min": float(z.min()),
            "max": float(z.max()),
            "mean": float(z.mean()),
            "std": float(z.std()),
            "quantiles": {f"q{int(ql*1000):04d}": v for ql, v in zip(quantile_levels, q)},
            "fraction_outside_interval": outside,
        }

    logit_stats_all = {
        "train_with_intercept": logit_stats(z_train_full, "train_with_intercept"),
        "val_with_intercept":   logit_stats(z_val_full,   "val_with_intercept"),
        "train_no_intercept":   logit_stats(z_train_ckks, "train_no_intercept"),
        "val_no_intercept":     logit_stats(z_val_ckks,   "val_no_intercept"),
    }

    for name, s in logit_stats_all.items():
        log(f"  {name}: min={s['min']:.3f}  max={s['max']:.3f}  "
            f"mean={s['mean']:.3f}  std={s['std']:.3f}")
    log(f"\n  Fraction of TRAINING logits (with intercept) outside candidate intervals:")
    for hw in [3, 5, 6, 8]:
        log(f"    |z|>{hw}: {logit_stats_all['train_with_intercept']['fraction_outside_interval'][f'|z|>{hw}']:.4f}")
    log(f"\n  Fraction of VAL logits (with intercept) outside candidate intervals:")
    for hw in [3, 5, 6, 8]:
        log(f"    |z|>{hw}: {logit_stats_all['val_with_intercept']['fraction_outside_interval'][f'|z|>{hw}']:.4f}")
    log(f"\n  NOTE: val max={z_val_full.max():.2f} suggests outliers beyond [-6,6]")

    # ?? 4. Evaluate MHEALTH polynomials on PTB-XL logits ?????????????????????
    log("\n[4] Evaluating MHEALTH polynomials on PTB-XL logits...")

    # Use training logits WITH intercept as the reference distribution
    # (this is what a CKKS model WITH bias would see; without intercept
    # is a separate column for CKKS-convention analysis)
    poly_eval_results = {}
    for deg, p in MHEALTH_POLYS.items():
        # Evaluate on train (with intercept) ? most representative
        r_tr = evaluate_poly_vs_sigmoid(z_train_full,
                                        p["c0"], p["c1"], p["c3"], p["c5"],
                                        f"deg{deg}_MHEALTH_train_full")
        # Also on the 99.9-percentile bounded subset
        z_p999 = np.percentile(np.abs(z_train_full), 99.9)
        z_bounded = z_train_full[np.abs(z_train_full) <= z_p999]
        r_bounded = evaluate_poly_vs_sigmoid(z_bounded,
                                             p["c0"], p["c1"], p["c3"], p["c5"],
                                             f"deg{deg}_MHEALTH_train_bounded")
        # On val
        r_val = evaluate_poly_vs_sigmoid(z_val_full,
                                         p["c0"], p["c1"], p["c3"], p["c5"],
                                         f"deg{deg}_MHEALTH_val")
        poly_eval_results[f"deg{deg}"] = {
            "coefficients": {"c0": p["c0"], "c1": p["c1"], "c3": p["c3"], "c5": p["c5"]},
            "mhealth_valid_interval": list(p["valid_interval"]),
            "depth_levels_poly_only": p["levels"],
            "train_full": r_tr,
            "train_bounded_p999": r_bounded,
            "val": r_val,
        }
        log(f"  Deg {deg} on train (full):    prob_maxAE={r_tr['prob_max_ae']:.4f}  "
            f"out_of_01={r_tr['prob_out_of_01']}  "
            f"grad_maxAE={r_tr['grad_factor_max_ae']:.4f}")
        log(f"  Deg {deg} on train (p999):    prob_maxAE={r_bounded['prob_max_ae']:.4f}  "
            f"out_of_01={r_bounded['prob_out_of_01']}  "
            f"grad_maxAE={r_bounded['grad_factor_max_ae']:.4f}")
        log(f"  Deg {deg} on val (full):      prob_maxAE={r_val['prob_max_ae']:.4f}  "
            f"out_of_01={r_val['prob_out_of_01']}  "
            f"grad_maxAE={r_val['grad_factor_max_ae']:.4f}")

    # ?? 5. Fit PTB-XL-specific polynomial candidates (train only) ?????????????
    log("\n[5] Fitting PTB-XL-specific polynomial candidates (training logits only)...")
    # Use CKKS-convention logits (no intercept) since CKKS circuit has no bias term
    z_fit = z_train_ckks     # fitted on training data only

    candidates: dict[str, dict] = {}
    for deg in [1, 3, 5]:
        t0 = time.perf_counter()
        fit = fit_minimax_odd_poly(z_fit, deg)
        elapsed = time.perf_counter() - t0

        # Evaluate on train (CKKS convention, no intercept)
        r_tr = evaluate_poly_vs_sigmoid(z_train_ckks, fit["c0"], fit["c1"],
                                        fit["c3"], fit["c5"],
                                        f"cand_deg{deg}_train")
        # Evaluate on val (CKKS convention, no intercept) ? candidate selection metric
        r_val = evaluate_poly_vs_sigmoid(z_val_ckks, fit["c0"], fit["c1"],
                                         fit["c3"], fit["c5"],
                                         f"cand_deg{deg}_val")

        depth_check = check_depth_feasibility(deg)
        candidates[f"deg{deg}"] = {
            "degree": deg,
            "fit_method": "least_squares_odd_polynomial_on_training_logits",
            "coefficients": {"c0": fit["c0"], "c1": fit["c1"],
                             "c3": fit["c3"], "c5": fit["c5"]},
            "fit_interval": fit["fit_interval"],
            "fit_n_grid": fit["fit_n_grid"],
            "fit_time_s": round(elapsed, 4),
            "depth": depth_check,
            "train_eval": r_tr,
            "val_eval": r_val,
        }
        log(f"  Candidate Deg {deg}: c1={fit['c1']:.6f} c3={fit['c3']:.6f} "
            f"c5={fit['c5']:.6f}  fit_interval={fit['fit_interval']}")
        log(f"    Train: prob_maxAE={r_tr['prob_max_ae']:.4f}  "
            f"out_of_01={r_tr['prob_out_of_01']}")
        log(f"    Val  : prob_maxAE={r_val['prob_max_ae']:.4f}  "
            f"out_of_01={r_val['prob_out_of_01']}")
        log(f"    Depth: fwd={depth_check['forward_levels']}  "
            f"remaining={depth_check['remaining_after_forward']}  "
            f"feasible={depth_check['feasible']}")

    # ?? 6. CKKS depth & numerical feasibility ?????????????????????????????????
    log("\n[6] CKKS depth & numerical feasibility check...")
    log(f"  CKKS total usable levels: {CKKS_TOTAL_LEVELS}")
    log(f"  Level consumption breakdown:")
    log(f"    Forward dot product (Ct?Ct): {LEVELS_FWD_DOT}")
    log(f"    Polynomial evaluation      : deg1={levels_for_polynomial(1)}  "
        f"deg3={levels_for_polynomial(3)}  deg5={levels_for_polynomial(5)}")
    log(f"    Backward gradient (Ct?Ct)  : {LEVELS_BACKWARD}")
    log(f"  Level budget check:")
    feasibility: dict[str, dict] = {}
    for deg in [1, 3, 5]:
        dc = check_depth_feasibility(deg)
        total_needed = dc["forward_levels"] + LEVELS_BACKWARD
        margin = CKKS_TOTAL_LEVELS - total_needed
        dc["backward_levels"] = LEVELS_BACKWARD
        dc["total_needed"] = total_needed
        dc["budget_margin"] = margin
        dc["feasible"] = margin >= 0
        feasibility[f"deg{deg}"] = dc
        log(f"    Deg {deg}: fwd={dc['forward_levels']}  +bwd={LEVELS_BACKWARD}  "
            f"total={total_needed}  margin={margin}  {'OK' if dc['feasible'] else 'INFEASIBLE'}")

    # ?? 7. Gradient comparison: polynomial vs exact sigmoid ???????????????????
    log("\n[7] Gradient comparison (poly vs exact sigmoid)...")
    # Use weight vectors: zero, two random, and a no-intercept reference
    # Do NOT use test data
    np.random.seed(SEED)

    # Fit a no-intercept plaintext reference model for fair comparison
    # (separate from saved baseline; does not overwrite anything)
    from sklearn.linear_model import LogisticRegression
    ref_noic = LogisticRegression(fit_intercept=False, C=100.0, solver="lbfgs",
                                  max_iter=2000, random_state=42)
    ref_noic.fit(X_train, y_train.astype(int))
    w_ref_noic = ref_noic.coef_[0].copy()

    grad_cases = [
        ("w_zero",       np.zeros(D)),
        ("w_rand1",      rng.normal(0, 0.05, D)),
        ("w_rand2",      rng.normal(0, 0.05, D)),
        ("w_baseline_coef", w_base_raw),
        ("w_noic_ref",   w_ref_noic),
    ]

    grad_results = []
    for poly_name, cands_entry in [
        ("deg1_ptbxl",  candidates["deg1"]),
        ("deg3_ptbxl",  candidates["deg3"]),
        ("deg5_ptbxl",  candidates["deg5"]),
        ("deg3_mhealth",{"coefficients": MHEALTH_POLYS[3]}),
    ]:
        cc = cands_entry["coefficients"]
        c0, c1, c3, c5 = cc["c0"], cc["c1"], cc["c3"], cc.get("c5", 0.0)
        for w_name, w_vec in grad_cases:
            g_exact = compute_gradient(X_train, y_train, w_vec, c0, c1, c3, c5,
                                       l2=CKKS_L2, exact_sigmoid=True)
            g_poly  = compute_gradient(X_train, y_train, w_vec, c0, c1, c3, c5,
                                       l2=CKKS_L2, exact_sigmoid=False)
            diff = np.abs(g_exact - g_poly)
            grad_results.append({
                "polynomial": poly_name,
                "weight_case": w_name,
                "grad_max_ae": float(diff.max()),
                "grad_mean_ae": float(diff.mean()),
                "g_exact_norm": float(np.linalg.norm(g_exact)),
                "g_poly_norm": float(np.linalg.norm(g_poly)),
            })
            log(f"  {poly_name:20s} {w_name:22s}  "
                f"maxAE={diff.max():.4e}  meanAE={diff.mean():.4e}")

    # ?? 8. Short plaintext polynomial-training sanity check ???????????????????
    log("\n[8] Short plaintext polynomial-training sanity check (50 epochs)...")
    log("    Using CKKS-convention: no intercept, lr=0.05, L2=0.01, clip=1.0")
    log("    Reporting train loss and VAL metrics ONLY. NO test data used.")

    train_sanity: dict[str, dict] = {}
    for poly_name, cands_entry in [
        ("deg1_ptbxl",   candidates["deg1"]),
        ("deg3_ptbxl",   candidates["deg3"]),
        ("deg5_ptbxl",   candidates["deg5"]),
        ("deg3_mhealth", {"coefficients": MHEALTH_POLYS[3]}),
    ]:
        cc = cands_entry["coefficients"]
        c0, c1, c3, c5 = cc["c0"], cc["c1"], cc["c3"], cc.get("c5", 0.0)
        t0 = time.perf_counter()
        result = plaintext_train(X_train, y_train, X_val, y_val,
                                 c0, c1, c3, c5,
                                 lr=0.05, l2=CKKS_L2, epochs=50, clip=1.0)
        elapsed = time.perf_counter() - t0
        final = result["history"][-1]
        stable = all(h.get("w_finite", True) and h.get("status", "") != "nan_grad"
                     for h in result["history"])
        train_sanity[poly_name] = {
            "poly": poly_name,
            "total_time_s": round(elapsed, 2),
            "stable": stable,
            "history": result["history"],
            "final_val_acc": final.get("val_acc", None),
            "final_val_loss": final.get("val_loss", None),
            "final_w_norm": final.get("w_norm", None),
            "final_grad_norm": final.get("grad_norm", None),
        }
        log(f"  {poly_name:20s} val_acc={final.get('val_acc', '?'):.4f}  "
            f"val_loss={final.get('val_loss', '?'):.4f}  "
            f"stable={stable}  time={elapsed:.1f}s")

    # ?? 9. Select best candidate using validation performance ?????????????????
    log("\n[9] Candidate selection (validation performance + depth feasibility)...")

    # Thresholds (configurable)
    EPSILON_POLY  = 0.10   # max allowable polynomial prob MaxAE vs sigmoid (on val)
    EPSILON_GRAD  = 0.20   # max allowable grad MaxAE vs exact sigmoid
    FEASIBILITY_REQUIRED = True

    selection_results = []
    for deg in [1, 3, 5]:
        cand = candidates[f"deg{deg}"]
        val_prob_err = cand["val_eval"]["prob_max_ae"]
        feas = feasibility[f"deg{deg}"]["feasible"]
        sanity = train_sanity[f"deg{deg}_ptbxl"]
        grad_err = max(r["grad_max_ae"] for r in grad_results
                       if r["polynomial"] == f"deg{deg}_ptbxl")
        passes = (val_prob_err <= EPSILON_POLY and feas and
                  sanity["stable"] and grad_err <= EPSILON_GRAD)
        selection_results.append({
            "degree": deg,
            "val_prob_max_ae": val_prob_err,
            "grad_max_ae": grad_err,
            "depth_feasible": feas,
            "training_stable": sanity["stable"],
            "val_acc_epoch50": sanity["final_val_acc"],
            "passes_all_checks": passes,
        })
        status = "PASS" if passes else "FAIL"
        log(f"  Deg {deg}: val_prob_maxAE={val_prob_err:.4f}  "
            f"grad_maxAE={grad_err:.4e}  feasible={feas}  stable={sanity['stable']}  "
            f"val_acc={sanity['final_val_acc']:.4f}  ? {status}")

    passing = [r for r in selection_results if r["passes_all_checks"]]
    if passing:
        best = max(passing, key=lambda r: r["val_acc_epoch50"])
        recommendation = f"degree_{best['degree']}"
        log(f"\n  RECOMMENDATION: {recommendation}  "
            f"(best val_acc={best['val_acc_epoch50']:.4f} among feasible candidates)")
    else:
        recommendation = "NO_FEASIBLE_CANDIDATE"
        log("\n  RECOMMENDATION: NO_FEASIBLE_CANDIDATE ? "
            "all candidates fail at least one numerical or depth constraint")

    # ?? 10. Write outputs ?????????????????????????????????????????????????????
    log("\n[10] Writing outputs...")

    results_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "ckks_poly_modulus_degree": 16384,
            "ckks_coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60],
            "ckks_scale": "2^40",
            "ckks_total_usable_levels": CKKS_TOTAL_LEVELS,
            "ckks_l2_lambda": CKKS_L2,
            "epsilon_poly": EPSILON_POLY,
            "epsilon_grad": EPSILON_GRAD,
        },
        "objective_audit": objective_audit,
        "baseline_model_audit": audit,
        "logit_distributions": logit_stats_all,
        "mhealth_poly_evaluation_on_ptbxl": poly_eval_results,
        "ptbxl_candidate_polynomials": candidates,
        "ckks_depth_feasibility": feasibility,
        "gradient_comparison": grad_results,
        "plaintext_training_sanity": {k: {
            "poly": v["poly"],
            "total_time_s": v["total_time_s"],
            "stable": v["stable"],
            "final_val_acc": v["final_val_acc"],
            "final_val_loss": v["final_val_loss"],
            "final_w_norm": v["final_w_norm"],
            "final_grad_norm": v["final_grad_norm"],
            "history": v["history"],
        } for k, v in train_sanity.items()},
        "candidate_selection": {
            "epsilon_poly": EPSILON_POLY,
            "epsilon_grad": EPSILON_GRAD,
            "results": selection_results,
            "recommendation": recommendation,
        },
    }

    json_file = OUT_DIR / "ptbxl_sigmoid_calibration.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    log(f"  Saved JSON : {json_file}")

    # CSV: polynomial evaluation table
    csv_rows = []
    for deg in [1, 3, 5]:
        for split in ["train_full", "train_bounded_p999", "val"]:
            r = poly_eval_results[f"deg{deg}"][split]
            row = {"source": "MHEALTH", "degree": deg, "split": split}
            row.update(r)
            csv_rows.append(row)
        for split in ["train_eval", "val_eval"]:
            r = candidates[f"deg{deg}"][split]
            row = {"source": "PTBXLfit", "degree": deg, "split": split.replace("_eval", "")}
            row.update(r)
            csv_rows.append(row)
    df_csv = pd.DataFrame(csv_rows)
    csv_file = OUT_DIR / "ptbxl_sigmoid_calibration.csv"
    df_csv.to_csv(csv_file, index=False)
    log(f"  Saved CSV  : {csv_file}")

    # ?? Write markdown documentation ??????????????????????????????????????????
    md_path = DOCS_DIR / "PTBXL_SIGMOID_CALIBRATION.md"
    _write_markdown(md_path, results_json, train_sanity, selection_results, recommendation)
    log(f"  Saved MD   : {md_path}")

    # Save execution log
    log_file = OUT_DIR / "ptbxl_sigmoid_calibration.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"  Saved log  : {log_file}")

    log("\n" + "=" * 72)
    log(f" FINAL RECOMMENDATION: {recommendation}")
    log("=" * 72)
    return 0


def _write_markdown(path: Path, data: dict, train_sanity: dict,
                    selection: list, recommendation: str):
    def _fmt(x, decimals=4):
        if x is None: return "?"
        return f"{x:.{decimals}f}" if isinstance(x, float) else str(x)

    lines = [
        "# PTB-XL Sigmoid Calibration & Objective-Consistency Audit",
        "",
        "> **Status**: Completed ? see `ptbxl_sigmoid_calibration.json` for full metrics.",
        "",
        f"**Generated**: {data['timestamp']}",
        "",
        "---",
        "",
        "## 1. Objective Consistency Audit",
        "",
        "| Property | Saved Baseline | CKKS Convention |",
        "| :--- | :--- | :--- |",
        f"| Intercept | Yes (`b ? {data['baseline_model_audit']['intercept_value']:.4f}`) | **No** |",
        f"| L2 regularization | sklearn C=1.0 (total, not per-sample) | ?=0.01 (per-sample) |",
        f"| Objective | sklearn LogLoss | (1/N)?X?err + ??w |",
        "| Compatible | ? | ? **Different objectives** |",
        "",
        "> [!IMPORTANT]",
        "> The saved baseline (`baseline_logistic_regression.joblib`) was fitted with `fit_intercept=True` and",
        "> `C=1.0`. The CKKS training circuit has no intercept. These are **different objectives**.",
        "> The saved baseline is **not overwritten**. A no-intercept plaintext reference is used for calibration.",
        "",
        "---",
        "",
        "## 2. PTB-XL Logit Distribution",
        "",
        "Logits computed from baseline model weights on **training and validation** sets only. Test fold excluded.",
        "",
        "| Split | Min | Max | Mean | Std | p1% | p50% | p99% |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for key in ["train_with_intercept", "val_with_intercept",
                 "train_no_intercept", "val_no_intercept"]:
        s = data["logit_distributions"][key]
        smin = s["min"]; smax = s["max"]; smean = s["mean"]; sstd = s["std"]
        q = s["quantiles"]
        q1 = q.get("q0100", 0.0); q50 = q.get("q0500", 0.0); q99 = q.get("q0990", 0.0)
        row = ("| " + key + " | " + str(round(smin, 2)) + " | " + str(round(smax, 2))
               + " | " + str(round(smean, 3)) + " | " + str(round(sstd, 3))
               + " | " + str(round(q1, 3)) + " | " + str(round(q50, 3))
               + " | " + str(round(q99, 3)) + " |")
        lines.append(row)

    lines += [
        "",
        "**Fraction outside approximation intervals (training, with intercept):**",
        "",
        "| Interval | Fraction Outside |",
        "| :--- | :--- |",
    ]
    frac = data["logit_distributions"]["train_with_intercept"]["fraction_outside_interval"]
    for k, v in frac.items():
        lines.append(f"| {k} | {v:.4f} ({v*100:.2f}%) |")

    lines += [
        "",
        "---",
        "",
        "## 3. MHEALTH Polynomial Evaluation on PTB-XL Logits",
        "",
        "| Degree | Split | Prob MaxAE | Prob MeanAE | Out-of-[0,1] | Grad MaxAE |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for deg in [1, 3, 5]:
        for split in ["train_full", "val"]:
            r = data["mhealth_poly_evaluation_on_ptbxl"][f"deg{deg}"][split]
            lines.append(
                f"| {deg} | {split} | {r['prob_max_ae']:.4f} | "
                f"{r['prob_mean_ae']:.4f} | {r['prob_out_of_01']} | "
                f"{r['grad_factor_max_ae']:.4f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 4. PTB-XL?Fitted Candidate Polynomials",
        "",
        "Fit performed on **training logits only** (CKKS convention, no intercept).",
        "Candidate selection uses **validation performance**. Test fold excluded.",
        "",
        "| Degree | c? | c? | c? | Fit Interval | Val Prob MaxAE | Val Out-of-[0,1] |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for deg in [1, 3, 5]:
        c = data["ptbxl_candidate_polynomials"][f"deg{deg}"]
        cc = c["coefficients"]
        ve = c["val_eval"]
        lines.append(
            f"| {deg} | {cc['c1']:.6f} | {cc['c3']:.6f} | {cc['c5']:.6f} | "
            f"{c['fit_interval']} | {ve['prob_max_ae']:.4f} | {ve['prob_out_of_01']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. CKKS Depth Feasibility",
        "",
        "CKKS configuration: N=16384, bits=[60,40,40,40,40,40,40,60], total usable levels=7.",
        "",
        "| Degree | Fwd Dot | Poly Levels | Bwd Grad | Total Used | Remaining | Feasible |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for deg in [1, 3, 5]:
        dc = data["ckks_depth_feasibility"][f"deg{deg}"]
        poly_lvl = levels_for_polynomial(deg)
        lines.append(
            f"| {deg} | {LEVELS_FWD_DOT} | {poly_lvl} | {LEVELS_BACKWARD} | "
            f"{dc['total_needed']} | {dc['budget_margin']} | "
            f"{'? YES' if dc['feasible'] else '? NO'} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Plaintext Training Sanity Check (50 epochs, val metrics only)",
        "",
        "CKKS convention: no intercept, lr=0.05, L2=0.01, clip=1.0. **No test data used.**",
        "",
        "| Polynomial | Val Acc (ep50) | Val Loss | W Norm | Grad Norm | Stable |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for poly, v in train_sanity.items():
        lines.append(
            f"| {poly} | {_fmt(v['final_val_acc'])} | {_fmt(v['final_val_loss'])} | "
            f"{_fmt(v['final_w_norm'])} | {_fmt(v['final_grad_norm'])} | "
            f"{'?' if v['stable'] else '?'} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 7. Final Candidate Assessment",
        "",
        f"Thresholds: ?_poly = {data['candidate_selection']['epsilon_poly']}, "
        f"?_grad = {data['candidate_selection']['epsilon_grad']}",
        "",
        "| Degree | Val Prob MaxAE | Grad MaxAE | CKKS Feasible | Training Stable | Val Acc@50 | Result |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in selection:
        status = "? PASS" if r["passes_all_checks"] else "? FAIL"
        lines.append(
            f"| {r['degree']} | {r['val_prob_max_ae']:.4f} | {r['grad_max_ae']:.4e} | "
            f"{'?' if r['depth_feasible'] else '?'} | "
            f"{'?' if r['training_stable'] else '?'} | "
            f"{_fmt(r['val_acc_epoch50'])} | {status} |"
        )

    lines += [
        "",
        f"> **Recommendation**: `{recommendation}`",
        "",
        "---",
        "",
        "## 8. Reproducibility",
        "",
        "```bash",
        "# Run calibration experiment",
        "python results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.py",
        "```",
        "",
        "Outputs written to `results/ptbxl/sigmoid_calibration/`:",
        "- `ptbxl_sigmoid_calibration.json` ? all measured values, coefficients, errors",
        "- `ptbxl_sigmoid_calibration.csv`  ? polynomial evaluation table",
        "- `ptbxl_sigmoid_calibration.log`  ? execution log",
        "- `docs/ptbxl/PTBXL_SIGMOID_CALIBRATION.md` ? this document",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
