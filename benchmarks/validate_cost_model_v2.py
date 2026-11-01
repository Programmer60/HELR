#!/usr/bin/env python3
"""
PTB-XL Phase 3A: Held-Out Boundary Validation of CostModel_v2
============================================================
Validates the context-aware CostModel_v2 against a new held-out validation grid
centered on SIMD batch boundaries:

Workload Grid (8 new boundary workloads):
  N in {8192, 8193, 16384, 16385}
  D in {12, 48}
  S = 8192
  B = ceil(N / S) in {1, 2, 2, 3}

Tests batch transitions:
  - 8192 (B=1, full batch) vs 8193 (B=2, boundary transition with 1 valid record in batch 2)
  - 16384 (B=2, full batches) vs 16385 (B=3, boundary transition with 1 valid record in batch 3)

Variants Benchmarked:
  - A-repeat, A-cache (Condition A: Deg-3, Grad L3, burn 4)
  - B-repeat, B-cache (Condition B: Deg-3, Grad L2, burn 5 via * 1.0)
  - C-repeat, C-cache (Condition C: Deg-5, Grad L2, burn 5)

Pre-Registration:
  Predictions from CostModel_v2 are generated and locked prior to running measurements.

Outputs:
  - results/ptbxl/scheduler/phase3a/preregistered_predictions.json
  - results/ptbxl/scheduler/phase3a/validation_grid_results.json
  - results/ptbxl/scheduler/phase3a/validation_grid_results.csv
  - results/ptbxl/scheduler/phase3a/validation_grid.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tenseal as ts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.cost_model_v2 import CKKSCostModelV2, ScheduleV2

RESULTS_DIR = PROJECT_ROOT / "results" / "ptbxl" / "scheduler" / "phase3a"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREREG_JSON = RESULTS_DIR / "preregistered_predictions.json"
OUT_JSON = RESULTS_DIR / "validation_grid_results.json"
OUT_CSV = RESULTS_DIR / "validation_grid_results.csv"
OUT_LOG = RESULTS_DIR / "validation_grid.log"

GLOBAL_SEED = 1042
SLOT_CAPACITY = 8192
L2_LAMBDA = 0.01
REPETITIONS = 5  # 5 measured repetitions per variant per workload

BOUNDARY_GRID = [
    {"workload_id": "VB1", "N": 8192, "D": 12, "B": 1, "desc": "B=1 boundary, full capacity"},
    {"workload_id": "VB2", "N": 8192, "D": 48, "B": 1, "desc": "B=1 boundary, full capacity"},
    {"workload_id": "VB3", "N": 8193, "D": 12, "B": 2, "desc": "B=2 boundary transition (+1 record)"},
    {"workload_id": "VB4", "N": 8193, "D": 48, "B": 2, "desc": "B=2 boundary transition (+1 record)"},
    {"workload_id": "VB5", "N": 16384, "D": 12, "B": 2, "desc": "B=2 boundary, full capacity"},
    {"workload_id": "VB6", "N": 16384, "D": 48, "B": 2, "desc": "B=2 boundary, full capacity"},
    {"workload_id": "VB7", "N": 16385, "D": 12, "B": 3, "desc": "B=3 boundary transition (+1 record)"},
    {"workload_id": "VB8", "N": 16385, "D": 48, "B": 3, "desc": "B=3 boundary transition (+1 record)"},
]

POLY = {
    3: {"c0": 0.5, "c1": 0.15012, "c3": -0.001593, "c5": 0.0},
    5: {"c0": 0.5, "c1": 0.217101, "c3": -0.007823, "c5": 0.000118},
}


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


def burn(v, n: int):
    for _ in range(n):
        v = v * 1.0
    return v


def poly_sigmoid(z: np.ndarray, deg: int) -> np.ndarray:
    p = POLY[deg]
    out = p["c0"] + p["c1"] * z + p["c3"] * (z**3)
    if deg == 5:
        out = out + p["c5"] * (z**5)
    return out


def enc_sigmoid(enc_z, deg: int):
    p = POLY[deg]
    if deg == 3:
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        t3 = z3 * p["c3"]
        t1 = burn(enc_z, 2) * p["c1"]
        return t3 + t1 + p["c0"]
    elif deg == 5:
        z2 = enc_z * enc_z
        z3 = z2 * burn(enc_z, 1)
        z5 = z3 * burn(z2, 1)
        t5 = z5 * p["c5"]
        t3 = burn(z3, 1) * p["c3"]
        t1 = burn(enc_z, 3) * p["c1"]
        return t5 + t3 + t1 + p["c0"]
    raise ValueError(f"Unsupported degree: {deg}")


def partition_synthetic_dataset(N: int, D: int, S: int = SLOT_CAPACITY):
    rng = np.random.default_rng(GLOBAL_SEED + N + D)
    X = rng.normal(0.0, 1.0, size=(N, D))
    y = (X[:, 0] > 0.0).astype(np.float64)
    w_init = rng.uniform(-0.05, 0.05, size=D)

    num_batches = int(np.ceil(N / S))
    batches = []
    for b in range(num_batches):
        start = b * S
        end = min(start + S, N)
        n_valid = end - start
        n_unused = S - n_valid

        X_valid = X[start:end]
        y_valid = y[start:end]

        if n_unused > 0:
            X_pad = np.pad(X_valid, ((0, n_unused), (0, 0)), mode="constant", constant_values=0.0)
            y_pad = np.pad(y_valid, (0, n_unused), mode="constant", constant_values=0.0)
        else:
            X_pad = X_valid.copy()
            y_pad = y_valid.copy()

        batches.append({
            "b_idx": b + 1,
            "n_valid": n_valid,
            "n_unused": n_unused,
            "X_pad": X_pad,
            "y_pad": y_pad,
        })

    return X, y, w_init, batches


def compute_pt_ref_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, deg: int, l2: float = L2_LAMBDA) -> np.ndarray:
    N = X.shape[0]
    z = X @ w
    p = poly_sigmoid(z, deg)
    err = p - y
    grad_unreg = (X.T @ err) / N
    return grad_unreg + l2 * w


def preregister_predictions() -> Dict[str, Any]:
    """Generates and locks all CostModel_v2 predictions before taking benchmark measurements."""
    model = CKKSCostModelV2()
    schedules = model.get_all_schedules()

    prereg_records = {}
    for w in BOUNDARY_GRID:
        wid = w["workload_id"]
        N = w["N"]
        D = w["D"]
        B = w["B"]

        prereg_records[wid] = {
            "workload_id": wid,
            "N": N,
            "D": D,
            "B": B,
            "desc": w["desc"],
            "persistent_worker": {},
            "fresh_worker": {},
        }

        for ctx in ["persistent_worker", "fresh_worker"]:
            for s in schedules:
                var_key = f"{s.condition}-{'repeat' if s.alignment_strategy == 'repeated' else 'cache'}"
                comp = model.predict_components(s, N=N, D=D, epoch_idx=1, context=ctx)
                train_res = model.predict_training(s, N=N, D=D, epochs=10, context=ctx)
                step_after_precompute = (
                    comp["total_epoch_time_s"] - comp["feature_alignment_s"]
                    if s.alignment_strategy == "cached"
                    else comp["total_epoch_time_s"]
                )

                prereg_records[wid][ctx][var_key] = {
                    "schedule_name": s.name,
                    "condition": s.condition,
                    "alignment_strategy": s.alignment_strategy,
                    "predicted_step_s": step_after_precompute,
                    "predicted_epoch1_s": comp["total_epoch_time_s"],
                    "predicted_10ep_s": train_res["total_training_time_s"],
                    "components_s": comp,
                }

            # Pre-registered schedule decision (Step 5: primary degree-3 choice)
            rec_sched = model.recommend_best_schedule(N=N, D=D, epochs=10, context=ctx)
            rec_var = f"{rec_sched.condition}-{'repeat' if rec_sched.alignment_strategy == 'repeated' else 'cache'}"
            prereg_records[wid][ctx]["recommended_primary_schedule"] = rec_var

    with open(PREREG_JSON, "w") as f:
        json.dump(prereg_records, f, indent=2)

    return prereg_records


def run_benchmark_for_workload(
    ctx: ts.Context,
    w_cfg: Dict[str, Any],
    reps: int = REPETITIONS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs empirical benchmark for a single boundary workload configuration."""
    wid = w_cfg["workload_id"]
    N = w_cfg["N"]
    D = w_cfg["D"]
    B = w_cfg["B"]

    X, y, w_init, batches = partition_synthetic_dataset(N, D, SLOT_CAPACITY)

    # Pre-encrypt dataset columns
    enc_batches = []
    for b in batches:
        enc_X_cols = [ts.ckks_vector(ctx, b["X_pad"][:, j].tolist()) for j in range(D)]
        enc_y = ts.ckks_vector(ctx, b["y_pad"].tolist())
        enc_batches.append({"enc_X": enc_X_cols, "enc_y": enc_y})

    # Plaintext gradient references
    ref_grad_deg3 = compute_pt_ref_gradient(X, y, w_init, 3)
    ref_grad_deg5 = compute_pt_ref_gradient(X, y, w_init, 5)

    conditions = [
        ("A", 3, 4),  # Deg 3, Grad L3, burn 4
        ("B", 3, 5),  # Deg 3, Grad L2, burn 5
        ("C", 5, 5),  # Deg 5, Grad L2, burn 5
    ]

    # Warm-up pass
    for cond, deg, burn_X in conditions:
        enc_w_cols = [ts.ckks_vector(ctx, [float(w_init[j])] * SLOT_CAPACITY) for j in range(D)]
        for b in enc_batches:
            enc_z = enc_batches[0]["enc_X"][0] * enc_w_cols[0]
            for j in range(1, D):
                enc_z = enc_z + (enc_batches[0]["enc_X"][j] * enc_w_cols[j])
            enc_p = enc_sigmoid(enc_z, deg)
            burn_y = 4 if deg == 3 else 5
            enc_err = enc_p - burn(enc_batches[0]["enc_y"], burn_y)
            enc_err_adj = (enc_err * 1.0) if cond == "B" else enc_err
            enc_X_al = [burn(enc_batches[0]["enc_X"][j], burn_X) for j in range(D)]
            sums = [(enc_err_adj * enc_X_al[j]).sum() for j in range(D)]
            _ = [s.decrypt()[0] for s in sums]

    # Measured repetitions
    raw_measurements = []
    variant_timings: Dict[str, Dict[str, List[float]]] = {
        f"{c}-{s}": {"step_s": [], "10ep_s": [], "grad_s": [], "fwd_s": [], "align_s": [], "max_ae": []}
        for c in ["A", "B", "C"]
        for s in ["repeat", "cache"]
    }
    setup_costs: Dict[str, List[float]] = {"A": [], "B": [], "C": []}

    for r in range(1, reps + 1):
        for cond, deg, burn_X in conditions:
            t0_step = time.perf_counter()

            # 1. Weight Encryption
            t0_wenc = time.perf_counter()
            enc_w_cols = [ts.ckks_vector(ctx, [float(w_init[j])] * SLOT_CAPACITY) for j in range(D)]
            t_wenc = time.perf_counter() - t0_wenc

            # 2. Alignment Precompute (measure setup cost for caching)
            t0_pre = time.perf_counter()
            for b in enc_batches:
                _ = [burn(b["enc_X"][j], burn_X) for j in range(D)]
            t_precompute = time.perf_counter() - t0_pre
            setup_costs[cond].append(t_precompute)

            unreg_sum = np.zeros(D, dtype=np.float64)
            t_fwd_total = 0.0
            t_poly_total = 0.0
            t_err_total = 0.0
            t_ladj_total = 0.0
            t_align_total = 0.0
            t_grad_total = 0.0
            t_dec_total = 0.0

            for b in enc_batches:
                # Forward linear
                t0_fwd = time.perf_counter()
                enc_z = b["enc_X"][0] * enc_w_cols[0]
                for j in range(1, D):
                    enc_z = enc_z + (b["enc_X"][j] * enc_w_cols[j])
                t_fwd_total += (time.perf_counter() - t0_fwd)

                # Poly eval
                t0_poly = time.perf_counter()
                enc_p = enc_sigmoid(enc_z, deg)
                t_poly_total += (time.perf_counter() - t0_poly)

                # Error comp
                t0_err = time.perf_counter()
                burn_y = 4 if deg == 3 else 5
                enc_err = enc_p - burn(b["enc_y"], burn_y)
                t_err_total += (time.perf_counter() - t0_err)

                # Level adj
                t0_ladj = time.perf_counter()
                enc_err_adj = (enc_err * 1.0) if cond == "B" else enc_err
                t_ladj_total += (time.perf_counter() - t0_ladj)

                # Alignment
                t0_al = time.perf_counter()
                enc_X_aligned = [burn(b["enc_X"][j], burn_X) for j in range(D)]
                t_align_total += (time.perf_counter() - t0_al)

                # Gradient
                t0_gr = time.perf_counter()
                enc_sums = [(enc_err_adj * enc_X_aligned[j]).sum() for j in range(D)]
                t_grad_total += (time.perf_counter() - t0_gr)

                # Decryption
                t0_dec = time.perf_counter()
                dec_sums = np.array([g.decrypt()[0] for g in enc_sums])
                unreg_sum += dec_sums
                t_dec_total += (time.perf_counter() - t0_dec)

            g_ckks = (unreg_sum / N) + L2_LAMBDA * w_init
            ref_grad = ref_grad_deg3 if deg == 3 else ref_grad_deg5
            max_ae = float(np.max(np.abs(g_ckks - ref_grad)))

            step_repeated = time.perf_counter() - t0_step
            step_cached = step_repeated - t_align_total
            ten_ep_repeated = 10.0 * step_repeated
            ten_ep_cached = t_precompute + 10.0 * step_cached

            var_rep = f"{cond}-repeat"
            var_cac = f"{cond}-cache"

            variant_timings[var_rep]["step_s"].append(step_repeated)
            variant_timings[var_rep]["10ep_s"].append(ten_ep_repeated)
            variant_timings[var_rep]["grad_s"].append(t_grad_total)
            variant_timings[var_rep]["fwd_s"].append(t_fwd_total)
            variant_timings[var_rep]["align_s"].append(t_align_total)
            variant_timings[var_rep]["max_ae"].append(max_ae)

            variant_timings[var_cac]["step_s"].append(step_cached)
            variant_timings[var_cac]["10ep_s"].append(ten_ep_cached)
            variant_timings[var_cac]["grad_s"].append(t_grad_total)
            variant_timings[var_cac]["fwd_s"].append(t_fwd_total)
            variant_timings[var_cac]["align_s"].append(0.0)
            variant_timings[var_cac]["max_ae"].append(max_ae)

            raw_measurements.append({
                "workload_id": wid,
                "N": N,
                "D": D,
                "B": B,
                "condition": cond,
                "repetition": r,
                "step_repeated_s": step_repeated,
                "step_cached_s": step_cached,
                "ten_ep_repeated_s": ten_ep_repeated,
                "ten_ep_cached_s": ten_ep_cached,
                "fwd_s": t_fwd_total,
                "poly_s": t_poly_total,
                "err_s": t_err_total,
                "ladj_s": t_ladj_total,
                "align_s": t_align_total,
                "precompute_s": t_precompute,
                "grad_s": t_grad_total,
                "dec_s": t_dec_total,
                "wenc_s": t_wenc,
                "max_abs_error": max_ae,
            })

    # Summary per variant
    summary: Dict[str, Any] = {}
    for var, data in variant_timings.items():
        summary[var] = {
            "mean_step_s": float(np.mean(data["step_s"])),
            "std_step_s": float(np.std(data["step_s"])),
            "mean_10ep_s": float(np.mean(data["10ep_s"])),
            "std_10ep_s": float(np.std(data["10ep_s"])),
            "mean_grad_s": float(np.mean(data["grad_s"])),
            "mean_fwd_s": float(np.mean(data["fwd_s"])),
            "mean_align_s": float(np.mean(data["align_s"])),
            "max_abs_error": float(np.max(data["max_ae"])),
            "numerical_pass": bool(np.max(data["max_ae"]) <= 1e-4),
        }

    caching_analysis = {
        "setup_cost_A_s": float(np.mean(setup_costs["A"])),
        "setup_cost_B_s": float(np.mean(setup_costs["B"])),
        "setup_cost_C_s": float(np.mean(setup_costs["C"])),
        "per_step_savings_B_s": summary["B-repeat"]["mean_align_s"],
        "break_even_exact_B": (
            float(np.mean(setup_costs["B"])) / summary["B-repeat"]["mean_align_s"]
            if summary["B-repeat"]["mean_align_s"] > 0
            else float("inf")
        ),
        "break_even_epoch_B": int(math.ceil(float(np.mean(setup_costs["B"])) / summary["B-repeat"]["mean_align_s"])),
    }

    return raw_measurements, {"variants": summary, "caching": caching_analysis}


def main():
    print("=" * 95)
    print(" PTB-XL PHASE 3A: HELD-OUT BOUNDARY VALIDATION OF COSTMODEL_V2")
    print("=" * 95)
    print(f"Platform: {platform.platform()} | Python: {platform.python_version()} | TenSEAL: {ts.__version__}")
    print(f"Workload Grid: 8 Boundary Workloads (N in {{8192, 8193, 16384, 16385}}, D in {{12, 48}})")
    print(f"Repetitions per variant: {REPETITIONS} (persistent process steady state)")
    print("=" * 95)

    # 1. Pre-register predictions
    print("\n[1/3] Pre-registering CostModel_v2 predictions across boundary grid...", flush=True)
    prereg = preregister_predictions()
    print(f"      Predictions locked and saved to: {PREREG_JSON}", flush=True)

    # 2. Run boundary benchmark
    print("\n[2/3] Executing empirical boundary benchmarks...", flush=True)
    ctx = build_context()
    all_raw_records = []
    workload_evaluations = []

    step_mapes = []
    ten_ep_mapes = []
    regrets = []

    for w_cfg in BOUNDARY_GRID:
        wid = w_cfg["workload_id"]
        N = w_cfg["N"]
        D = w_cfg["D"]
        B = w_cfg["B"]
        desc = w_cfg["desc"]

        print(f"\n>>> Running Workload {wid}: N={N:,}, D={D}, Batches={B} ({desc})...", flush=True)
        t0_w = time.perf_counter()
        raw_recs, w_summary = run_benchmark_for_workload(ctx, w_cfg, reps=REPETITIONS)
        t_w = time.perf_counter() - t0_w
        print(f"    Completed in {t_w:.2f}s", flush=True)

        all_raw_records.extend(raw_recs)

        # Compare predicted vs measured for each variant
        w_eval_variants = {}
        for var in ["A-repeat", "A-cache", "B-repeat", "B-cache", "C-repeat", "C-cache"]:
            meas_s = w_summary["variants"][var]
            pred_s = prereg[wid]["persistent_worker"][var]

            actual_step = meas_s["mean_step_s"]
            actual_10ep = meas_s["mean_10ep_s"]
            pred_step = pred_s["predicted_step_s"]
            pred_10ep = pred_s["predicted_10ep_s"]

            step_mape = abs(pred_step - actual_step) / actual_step * 100.0
            ep10_mape = abs(pred_10ep - actual_10ep) / actual_10ep * 100.0

            step_mapes.append(step_mape)
            ten_ep_mapes.append(ep10_mape)

            w_eval_variants[var] = {
                "measured_step_s": actual_step,
                "predicted_step_s": pred_step,
                "step_mape_pct": step_mape,
                "measured_10ep_s": actual_10ep,
                "predicted_10ep_s": pred_10ep,
                "10ep_mape_pct": ep10_mape,
                "measured_grad_s": meas_s["mean_grad_s"],
                "measured_fwd_s": meas_s["mean_fwd_s"],
                "max_abs_error": meas_s["max_abs_error"],
                "numerical_pass": meas_s["numerical_pass"],
            }

            print(f"    {var:<10s} | Meas Step: {actual_step:6.3f}s | Pred Step: {pred_step:6.3f}s | MAPE: {step_mape:5.2f}% | Meas 10Ep: {actual_10ep:7.3f}s | Pred 10Ep: {pred_10ep:7.3f}s | MAPE: {ep10_mape:5.2f}%")

        # Regret Analysis (Primary Degree-3 set: A-repeat, A-cache, B-repeat, B-cache)
        primary_vars = ["A-repeat", "A-cache", "B-repeat", "B-cache"]
        pred_best_var = min(primary_vars, key=lambda k: prereg[wid]["persistent_worker"][k]["predicted_10ep_s"])
        meas_best_var = min(primary_vars, key=lambda k: w_summary["variants"][k]["mean_10ep_s"])

        meas_time_selected = w_summary["variants"][pred_best_var]["mean_10ep_s"]
        meas_time_best = w_summary["variants"][meas_best_var]["mean_10ep_s"]
        regret_pct = ((meas_time_selected / meas_time_best) - 1.0) * 100.0
        regrets.append(regret_pct)

        print(f"    -> Decision: Selected={pred_best_var} | Best={meas_best_var} | Regret={regret_pct:.2f}% (Pass <= 5%: {regret_pct <= 5.0})")

        workload_evaluations.append({
            "workload_id": wid,
            "N": N,
            "D": D,
            "B": B,
            "desc": desc,
            "selected_schedule": pred_best_var,
            "best_schedule": meas_best_var,
            "regret_pct": regret_pct,
            "regret_pass": bool(regret_pct <= 5.0),
            "variants": w_eval_variants,
            "caching": w_summary["caching"],
        })

    # 3. Overall Summary & Target Assessment
    mean_step_mape = float(np.mean(step_mapes))
    max_step_mape = float(np.max(step_mapes))
    mean_10ep_mape = float(np.mean(ten_ep_mapes))
    max_10ep_mape = float(np.max(ten_ep_mapes))
    mean_regret = float(np.mean(regrets))
    max_regret = float(np.max(regrets))
    max_error_all = float(max(r["max_abs_error"] for r in all_raw_records))
    all_numerical_pass = bool(max_error_all <= 1e-4)

    passed_mape = bool(mean_step_mape <= 15.0 and mean_10ep_mape <= 15.0)
    passed_regret = bool(max_regret <= 5.0)
    passed_numerical = all_numerical_pass

    print("\n" + "=" * 95)
    print(" HELD-OUT BOUNDARY VALIDATION EXECUTIVE SUMMARY")
    print("=" * 95)
    print(f"Boundary Workloads Evaluated       : 8 ({[w['workload_id'] for w in BOUNDARY_GRID]})")
    print(f"Mean Step Runtime MAPE             : {mean_step_mape:.2f}% (Target: <= 15.0%) -> {'PASS' if mean_step_mape <= 15.0 else 'FAIL'}")
    print(f"Max Step Runtime MAPE              : {max_step_mape:.2f}%")
    print(f"Mean 10-Epoch Runtime MAPE         : {mean_10ep_mape:.2f}% (Target: <= 15.0%) -> {'PASS' if mean_10ep_mape <= 15.0 else 'FAIL'}")
    print(f"Max 10-Epoch Runtime MAPE          : {max_10ep_mape:.2f}%")
    print(f"Mean Decision Regret               : {mean_regret:.2f}%")
    print(f"Max Decision Regret                : {max_regret:.2f}% (Target: <= 5.0%) -> {'PASS' if passed_regret else 'FAIL'}")
    print(f"Max Absolute Gradient Error vs Ref : {max_error_all:.2e} (Target: <= 1e-4) -> {'PASS' if passed_numerical else 'FAIL'}")
    print(f"Overall Phase 3A Validation Status : {'PASS' if (passed_mape and passed_regret and passed_numerical) else 'FAIL'}")

    final_results = {
        "status": "PASS" if (passed_mape and passed_regret and passed_numerical) else "FAIL",
        "criteria": {
            "mape_target_pct": 15.0,
            "regret_target_pct": 5.0,
            "grad_error_target": 1e-4,
        },
        "summary_metrics": {
            "mean_step_mape_pct": mean_step_mape,
            "max_step_mape_pct": max_step_mape,
            "mean_10ep_mape_pct": mean_10ep_mape,
            "max_10ep_mape_pct": max_10ep_mape,
            "mean_regret_pct": mean_regret,
            "max_regret_pct": max_regret,
            "max_abs_gradient_error": max_error_all,
            "passed_mape": passed_mape,
            "passed_regret": passed_regret,
            "passed_numerical": passed_numerical,
        },
        "workloads": workload_evaluations,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(final_results, f, indent=2)

    df_raw = pd.DataFrame(all_raw_records)
    df_raw.to_csv(OUT_CSV, index=False)

    print(f"\nArtifacts saved successfully:")
    print(f"  JSON: {OUT_JSON}")
    print(f"  CSV : {OUT_CSV}")


if __name__ == "__main__":
    main()
