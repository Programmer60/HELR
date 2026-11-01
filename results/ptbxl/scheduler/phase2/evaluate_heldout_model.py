#!/usr/bin/env python3
"""
PTB-XL Phase 2: Held-Out Model Evaluation & Decision Regret Analysis
===================================================================
Evaluates the frozen Phase 1 CKKSCostModel on the out-of-workload benchmark
grid results (9 workloads x 6 variants).

Analyses:
  1. Prediction accuracy out-of-workload:
     - Separates known reference configuration (W9: N=17084, D=48, B=3)
       from the 8 held-out configurations (W1-W8).
     - Calculates Absolute Error, Relative Error, and MAPE for each workload.
     - Evaluates against the 15.0% MAPE project acceptance criterion.
     - Factorial breakdown by N (batch count B), D (feature count),
       circuit schedule (A, B, C), and alignment policy (repeat, cache).
  2. Schedule Decision Regret:
     - Evaluates whether the scheduler selects the empirical best schedule.
     - Regret = (Measured_Time(Selected) / Measured_Time(Best)) - 1.
     - Evaluates against the 5.0% regret target.
  3. Feature-Alignment Caching Break-Even:
     - Evaluates empirical setup cost C vs per-step savings S across all 9 workloads.
  4. Numerical Feasibility & Pass Rate:
     - Verifies max_abs_gradient_error <= 1e-4.

Outputs:
  - results/ptbxl/scheduler/phase2/heldout_evaluation_report.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.cost_model import CKKSCostModel, Schedule

RESULTS_DIR = PROJECT_ROOT / "results" / "ptbxl" / "scheduler" / "phase2"
GRID_JSON = RESULTS_DIR / "workload_grid_results.json"
OUT_REPORT_JSON = RESULTS_DIR / "heldout_evaluation_report.json"


def evaluate_heldout_model():
    if not GRID_JSON.exists():
        raise FileNotFoundError(f"Workload grid results not found at: {GRID_JSON}")

    with open(GRID_JSON, "r") as f:
        grid_data = json.load(f)

    model = CKKSCostModel()
    summaries = grid_data["summary"]

    # Map summary entries by (workload_id, variant)
    meas_map = {(s["workload_id"], s["variant"]): s for s in summaries}

    workload_evaluations = []
    regret_records = []

    heldout_step_mapes = []
    heldout_10ep_mapes = []
    reference_step_mapes = []
    reference_10ep_mapes = []

    print("=" * 95)
    print(" PTB-XL PHASE 2: HELD-OUT MODEL EVALUATION & DECISION REGRET ANALYSIS")
    print("=" * 95)

    # Evaluate each workload
    for w_cfg in grid_data["workload_grid"]:
        wid = w_cfg["workload_id"]
        N = w_cfg["N"]
        D = w_cfg["D"]
        B = w_cfg["B"]
        is_ref = w_cfg["is_reference"]
        ref_label = " [REFERENCE]" if is_ref else " [HELD-OUT]"

        print(f"\n>>> Evaluating Workload {wid}{ref_label}: N={N:,}, D={D}, Batches={B}")
        print(f"{'Variant':<10s} | {'Meas Step(s)':<12s} | {'Pred Step(s)':<12s} | {'Step MAPE':<10s} | {'Meas 10Ep(s)':<12s} | {'Pred 10Ep(s)':<12s} | {'10Ep MAPE':<10s}")
        print("-" * 88)

        variant_predictions = {}
        variant_records = []

        for cond, deg, lvl, burn_n in [("A", 3, 3, 4), ("B", 3, 2, 5), ("C", 5, 2, 5)]:
            for strat in ["repeated", "cached"]:
                var_name = f"{cond}-repeat" if strat == "repeated" else f"{cond}-cache"
                sched = Schedule(condition=cond, poly_degree=deg, grad_level=lvl, burn_n=burn_n, alignment_strategy=strat)

                meas = meas_map[(wid, var_name)]
                actual_step = meas["total_step_mean_s"]
                actual_10ep = meas["ten_epoch_mean_s"]

                # Step prediction (for cached: step after precompute has alignment=0)
                step_epoch_idx = 1 if strat == "repeated" else 2
                pred_step = model.predict_epoch(sched, N=N, D=D, slot_capacity=8192, epoch_idx=step_epoch_idx)

                # 10-Epoch prediction
                pred_10ep_res = model.predict_training(sched, N=N, D=D, slot_capacity=8192, epochs=10)
                pred_10ep = pred_10ep_res["total_training_time_s"]

                step_mape = abs(pred_step - actual_step) / actual_step * 100.0
                ep10_mape = abs(pred_10ep - actual_10ep) / actual_10ep * 100.0

                if is_ref:
                    reference_step_mapes.append(step_mape)
                    reference_10ep_mapes.append(ep10_mape)
                else:
                    heldout_step_mapes.append(step_mape)
                    heldout_10ep_mapes.append(ep10_mape)

                variant_predictions[var_name] = {
                    "sched": sched,
                    "pred_10ep": pred_10ep,
                    "meas_10ep": actual_10ep,
                    "pred_step": pred_step,
                    "meas_step": actual_step,
                    "step_mape": step_mape,
                    "10ep_mape": ep10_mape,
                }

                print(f"{var_name:<10s} | {actual_step:12.4f} | {pred_step:12.4f} | {step_mape:9.2f}% | {actual_10ep:12.4f} | {pred_10ep:12.4f} | {ep10_mape:9.2f}%")

                variant_records.append({
                    "variant": var_name,
                    "condition": cond,
                    "alignment_strategy": strat,
                    "actual_step_s": actual_step,
                    "predicted_step_s": pred_step,
                    "step_abs_error_s": abs(pred_step - actual_step),
                    "step_mape_percent": step_mape,
                    "actual_10ep_s": actual_10ep,
                    "predicted_10ep_s": pred_10ep,
                    "10ep_abs_error_s": abs(pred_10ep - actual_10ep),
                    "10ep_mape_percent": ep10_mape,
                    "max_abs_error_vs_ref": meas["max_abs_error_vs_ref"],
                    "numerical_pass": bool(meas["max_abs_error_vs_ref"] <= 1e-4),
                })

        # Decision Regret Analysis for 10-epoch training
        # Selected = schedule with minimum predicted 10-epoch time
        # Best = schedule with minimum measured 10-epoch time
        selected_var = min(variant_predictions.keys(), key=lambda k: variant_predictions[k]["pred_10ep"])
        best_var = min(variant_predictions.keys(), key=lambda k: variant_predictions[k]["meas_10ep"])

        meas_time_selected = variant_predictions[selected_var]["meas_10ep"]
        meas_time_best = variant_predictions[best_var]["meas_10ep"]
        pred_time_selected = variant_predictions[selected_var]["pred_10ep"]

        regret = (meas_time_selected / meas_time_best) - 1.0
        regret_pct = regret * 100.0

        regret_records.append({
            "workload_id": wid,
            "N": N,
            "D": D,
            "B": B,
            "is_reference": is_ref,
            "predicted_best_variant": selected_var,
            "measured_best_variant": best_var,
            "correct_selection": bool(selected_var == best_var),
            "predicted_time_selected_s": pred_time_selected,
            "measured_time_selected_s": meas_time_selected,
            "measured_time_best_s": meas_time_best,
            "regret_ratio": regret,
            "regret_percent": regret_pct,
            "regret_pass": bool(regret_pct <= 5.0),
        })

        print(f"  -> Decision: Selected={selected_var} | Best={best_var} | Regret={regret_pct:.2f}% (Pass <= 5%: {regret_pct <= 5.0})")

        # Caching break-even calculation for Condition B
        meas_b_rep = meas_map[(wid, "B-repeat")]
        c_setup = meas_b_rep["alignment_precompute_mean_s"]
        s_per_ep = meas_b_rep["feature_alignment_mean_s"]
        break_even_exact = c_setup / s_per_ep if s_per_ep > 0 else float("inf")
        break_even_epoch = int(math.ceil(break_even_exact))

        workload_evaluations.append({
            "workload_id": wid,
            "N": N,
            "D": D,
            "B": B,
            "is_reference": is_ref,
            "selected_variant": selected_var,
            "best_variant": best_var,
            "regret_percent": regret_pct,
            "variants": variant_records,
            "caching_break_even": {
                "setup_cost_s": c_setup,
                "per_epoch_savings_s": s_per_ep,
                "exact_ratio": break_even_exact,
                "break_even_epoch_threshold": break_even_epoch,
            },
        })

    # Summary Statistics
    heldout_regrets = [r["regret_percent"] for r in regret_records if not r["is_reference"]]
    mean_heldout_regret = float(np.mean(heldout_regrets))
    max_heldout_regret = float(np.max(heldout_regrets))
    heldout_selection_acc = float(np.mean([r["correct_selection"] for r in regret_records if not r["is_reference"]])) * 100.0

    mean_heldout_step_mape = float(np.mean(heldout_step_mapes))
    max_heldout_step_mape = float(np.max(heldout_step_mapes))
    mean_heldout_10ep_mape = float(np.mean(heldout_10ep_mapes))
    max_heldout_10ep_mape = float(np.max(heldout_10ep_mapes))

    mean_ref_step_mape = float(np.mean(reference_step_mapes))
    mean_ref_10ep_mape = float(np.mean(reference_10ep_mapes))

    passed_mape_criterion = bool(mean_heldout_step_mape <= 15.0 and mean_heldout_10ep_mape <= 15.0)
    passed_regret_criterion = bool(max_heldout_regret <= 5.0)

    print("\n" + "=" * 95)
    print(" HELD-OUT VALIDATION EXECUTIVE SUMMARY")
    print("=" * 95)
    print(f"Held-Out Workloads Evaluated       : 8 (W1 through W8)")
    print(f"Known Reference Workload           : 1 (W9: N=17,084, D=48, B=3)")
    print(f"Held-Out Mean Step MAPE            : {mean_heldout_step_mape:.2f}% (Threshold: <= 15.0%)")
    print(f"Held-Out Max Step MAPE             : {max_heldout_step_mape:.2f}%")
    print(f"Held-Out Mean 10-Epoch MAPE        : {mean_heldout_10ep_mape:.2f}% (Threshold: <= 15.0%)")
    print(f"Held-Out Max 10-Epoch MAPE         : {max_heldout_10ep_mape:.2f}%")
    print(f"Reference Workload Mean Step MAPE  : {mean_ref_step_mape:.2f}%")
    print(f"Reference Workload Mean 10Ep MAPE  : {mean_ref_10ep_mape:.2f}%")
    print(f"Overall 15% MAPE Target Status     : {'PASS' if passed_mape_criterion else 'FAIL'}")
    print("-" * 95)
    print(f"Schedule Decision Accuracy         : {heldout_selection_acc:.1f}%")
    print(f"Mean Decision Regret (Held-Out)    : {mean_heldout_regret:.2f}%")
    print(f"Max Decision Regret (Held-Out)     : {max_heldout_regret:.2f}% (Target: <= 5.0%)")
    print(f"Overall 5% Regret Target Status    : {'PASS' if passed_regret_criterion else 'FAIL'}")

    final_report = {
        "status": "PASS" if (passed_mape_criterion and passed_regret_criterion) else "FAIL",
        "acceptance_criteria": {
            "mape_threshold_percent": 15.0,
            "regret_threshold_percent": 5.0,
            "gradient_error_threshold": 1e-4,
        },
        "heldout_metrics": {
            "mean_step_mape_percent": mean_heldout_step_mape,
            "max_step_mape_percent": max_heldout_step_mape,
            "mean_10ep_mape_percent": mean_heldout_10ep_mape,
            "max_10ep_mape_percent": max_heldout_10ep_mape,
            "mape_criterion_passed": passed_mape_criterion,
            "mean_regret_percent": mean_heldout_regret,
            "max_regret_percent": max_heldout_regret,
            "regret_criterion_passed": passed_regret_criterion,
            "selection_accuracy_percent": heldout_selection_acc,
        },
        "reference_metrics": {
            "workload_id": "W9",
            "mean_step_mape_percent": mean_ref_step_mape,
            "mean_10ep_mape_percent": mean_ref_10ep_mape,
        },
        "workloads": workload_evaluations,
        "regret_analysis": regret_records,
    }

    with open(OUT_REPORT_JSON, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nHeld-out evaluation report saved to: {OUT_REPORT_JSON}")
    return passed_mape_criterion and passed_regret_criterion


if __name__ == "__main__":
    success = evaluate_heldout_model()
    if not success:
        sys.exit(1)
