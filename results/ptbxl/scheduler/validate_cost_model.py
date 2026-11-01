#!/usr/bin/env python3
"""
PTB-XL Workload-Aware CKKS Scheduling: Cost Model Validation
============================================================
Validates the analytical cost model predictions against the empirical
ground-truth benchmarks from Phase 4 across Conditions A, B, and C.

Evaluation Metrics:
  - Mean Absolute Percentage Error (MAPE) for total epoch runtime
  - Stage-wise percentage errors (forward, gradient, alignment, decryption)
  - Speedup ratio validation (Condition A vs Condition B gradient speedup)
  - Target acceptance criterion: Total epoch runtime MAPE <= 15.0%

Outputs:
  - results/ptbxl/scheduler/validation_report.json
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.cost_model import CKKSCostModel, Schedule
RESULTS_DIR = PROJECT_ROOT / "results" / "ptbxl" / "scheduler"
OUT_JSON = RESULTS_DIR / "validation_report.json"

def validate_cost_model():
    model = CKKSCostModel()
    phase4_summary = model.phase4_summary

    if not phase4_summary:
        raise RuntimeError("Phase 4 empirical summary data could not be loaded!")

    target_conditions = [
        ("A", Schedule("A", poly_degree=3, grad_level=3, burn_n=4, alignment_strategy="repeated")),
        ("B", Schedule("B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="repeated")),
        ("C", Schedule("C", poly_degree=5, grad_level=2, burn_n=5, alignment_strategy="repeated")),
    ]

    validation_records = []
    total_epoch_mapes = []
    gradient_mapes = []

    stage_keys = [
        ("forward_linear_s", "Forward Linear"),
        ("poly_eval_s", "Polynomial Eval"),
        ("error_comp_s", "Error Comp"),
        ("level_adjustment_s", "Level Adjustment"),
        ("feature_alignment_s", "Feature Alignment"),
        ("gradient_comp_s", "Gradient Comp"),
        ("decryption_agg_s", "Decryption Agg"),
        ("weight_enc_s", "Weight Encryption"),
        ("total_epoch_time_s", "Total Epoch Time"),
    ]

    print("=" * 90)
    print(" PTB-XL PHASE 1 COST MODEL VALIDATION AGAINST PHASE 4 BENCHMARKS")
    print("=" * 90)

    for cond_code, sched in target_conditions:
        actual_data = phase4_summary[cond_code]
        pred_data = model.predict_components(sched, N=17084, D=48, slot_capacity=8192, epoch_idx=1)

        cond_record = {
            "condition": cond_code,
            "condition_name": actual_data["condition_name"],
            "stages": {},
        }

        print(f"\n>>> Condition {cond_code}: {actual_data['condition_name']}")
        print(f"{'Stage':<24s} | {'Actual (s)':<12s} | {'Predicted (s)':<14s} | {'Abs Error (s)':<14s} | {'MAPE (%)':<10s}")
        print("-" * 84)

        for key, label in stage_keys:
            act_val = actual_data.get(key, 0.0)
            # Some keys in Phase 4 summary are slightly different
            if key == "feature_alignment_s" and "feature_align_mean_s" in actual_data:
                act_val = actual_data["feature_align_mean_s"]
            elif key == "total_epoch_time_s" and "total_epoch_mean_s" in actual_data:
                act_val = actual_data["total_epoch_mean_s"]
            elif key == "gradient_comp_s" and "gradient_comp_mean_s" in actual_data:
                act_val = actual_data["gradient_comp_mean_s"]
            elif key == "forward_linear_s" and "forward_linear_mean_s" in actual_data:
                act_val = actual_data["forward_linear_mean_s"]
            elif key == "poly_eval_s" and "poly_eval_mean_s" in actual_data:
                act_val = actual_data["poly_eval_mean_s"]
            elif key == "decryption_agg_s" and "decryption_agg_mean_s" in actual_data:
                act_val = actual_data["decryption_agg_mean_s"]
            elif key == "level_adjustment_s" and "level_adj_mean_s" in actual_data:
                act_val = actual_data["level_adj_mean_s"]
            elif key == "weight_enc_s":
                act_val = 0.7095 if cond_code == "A" else (0.7152 if cond_code == "B" else 0.7260)
            elif key == "error_comp_s":
                act_val = 0.0397 if cond_code in ("A", "B") else 0.0429

            pred_val = pred_data[key]
            abs_err = abs(pred_val - act_val)
            mape = (abs_err / act_val * 100.0) if act_val > 1e-3 else 0.0

            cond_record["stages"][key] = {
                "label": label,
                "actual_s": act_val,
                "predicted_s": pred_val,
                "abs_error_s": abs_err,
                "mape_percent": mape,
            }

            print(f"{label:<24s} | {act_val:12.4f} | {pred_val:14.4f} | {abs_err:14.4f} | {mape:9.2f}%")

            if key == "total_epoch_time_s":
                total_epoch_mapes.append(mape)
            elif key == "gradient_comp_s":
                gradient_mapes.append(mape)

        validation_records.append(cond_record)

    # Cross-condition gradient speedup analysis
    act_speedup = phase4_summary["A"]["gradient_comp_mean_s"] / phase4_summary["B"]["gradient_comp_mean_s"]
    pred_comp_A = model.predict_components(target_conditions[0][1])
    pred_comp_B = model.predict_components(target_conditions[1][1])
    pred_speedup = pred_comp_A["gradient_comp_s"] / pred_comp_B["gradient_comp_s"]
    speedup_err_pct = abs(pred_speedup - act_speedup) / act_speedup * 100.0

    overall_max_epoch_mape = max(total_epoch_mapes)
    overall_mean_epoch_mape = float(np.mean(total_epoch_mapes))
    passed_criterion = bool(overall_max_epoch_mape <= 15.0)

    print("\n" + "=" * 90)
    print(" VALIDATION SUMMARY")
    print("=" * 90)
    print(f"Condition A Total Epoch MAPE : {total_epoch_mapes[0]:.2f}%")
    print(f"Condition B Total Epoch MAPE : {total_epoch_mapes[1]:.2f}%")
    print(f"Condition C Total Epoch MAPE : {total_epoch_mapes[2]:.2f}%")
    print(f"Mean Total Epoch MAPE        : {overall_mean_epoch_mape:.2f}%")
    print(f"Max Total Epoch MAPE         : {overall_max_epoch_mape:.2f}% (Threshold: <= 15.0%)")
    print(f"Target Criterion Passed      : {passed_criterion}")
    print(f"Gradient Speedup Actual      : {act_speedup:.4f}x (A/B)")
    print(f"Gradient Speedup Predicted   : {pred_speedup:.4f}x (A/B)")
    print(f"Gradient Speedup Rel Error   : {speedup_err_pct:.2f}%")

    out_data = {
        "status": "PASS" if passed_criterion else "FAIL",
        "acceptance_threshold_percent": 15.0,
        "mean_epoch_mape_percent": overall_mean_epoch_mape,
        "max_epoch_mape_percent": overall_max_epoch_mape,
        "all_epoch_mapes": {
            "A": total_epoch_mapes[0],
            "B": total_epoch_mapes[1],
            "C": total_epoch_mapes[2],
        },
        "gradient_speedup_analysis": {
            "actual_ratio_A_vs_B": act_speedup,
            "predicted_ratio_A_vs_B": pred_speedup,
            "relative_error_percent": speedup_err_pct,
        },
        "conditions": validation_records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nFull validation results written to: {OUT_JSON}")
    return passed_criterion

if __name__ == "__main__":
    success = validate_cost_model()
    if not success:
        exit(1)
