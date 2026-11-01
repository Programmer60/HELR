#!/usr/bin/env python3
"""
results/ptbxl/scheduler/phase3b/analyze_phase3b_results.py
==========================================================
Analyzes the results of the Phase 3B matched encrypted training experiments.
Computes:
  1. Mean ± std and speedups for:
     - A-repeat vs B-repeat (Level intervention effect alone)
     - A-cache vs B-cache (Level intervention effect with caching held fixed)
     - A-repeat vs A-cache (Alignment caching effect alone under Condition A)
     - B-repeat vs B-cache (Alignment caching effect alone under Condition B)
     - A-repeat vs B-cache (Combined scheduling optimization effect)
  2. Model accuracy / convergence comparisons across configurations
  3. Cost model prediction accuracy (MAPE, error in seconds)
  4. Decision regret of the scheduler policy vs the oracle best schedule
  5. Formatted Markdown report tables for SCHEDULER_PHASE3B_TRAINER_INTEGRATION.md
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PHASE3B_DIR = Path(__file__).resolve().parent
RESULTS_CSV = PHASE3B_DIR / "phase3b_training_results.csv"
RESULTS_JSON = PHASE3B_DIR / "phase3b_training_results.json"
TIMING_CSV = PHASE3B_DIR / "phase3b_timing_breakdown.csv"
DECISIONS_JSON = PHASE3B_DIR / "phase3b_scheduler_decisions.json"
OUT_SUMMARY_JSON = PHASE3B_DIR / "phase3b_analysis_summary.json"


def main():
    if not RESULTS_CSV.exists():
        print(f"Error: {RESULTS_CSV} does not exist yet. Run training experiments first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} experiment runs from {RESULTS_CSV}")

    # Summary table by schedule_id
    grouped = df.groupby("schedule_id")
    stats = {}
    for sched_id, grp in grouped:
        stats[sched_id] = {
            "n_runs": len(grp),
            "cum_time_mean": float(grp["cumulative_training_s"].mean()),
            "cum_time_std": float(grp["cumulative_training_s"].std()),
            "epoch_time_mean": float(grp["mean_epoch_time_s"].mean()),
            "epoch_time_std": float(grp["mean_epoch_time_s"].std()),
            "grad_time_mean": float(grp["mean_gradient_comp_s"].mean()),
            "grad_time_std": float(grp["mean_gradient_comp_s"].std()),
            "pred_time_mean": float(grp["predicted_total_s"].mean()),
            "mape_mean": float(grp["prediction_mape_percent"].mean()),
            "mape_std": float(grp["prediction_mape_percent"].std()),
            "val_auc_mean": float(grp["final_val_auc"].mean()),
            "val_auc_std": float(grp["final_val_auc"].std()),
            "best_auc_mean": float(grp["best_val_auc"].mean()),
            "best_auc_std": float(grp["best_val_auc"].std()),
            "max_grad_ae": float(grp["max_grad_ae"].max()),
        }

    # Pairwise Comparisons (Matched across seeds)
    seeds = sorted(df["seed"].unique())
    comparisons = {
        "level_effect_repeated": [],   # A-repeat / B-repeat
        "level_effect_cached": [],     # A-cache / B-cache
        "cache_effect_cond_A": [],     # A-repeat / A-cache
        "cache_effect_cond_B": [],     # B-repeat / B-cache
        "combined_effect": [],         # A-repeat / B-cache
    }

    for s in seeds:
        sub = df[df["seed"] == s].set_index("schedule_id")
        t_a_rep = sub.loc["A-repeat", "cumulative_training_s"]
        t_a_cac = sub.loc["A-cache", "cumulative_training_s"]
        t_b_rep = sub.loc["B-repeat", "cumulative_training_s"]
        t_b_cac = sub.loc["B-cache", "cumulative_training_s"]

        comparisons["level_effect_repeated"].append(t_a_rep / t_b_rep)
        comparisons["level_effect_cached"].append(t_a_cac / t_b_cac)
        comparisons["cache_effect_cond_A"].append(t_a_rep / t_a_cac)
        comparisons["cache_effect_cond_B"].append(t_b_rep / t_b_cac)
        comparisons["combined_effect"].append(t_a_rep / t_b_cac)

    comp_summary = {}
    for k, v in comparisons.items():
        comp_summary[k] = {
            "mean_ratio": float(np.mean(v)),
            "std_ratio": float(np.std(v)),
            "min_ratio": float(np.min(v)),
            "max_ratio": float(np.max(v)),
        }

    # Oracle schedule and Decision Regret
    regrets = []
    for s in seeds:
        sub = df[df["seed"] == s]
        min_time = sub["cumulative_training_s"].min()
        oracle_sched = sub.loc[sub["cumulative_training_s"].idxmin(), "schedule_id"]
        chosen_time = sub.loc[sub["schedule_id"] == "B-cache", "cumulative_training_s"].values[0]
        regret_s = chosen_time - min_time
        regret_pct = (regret_s / min_time) * 100.0
        regrets.append({
            "seed": int(s),
            "oracle_schedule": oracle_sched,
            "oracle_time_s": float(min_time),
            "chosen_schedule": "B-cache",
            "chosen_time_s": float(chosen_time),
            "regret_s": float(regret_s),
            "regret_percent": float(regret_pct),
        })

    out_data = {
        "schedule_stats": stats,
        "pairwise_comparisons": comp_summary,
        "regret_evaluation": regrets,
    }

    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved analysis summary to {OUT_SUMMARY_JSON}")

    # Print textual summary
    print("\n" + "=" * 70)
    print("PHASE 3B EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    for s_id, s in stats.items():
        print(f"Schedule: {s_id:<10} | Total Time: {s['cum_time_mean']:6.2f} ± {s['cum_time_std']:4.2f}s | "
              f"Epoch Time: {s['epoch_time_mean']:5.2f}s | Grad Time: {s['grad_time_mean']:5.2f}s | "
              f"MAPE: {s['mape_mean']:4.1f}% | AUC: {s['val_auc_mean']:.4f}")

    print("\nPairwise Speedup Ratios (Mean ± Std):")
    print(f"  Level effect (repeated) [A-repeat / B-repeat]: {comp_summary['level_effect_repeated']['mean_ratio']:.3f}x")
    print(f"  Level effect (cached)   [A-cache / B-cache]  : {comp_summary['level_effect_cached']['mean_ratio']:.3f}x")
    print(f"  Cache effect (Cond A)   [A-repeat / A-cache] : {comp_summary['cache_effect_cond_A']['mean_ratio']:.3f}x")
    print(f"  Cache effect (Cond B)   [B-repeat / B-cache] : {comp_summary['cache_effect_cond_B']['mean_ratio']:.3f}x")
    print(f"  Combined Optimization   [A-repeat / B-cache] : {comp_summary['combined_effect']['mean_ratio']:.3f}x")


if __name__ == "__main__":
    main()
