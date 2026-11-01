#!/usr/bin/env python3
"""
results/ptbxl/scheduler/phase3b/verify_phase3b_statistics.py
============================================================
Phase 4B Statistical Verification Script.

Reads ONLY the existing Phase 3B raw JSON/CSV experimental logs:
  - results/ptbxl/scheduler/phase3b/phase3b_training_results.json
  - results/ptbxl/scheduler/phase3b/phase3b_training_results.csv

Calculates and prints all project statistics with explicit mathematical formulas:
  1. Mean and Population Standard Deviation (ddof=0) for cumulative training time.
  2. Seed-matched pairwise speedup ratios:
       R_s = T_{baseline, s} / T_{optimized, s}
       Mean = (1/3) * sum(R_s)
       Pop Std = sqrt( (1/3) * sum((R_s - Mean)^2) )
  3. Ratio of aggregate means:
       R_agg = Mean(T_baseline) / Mean(T_optimized)
  4. Validation accuracy and ROC-AUC per seed, mean, and population standard deviation.
  5. Decision regret per seed.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RESULTS_JSON = ROOT / "results" / "ptbxl" / "scheduler" / "phase3b" / "phase3b_training_results.json"
RESULTS_CSV = ROOT / "results" / "ptbxl" / "scheduler" / "phase3b" / "phase3b_training_results.csv"


def main():
    print("=" * 80)
    print(" PHASE 4B: INDEPENDENT STATISTICAL VERIFICATION")
    print("=" * 80)
    print(f"Reading raw results from: {RESULTS_JSON}")

    with open(RESULTS_JSON, "r") as f:
        runs = json.load(f)

    df = pd.DataFrame([
        {
            "schedule_id": r["schedule_id"],
            "seed": r["seed"],
            "setup_alignment_time_s": r["setup_alignment_time_s"],
            "cumulative_training_s": r["cumulative_training_s"],
            "mean_epoch_time_s": r["mean_epoch_time_s"],
            "mean_gradient_comp_s": r["mean_gradient_comp_s"],
            "val_auc": r["final_val_auc_exact"],
            "val_acc": r["final_val_accuracy_exact"],
            "max_grad_ae": r["max_grad_ae"],
        }
        for r in runs
    ])

    seeds = sorted(df["seed"].unique())
    schedules = ["A-repeat", "A-cache", "B-repeat", "B-cache"]

    print(f"\n1. PER-SCHEDULE AGGREGATE TIMINGS (N = 3 seeds: {seeds})")
    print("   Convention: Mean +/- Population Standard Deviation (ddof=0)")
    print("-" * 80)
    print(f"{'Schedule':<10} | {'Cum Time Mean +/- PopStd (s)':<30} | {'Mean Epoch (s)':<15} | {'Mean Grad (s)':<15}")
    print("-" * 80)

    sched_means = {}
    for s_id in schedules:
        sub = df[df["schedule_id"] == s_id]
        cum_mean = float(np.mean(sub["cumulative_training_s"]))
        cum_std_pop = float(np.std(sub["cumulative_training_s"], ddof=0))
        cum_std_samp = float(np.std(sub["cumulative_training_s"], ddof=1))
        ep_mean = float(np.mean(sub["mean_epoch_time_s"]))
        gr_mean = float(np.mean(sub["mean_gradient_comp_s"]))
        sched_means[s_id] = cum_mean

        print(f"{s_id:<10} | {cum_mean:8.2f} +/- {cum_std_pop:5.2f} (samp: {cum_std_samp:5.2f})s | {ep_mean:10.2f} s   | {gr_mean:10.2f} s")

    print("\n2. SEED-MATCHED PAIRWISE SPEEDUP RATIOS")
    print("   Formula: R_s = T_{baseline, s} / T_{optimized, s}")
    print("   Summary: Mean(R_s) +/- PopStd(R_s, ddof=0)")
    print("-" * 80)

    comparisons = [
        ("Level Effect (repeated) [A-rep / B-rep]", "A-repeat", "B-repeat"),
        ("Level Effect (cached)   [A-cac / B-cac]", "A-cache", "B-cache"),
        ("Cache Effect (Cond A)   [A-rep / A-cac]", "A-repeat", "A-cache"),
        ("Cache Effect (Cond B)   [B-rep / B-cac]", "B-repeat", "B-cache"),
        ("Combined Optimization   [A-rep / B-cac]", "A-repeat", "B-cache"),
    ]

    for label, base, opt in comparisons:
        ratios = []
        per_seed_str = []
        for s in seeds:
            sub = df[df["seed"] == s].set_index("schedule_id")
            t_base = sub.loc[base, "cumulative_training_s"]
            t_opt = sub.loc[opt, "cumulative_training_s"]
            r = t_base / t_opt
            ratios.append(r)
            per_seed_str.append(f"Seed {s}: {r:.3f} ({t_base:.2f}s/{t_opt:.2f}s)")

        r_mean = float(np.mean(ratios))
        r_std_pop = float(np.std(ratios, ddof=0))
        r_std_samp = float(np.std(ratios, ddof=1))
        ratio_of_means = sched_means[base] / sched_means[opt]

        print(f"\n{label}:")
        for s_str in per_seed_str:
            print(f"  {s_str}")
        print(f"  -> Seed-Matched Mean Ratio : {r_mean:.3f} +/- {r_std_pop:.3f} (pop) [sample std: {r_std_samp:.3f}]")
        print(f"  -> Ratio of Aggregate Means: {ratio_of_means:.3f}x ({sched_means[base]:.2f}s / {sched_means[opt]:.2f}s)")

    print("\n3. VALIDATION METRIC CONCORDANCE ACROSS SCHEDULES & SEEDS")
    print("-" * 80)
    for s in seeds:
        sub = df[df["seed"] == s]
        auc_vals = sub["val_auc"].tolist()
        acc_vals = sub["val_acc"].tolist()
        print(f"Seed {s:4d}:")
        print(f"  ROC-AUC across 4 schedules : {[round(v, 6) for v in auc_vals]}")
        print(f"  Accuracy across 4 schedules: {[round(v, 6) for v in acc_vals]}")
        # Verify no observed difference
        assert max(auc_vals) - min(auc_vals) < 1e-5, f"Discrepancy in AUC for Seed {s}!"
        assert max(acc_vals) - min(acc_vals) < 1e-5, f"Discrepancy in Acc for Seed {s}!"

    # Aggregate across the 3 seeds
    unique_seed_accs = [df[df["seed"] == s]["val_acc"].iloc[0] for s in seeds]
    unique_seed_aucs = [df[df["seed"] == s]["val_auc"].iloc[0] for s in seeds]

    acc_mean = float(np.mean(unique_seed_accs))
    acc_std_pop = float(np.std(unique_seed_accs, ddof=0))
    acc_std_samp = float(np.std(unique_seed_accs, ddof=1))

    auc_mean = float(np.mean(unique_seed_aucs))
    auc_std_pop = float(np.std(unique_seed_aucs, ddof=0))
    auc_std_samp = float(np.std(unique_seed_aucs, ddof=1))

    print(f"\nAcross-Seed Summary (N = 3):")
    print(f"  Validation Accuracy: {acc_mean:.4f} +/- {acc_std_pop:.4f} (pop) [sample: {acc_std_samp:.4f}]")
    print(f"    Per-seed values  : Seed 42 = {unique_seed_accs[0]:.4f}, Seed 123 = {unique_seed_accs[1]:.4f}, Seed 2026 = {unique_seed_accs[2]:.4f}")
    print(f"  Validation ROC-AUC : {auc_mean:.4f} +/- {auc_std_pop:.4f} (pop) [sample: {auc_std_samp:.4f}]")
    print(f"    Per-seed values  : Seed 42 = {unique_seed_aucs[0]:.4f}, Seed 123 = {unique_seed_aucs[1]:.4f}, Seed 2026 = {unique_seed_aucs[2]:.4f}")

    print("\n4. DECISION REGRET & ORACLE AGREEMENT")
    print("-" * 80)
    for s in seeds:
        sub = df[df["seed"] == s]
        min_time = sub["cumulative_training_s"].min()
        oracle_sched = sub.loc[sub["cumulative_training_s"].idxmin(), "schedule_id"]
        chosen_time = sub.loc[sub["schedule_id"] == "B-cache", "cumulative_training_s"].iloc[0]
        regret_s = chosen_time - min_time
        regret_pct = (regret_s / min_time) * 100.0
        print(f"Seed {s:4d}: Oracle={oracle_sched:<8} ({min_time:6.2f}s) | Policy=B-cache ({chosen_time:6.2f}s) | Regret={regret_s:.2f}s ({regret_pct:.2f}%)")

    print("\n5. TIMING SETUP ACCOUNTING VERIFICATION")
    print("-" * 80)
    with open(ROOT / "results" / "ptbxl" / "scheduler" / "phase3b" / "phase3b_timing_breakdown.csv") as f:
        breakdown_df = pd.read_csv(f)

    for s in seeds:
        for s_id in schedules:
            sub_ep = breakdown_df[(breakdown_df["seed"] == s) & (breakdown_df["schedule_id"] == s_id)]
            sub_run = df[(df["seed"] == s) & (df["schedule_id"] == s_id)].iloc[0]
            setup_t = sub_run["setup_alignment_time_s"]
            sum_ep_t = sub_ep["total_epoch_time_s"].sum()
            cum_t = sub_run["cumulative_training_s"]
            diff = abs(cum_t - (setup_t + sum_ep_t))
            assert diff < 1e-9, f"Timing accounting mismatch for {s_id} seed {s}: {diff}"

            # Verify cached alignment time during epoch is 0
            if "cache" in s_id:
                assert (sub_ep["feature_alignment_s"] == 0.0).all(), f"Non-zero epoch alignment in cached run {s_id} seed {s}!"
            else:
                assert (sub_ep["feature_alignment_s"] > 0.0).all(), f"Zero epoch alignment in repeated run {s_id} seed {s}!"

    print("ALL STATISTICAL AND ACCOUNTING INVARIANTS VERIFIED!")


if __name__ == "__main__":
    main()
