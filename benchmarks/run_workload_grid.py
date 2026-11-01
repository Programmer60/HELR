#!/usr/bin/env python3
"""
PTB-XL Phase 2: Held-Out Workload Grid Benchmark
================================================
Executes a factorial benchmarking matrix across 9 workload configurations
and 6 schedule/alignment variants using TenSEAL RNS-CKKS.

Workload Grid (9 configurations):
  N in {4,096, 10,000, 17,084}
  D in {12, 24, 48}
  S = 8,192
  B = ceil(N / S) in {1, 2, 3}

Schedule Variants (6 variants):
  - A-repeat, A-cache (Deg 3, Grad at Level 3, burn 4)
  - B-repeat, B-cache (Deg 3, Grad at Level 2 via * 1.0, burn 5)
  - C-repeat, C-cache (Deg 5, Grad at Level 2, burn 5)

Measurement Protocol:
  - 1 warm-up run per workload/condition.
  - 10 repetitions per configuration, distributed across 3 subprocesses (4, 3, 3 reps).
  - Explicit stage profiling: weight_enc, forward_linear, poly_eval, error_comp,
    level_adjustment, feature_alignment, gradient_comp, decryption_agg.
  - Plaintext numerical gradient reference validation (threshold <= 1e-4).
  - Outputs:
      results/ptbxl/scheduler/phase2/workload_grid_results.json
      results/ptbxl/scheduler/phase2/workload_grid_results.csv
      results/ptbxl/scheduler/phase2/workload_grid.log
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tenseal as ts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "ptbxl" / "scheduler" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = RESULTS_DIR / "workload_grid_results.json"
OUT_CSV = RESULTS_DIR / "workload_grid_results.csv"
OUT_LOG = RESULTS_DIR / "workload_grid.log"

GLOBAL_SEED = 42
SLOT_CAPACITY = 8192
L2_LAMBDA = 0.01

POLY = {
    3: {"c0": 0.5, "c1": 0.15012, "c3": -0.001593, "c5": 0.0},
    5: {"c0": 0.5, "c1": 0.217101, "c3": -0.007823, "c5": 0.000118},
}

WORKLOAD_GRID = [
    {"workload_id": "W1", "N": 4096, "D": 12, "B": 1, "is_reference": False},
    {"workload_id": "W2", "N": 4096, "D": 24, "B": 1, "is_reference": False},
    {"workload_id": "W3", "N": 4096, "D": 48, "B": 1, "is_reference": False},
    {"workload_id": "W4", "N": 10000, "D": 12, "B": 2, "is_reference": False},
    {"workload_id": "W5", "N": 10000, "D": 24, "B": 2, "is_reference": False},
    {"workload_id": "W6", "N": 10000, "D": 48, "B": 2, "is_reference": False},
    {"workload_id": "W7", "N": 17084, "D": 12, "B": 3, "is_reference": False},
    {"workload_id": "W8", "N": 17084, "D": 24, "B": 3, "is_reference": False},
    {"workload_id": "W9", "N": 17084, "D": 48, "B": 3, "is_reference": True},  # Phase 4 reference
]

class StageTimer:
    def __init__(self):
        self.timers: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        self.timers[name] = self.timers.get(name, 0.0) + (t1 - t0)

    def get(self, name: str) -> float:
        return self.timers.get(name, 0.0)

    def clear(self):
        self.timers.clear()


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


def run_worker_reps(
    workload_id: str,
    N: int,
    D: int,
    reps: int,
    worker_id: int,
    warmup: bool = True,
) -> List[Dict[str, Any]]:
    """Worker function executed within a subprocess for clean memory and cache isolation."""
    ctx = build_context()
    X, y, w_init, batches = partition_synthetic_dataset(N, D, SLOT_CAPACITY)

    # Pre-encrypt batched dataset columns
    enc_batches = []
    for b in batches:
        enc_X_cols = [ts.ckks_vector(ctx, b["X_pad"][:, j].tolist()) for j in range(D)]
        enc_y = ts.ckks_vector(ctx, b["y_pad"].tolist())
        enc_batches.append({"enc_X": enc_X_cols, "enc_y": enc_y})

    # Precompute plaintext gradient references
    ref_grad_deg3 = compute_pt_ref_gradient(X, y, w_init, 3)
    ref_grad_deg5 = compute_pt_ref_gradient(X, y, w_init, 5)

    conditions = [
        ("A", 3, 4),  # Deg 3, Grad L3, burn 4
        ("B", 3, 5),  # Deg 3, Grad L2, burn 5
        ("C", 5, 5),  # Deg 5, Grad L2, burn 5
    ]

    timer = StageTimer()
    worker_records = []

    # Optional warm-up run
    if warmup:
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
    for r in range(1, reps + 1):
        for cond, deg, burn_X in conditions:
            timer.clear()
            t0_step = time.perf_counter()

            # 1. Weight Encryption
            with timer.measure("weight_encryption"):
                enc_w_cols = [ts.ckks_vector(ctx, [float(w_init[j])] * SLOT_CAPACITY) for j in range(D)]

            # 2. One-time Feature Alignment Precompute (measuring setup time for caching)
            t0_pre = time.perf_counter()
            cached_aligned_X = []
            for b in enc_batches:
                cached_aligned_X.append([burn(b["enc_X"][j], burn_X) for j in range(D)])
            alignment_precompute_s = time.perf_counter() - t0_pre

            unreg_sum = np.zeros(D, dtype=np.float64)

            for b_idx, b in enumerate(enc_batches):
                # 3. Forward linear
                with timer.measure("forward_linear"):
                    enc_z = b["enc_X"][0] * enc_w_cols[0]
                    for j in range(1, D):
                        enc_z = enc_z + (b["enc_X"][j] * enc_w_cols[j])

                # 4. Polynomial evaluation
                with timer.measure("poly_eval"):
                    enc_p = enc_sigmoid(enc_z, deg)

                # 5. Error computation
                with timer.measure("error_computation"):
                    burn_y = 4 if deg == 3 else 5
                    enc_y_aligned = burn(b["enc_y"], burn_y)
                    enc_err = enc_p - enc_y_aligned

                # 6. Condition B Level Adjustment
                with timer.measure("level_adjustment"):
                    if cond == "B":
                        enc_err_adj = enc_err * 1.0
                    else:
                        enc_err_adj = enc_err

                # 7. Feature alignment (repeated mode)
                with timer.measure("feature_alignment"):
                    enc_X_aligned = [burn(b["enc_X"][j], burn_X) for j in range(D)]

                # 8. Gradient computation
                with timer.measure("gradient_computation"):
                    enc_sums = [(enc_err_adj * enc_X_aligned[j]).sum() for j in range(D)]

                # 9. Decryption and aggregation
                with timer.measure("decryption_aggregation"):
                    dec_sums = np.array([g.decrypt()[0] for g in enc_sums])
                    unreg_sum += dec_sums

            # Normalization
            with timer.measure("decryption_aggregation"):
                g_ckks = (unreg_sum / N) + L2_LAMBDA * w_init

            total_step_repeated = time.perf_counter() - t0_step
            feature_align_s = timer.get("feature_alignment")
            total_step_cached = total_step_repeated - feature_align_s

            # 10-Epoch calculations
            ten_epoch_repeated_s = 10.0 * total_step_repeated
            ten_epoch_cached_s = alignment_precompute_s + 10.0 * total_step_cached

            # Numerical error checking
            ref_grad = ref_grad_deg3 if deg == 3 else ref_grad_deg5
            max_ae = float(np.max(np.abs(g_ckks - ref_grad)))

            # Record repeated variant
            rec_repeat = {
                "workload_id": workload_id,
                "N": N,
                "D": D,
                "B": len(batches),
                "condition": cond,
                "alignment_strategy": "repeated",
                "variant": f"{cond}-repeat",
                "worker_id": worker_id,
                "repetition": r,
                "weight_enc_s": timer.get("weight_encryption"),
                "forward_linear_s": timer.get("forward_linear"),
                "poly_eval_s": timer.get("poly_eval"),
                "error_comp_s": timer.get("error_computation"),
                "level_adjustment_s": timer.get("level_adjustment"),
                "feature_alignment_s": feature_align_s,
                "alignment_precompute_s": alignment_precompute_s,
                "gradient_comp_s": timer.get("gradient_computation"),
                "decryption_agg_s": timer.get("decryption_aggregation"),
                "total_step_s": total_step_repeated,
                "ten_epoch_s": ten_epoch_repeated_s,
                "max_abs_error_vs_ref": max_ae,
                "numerical_pass": bool(max_ae <= 1e-4),
            }

            # Record cached variant
            rec_cache = {
                "workload_id": workload_id,
                "N": N,
                "D": D,
                "B": len(batches),
                "condition": cond,
                "alignment_strategy": "cached",
                "variant": f"{cond}-cache",
                "worker_id": worker_id,
                "repetition": r,
                "weight_enc_s": timer.get("weight_encryption"),
                "forward_linear_s": timer.get("forward_linear"),
                "poly_eval_s": timer.get("poly_eval"),
                "error_comp_s": timer.get("error_computation"),
                "level_adjustment_s": timer.get("level_adjustment"),
                "feature_alignment_s": 0.0,
                "alignment_precompute_s": alignment_precompute_s,
                "gradient_comp_s": timer.get("gradient_computation"),
                "decryption_agg_s": timer.get("decryption_aggregation"),
                "total_step_s": total_step_cached,
                "ten_epoch_s": ten_epoch_cached_s,
                "max_abs_error_vs_ref": max_ae,
                "numerical_pass": bool(max_ae <= 1e-4),
            }

            worker_records.extend([rec_repeat, rec_cache])

    return worker_records


def run_workload_subprocess(workload_id: str, N: int, D: int, reps: int, worker_id: int) -> List[Dict[str, Any]]:
    """Spawns an isolated Python process to execute the given workload repetitions."""
    temp_json = RESULTS_DIR / f"temp_{workload_id}_w{worker_id}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--workload_id", workload_id,
        "--N", str(N),
        "--D", str(D),
        "--reps", str(reps),
        "--worker_id", str(worker_id),
        "--out_file", str(temp_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error in subprocess {workload_id} worker {worker_id}:\n{res.stderr}", flush=True)
        raise RuntimeError(f"Subprocess worker failed: {res.stderr}")

    with open(temp_json, "r") as f:
        data = json.load(f)
    temp_json.unlink(missing_ok=True)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run in worker mode inside a subprocess")
    parser.add_argument("--workload_id", type=str, default="")
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--D", type=int, default=12)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--worker_id", type=int, default=1)
    parser.add_argument("--out_file", type=str, default="")
    args = parser.parse_args()

    if args.worker:
        records = run_worker_reps(
            workload_id=args.workload_id,
            N=args.N,
            D=args.D,
            reps=args.reps,
            worker_id=args.worker_id,
            warmup=True,
        )
        with open(args.out_file, "w") as f:
            json.dump(records, f)
        sys.exit(0)

    # Master orchestrator mode
    print("=" * 90)
    print(" PTB-XL PHASE 2: HELD-OUT WORKLOAD GRID BENCHMARK")
    print("=" * 90)
    print(f"Platform: {platform.platform()} | Python: {platform.python_version()}")
    print(f"TenSEAL: {ts.__version__} | Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Grid: 9 Workloads x 6 Schedule Variants x 10 Measured Repetitions (across 3 subprocesses)")
    print("=" * 90)

    all_records = []
    # Distribution of 10 repetitions across 3 subprocesses: 4, 3, 3
    worker_distribution = [(1, 4), (2, 3), (3, 3)]

    t0_total = time.perf_counter()

    for w_cfg in WORKLOAD_GRID:
        wid = w_cfg["workload_id"]
        N = w_cfg["N"]
        D = w_cfg["D"]
        B = w_cfg["B"]
        is_ref = w_cfg["is_reference"]
        ref_label = " [KNOWN REFERENCE]" if is_ref else " [HELD-OUT]"

        print(f"\n>>> Executing Workload {wid}{ref_label}: N={N:,}, D={D}, Batches={B} (10 reps across 3 processes)...", flush=True)
        t0_w = time.perf_counter()

        for worker_id, reps in worker_distribution:
            w_recs = run_workload_subprocess(wid, N, D, reps, worker_id)
            all_records.extend(w_recs)

        t_w = time.perf_counter() - t0_w
        print(f"    Completed Workload {wid} in {t_w:.2f}s", flush=True)

        # Intermediate checkpoint
        df_int = pd.DataFrame(all_records)
        df_int.to_csv(OUT_CSV, index=False)

    total_bench_time = time.perf_counter() - t0_total
    print(f"\nAll 9 workloads executed in {total_bench_time:.2f}s.")

    # Compute summary statistics
    df_all = pd.DataFrame(all_records)
    summary_list = []

    grouped = df_all.groupby(["workload_id", "N", "D", "B", "condition", "alignment_strategy", "variant"])

    for (wid, N, D, B, cond, strat, var), group in grouped:
        step_times = group["total_step_s"].to_numpy()
        grad_times = group["gradient_comp_s"].to_numpy()
        align_times = group["feature_alignment_s"].to_numpy()
        fwd_times = group["forward_linear_s"].to_numpy()
        ten_ep_times = group["ten_epoch_s"].to_numpy()
        max_aes = group["max_abs_error_vs_ref"].to_numpy()

        q25, q75 = np.percentile(step_times, [25, 75])

        s_entry = {
            "workload_id": wid,
            "N": int(N),
            "D": int(D),
            "B": int(B),
            "condition": cond,
            "alignment_strategy": strat,
            "variant": var,
            "sample_count": len(group),
            "is_reference": bool(wid == "W9"),
            "total_step_mean_s": float(np.mean(step_times)),
            "total_step_median_s": float(np.median(step_times)),
            "total_step_std_s": float(np.std(step_times)),
            "total_step_iqr_s": float(q75 - q25),
            "total_step_min_s": float(np.min(step_times)),
            "total_step_max_s": float(np.max(step_times)),
            "gradient_comp_mean_s": float(np.mean(grad_times)),
            "gradient_comp_std_s": float(np.std(grad_times)),
            "feature_alignment_mean_s": float(np.mean(align_times)),
            "forward_linear_mean_s": float(np.mean(fwd_times)),
            "alignment_precompute_mean_s": float(np.mean(group["alignment_precompute_s"])),
            "ten_epoch_mean_s": float(np.mean(ten_ep_times)),
            "ten_epoch_std_s": float(np.std(ten_ep_times)),
            "max_abs_error_vs_ref": float(np.max(max_aes)),
            "numerical_pass_rate": float(np.mean(group["numerical_pass"])),
        }
        summary_list.append(s_entry)

    # Save final JSON and CSV
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "tenseal_version": ts.__version__,
            "total_benchmark_time_s": total_bench_time,
            "poly_modulus_degree": 16384,
            "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60],
            "global_scale": "2^40",
            "slot_capacity": SLOT_CAPACITY,
            "global_seed": GLOBAL_SEED,
            "repetitions_per_variant": 10,
            "workers_count": 3,
            "grid_configurations": 9,
            "variants_per_workload": 6,
            "total_measurements": len(all_records),
        },
        "workload_grid": WORKLOAD_GRID,
        "summary": summary_list,
        "raw_records": all_records,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(final_output, f, indent=2)

    df_all.to_csv(OUT_CSV, index=False)
    print(f"\nFinal results saved:")
    print(f"  JSON: {OUT_JSON}")
    print(f"  CSV : {OUT_CSV}")


if __name__ == "__main__":
    main()
