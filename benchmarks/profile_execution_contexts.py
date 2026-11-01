#!/usr/bin/env python3
"""
PTB-XL Phase 3A: Execution Context Profiler
==========================================
Systematically characterizes and quantifies the latency divergence between two
distinct execution modes for CKKS logistic regression workloads:

  Mode A: Persistent Worker
    - One long-lived Python process.
    - Context initialization and key generation executed once.
    - Prescribed warm-up iterations executed once.
    - Repeated measurements (>= 30 samples) collected in steady-state within the same process.

  Mode B: Fresh Worker
    - Independent subprocess launches.
    - Full initialization (module imports, CKKS context, Galois/Relin keys) per launch.
    - Prescribed warm-up executed per launch.
    - Measurements collected across multiple independent subprocess launches (>= 30 samples total).

Operations Profiled at all admissible levels:
  1. ct_ct_mul_rescale (Levels 7, 6, 5, 4, 3, 2; Level 1 unsupported)
  2. pt_mul_one_rescale_to_next (Levels 7, 6, 5, 4, 3, 2; Level 1 unsupported)
  3. slot_sum (Levels 7, 6, 5, 4, 3, 2, 1)
  4. gradient_dot_product: (err * X_j).sum() (Level 3 -> Level 2, Level 2 -> Level 1)
  5. feature_alignment_burn4 (Level 7 -> Level 3)
  6. feature_alignment_burn5 (Level 7 -> Level 2)
  7. condition_b_level_adjustment: err * 1.0 (Level 3 -> Level 2)
  8. decryption_scalar (Levels 2, 1)
  9. weight_encryption: broadcast scalar into S slots
  10. forward_linear_term: ct_ct_mul + ct_ct_add

Outputs:
  - results/ptbxl/scheduler/phase3a/execution_context_latencies.json
  - results/ptbxl/scheduler/phase3a/execution_context_latencies.csv
  - results/ptbxl/scheduler/phase3a/execution_context_profile.log
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import tenseal as ts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "ptbxl" / "scheduler" / "phase3a"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = RESULTS_DIR / "execution_context_latencies.json"
OUT_CSV = RESULTS_DIR / "execution_context_latencies.csv"
OUT_LOG = RESULTS_DIR / "execution_context_profile.log"

GLOBAL_SEED = 42
SLOT_CAPACITY = 8192
WARMUP_REPS = 5
SAMPLES_PER_OP = 30
FRESH_WORKER_LAUNCHES = 5
REPS_PER_FRESH_LAUNCH = 6  # 5 launches * 6 reps = 30 samples total


def get_system_metadata() -> Dict[str, Any]:
    freq = psutil.cpu_freq()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "python_version": platform.python_version(),
        "tenseal_version": ts.__version__,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "cpu_max_freq_mhz": freq.max if freq else None,
        "cpu_current_freq_mhz": freq.current if freq else None,
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60],
        "total_modulus_bits": 360,
        "scale": "2^40",
        "slot_capacity": SLOT_CAPACITY,
    }


def build_ckks_context() -> Tuple[ts.Context, float, float]:
    """Returns (context, context_creation_ms, key_gen_ms)"""
    t0 = time.perf_counter()
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    t1 = time.perf_counter()

    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    t2 = time.perf_counter()

    ctx_create_ms = (t1 - t0) * 1000.0
    key_gen_ms = (t2 - t1) * 1000.0
    return ctx, ctx_create_ms, key_gen_ms


def get_level(enc: ts.CKKSVector) -> int:
    return enc.ciphertext()[0].coeff_modulus_size()


def get_parms_id(enc: ts.CKKSVector) -> List[int]:
    return list(enc.ciphertext()[0].parms_id())


def get_scale(enc: ts.CKKSVector) -> float:
    return float(enc.ciphertext()[0].scale)


def burn(v: ts.CKKSVector, n: int) -> ts.CKKSVector:
    for _ in range(n):
        v = v * 1.0
    return v


def compute_stats(samples: List[float]) -> Dict[str, float]:
    arr = np.array(samples, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "iqr_ms": float(q75 - q25),
        "sample_count": len(samples),
    }


def execute_benchmarks_in_current_process(
    ctx: ts.Context,
    reps: int = 30,
    warmup: int = 5,
    seed: int = GLOBAL_SEED,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Executes benchmark measurements within the current process.
    Returns (records, warmup_time_ms).
    """
    rng = np.random.default_rng(seed)
    raw_x = rng.uniform(-1.0, 1.0, size=SLOT_CAPACITY).tolist()
    raw_w = rng.uniform(-0.1, 0.1, size=SLOT_CAPACITY).tolist()
    raw_y = rng.integers(0, 2, size=SLOT_CAPACITY).astype(float).tolist()

    records = []
    total_warmup_t0 = time.perf_counter()

    # Generic benchmark runner
    def run_op(
        op_name: str,
        start_level: int,
        prep_fn,
        exec_fn,
        verify_fn=None,
        supported: bool = True,
        notes: str = "",
    ):
        if not supported:
            records.append({
                "operation": op_name,
                "start_level": start_level,
                "end_level": None,
                "supported": False,
                "samples_ms": [],
                "stats": None,
                "max_abs_error": None,
                "notes": notes or "Unsupported at this level",
            })
            return

        # Warm-up iterations
        for _ in range(warmup):
            inputs = prep_fn()
            _ = exec_fn(*inputs)

        # Measured iterations
        samples = []
        last_out = None
        for _ in range(reps):
            inputs = prep_fn()
            t0 = time.perf_counter()
            out = exec_fn(*inputs)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)
            last_out = out

        end_level = get_level(last_out) if isinstance(last_out, ts.CKKSVector) else None
        max_ae = None
        if verify_fn is not None and last_out is not None:
            max_ae = verify_fn(last_out)

        records.append({
            "operation": op_name,
            "start_level": start_level,
            "end_level": end_level,
            "supported": True,
            "samples_ms": samples,
            "stats": compute_stats(samples),
            "max_abs_error": max_ae,
            "notes": notes,
        })

    # 1. Ct-Ct Multiplication + Rescale (c1 * c2)
    for lvl in range(7, 0, -1):
        if lvl >= 2:
            burn_n = 7 - lvl
            c1_base = burn(ts.ckks_vector(ctx, raw_x), burn_n)
            c2_base = burn(ts.ckks_vector(ctx, raw_w), burn_n)

            def prep_mul(c1=c1_base, c2=c2_base):
                return c1, c2

            def exec_mul(c1, c2):
                return c1 * c2

            def verify_mul(out):
                dec = np.array(out.decrypt())
                expected = np.array(raw_x) * np.array(raw_w)
                return float(np.max(np.abs(dec - expected)))

            run_op(
                "ct_ct_mul_rescale",
                start_level=lvl,
                prep_fn=prep_mul,
                exec_fn=exec_mul,
                verify_fn=verify_mul,
                supported=True,
                notes=f"Ct-Ct mul + relin + rescale (L{lvl} -> L{lvl-1})",
            )
        else:
            run_op(
                "ct_ct_mul_rescale",
                start_level=1,
                prep_fn=None,
                exec_fn=None,
                supported=False,
                notes="Cannot rescale below Level 1",
            )

    # 2. Plaintext Multiplication by 1.0 + Rescale (pt_mul_one)
    for lvl in range(7, 0, -1):
        if lvl >= 2:
            burn_n = 7 - lvl
            c_base = burn(ts.ckks_vector(ctx, raw_x), burn_n)

            def prep_pt(c=c_base):
                return (c,)

            def exec_pt(c):
                return c * 1.0

            def verify_pt(out):
                dec = np.array(out.decrypt())
                return float(np.max(np.abs(dec - np.array(raw_x))))

            run_op(
                "pt_mul_one_rescale_to_next",
                start_level=lvl,
                prep_fn=prep_pt,
                exec_fn=exec_pt,
                verify_fn=verify_pt,
                supported=True,
                notes=f"Pt mul by 1.0 + rescale (L{lvl} -> L{lvl-1})",
            )
        else:
            run_op(
                "pt_mul_one_rescale_to_next",
                start_level=1,
                prep_fn=None,
                exec_fn=None,
                supported=False,
                notes="Cannot rescale below Level 1",
            )

    # 3. Slot Summation (.sum()) across S slots
    for lvl in range(7, 0, -1):
        burn_n = 7 - lvl
        c_base = burn(ts.ckks_vector(ctx, raw_x), burn_n)

        def prep_sum(c=c_base):
            return (c,)

        def exec_sum(c):
            return c.sum()

        def verify_sum(out):
            dec = out.decrypt()[0]
            expected = float(np.sum(raw_x))
            return float(np.abs(dec - expected))

        run_op(
            "slot_sum",
            start_level=lvl,
            prep_fn=prep_sum,
            exec_fn=exec_sum,
            verify_fn=verify_sum,
            supported=True,
            notes=f"Slot summation across {SLOT_CAPACITY} slots at L{lvl}",
        )

    # 4. Gradient Dot Product: (err * X_j).sum()
    for grad_lvl in [3, 2]:
        burn_n = 7 - grad_lvl
        c_err_base = burn(ts.ckks_vector(ctx, raw_y), burn_n)
        c_x_base = burn(ts.ckks_vector(ctx, raw_x), burn_n)

        def prep_dot(ce=c_err_base, cx=c_x_base):
            return ce, cx

        def exec_dot(ce, cx):
            return (ce * cx).sum()

        def verify_dot(out):
            dec = out.decrypt()[0]
            expected = float(np.dot(raw_y, raw_x))
            return float(np.abs(dec - expected))

        label = "Condition A" if grad_lvl == 3 else "Condition B/C"
        run_op(
            "gradient_dot_product",
            start_level=grad_lvl,
            prep_fn=prep_dot,
            exec_fn=exec_dot,
            verify_fn=verify_dot,
            supported=True,
            notes=f"Complete gradient dot product at L{grad_lvl} -> L{grad_lvl-1} ({label})",
        )

    # 5. Feature Alignment Sequences (burn 4 and burn 5)
    c_fresh = ts.ckks_vector(ctx, raw_x)
    for burn_count in [4, 5]:
        target_lvl = 7 - burn_count

        def prep_burn(c=c_fresh):
            return (c,)

        def exec_burn(c, count=burn_count):
            return burn(c, count)

        def verify_burn(out):
            dec = np.array(out.decrypt())
            return float(np.max(np.abs(dec - np.array(raw_x))))

        run_op(
            f"feature_alignment_burn{burn_count}",
            start_level=7,
            prep_fn=prep_burn,
            exec_fn=exec_burn,
            verify_fn=verify_burn,
            supported=True,
            notes=f"Feature alignment burn {burn_count} (L7 -> L{target_lvl})",
        )

    # 6. Condition B Level Adjustment (err * 1.0 at L3 -> L2)
    c_err_l3 = burn(ts.ckks_vector(ctx, raw_y), 4)

    def prep_adj(c=c_err_l3):
        return (c,)

    def exec_adj(c):
        return c * 1.0

    def verify_adj(out):
        dec = np.array(out.decrypt())
        return float(np.max(np.abs(dec - np.array(raw_y))))

    run_op(
        "condition_b_level_adjustment",
        start_level=3,
        prep_fn=prep_adj,
        exec_fn=exec_adj,
        verify_fn=verify_adj,
        supported=True,
        notes="Condition B level adjustment via * 1.0 (L3 -> L2)",
    )

    # 7. Scalar Decryption
    # For Level 2 (Condition A gradient result): Level 3 * Level 3 -> Level 2 -> sum
    c_dec_l2 = (burn(ts.ckks_vector(ctx, raw_y), 4) * burn(ts.ckks_vector(ctx, raw_x), 4)).sum()
    # For Level 1 (Condition B/C gradient result): Level 2 * Level 2 -> Level 1 -> sum
    c_dec_l1 = (burn(ts.ckks_vector(ctx, raw_y), 5) * burn(ts.ckks_vector(ctx, raw_x), 5)).sum()

    for dec_lvl in [2, 1]:
        c_dec_base = c_dec_l2 if dec_lvl == 2 else c_dec_l1

        def prep_dec(c=c_dec_base):
            return (c,)

        def exec_dec(c):
            return c.decrypt()[0]

        run_op(
            "decryption_scalar",
            start_level=dec_lvl,
            prep_fn=prep_dec,
            exec_fn=exec_dec,
            verify_fn=None,
            supported=True,
            notes=f"Scalar decryption at L{dec_lvl}",
        )

    # 8. Weight Encryption (broadcast scalar into S slots)
    def prep_wenc():
        return (0.01,)

    def exec_wenc(scalar_val):
        return ts.ckks_vector(ctx, [scalar_val] * SLOT_CAPACITY)

    run_op(
        "weight_encryption_scalar",
        start_level=7,
        prep_fn=prep_wenc,
        exec_fn=exec_wenc,
        verify_fn=None,
        supported=True,
        notes=f"Encrypt scalar broadcast to {SLOT_CAPACITY} slots (L7)",
    )

    # 9. Forward Linear Term (c_X * c_w + c_acc)
    c_x_l7 = ts.ckks_vector(ctx, raw_x)
    c_w_l7 = ts.ckks_vector(ctx, raw_w)
    c_acc_l6 = c_x_l7 * c_w_l7

    def prep_fwd(cx=c_x_l7, cw=c_w_l7, cacc=c_acc_l6):
        return cx, cw, cacc

    def exec_fwd(cx, cw, cacc):
        return cacc + (cx * cw)

    run_op(
        "forward_linear_accumulation_step",
        start_level=7,
        prep_fn=prep_fwd,
        exec_fn=exec_fwd,
        verify_fn=None,
        supported=True,
        notes="Forward linear step: Ct-Ct mul L7->L6 + Ct-Ct add at L6",
    )

    total_warmup_ms = (time.perf_counter() - total_warmup_t0) * 1000.0
    return records, total_warmup_ms


def run_fresh_worker_subprocess(reps: int, launch_idx: int) -> Dict[str, Any]:
    """Launches an independent subprocess to collect fresh-worker measurements."""
    temp_json = RESULTS_DIR / f"temp_fresh_worker_{launch_idx}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--subworker",
        "--reps", str(reps),
        "--launch_idx", str(launch_idx),
        "--out_file", str(temp_json),
    ]
    t0_launch = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1_launch = time.perf_counter()

    if res.returncode != 0:
        print(f"Error in fresh worker subprocess launch {launch_idx}:\n{res.stderr}", flush=True)
        raise RuntimeError(f"Fresh worker subprocess failed: {res.stderr}")

    with open(temp_json, "r") as f:
        data = json.load(f)
    temp_json.unlink(missing_ok=True)

    data["total_subprocess_wall_clock_ms"] = (t1_launch - t0_launch) * 1000.0
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subworker", action="store_true", help="Run in subworker subprocess mode")
    parser.add_argument("--reps", type=int, default=REPS_PER_FRESH_LAUNCH)
    parser.add_argument("--launch_idx", type=int, default=1)
    parser.add_argument("--out_file", type=str, default="")
    args = parser.parse_args()

    # Subworker subprocess mode
    if args.subworker:
        t0_sub_start = time.perf_counter()
        ctx, ctx_create_ms, key_gen_ms = build_ckks_context()
        records, warmup_ms = execute_benchmarks_in_current_process(
            ctx=ctx,
            reps=args.reps,
            warmup=WARMUP_REPS,
            seed=GLOBAL_SEED + args.launch_idx,
        )
        total_sub_exec_ms = (time.perf_counter() - t0_sub_start) * 1000.0

        subworker_result = {
            "launch_idx": args.launch_idx,
            "ctx_creation_ms": ctx_create_ms,
            "key_generation_ms": key_gen_ms,
            "warmup_total_ms": warmup_ms,
            "total_sub_exec_ms": total_sub_exec_ms,
            "records": records,
        }
        with open(args.out_file, "w") as f:
            json.dump(subworker_result, f)
        sys.exit(0)

    # Main Orchestrator Mode
    print("=" * 95)
    print(" PTB-XL PHASE 3A: RUNTIME EXECUTION CONTEXT PROFILER")
    print("=" * 95)
    sys_meta = get_system_metadata()
    print(f"Platform: {sys_meta['platform']} | Python: {sys_meta['python_version']} | TenSEAL: {sys_meta['tenseal_version']}")
    print(f"CPUs: {sys_meta['logical_cpus']} logical, {sys_meta['physical_cpus']} physical | Clock: {sys_meta['cpu_current_freq_mhz']} MHz")
    print(f"Protocol: Persistent Worker (30 reps) vs Fresh Worker ({FRESH_WORKER_LAUNCHES} launches x {REPS_PER_FRESH_LAUNCH} reps = 30 samples)")
    print("=" * 95)

    # -------------------------------------------------------------
    # 1. Profile Mode A: Persistent Worker
    # -------------------------------------------------------------
    print("\n[1/3] Profiling Mode A: Persistent Worker (single long-lived process, 30 reps)...", flush=True)
    t0_persistent = time.perf_counter()
    ctx_persistent, ctx_create_p_ms, key_gen_p_ms = build_ckks_context()
    persistent_records, warmup_p_ms = execute_benchmarks_in_current_process(
        ctx=ctx_persistent,
        reps=SAMPLES_PER_OP,
        warmup=WARMUP_REPS,
        seed=GLOBAL_SEED,
    )
    t_persistent_total = (time.perf_counter() - t0_persistent) * 1000.0
    print(f"      Mode A Completed in {t_persistent_total/1000.0:.2f}s (Init: {ctx_create_p_ms:.1f}ms, Keys: {key_gen_p_ms:.1f}ms, Warmup: {warmup_p_ms:.1f}ms)", flush=True)

    # -------------------------------------------------------------
    # 2. Profile Mode B: Fresh Worker
    # -------------------------------------------------------------
    print(f"\n[2/3] Profiling Mode B: Fresh Worker ({FRESH_WORKER_LAUNCHES} independent subprocess launches)...", flush=True)
    fresh_worker_outputs = []
    t0_fresh_total = time.perf_counter()

    for l_idx in range(1, FRESH_WORKER_LAUNCHES + 1):
        print(f"      Launching fresh worker subprocess {l_idx}/{FRESH_WORKER_LAUNCHES}...", flush=True)
        fw_res = run_fresh_worker_subprocess(reps=REPS_PER_FRESH_LAUNCH, launch_idx=l_idx)
        fresh_worker_outputs.append(fw_res)

    t_fresh_total = (time.perf_counter() - t0_fresh_total) * 1000.0
    print(f"      Mode B Completed in {t_fresh_total/1000.0:.2f}s across {FRESH_WORKER_LAUNCHES} launches", flush=True)

    # Pool fresh worker samples across all launches
    fresh_pooled_records: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
    for fw in fresh_worker_outputs:
        for rec in fw["records"]:
            key = (rec["operation"], rec["start_level"])
            if key not in fresh_pooled_records:
                fresh_pooled_records[key] = {
                    "operation": rec["operation"],
                    "start_level": rec["start_level"],
                    "end_level": rec["end_level"],
                    "supported": rec["supported"],
                    "samples_ms": [],
                    "max_abs_errors": [],
                    "notes": rec["notes"],
                }
            if rec["supported"]:
                fresh_pooled_records[key]["samples_ms"].extend(rec["samples_ms"])
                if rec["max_abs_error"] is not None:
                    fresh_pooled_records[key]["max_abs_errors"].append(rec["max_abs_error"])

    # Compute pooled fresh stats
    for key, item in fresh_pooled_records.items():
        if item["supported"]:
            item["stats"] = compute_stats(item["samples_ms"])
            item["max_abs_error"] = float(np.max(item["max_abs_errors"])) if item["max_abs_errors"] else None
        else:
            item["stats"] = None
            item["max_abs_error"] = None

    # -------------------------------------------------------------
    # 3. Ratio Analysis (gamma_op = persistent / fresh and fresh / persistent)
    # -------------------------------------------------------------
    print("\n[3/3] Analyzing Operation Latency Divergence and Context Ratios...", flush=True)
    comparison_table = []

    print("-" * 105)
    print(f"{'Operation':<34s} | {'Lvl':<3s} | {'Persistent(ms)':<14s} | {'Fresh(ms)':<12s} | {'P / F Ratio':<12s} | {'F / P Ratio':<12s} | {'Status'}")
    print("-" * 105)

    persistent_map = {(r["operation"], r["start_level"]): r for r in persistent_records}

    for key, p_rec in persistent_map.items():
        op_name, lvl = key
        f_rec = fresh_pooled_records.get(key)

        if not p_rec["supported"]:
            print(f"{op_name:<34s} | L{lvl or '?':<2} | {'UNSUPPORTED':<14s} | {'UNSUPPORTED':<12s} | {'N/A':<12s} | {'N/A':<12s} | Unsupported")
            continue

        p_mean = p_rec["stats"]["mean_ms"]
        f_mean = f_rec["stats"]["mean_ms"]

        ratio_p_over_f = p_mean / f_mean if f_mean > 0 else 0.0
        ratio_f_over_p = f_mean / p_mean if p_mean > 0 else 0.0

        comparison_table.append({
            "operation": op_name,
            "start_level": lvl,
            "end_level": p_rec["end_level"],
            "persistent_mean_ms": p_mean,
            "persistent_median_ms": p_rec["stats"]["median_ms"],
            "persistent_std_ms": p_rec["stats"]["std_ms"],
            "persistent_iqr_ms": p_rec["stats"]["iqr_ms"],
            "fresh_mean_ms": f_mean,
            "fresh_median_ms": f_rec["stats"]["median_ms"],
            "fresh_std_ms": f_rec["stats"]["std_ms"],
            "fresh_iqr_ms": f_rec["stats"]["iqr_ms"],
            "ratio_persistent_over_fresh": ratio_p_over_f,
            "ratio_fresh_over_persistent": ratio_f_over_p,
            "max_abs_error_p": p_rec["max_abs_error"],
            "max_abs_error_f": f_rec["max_abs_error"],
        })

        print(f"{op_name:<34s} | L{lvl or '?':<2} | {p_mean:14.3f} | {f_mean:12.3f} | {ratio_p_over_f:12.4f} | {ratio_f_over_p:12.4f} | Supported")

    # Average startup metrics across fresh launches
    avg_fresh_ctx_init_ms = float(np.mean([fw["ctx_creation_ms"] for fw in fresh_worker_outputs]))
    avg_fresh_key_gen_ms = float(np.mean([fw["key_generation_ms"] for fw in fresh_worker_outputs]))
    avg_fresh_warmup_ms = float(np.mean([fw["warmup_total_ms"] for fw in fresh_worker_outputs]))
    avg_fresh_wall_ms = float(np.mean([fw["total_subprocess_wall_clock_ms"] for fw in fresh_worker_outputs]))

    # Analyze family ratios
    ratios_f_over_p = [c["ratio_fresh_over_persistent"] for c in comparison_table]
    mean_f_over_p = float(np.mean(ratios_f_over_p))
    std_f_over_p = float(np.std(ratios_f_over_p))
    min_f_over_p = float(np.min(ratios_f_over_p))
    max_f_over_p = float(np.max(ratios_f_over_p))

    print("\n" + "=" * 95)
    print(" EXECUTION CONTEXT RATIO ANALYSIS SUMMARY")
    print("=" * 95)
    print(f"Mean Fresh / Persistent Ratio : {mean_f_over_p:.4f} +/- {std_f_over_p:.4f}")
    print(f"Min Ratio                      : {min_f_over_p:.4f} ({comparison_table[int(np.argmin(ratios_f_over_p))]['operation']})")
    print(f"Max Ratio                      : {max_f_over_p:.4f} ({comparison_table[int(np.argmax(ratios_f_over_p))]['operation']})")
    print(f"Spread (Max / Min)             : {max_f_over_p / min_f_over_p:.2f}x divergence across operation families")
    print(f"Fresh Process Startup Overhead : Init={avg_fresh_ctx_init_ms:.1f}ms, Keys={avg_fresh_key_gen_ms:.1f}ms, Warmup={avg_fresh_warmup_ms:.1f}ms, Total Wall={avg_fresh_wall_ms:.1f}ms")

    # Save output artifacts
    output_data = {
        "metadata": sys_meta,
        "persistent_worker": {
            "ctx_creation_ms": ctx_create_p_ms,
            "key_generation_ms": key_gen_p_ms,
            "warmup_total_ms": warmup_p_ms,
            "total_benchmark_time_ms": t_persistent_total,
            "reps_per_op": SAMPLES_PER_OP,
            "operations": persistent_records,
        },
        "fresh_worker": {
            "num_launches": FRESH_WORKER_LAUNCHES,
            "reps_per_launch": REPS_PER_FRESH_LAUNCH,
            "total_samples_per_op": FRESH_WORKER_LAUNCHES * REPS_PER_FRESH_LAUNCH,
            "avg_ctx_creation_ms": avg_fresh_ctx_init_ms,
            "avg_key_generation_ms": avg_fresh_key_gen_ms,
            "avg_warmup_total_ms": avg_fresh_warmup_ms,
            "avg_subprocess_wall_clock_ms": avg_fresh_wall_ms,
            "operations": list(fresh_pooled_records.values()),
        },
        "comparison_table": comparison_table,
        "ratio_summary": {
            "mean_fresh_over_persistent": mean_f_over_p,
            "std_fresh_over_persistent": std_f_over_p,
            "min_fresh_over_persistent": min_f_over_p,
            "max_fresh_over_persistent": max_f_over_p,
            "spread_ratio": max_f_over_p / min_f_over_p,
            "universal_gamma_justified": bool((max_f_over_p / min_f_over_p) < 1.25),
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    df_cmp = pd.DataFrame(comparison_table)
    df_cmp.to_csv(OUT_CSV, index=False)

    print(f"\nArtifacts saved successfully:")
    print(f"  JSON: {OUT_JSON}")
    print(f"  CSV : {OUT_CSV}")


if __name__ == "__main__":
    main()
