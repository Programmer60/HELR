#!/usr/bin/env python3
"""
CKKS Microbenchmark Profiler for Workload-Aware Circuit Scheduling
==================================================================
Empirically benchmarks the elementary cryptographic and SIMD operations
comprising the Homomorphic Encrypted Logistic Regression (HELR) circuit
under TenSEAL / Microsoft SEAL RNS-CKKS across all supported modulus levels.

Outputs:
  - results/ptbxl/scheduler/microbenchmark_results.json
  - results/ptbxl/scheduler/microbenchmark_results.csv
  - results/ptbxl/scheduler/microbenchmark.log
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

import numpy as np
import pandas as pd
import tenseal as ts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "ptbxl" / "scheduler"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen Parameter Set
POLY_MODULUS_DEGREE = 16384
COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 40, 40, 60]
GLOBAL_SCALE = 2**40
SLOT_CAPACITY = 8192
SEED = 42

POLY3 = {"c0": 0.5, "c1": 0.15012, "c3": -0.001593}
POLY5 = {"c0": 0.5, "c1": 0.217101, "c3": -0.007823, "c5": 0.000118}


def build_ckks_context() -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def get_level(enc: ts.CKKSVector) -> int:
    """Returns SEAL Ciphertext::coeff_modulus_size()."""
    return enc.ciphertext()[0].coeff_modulus_size()


def get_parms_id(enc: ts.CKKSVector) -> list[int]:
    return list(enc.ciphertext()[0].parms_id())


def get_scale(enc: ts.CKKSVector) -> float:
    return float(enc.ciphertext()[0].scale)


def burn(v: ts.CKKSVector, n: int) -> ts.CKKSVector:
    """Drops n modulus levels via successive scalar multiplication by 1.0."""
    for _ in range(n):
        v = v * 1.0
    return v


def compute_stats(samples_ms: list[float]) -> dict[str, float]:
    arr = np.array(samples_ms, dtype=np.float64)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(q50),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "iqr_ms": float(q75 - q25),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "samples_count": len(samples_ms),
    }


def run_single_process_benchmarks(reps: int = 30, warmup: int = 5, seed: int = 42) -> dict:
    ctx = build_ckks_context()
    rng = np.random.default_rng(seed)

    raw_x = rng.uniform(-1.0, 1.0, size=SLOT_CAPACITY).tolist()
    raw_w = rng.uniform(-0.1, 0.1, size=SLOT_CAPACITY).tolist()
    raw_y = rng.integers(0, 2, size=SLOT_CAPACITY).astype(float).tolist()

    records = []

    # Helper to measure an operation with pre-prepared inputs
    def measure_op(
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
                "notes": notes or "Unsupported at this modulus level",
            })
            return

        # Warm-up runs
        for _ in range(warmup):
            inputs = prep_fn()
            _ = exec_fn(*inputs)

        # Timed runs
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

    # -------------------------------------------------------------
    # 1. Ciphertext-Ciphertext Multiplication + Rescale (c1 * c2)
    # Supported at levels 7, 6, 5, 4, 3, 2; Unsupported at level 1
    # -------------------------------------------------------------
    for lvl in range(7, 0, -1):
        if lvl >= 2:
            burn_n = 7 - lvl

            def prep():
                c1 = burn(ts.ckks_vector(ctx, raw_x), burn_n)
                c2 = burn(ts.ckks_vector(ctx, raw_w), burn_n)
                return c1, c2

            def exec_fn(c1, c2):
                return c1 * c2

            def verify(out):
                dec = np.array(out.decrypt())
                expected = np.array(raw_x) * np.array(raw_w)
                return float(np.max(np.abs(dec - expected)))

            measure_op(
                op_name="ct_ct_mul_rescale",
                start_level=lvl,
                prep_fn=prep,
                exec_fn=exec_fn,
                verify_fn=verify,
                supported=True,
                notes=f"Ct-Ct multiply + relin + rescale (drops {lvl} -> {lvl-1})",
            )
        else:
            measure_op(
                op_name="ct_ct_mul_rescale",
                start_level=1,
                prep_fn=None,
                exec_fn=None,
                supported=False,
                notes="Cannot rescale below Level 1 (only 1 prime left in chain)",
            )

    # -------------------------------------------------------------
    # 2. Plaintext Multiplication by 1.0 + Rescale (pt_mul_one_rescale)
    # The exact mechanism of Condition B and burn(X, n)
    # Supported at levels 7, 6, 5, 4, 3, 2; Unsupported at level 1
    # -------------------------------------------------------------
    for lvl in range(7, 0, -1):
        if lvl >= 2:
            burn_n = 7 - lvl

            def prep():
                c = burn(ts.ckks_vector(ctx, raw_x), burn_n)
                return (c,)

            def exec_fn(c):
                return c * 1.0

            def verify(out):
                dec = np.array(out.decrypt())
                expected = np.array(raw_x)
                return float(np.max(np.abs(dec - expected)))

            measure_op(
                op_name="pt_mul_one_rescale_to_next",
                start_level=lvl,
                prep_fn=prep,
                exec_fn=exec_fn,
                verify_fn=verify,
                supported=True,
                notes=f"Plaintext multiply by 1.0 + rescale (drops {lvl} -> {lvl-1})",
            )
        else:
            measure_op(
                op_name="pt_mul_one_rescale_to_next",
                start_level=1,
                prep_fn=None,
                exec_fn=None,
                supported=False,
                notes="Cannot rescale below Level 1",
            )

    # -------------------------------------------------------------
    # 3. Plaintext Multiplication by Arbitrary Scalar + Rescale
    # (e.g. polynomial coefficient scaling c * 0.15012)
    # -------------------------------------------------------------
    scalar_val = float(POLY3["c1"])
    for lvl in range(7, 0, -1):
        if lvl >= 2:
            burn_n = 7 - lvl

            def prep():
                c = burn(ts.ckks_vector(ctx, raw_x), burn_n)
                return (c,)

            def exec_fn(c):
                return c * scalar_val

            def verify(out):
                dec = np.array(out.decrypt())
                expected = np.array(raw_x) * scalar_val
                return float(np.max(np.abs(dec - expected)))

            measure_op(
                op_name="pt_mul_scalar_rescale",
                start_level=lvl,
                prep_fn=prep,
                exec_fn=exec_fn,
                verify_fn=verify,
                supported=True,
                notes=f"Plaintext scalar multiply (c1={scalar_val}) + rescale",
            )
        else:
            measure_op(
                op_name="pt_mul_scalar_rescale",
                start_level=1,
                prep_fn=None,
                exec_fn=None,
                supported=False,
                notes="Cannot rescale below Level 1",
            )

    # -------------------------------------------------------------
    # 4. Ciphertext-Ciphertext Addition (ct_ct_add)
    # Supported at all levels 7..1
    # -------------------------------------------------------------
    for lvl in range(7, 0, -1):
        burn_n = 7 - lvl

        def prep():
            c1 = burn(ts.ckks_vector(ctx, raw_x), burn_n)
            c2 = burn(ts.ckks_vector(ctx, raw_w), burn_n)
            return c1, c2

        def exec_fn(c1, c2):
            return c1 + c2

        def verify(out):
            dec = np.array(out.decrypt())
            expected = np.array(raw_x) + np.array(raw_w)
            return float(np.max(np.abs(dec - expected)))

        measure_op(
            op_name="ct_ct_add",
            start_level=lvl,
            prep_fn=prep,
            exec_fn=exec_fn,
            verify_fn=verify,
            supported=True,
            notes=f"Ct-Ct addition at Level {lvl}",
        )

    # -------------------------------------------------------------
    # 5. Ciphertext-Ciphertext Subtraction (ct_ct_sub)
    # Supported at all levels 7..1
    # -------------------------------------------------------------
    for lvl in range(7, 0, -1):
        burn_n = 7 - lvl

        def prep():
            c1 = burn(ts.ckks_vector(ctx, raw_x), burn_n)
            c2 = burn(ts.ckks_vector(ctx, raw_w), burn_n)
            return c1, c2

        def exec_fn(c1, c2):
            return c1 - c2

        def verify(out):
            dec = np.array(out.decrypt())
            expected = np.array(raw_x) - np.array(raw_w)
            return float(np.max(np.abs(dec - expected)))

        measure_op(
            op_name="ct_ct_sub",
            start_level=lvl,
            prep_fn=prep,
            exec_fn=exec_fn,
            verify_fn=verify,
            supported=True,
            notes=f"Ct-Ct subtraction at Level {lvl}",
        )

    # -------------------------------------------------------------
    # 6. In-Slot Summation via Rotation Tree (slot_sum)
    # Supported at all levels 7..1
    # -------------------------------------------------------------
    for lvl in range(7, 0, -1):
        burn_n = 7 - lvl

        def prep():
            c = burn(ts.ckks_vector(ctx, raw_x), burn_n)
            return (c,)

        def exec_fn(c):
            return c.sum()

        def verify(out):
            dec = out.decrypt()[0]
            expected = float(np.sum(raw_x))
            return float(np.abs(dec - expected))

        measure_op(
            op_name="slot_sum",
            start_level=lvl,
            prep_fn=prep,
            exec_fn=exec_fn,
            verify_fn=verify,
            supported=True,
            notes=f"Slot summation across 8,192 slots at Level {lvl}",
        )

    # -------------------------------------------------------------
    # 7. Complete Feature-Gradient Dot Product: (err * X_j).sum()
    # At Level 3 (Condition A): Level 3 * Level 3 -> Level 2 -> sum() at Level 2
    # At Level 2 (Conditions B & C): Level 2 * Level 2 -> Level 1 -> sum() at Level 1
    # -------------------------------------------------------------
    for grad_lvl in [3, 2]:
        burn_n = 7 - grad_lvl

        def prep():
            c_err = burn(ts.ckks_vector(ctx, raw_y), burn_n)
            c_x = burn(ts.ckks_vector(ctx, raw_x), burn_n)
            return c_err, c_x

        def exec_fn(c_err, c_x):
            return (c_err * c_x).sum()

        def verify(out):
            dec = out.decrypt()[0]
            expected = float(np.dot(raw_y, raw_x))
            return float(np.abs(dec - expected))

        cond_label = "Condition A" if grad_lvl == 3 else "Conditions B & C"
        measure_op(
            op_name="gradient_dot_product",
            start_level=grad_lvl,
            prep_fn=prep,
            exec_fn=exec_fn,
            verify_fn=verify,
            supported=True,
            notes=f"Complete dot product at Level {grad_lvl}->{grad_lvl-1} ({cond_label})",
        )

    # -------------------------------------------------------------
    # 8. Complete Feature-Alignment Sequences
    # burn(X, 4): Level 7 -> Level 3 (Condition A)
    # burn(X, 5): Level 7 -> Level 2 (Conditions B & C)
    # -------------------------------------------------------------
    for burn_count in [4, 5]:
        target_lvl = 7 - burn_count

        def prep():
            c = ts.ckks_vector(ctx, raw_x)
            return (c,)

        def exec_fn(c):
            return burn(c, burn_count)

        def verify(out):
            dec = np.array(out.decrypt())
            expected = np.array(raw_x)
            return float(np.max(np.abs(dec - expected)))

        measure_op(
            op_name=f"feature_alignment_burn{burn_count}",
            start_level=7,
            prep_fn=prep,
            exec_fn=exec_fn,
            verify_fn=verify,
            supported=True,
            notes=f"Sequence of {burn_count} plaintext mults by 1.0 (Level 7 -> Level {target_lvl})",
        )

    # -------------------------------------------------------------
    # 9. Condition B Adjustment (err * 1.0 at Level 3 -> Level 2)
    # -------------------------------------------------------------
    def prep_cond_b():
        c_err = burn(ts.ckks_vector(ctx, raw_y), 4)  # err is at level 3
        return (c_err,)

    def exec_cond_b(c_err):
        return c_err * 1.0

    def verify_cond_b(out):
        dec = np.array(out.decrypt())
        expected = np.array(raw_y)
        return float(np.max(np.abs(dec - expected)))

    measure_op(
        op_name="condition_b_level_adjustment",
        start_level=3,
        prep_fn=prep_cond_b,
        exec_fn=exec_cond_b,
        verify_fn=verify_cond_b,
        supported=True,
        notes="Condition B single scalar mult by 1.0 on err (Level 3 -> Level 2)",
    )

    # -------------------------------------------------------------
    # 10. Scalar Decryption: out.decrypt()[0]
    # -------------------------------------------------------------
    for lvl in [2, 1]:
        burn_n = 7 - lvl

        def prep():
            c = burn(ts.ckks_vector(ctx, raw_x), burn_n)
            return (c,)

        def exec_fn(c):
            return c.decrypt()[0]

        measure_op(
            op_name="decryption_scalar",
            start_level=lvl,
            prep_fn=prep,
            exec_fn=exec_fn,
            verify_fn=None,
            supported=True,
            notes=f"Decryption of scalar result at Level {lvl}",
        )

    return {"records": records}


def run_multi_process_benchmark(runs: int = 3, reps: int = 30, warmup: int = 5) -> dict:
    all_process_runs = []
    print(f"Executing {runs} independent fresh process runs ({reps} reps each)...")

    for r in range(1, runs + 1):
        print(f"\n--- Process Run {r}/{runs} ---")
        cmd = [
            sys.executable,
            __file__,
            "--sub-process",
            "--reps",
            str(reps),
            "--warmup",
            str(warmup),
            "--seed",
            str(SEED + r * 10),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split("\n")
        json_line = None
        for line in reversed(lines):
            if line.startswith("JSON_OUTPUT:"):
                json_line = line[len("JSON_OUTPUT:") :]
                break
        if not json_line:
            raise RuntimeError(f"Subprocess run {r} did not produce JSON_OUTPUT marker.\n{res.stdout}\n{res.stderr}")
        run_data = json.loads(json_line)
        all_process_runs.append(run_data["records"])
        print(f"Run {r} completed successfully ({len(run_data['records'])} operations benchmarked).")

    # Aggregate samples across all runs
    num_ops = len(all_process_runs[0])
    combined_records = []

    for i in range(num_ops):
        op_info = all_process_runs[0][i]
        op_name = op_info["operation"]
        start_lvl = op_info["start_level"]
        supported = op_info["supported"]
        notes = op_info["notes"]

        if not supported:
            combined_records.append({
                "operation": op_name,
                "start_level": start_lvl,
                "end_level": None,
                "supported": False,
                "pooled_stats": None,
                "per_run_means_ms": [],
                "max_abs_error": None,
                "notes": notes,
            })
            continue

        pooled_samples = []
        run_means = []
        max_errors = []

        for r_idx in range(runs):
            r_rec = all_process_runs[r_idx][i]
            pooled_samples.extend(r_rec["samples_ms"])
            run_means.append(r_rec["stats"]["mean_ms"])
            if r_rec["max_abs_error"] is not None:
                max_errors.append(r_rec["max_abs_error"])

        combined_records.append({
            "operation": op_name,
            "start_level": start_lvl,
            "end_level": op_info["end_level"],
            "supported": True,
            "pooled_stats": compute_stats(pooled_samples),
            "per_run_means_ms": run_means,
            "max_abs_error": float(np.max(max_errors)) if max_errors else None,
            "notes": notes,
        })

    return {
        "environment": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
            "tenseal_version": ts.__version__,
            "logical_cpus": os.cpu_count(),
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
            "data_primes_count": 7,
            "special_prime_bits": 60,
            "scale": "2^40",
            "slots": SLOT_CAPACITY,
        },
        "meta": {
            "process_runs": runs,
            "reps_per_run": reps,
            "warmup_reps": warmup,
            "total_samples_per_supported_op": runs * reps,
        },
        "operations": combined_records,
    }


def main():
    parser = argparse.ArgumentParser(description="CKKS Microbenchmark Profiler")
    parser.add_argument("--sub-process", action="store_true", help="Internal flag for child process")
    parser.add_argument("--runs", type=int, default=3, help="Number of fresh process runs")
    parser.add_argument("--reps", type=int, default=30, help="Measured repetitions per run")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up repetitions")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    if args.sub_process:
        res = run_single_process_benchmarks(reps=args.reps, warmup=args.warmup, seed=args.seed)
        print("JSON_OUTPUT:" + json.dumps(res))
        sys.exit(0)

    print("=" * 80)
    print(" CKKS MICROBENCHMARK PROFILER FOR WORKLOAD-AWARE SCHEDULING")
    print("=" * 80)
    print(f"Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform    : {platform.platform()} | Python: {sys.version.split()[0]} | TenSEAL: {ts.__version__}")
    print(f"Protocol    : {args.runs} fresh process runs x {args.reps} repetitions ({args.runs * args.reps} samples/op)")
    print(f"Parameters  : N=16384, coeff_mod=[60,40,40,40,40,40,40,60], scale=2^40, slots=8192")

    t0_all = time.perf_counter()
    full_results = run_multi_process_benchmark(runs=args.runs, reps=args.reps, warmup=args.warmup)
    total_bench_time = time.perf_counter() - t0_all
    full_results["meta"]["total_profiling_wall_clock_s"] = total_bench_time

    # Save JSON
    json_path = OUT_DIR / "microbenchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n[OK] Saved microbenchmark JSON: {json_path}")

    # Build and Save CSV
    csv_rows = []
    log_lines = [
        "=" * 80,
        " CKKS MICROBENCHMARK PROFILER SUMMARY",
        "=" * 80,
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform : {platform.platform()} | CPUs: {os.cpu_count()}",
        "",
        f"{'Operation':32s} | {'Lvl':3s} | {'Mean (ms)':10s} | {'Std (ms)':9s} | {'Median':9s} | {'IQR (ms)':9s} | {'Max AE':9s} | Notes",
        "-" * 120,
    ]

    for op in full_results["operations"]:
        st = op["pooled_stats"]
        row = {
            "operation": op["operation"],
            "start_level": op["start_level"],
            "end_level": op["end_level"],
            "supported": op["supported"],
            "mean_ms": st["mean_ms"] if st else None,
            "std_ms": st["std_ms"] if st else None,
            "median_ms": st["median_ms"] if st else None,
            "q25_ms": st["q25_ms"] if st else None,
            "q75_ms": st["q75_ms"] if st else None,
            "iqr_ms": st["iqr_ms"] if st else None,
            "min_ms": st["min_ms"] if st else None,
            "max_ms": st["max_ms"] if st else None,
            "max_abs_error": op["max_abs_error"],
            "notes": op["notes"],
        }
        csv_rows.append(row)

        if op["supported"]:
            err_str = f"{op['max_abs_error']:.2e}" if op["max_abs_error"] is not None else "N/A"
            log_line = (
                f"{op['operation']:32s} | "
                f"L{op['start_level']:<2d} | "
                f"{st['mean_ms']:10.3f} | "
                f"{st['std_ms']:9.3f} | "
                f"{st['median_ms']:9.3f} | "
                f"{st['iqr_ms']:9.3f} | "
                f"{err_str:9s} | "
                f"{op['notes']}"
            )
        else:
            log_line = (
                f"{op['operation']:32s} | "
                f"L{op['start_level']:<2d} | "
                f"{'UNSUPPORTED':10s} | "
                f"{'-':9s} | "
                f"{'-':9s} | "
                f"{'-':9s} | "
                f"{'-':9s} | "
                f"{op['notes']}"
            )
        log_lines.append(log_line)

    csv_path = OUT_DIR / "microbenchmark_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"[OK] Saved microbenchmark CSV : {csv_path}")

    log_path = OUT_DIR / "microbenchmark.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"[OK] Saved microbenchmark Log : {log_path}")

    print("\nSummary Table:")
    for line in log_lines[6:]:
        print(line)

    print(f"\nTotal profiling time: {total_bench_time:.2f}s")


if __name__ == "__main__":
    main()
