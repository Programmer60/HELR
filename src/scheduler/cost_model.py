#!/usr/bin/env python3
"""
PTB-XL Workload-Aware CKKS Scheduling: Analytical Cost Model
============================================================
An interpretable, component-decomposed analytical cost model that predicts
the per-epoch and total training runtime for homomorphic logistic regression
under various CKKS modulus-level scheduling strategies.

Mathematical Formulation:
  T_epoch(sched, N, D) = T_wenc(D)
                       + T_fwd(B, D)
                       + T_poly(B, deg)
                       + T_err(B, deg)
                       + T_ladj(B, sched)
                       + T_align(B, D, sched, epoch_idx)
                       + T_grad(B, D, lvl)
                       + T_dec(B, D, lvl)

where:
  B = ceil(N / S) is the number of ciphertext batches (S = slot capacity, default 8192)
  D is the number of features (default 48)
  lvl is the ciphertext modulus level entering the gradient dot product (Level 3 or 2)
  gamma is the loop execution throughput factor (accounting for cache warmup and OpenMP reuse)

Supported Schedules:
  - Condition A: Degree-3 polynomial, gradient at Level 3 (Control)
  - Condition B: Degree-3 polynomial, gradient at Level 2 via controlled level adjustment (Intervention)
  - Condition C: Degree-5 polynomial, gradient at Level 2 (Control)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Default repository paths
DEFAULT_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "results" / "ptbxl" / "scheduler" / "microbenchmark_results.json"
DEFAULT_PHASE4_PATH = Path(__file__).resolve().parents[2] / "results" / "ptbxl" / "phase4" / "ptbxl_phase4_results.json"


@dataclass(frozen=True)
class Schedule:
    condition: Literal["A", "B", "C"]
    poly_degree: int
    grad_level: int
    burn_n: int
    alignment_strategy: Literal["repeated", "cached"] = "repeated"
    name: str = ""

    def __post_init__(self):
        if not self.name:
            strat_label = "Cached" if self.alignment_strategy == "cached" else "Repeated"
            object.__setattr__(
                self,
                "name",
                f"Condition {self.condition} (Deg-{self.poly_degree}, Grad L{self.grad_level}, {strat_label})"
            )


class CKKSCostModel:
    """
    Analytical performance model calibrated against measured microbenchmarks
    and empirical Phase 4 pipeline benchmarks.
    """

    def __init__(
        self,
        benchmark_path: Optional[Path | str] = None,
        phase4_path: Optional[Path | str] = None,
    ):
        self.bench_path = Path(benchmark_path) if benchmark_path else DEFAULT_BENCHMARK_PATH
        self.phase4_path = Path(phase4_path) if phase4_path else DEFAULT_PHASE4_PATH

        self._load_benchmarks()
        self._load_phase4_reference()

        # Batch loop execution throughput factor (OpenMP + cache warming ratio)
        # Ratio of isolated microbenchmark operation latency to tight-loop batch latency
        self.loop_factor = 2.1627

    def _load_benchmarks(self) -> None:
        if not self.bench_path.exists():
            raise FileNotFoundError(f"Microbenchmark results not found at: {self.bench_path}")

        with open(self.bench_path, "r") as f:
            data = json.load(f)

        self.benchmarks: Dict[tuple[str, Optional[int]], Dict[str, Any]] = {}
        for op in data["operations"]:
            key = (op["operation"], op.get("start_level"))
            self.benchmarks[key] = op

    def _load_phase4_reference(self) -> None:
        self.phase4_summary: Dict[str, Dict[str, float]] = {}
        if self.phase4_path.exists():
            with open(self.phase4_path, "r") as f:
                data = json.load(f)
            for item in data.get("summary", []):
                self.phase4_summary[item["condition"]] = item

    @staticmethod
    def get_num_batches(N: int, slot_capacity: int = 8192) -> int:
        if N <= 0 or slot_capacity <= 0:
            raise ValueError(f"N ({N}) and slot_capacity ({slot_capacity}) must be positive")
        return int(math.ceil(N / slot_capacity))

    def validate_schedule(self, schedule: Schedule) -> None:
        """
        Validates the mathematical and cryptographic validity of a schedule.
        Rejects invalid states such as unsupported Level 1 multiplications.
        """
        if schedule.condition not in ("A", "B", "C"):
            raise ValueError(f"Unknown condition '{schedule.condition}'. Must be 'A', 'B', or 'C'.")

        if schedule.poly_degree not in (3, 5):
            raise ValueError(f"Unsupported polynomial degree: {schedule.poly_degree}. Supported: 3, 5.")

        if schedule.grad_level not in (2, 3):
            raise ValueError(
                f"Unsupported gradient level: {schedule.grad_level}. "
                "Level 1 multiplication is unsupported by RNS-CKKS (cannot rescale with 1 prime left)."
            )

        if schedule.condition == "A":
            if schedule.poly_degree != 3 or schedule.grad_level != 3 or schedule.burn_n != 4:
                raise ValueError("Condition A requires poly_degree=3, grad_level=3, burn_n=4")
        elif schedule.condition == "B":
            if schedule.poly_degree != 3 or schedule.grad_level != 2 or schedule.burn_n != 5:
                raise ValueError("Condition B requires poly_degree=3, grad_level=2, burn_n=5")
        elif schedule.condition == "C":
            if schedule.poly_degree != 5 or schedule.grad_level != 2 or schedule.burn_n != 5:
                raise ValueError("Condition C requires poly_degree=5, grad_level=2, burn_n=5")

        if schedule.alignment_strategy not in ("repeated", "cached"):
            raise ValueError(f"Unknown alignment strategy: {schedule.alignment_strategy}")

    def predict_components(
        self,
        schedule: Schedule,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epoch_idx: int = 1,
    ) -> Dict[str, float]:
        """
        Predicts the time (in seconds) for each pipeline stage for a single epoch.
        """
        self.validate_schedule(schedule)
        B = self.get_num_batches(N, slot_capacity)

        # 1. Weight encryption: O(D) broadcast encryptions once per epoch
        # Microbenchmark baseline / Phase 4 empirical: 0.7169s for D=48
        t_weight_enc = (0.7169 / 48.0) * D

        # 2. Forward linear combination: B * D ct-ct multiplies at Level 7
        fwd_bench_ms = self.benchmarks[("ct_ct_mul_rescale", 7)]["pooled_stats"]["mean_ms"]
        t_forward = (B * D * (fwd_bench_ms / 1000.0)) / self.loop_factor

        # 3. Polynomial evaluation: B polynomial evaluations
        # Phase 4 empirical: Deg 3 = 0.1338s for B=3 (0.0446s/batch); Deg 5 = 0.1891s for B=3 (0.0630s/batch)
        poly_per_batch = 0.0446 if schedule.poly_degree == 3 else 0.0630
        t_poly = B * poly_per_batch

        # 4. Error computation: B ciphertext subtractions
        # Phase 4 empirical: 0.0397s for Deg 3 (B=3); 0.0429s for Deg 5 (B=3)
        err_per_batch = 0.0132 if schedule.poly_degree == 3 else 0.0143
        t_error = B * err_per_batch

        # 5. Controlled Level Adjustment (Condition B only): B * pt_mul_one at Level 3
        if schedule.condition == "B":
            ladj_bench_ms = self.benchmarks[("condition_b_level_adjustment", 3)]["pooled_stats"]["mean_ms"]
            t_level_adj = (B * (ladj_bench_ms / 1000.0)) / self.loop_factor
        else:
            t_level_adj = 0.0

        # 6. Feature alignment: B * D column burns
        # In cached mode: precomputed in epoch 1, 0 in subsequent epochs
        if schedule.alignment_strategy == "cached" and epoch_idx > 1:
            t_align = 0.0
        else:
            burn_op = ("feature_alignment_burn4", 7) if schedule.burn_n == 4 else ("feature_alignment_burn5", 7)
            align_bench_ms = self.benchmarks[burn_op]["pooled_stats"]["mean_ms"]
            t_align = (B * D * (align_bench_ms / 1000.0)) / self.loop_factor

        # 7. Isolated Gradient computation: B * D encrypted dot products (c_err * X_j).sum()
        grad_bench_ms = self.benchmarks[("gradient_dot_product", schedule.grad_level)]["pooled_stats"]["mean_ms"]
        t_gradient = (B * D * (grad_bench_ms / 1000.0)) / self.loop_factor

        # 8. Decryption & aggregation: B * D scalar decryptions
        dec_lvl = schedule.grad_level - 1  # Result of grad dot product drops 1 level
        dec_bench_ms = self.benchmarks[("decryption_scalar", dec_lvl)]["pooled_stats"]["mean_ms"]
        t_dec = (B * D * (dec_bench_ms / 1000.0)) / self.loop_factor

        t_total = (
            t_weight_enc
            + t_forward
            + t_poly
            + t_error
            + t_level_adj
            + t_align
            + t_gradient
            + t_dec
        )

        return {
            "weight_enc_s": t_weight_enc,
            "forward_linear_s": t_forward,
            "poly_eval_s": t_poly,
            "error_comp_s": t_error,
            "level_adjustment_s": t_level_adj,
            "feature_alignment_s": t_align,
            "gradient_comp_s": t_gradient,
            "decryption_agg_s": t_dec,
            "total_epoch_time_s": t_total,
        }

    def predict_epoch(
        self,
        schedule: Schedule,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epoch_idx: int = 1,
    ) -> float:
        """Convenience method returning predicted total epoch runtime in seconds."""
        components = self.predict_components(schedule, N, D, slot_capacity, epoch_idx)
        return components["total_epoch_time_s"]

    def predict_training(
        self,
        schedule: Schedule,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """Predicts cumulative training latency over multiple epochs."""
        epoch_latencies = []
        for ep in range(1, epochs + 1):
            comp = self.predict_components(schedule, N, D, slot_capacity, ep)
            epoch_latencies.append(comp["total_epoch_time_s"])

        total_time_s = sum(epoch_latencies)
        mean_epoch_s = total_time_s / epochs

        return {
            "schedule": schedule.name,
            "condition": schedule.condition,
            "alignment_strategy": schedule.alignment_strategy,
            "epochs": epochs,
            "total_training_time_s": total_time_s,
            "mean_epoch_time_s": mean_epoch_s,
            "per_epoch_latencies_s": epoch_latencies,
        }

    @staticmethod
    def get_standard_schedules() -> List[Schedule]:
        """Returns the canonical schedules for Conditions A, B, and C."""
        return [
            Schedule(condition="A", poly_degree=3, grad_level=3, burn_n=4, alignment_strategy="repeated"),
            Schedule(condition="B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="repeated"),
            Schedule(condition="C", poly_degree=5, grad_level=2, burn_n=5, alignment_strategy="repeated"),
            Schedule(condition="A", poly_degree=3, grad_level=3, burn_n=4, alignment_strategy="cached"),
            Schedule(condition="B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="cached"),
            Schedule(condition="C", poly_degree=5, grad_level=2, burn_n=5, alignment_strategy="cached"),
        ]

    def recommend_best_schedule(
        self,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epochs: int = 10,
    ) -> Schedule:
        """
        Evaluates all valid candidate schedules and selects the policy
        with minimum cumulative execution time.
        """
        schedules = self.get_standard_schedules()
        best_sched = None
        best_time = float("inf")

        for sched in schedules:
            res = self.predict_training(sched, N, D, slot_capacity, epochs)
            if res["total_training_time_s"] < best_time:
                best_time = res["total_training_time_s"]
                best_sched = sched

        assert best_sched is not None
        return best_sched
