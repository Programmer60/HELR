#!/usr/bin/env python3
"""
PTB-XL Workload-Aware CKKS Scheduling: CostModel_v2 (Runtime-Context-Aware)
==========================================================================
Second-generation analytical cost model for CKKS logistic regression training.
Explicitly parametrizes runtime execution context:
  - 'persistent_worker': Long-lived Python process (steady-state cache, warm thread pool)
  - 'fresh_worker': Ephemeral subprocess execution (fresh process initialization, cold start)

Component Decomposition Equation:
  T_epoch = T_weight_enc(D)
          + T_fwd(B, D)
          + T_poly(B, deg)
          + T_err(B, deg)
          + T_ladj(B, sched)
          + T_align(B, D, sched, epoch_idx)
          + T_grad(B, D, lvl)
          + T_dec(B, D, lvl)

Primary Scheduling Decision (Step 5):
  Evaluates mathematical degree-3 logistic regression between:
    - Condition A: Gradient evaluated at Level 3 (burn 4)
    - Condition B: Gradient evaluated at Level 2 (burn 5 via enc_err * 1.0)
  With alignment strategies:
    - repeated: Re-burns feature columns on every epoch
    - cached: Precomputes and caches burned feature ciphertexts once
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Default repository paths
DEFAULT_PROFILE_V2_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "ptbxl"
    / "scheduler"
    / "phase3a"
    / "execution_context_latencies.json"
)

ExecutionContext = Literal["persistent_worker", "fresh_worker"]
Condition = Literal["A", "B", "C"]
AlignmentStrategy = Literal["repeated", "cached"]


@dataclass(frozen=True)
class ScheduleV2:
    condition: Condition
    poly_degree: int
    grad_level: int
    burn_n: int
    alignment_strategy: AlignmentStrategy = "repeated"
    name: str = ""

    def __post_init__(self):
        if not self.name:
            strat_label = "Cached" if self.alignment_strategy == "cached" else "Repeated"
            object.__setattr__(
                self,
                "name",
                f"Condition {self.condition} (Deg-{self.poly_degree}, Grad L{self.grad_level}, {strat_label})",
            )


class CKKSCostModelV2:
    """
    Context-aware analytical cost model implementing operation-specific calibration
    for both persistent and fresh worker execution modes.
    """

    def __init__(
        self,
        profile_path: Optional[Path | str] = None,
        default_context: ExecutionContext = "persistent_worker",
    ):
        self.profile_path = Path(profile_path) if profile_path else DEFAULT_PROFILE_V2_PATH
        self.default_context: ExecutionContext = default_context
        self._load_profiles()

    def _load_profiles(self) -> None:
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Execution context profile not found at: {self.profile_path}")

        with open(self.profile_path, "r") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.ratio_summary = data.get("ratio_summary", {})

        # Build fast lookup tables for persistent and fresh profiles
        self.persistent_ops: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
        for op in data["persistent_worker"]["operations"]:
            self.persistent_ops[(op["operation"], op.get("start_level"))] = op

        self.fresh_ops: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
        for op in data["fresh_worker"]["operations"]:
            self.fresh_ops[(op["operation"], op.get("start_level"))] = op

        # Process startup overheads for fresh worker
        fw_meta = data["fresh_worker"]
        self.fresh_startup_overhead = {
            "ctx_creation_ms": fw_meta.get("avg_ctx_creation_ms", 901.5),
            "key_generation_ms": fw_meta.get("avg_key_generation_ms", 1464.7),
            "warmup_total_ms": fw_meta.get("avg_warmup_total_ms", 11412.8),
            "total_wall_clock_ms": fw_meta.get("avg_subprocess_wall_clock_ms", 14887.5),
        }

    @staticmethod
    def get_num_batches(N: int, slot_capacity: int = 8192) -> int:
        if N <= 0 or slot_capacity <= 0:
            raise ValueError(f"N ({N}) and slot_capacity ({slot_capacity}) must be positive integers.")
        return int(math.ceil(N / slot_capacity))

    def validate_schedule(self, schedule: ScheduleV2) -> None:
        """Validates schedule consistency and cryptographically admissible levels."""
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
                raise ValueError("Condition A requires poly_degree=3, grad_level=3, burn_n=4.")
        elif schedule.condition == "B":
            if schedule.poly_degree != 3 or schedule.grad_level != 2 or schedule.burn_n != 5:
                raise ValueError("Condition B requires poly_degree=3, grad_level=2, burn_n=5.")
        elif schedule.condition == "C":
            if schedule.poly_degree != 5 or schedule.grad_level != 2 or schedule.burn_n != 5:
                raise ValueError("Condition C requires poly_degree=5, grad_level=2, burn_n=5.")

        if schedule.alignment_strategy not in ("repeated", "cached"):
            raise ValueError(f"Unknown alignment strategy: {schedule.alignment_strategy}")

    def _get_op_mean_s(
        self,
        op_name: str,
        start_level: Optional[int],
        context: ExecutionContext,
    ) -> float:
        op_dict = self.persistent_ops if context == "persistent_worker" else self.fresh_ops
        entry = op_dict.get((op_name, start_level))
        if entry is None or not entry.get("supported", False):
            raise ValueError(f"Operation '{op_name}' at level {start_level} is unsupported or missing.")
        return float(entry["stats"]["mean_ms"]) / 1000.0

    def predict_components(
        self,
        schedule: ScheduleV2,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epoch_idx: int = 1,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, float]:
        """
        Predicts runtime (in seconds) for each pipeline stage for a single epoch.
        """
        self.validate_schedule(schedule)
        ctx = context or self.default_context
        B = self.get_num_batches(N, slot_capacity)

        # 1. Weight Encryption: O(D) broadcast encryptions once per epoch
        t_wenc_single = self._get_op_mean_s("weight_encryption_scalar", 7, ctx)
        t_weight_enc = D * t_wenc_single

        # 2. Forward Linear Combination: B * D Ct-Ct multiply-accumulate steps
        t_fwd_single = self._get_op_mean_s("forward_linear_accumulation_step", 7, ctx)
        t_forward = B * D * t_fwd_single

        # 3. Polynomial Evaluation: B polynomial evaluations
        # Calibrated per batch from operation profile components
        if schedule.poly_degree == 3:
            # z^2 (mul L6) + z^3 (mul L5 + pt_mul L6) + t3 (pt_mul L4) + t1 (burn 2 + pt_mul L4) + adds
            t_mul_l6 = self._get_op_mean_s("ct_ct_mul_rescale", 6, ctx)
            t_mul_l5 = self._get_op_mean_s("ct_ct_mul_rescale", 5, ctx)
            t_pt_l6 = self._get_op_mean_s("pt_mul_one_rescale_to_next", 6, ctx)
            t_poly_batch = t_mul_l6 + t_mul_l5 + t_pt_l6 + 0.012  # plus scalar mults & adds
        else:
            # Degree 5 adds z^5 (mul L4 + burn) + scalar mults & adds
            t_mul_l6 = self._get_op_mean_s("ct_ct_mul_rescale", 6, ctx)
            t_mul_l5 = self._get_op_mean_s("ct_ct_mul_rescale", 5, ctx)
            t_mul_l4 = self._get_op_mean_s("ct_ct_mul_rescale", 4, ctx)
            t_poly_batch = t_mul_l6 + t_mul_l5 + t_mul_l4 + 0.024
        t_poly = B * t_poly_batch

        # 4. Error Computation: B label burns and ciphertext subtractions
        # Label burn: burn 4 (deg 3) or burn 5 (deg 5)
        burn_y_op = "feature_alignment_burn4" if schedule.poly_degree == 3 else "feature_alignment_burn5"
        t_err_burn = self._get_op_mean_s(burn_y_op, 7, ctx)
        t_error = B * (t_err_burn + 0.0005)

        # 5. Condition B Level Adjustment: B * pt_mul_one at Level 3
        if schedule.condition == "B":
            t_ladj_single = self._get_op_mean_s("condition_b_level_adjustment", 3, ctx)
            t_level_adj = B * t_ladj_single
        else:
            t_level_adj = 0.0

        # 6. Feature Alignment: B * D column burns
        burn_op = "feature_alignment_burn4" if schedule.burn_n == 4 else "feature_alignment_burn5"
        t_align_single = self._get_op_mean_s(burn_op, 7, ctx)
        t_precompute = B * D * t_align_single

        if schedule.alignment_strategy == "cached" and epoch_idx > 1:
            t_align = 0.0
        else:
            t_align = t_precompute

        # 7. Isolated Gradient Computation: B * D encrypted dot products (err * X_j).sum()
        t_grad_single = self._get_op_mean_s("gradient_dot_product", schedule.grad_level, ctx)
        t_gradient = B * D * t_grad_single

        # 8. Decryption & Aggregation: B * D scalar decryptions
        dec_lvl = schedule.grad_level - 1
        t_dec_single = self._get_op_mean_s("decryption_scalar", dec_lvl, ctx)
        t_dec = B * D * t_dec_single

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
            "alignment_precompute_s": t_precompute,
            "gradient_comp_s": t_gradient,
            "decryption_agg_s": t_dec,
            "total_epoch_time_s": t_total,
        }

    def predict_epoch(
        self,
        schedule: ScheduleV2,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epoch_idx: int = 1,
        context: Optional[ExecutionContext] = None,
    ) -> float:
        """Returns predicted total epoch runtime in seconds."""
        comp = self.predict_components(schedule, N, D, slot_capacity, epoch_idx, context)
        return comp["total_epoch_time_s"]

    def predict_training(
        self,
        schedule: ScheduleV2,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epochs: int = 10,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """Predicts cumulative training latency over multiple epochs."""
        epoch_latencies = []
        for ep in range(1, epochs + 1):
            t_ep = self.predict_epoch(schedule, N, D, slot_capacity, ep, context)
            epoch_latencies.append(t_ep)

        total_time_s = sum(epoch_latencies)
        mean_epoch_s = total_time_s / epochs

        return {
            "schedule": schedule.name,
            "condition": schedule.condition,
            "alignment_strategy": schedule.alignment_strategy,
            "context": context or self.default_context,
            "epochs": epochs,
            "total_training_time_s": total_time_s,
            "mean_epoch_time_s": mean_epoch_s,
            "per_epoch_latencies_s": epoch_latencies,
        }

    @staticmethod
    def get_primary_schedules() -> List[ScheduleV2]:
        """
        Returns the primary mathematical degree-3 schedules (Step 5):
        Condition A vs Condition B, with repeated vs cached alignment.
        """
        return [
            ScheduleV2(condition="A", poly_degree=3, grad_level=3, burn_n=4, alignment_strategy="repeated"),
            ScheduleV2(condition="B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="repeated"),
            ScheduleV2(condition="A", poly_degree=3, grad_level=3, burn_n=4, alignment_strategy="cached"),
            ScheduleV2(condition="B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="cached"),
        ]

    @staticmethod
    def get_all_schedules() -> List[ScheduleV2]:
        """Returns all schedules including exploratory Degree-5 (Condition C)."""
        primary = CKKSCostModelV2.get_primary_schedules()
        degree5 = [
            ScheduleV2(condition="C", poly_degree=5, grad_level=2, burn_n=5, alignment_strategy="repeated"),
            ScheduleV2(condition="C", poly_degree=5, grad_level=2, burn_n=5, alignment_strategy="cached"),
        ]
        return primary + degree5

    def recommend_best_schedule(
        self,
        N: int = 17084,
        D: int = 48,
        slot_capacity: int = 8192,
        epochs: int = 10,
        context: Optional[ExecutionContext] = None,
        candidate_schedules: Optional[List[ScheduleV2]] = None,
    ) -> ScheduleV2:
        """
        Evaluates candidate schedules and selects the policy with minimum cumulative time.
        Defaults strictly to the primary mathematical degree-3 set (Step 5).
        """
        schedules = candidate_schedules or self.get_primary_schedules()
        best_sched = None
        best_time = float("inf")

        for sched in schedules:
            res = self.predict_training(sched, N, D, slot_capacity, epochs, context)
            if res["total_training_time_s"] < best_time:
                best_time = res["total_training_time_s"]
                best_sched = sched

        assert best_sched is not None
        return best_sched
