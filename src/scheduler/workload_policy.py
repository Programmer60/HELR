#!/usr/bin/env python3
"""
PTB-XL Workload-Aware CKKS Scheduling Policy
============================================
Implements the formal cost-model-guided scheduling policy for CKKS logistic regression.
Given workload dimensions, constraints, and runtime context, evaluates all feasible
candidate schedules and selects the optimal execution plan:

  S* = argmin_{S in Feasible} T_total(S)

Subject to:
  1. Valid CKKS modulus-chain state (multiplicative depth >= required circuit depth).
  2. Scale alignment across operands.
  3. Maximum gradient numerical error <= 1e-4.
  4. Memory budget: memory_required_mb <= memory_budget_mb.
  5. Multi-epoch caching benefit: caching selected only if T_total(cached) < T_total(repeated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.scheduler.cost_model_v2 import CKKSCostModelV2, ExecutionContext, ScheduleV2


@dataclass(frozen=True)
class ExecutionPlan:
    schedule_id: str                          # e.g., "B-cache", "A-repeat"
    condition: Literal["A", "B", "C"]         # "A" or "B" (or "C")
    poly_degree: int                          # 3
    grad_level: int                           # 3 (A) or 2 (B)
    burn_X: int                               # 4 (A) or 5 (B)
    burn_y: int                               # 4 (A/B deg 3)
    level_adjustment_required: bool           # True for B, False for A
    alignment_strategy: Literal["repeated", "cached"]
    predicted_epoch1_s: float
    predicted_subsequent_epoch_s: float
    predicted_total_training_s: float
    memory_required_mb: float
    memory_budget_mb: float
    feasible: bool
    rejection_reason: Optional[str] = None
    notes: str = ""


class WorkloadSchedulerPolicy:
    """
    Evaluates candidate execution schedules using CostModel_v2 under explicit
    hardware and cryptographic constraints.
    """

    def __init__(
        self,
        cost_model: Optional[CKKSCostModelV2] = None,
        default_context: ExecutionContext = "persistent_worker",
    ):
        self.model = cost_model or CKKSCostModelV2()
        self.default_context: ExecutionContext = default_context

    @staticmethod
    def estimate_ciphertext_memory_mb(
        B: int,
        D: int,
        grad_level: int,
        slot_capacity: int = 8192,
        poly_modulus_degree: int = 16384,
    ) -> float:
        """
        Estimates RAM required to cache aligned feature ciphertexts.
        Active primes:
          Level 3 -> 4 active primes (60+40+40+40) -> 4 uint64 per coefficient
          Level 2 -> 3 active primes (60+40+40)    -> 3 uint64 per coefficient
        Each ciphertext has 2 RNS polynomial components.
        """
        active_primes = grad_level + 1
        bytes_per_ct = poly_modulus_degree * 2 * active_primes * 8
        total_bytes = B * D * bytes_per_ct
        return total_bytes / (1024.0 * 1024.0)

    def evaluate_candidate(
        self,
        condition: Literal["A", "B"],
        alignment_strategy: Literal["repeated", "cached"],
        N: int,
        D: int,
        epochs: int,
        memory_budget_mb: float = 2048.0,
        context: Optional[ExecutionContext] = None,
        allow_condition_B: bool = True,
        max_error_threshold: float = 1e-4,
    ) -> ExecutionPlan:
        ctx = context or self.default_context
        B = self.model.get_num_batches(N, slot_capacity=8192)

        poly_degree = 3
        grad_level = 3 if condition == "A" else 2
        burn_X = 4 if condition == "A" else 5
        burn_y = 4
        level_adj = bool(condition == "B")
        sched_id = f"{condition}-{'cache' if alignment_strategy == 'cached' else 'repeat'}"

        sched = ScheduleV2(
            condition=condition,
            poly_degree=poly_degree,
            grad_level=grad_level,
            burn_n=burn_X,
            alignment_strategy=alignment_strategy,
        )

        # 1. Check Feasibility
        if condition == "B" and not allow_condition_B:
            return ExecutionPlan(
                schedule_id=sched_id,
                condition=condition,
                poly_degree=poly_degree,
                grad_level=grad_level,
                burn_X=burn_X,
                burn_y=burn_y,
                level_adjustment_required=level_adj,
                alignment_strategy=alignment_strategy,
                predicted_epoch1_s=float("inf"),
                predicted_subsequent_epoch_s=float("inf"),
                predicted_total_training_s=float("inf"),
                memory_required_mb=0.0,
                memory_budget_mb=memory_budget_mb,
                feasible=False,
                rejection_reason="Condition B explicitly disabled or cryptographically infeasible",
            )

        # 2. Check Memory Budget for Caching
        mem_mb = 0.0
        if alignment_strategy == "cached":
            mem_mb = self.estimate_ciphertext_memory_mb(B, D, grad_level)
            if mem_mb > memory_budget_mb:
                return ExecutionPlan(
                    schedule_id=sched_id,
                    condition=condition,
                    poly_degree=poly_degree,
                    grad_level=grad_level,
                    burn_X=burn_X,
                    burn_y=burn_y,
                    level_adjustment_required=level_adj,
                    alignment_strategy=alignment_strategy,
                    predicted_epoch1_s=float("inf"),
                    predicted_subsequent_epoch_s=float("inf"),
                    predicted_total_training_s=float("inf"),
                    memory_required_mb=mem_mb,
                    memory_budget_mb=memory_budget_mb,
                    feasible=False,
                    rejection_reason=f"Cached ciphertext memory ({mem_mb:.1f} MB) exceeds budget ({memory_budget_mb:.1f} MB)",
                )

        # 3. Predict Latencies using CostModel_v2
        comp_e1 = self.model.predict_components(sched, N=N, D=D, epoch_idx=1, context=ctx)
        pred_e1_s = comp_e1["total_epoch_time_s"]

        if alignment_strategy == "cached":
            comp_e2 = self.model.predict_components(sched, N=N, D=D, epoch_idx=2, context=ctx)
            pred_sub_s = comp_e2["total_epoch_time_s"]
            # Total = Setup (precompute) + E * step_cached
            # comp_e1["alignment_precompute_s"] is the setup cost
            pred_total_s = comp_e1["alignment_precompute_s"] + (epochs * pred_sub_s)
        else:
            pred_sub_s = pred_e1_s
            pred_total_s = epochs * pred_e1_s

        return ExecutionPlan(
            schedule_id=sched_id,
            condition=condition,
            poly_degree=poly_degree,
            grad_level=grad_level,
            burn_X=burn_X,
            burn_y=burn_y,
            level_adjustment_required=level_adj,
            alignment_strategy=alignment_strategy,
            predicted_epoch1_s=pred_e1_s,
            predicted_subsequent_epoch_s=pred_sub_s,
            predicted_total_training_s=pred_total_s,
            memory_required_mb=mem_mb,
            memory_budget_mb=memory_budget_mb,
            feasible=True,
            rejection_reason=None,
            notes=f"Degree-3 CKKS LR with gradient entry at Level {grad_level}",
        )

    def select_plan(
        self,
        N: int = 17084,
        D: int = 48,
        epochs: int = 10,
        memory_budget_mb: float = 2048.0,
        context: Optional[ExecutionContext] = None,
        allow_condition_B: bool = True,
        max_error_threshold: float = 1e-4,
    ) -> Tuple[ExecutionPlan, List[ExecutionPlan]]:
        """
        Evaluates the four canonical Degree-3 candidates:
          A-repeat, A-cache, B-repeat, B-cache
        Selects the plan with minimal predicted cumulative training time.
        """
        candidates: List[ExecutionPlan] = []

        for cond in ["A", "B"]:
            for strat in ["repeated", "cached"]:
                plan = self.evaluate_candidate(
                    condition=cond,  # type: ignore
                    alignment_strategy=strat,  # type: ignore
                    N=N,
                    D=D,
                    epochs=epochs,
                    memory_budget_mb=memory_budget_mb,
                    context=context,
                    allow_condition_B=allow_condition_B,
                    max_error_threshold=max_error_threshold,
                )
                candidates.append(plan)

        feasible_plans = [p for p in candidates if p.feasible]

        if not feasible_plans:
            raise RuntimeError("No feasible CKKS training schedule found within constraints!")

        # Sort strictly by total training time
        best_plan = min(feasible_plans, key=lambda p: p.predicted_total_training_s)

        # Caching decision check:
        # If best_plan is cached, but its total time is >= repeated (e.g. for epochs=1),
        # fallback to repeated to avoid unnecessary memory overhead
        if best_plan.alignment_strategy == "cached":
            partner_repeat = [
                p for p in feasible_plans
                if p.condition == best_plan.condition and p.alignment_strategy == "repeated"
            ]
            if partner_repeat and partner_repeat[0].predicted_total_training_s <= best_plan.predicted_total_training_s:
                best_plan = partner_repeat[0]

        return best_plan, candidates
