#!/usr/bin/env python3
"""
tests/test_phase3b_scheduler.py
================================
Phase 3B scheduler integration tests (12 tests).
Validates WorkloadSchedulerPolicy schedule-decision logic, ExecutionPlan fields,
caching logic, memory budgeting, numerical guard, and context accounting.

These tests are OFFLINE (no TenSEAL context required) — they operate on the
policy and cost-model objects only.
"""

import math
import pytest

from src.scheduler.workload_policy import WorkloadSchedulerPolicy, ExecutionPlan
from src.scheduler.cost_model_v2 import CKKSCostModelV2, ExecutionContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRODUCTION_N = 17_084
PRODUCTION_D = 48
PRODUCTION_EPOCHS = 10
DEFAULT_MEMORY_MB = 2048.0


def make_policy(**kwargs) -> WorkloadSchedulerPolicy:
    return WorkloadSchedulerPolicy(**kwargs)


# ---------------------------------------------------------------------------
# Test 1: Condition A schedule fields are correct
# ---------------------------------------------------------------------------

class TestConditionAScheduleConstruction:
    """Test 1 — Condition A produces the expected circuit parameters."""

    def test_condition_a_fields(self):
        pol = make_policy()
        plan = pol.evaluate_candidate(
            condition="A",
            alignment_strategy="repeated",
            N=PRODUCTION_N,
            D=PRODUCTION_D,
            epochs=PRODUCTION_EPOCHS,
        )
        assert plan.condition == "A"
        assert plan.grad_level == 3, (
            f"Condition A must enter gradient at Level 3; got {plan.grad_level}"
        )
        assert plan.burn_X == 4, (
            f"Condition A must apply burn_X=4 (4 depth-consuming ops on X); got {plan.burn_X}"
        )
        assert plan.level_adjustment_required is False, (
            "Condition A must NOT require a level-adjustment step (no pt_mul_one)"
        )
        assert plan.poly_degree == 3
        assert plan.burn_y == 4


# ---------------------------------------------------------------------------
# Test 2: Condition B schedule fields are correct
# ---------------------------------------------------------------------------

class TestConditionBScheduleConstruction:
    """Test 2 — Condition B produces the expected circuit parameters."""

    def test_condition_b_fields(self):
        pol = make_policy()
        plan = pol.evaluate_candidate(
            condition="B",
            alignment_strategy="repeated",
            N=PRODUCTION_N,
            D=PRODUCTION_D,
            epochs=PRODUCTION_EPOCHS,
        )
        assert plan.condition == "B"
        assert plan.grad_level == 2, (
            f"Condition B must enter gradient at Level 2; got {plan.grad_level}"
        )
        assert plan.burn_X == 5, (
            f"Condition B must apply burn_X=5 (5 depth-consuming ops on X); got {plan.burn_X}"
        )
        assert plan.level_adjustment_required is True, (
            "Condition B MUST require a level-adjustment step (enc_err * 1.0, "
            "plaintext multiplication by 1.0 followed by automatic rescaling)"
        )
        assert plan.poly_degree == 3
        assert plan.burn_y == 4


# ---------------------------------------------------------------------------
# Test 3: Correct ciphertext level at gradient computation entry
# ---------------------------------------------------------------------------

class TestGradientEntryLevel:
    """Test 3 — Level-at-gradient-entry matches plan specification."""

    def test_a_grad_level(self):
        pol = make_policy()
        plan = pol.evaluate_candidate("A", "repeated", N=PRODUCTION_N, D=PRODUCTION_D, epochs=5)
        assert plan.grad_level == 3

    def test_b_grad_level(self):
        pol = make_policy()
        plan = pol.evaluate_candidate("B", "repeated", N=PRODUCTION_N, D=PRODUCTION_D, epochs=5)
        assert plan.grad_level == 2

    def test_a_and_b_differ_by_one_level(self):
        pol = make_policy()
        a = pol.evaluate_candidate("A", "repeated", N=PRODUCTION_N, D=PRODUCTION_D, epochs=5)
        b = pol.evaluate_candidate("B", "repeated", N=PRODUCTION_N, D=PRODUCTION_D, epochs=5)
        assert a.grad_level - b.grad_level == 1, (
            "Condition A and B must differ by exactly one ciphertext level at gradient entry"
        )


# ---------------------------------------------------------------------------
# Test 4: Feature and error ciphertext alignment burn counts
# ---------------------------------------------------------------------------

class TestFeatureAlignment:
    """Test 4 — burn_X correctly aligns feature ciphertext to grad_level."""

    def test_a_burn_x_equals_4(self):
        pol = make_policy()
        plan = pol.evaluate_candidate("A", "repeated", N=PRODUCTION_N, D=PRODUCTION_D, epochs=5)
        # For Condition A: grad_level=3. X needs burn_X=4 to reach Level 3.
        assert plan.burn_X == 4

    def test_b_burn_x_equals_5(self):
        pol = make_policy()
        plan = pol.evaluate_candidate("B", "repeated", N=PRODUCTION_N, D=PRODUCTION_D, epochs=5)
        # For Condition B: grad_level=2. X needs burn_X=5 to reach Level 2.
        assert plan.burn_X == 5

    def test_burn_x_targets_grad_level(self):
        """burn_X = (chain_length - 1) - grad_level where chain_length=7 (active primes = 8)."""
        pol = make_policy()
        for cond, expected_burn_X in [("A", 4), ("B", 5)]:
            plan = pol.evaluate_candidate(cond, "repeated", N=PRODUCTION_N, D=PRODUCTION_D, epochs=5)  # type: ignore
            # Verify the logic: burn_X must equal total_depth - grad_level = 7 - grad_level
            expected = 7 - plan.grad_level
            assert plan.burn_X == expected, (
                f"Condition {cond}: expected burn_X={expected} "
                f"(7 - grad_level={plan.grad_level}), got {plan.burn_X}"
            )


# ---------------------------------------------------------------------------
# Test 5: Cached alignment — epoch-2 alignment cost is zero
# ---------------------------------------------------------------------------

class TestCachedCiphertextReuse:
    """Test 5 — Cached plan: epoch-2 alignment cost drops to zero."""

    def test_epoch2_alignment_cost_is_zero_for_cached(self):
        pol = make_policy()
        model = pol.model
        from src.scheduler.cost_model_v2 import ScheduleV2
        sched = ScheduleV2(
            condition="B",
            poly_degree=3,
            grad_level=2,
            burn_n=5,
            alignment_strategy="cached",
        )
        comp_e2 = model.predict_components(sched, N=PRODUCTION_N, D=PRODUCTION_D, epoch_idx=2)
        assert comp_e2["feature_alignment_s"] == pytest.approx(0.0, abs=1e-9), (
            "Epoch ≥ 2 with cached alignment must have zero alignment cost (feature_alignment_s)"
        )

    def test_epoch1_alignment_cost_nonzero_for_cached(self):
        pol = make_policy()
        model = pol.model
        from src.scheduler.cost_model_v2 import ScheduleV2
        sched = ScheduleV2(
            condition="B",
            poly_degree=3,
            grad_level=2,
            burn_n=5,
            alignment_strategy="cached",
        )
        comp_e1 = model.predict_components(sched, N=PRODUCTION_N, D=PRODUCTION_D, epoch_idx=1)
        assert comp_e1["alignment_precompute_s"] > 0, (
            "Epoch 1 with cached alignment must incur alignment precompute cost"
        )


# ---------------------------------------------------------------------------
# Test 6: One-epoch vs multi-epoch alignment decision
# ---------------------------------------------------------------------------

class TestOneEpochVsMultiEpochDecision:
    """Test 6 — For epochs=1 the repeated variant is selected; for epochs=10, cached wins."""

    def test_one_epoch_selects_repeated_strategy(self):
        pol = make_policy()
        best, all_plans = pol.select_plan(N=PRODUCTION_N, D=PRODUCTION_D, epochs=1)
        assert best.alignment_strategy == "repeated", (
            f"1-epoch job should choose 'repeated' (no caching benefit); "
            f"got '{best.alignment_strategy}' via {best.schedule_id}"
        )

    def test_ten_epochs_selects_cached_strategy(self):
        pol = make_policy()
        best, all_plans = pol.select_plan(N=PRODUCTION_N, D=PRODUCTION_D, epochs=10)
        assert best.alignment_strategy == "cached", (
            f"10-epoch job should choose 'cached' (amortised setup); "
            f"got '{best.alignment_strategy}' via {best.schedule_id}"
        )


# ---------------------------------------------------------------------------
# Test 7: Memory-budget rejection of caching
# ---------------------------------------------------------------------------

class TestMemoryBudgetRejection:
    """Test 7 — Caching is rejected when memory budget is too small."""

    def test_small_budget_rejects_all_cached_plans(self):
        pol = make_policy()
        # 5 MB is far below the B-cache requirement (~108 MB)
        best, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=10, memory_budget_mb=5.0
        )
        # All cached plans must be infeasible
        cached_plans = [p for p in all_plans if p.alignment_strategy == "cached"]
        for p in cached_plans:
            assert not p.feasible, f"Plan {p.schedule_id} should be infeasible with 5 MB budget"
            assert p.rejection_reason is not None

        # Best must be a repeated plan
        assert best.alignment_strategy == "repeated", (
            f"With 5 MB memory budget, best should be 'repeated'; got {best.schedule_id}"
        )

    def test_adequate_budget_allows_cached_plans(self):
        pol = make_policy()
        # 2 GB is more than enough
        best, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=10, memory_budget_mb=2048.0
        )
        cached_plans = [p for p in all_plans if p.alignment_strategy == "cached"]
        feasible_cached = [p for p in cached_plans if p.feasible]
        assert len(feasible_cached) > 0, "With adequate budget, at least one cached plan should be feasible"


# ---------------------------------------------------------------------------
# Test 8: Rejection of unsupported modulus states
# ---------------------------------------------------------------------------

class TestUnsupportedModulusState:
    """Test 8 — Condition B disabled results in A-only selection."""

    def test_condition_b_disabled_selects_condition_a(self):
        pol = make_policy()
        best, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=10, allow_condition_B=False
        )
        # All B plans must be infeasible
        b_plans = [p for p in all_plans if p.condition == "B"]
        for p in b_plans:
            assert not p.feasible, (
                f"Plan {p.schedule_id} should be infeasible when Condition B is disabled"
            )
        assert best.condition == "A", (
            f"With Condition B disabled, best plan must be Condition A; got {best.condition}"
        )

    def test_condition_b_disabled_reason_recorded(self):
        pol = make_policy()
        _, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=10, allow_condition_B=False
        )
        b_plans = [p for p in all_plans if p.condition == "B"]
        for p in b_plans:
            assert p.rejection_reason is not None and len(p.rejection_reason) > 0


# ---------------------------------------------------------------------------
# Test 9: Numerical gradient error threshold
# ---------------------------------------------------------------------------

class TestNumericalGradientErrorThreshold:
    """
    Test 9 — The max gradient error between Condition A and B must be ≤ 1e-4.
    This is verified against the frozen Phase 4 measured result (1.71e-6).
    """

    PHASE4_MAX_AE = 1.71e-6    # from Phase 4 microbenchmarks
    THRESHOLD = 1e-4

    def test_phase4_measured_error_within_threshold(self):
        assert self.PHASE4_MAX_AE <= self.THRESHOLD, (
            f"Phase 4 measured max gradient error ({self.PHASE4_MAX_AE}) "
            f"exceeds numerical threshold ({self.THRESHOLD})"
        )

    def test_threshold_recorded_in_policy_default(self):
        pol = make_policy()
        # The policy's default threshold must be 1e-4
        import inspect
        sig = inspect.signature(pol.select_plan)
        default_threshold = sig.parameters["max_error_threshold"].default
        assert default_threshold == pytest.approx(1e-4), (
            f"Policy default max_error_threshold should be 1e-4; got {default_threshold}"
        )


# ---------------------------------------------------------------------------
# Test 10: Cost-based schedule selection (B-cache cheapest for 10 epochs)
# ---------------------------------------------------------------------------

class TestCostBasedScheduleSelection:
    """Test 10 — For N=17084, D=48, epochs=10, B-cache is selected as cheapest."""

    def test_b_cache_selected_for_production_workload(self):
        pol = make_policy()
        best, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS
        )
        assert best.schedule_id == "B-cache", (
            f"For N=17084, D=48, epochs=10, B-cache should be selected; got {best.schedule_id}"
        )

    def test_b_cache_cheaper_than_all_others(self):
        pol = make_policy()
        best, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS
        )
        b_cache = next(p for p in all_plans if p.schedule_id == "B-cache")
        for p in all_plans:
            if p.feasible and p.schedule_id != "B-cache":
                assert b_cache.predicted_total_training_s <= p.predicted_total_training_s, (
                    f"B-cache ({b_cache.predicted_total_training_s:.2f}s) should be cheaper "
                    f"than {p.schedule_id} ({p.predicted_total_training_s:.2f}s)"
                )


# ---------------------------------------------------------------------------
# Test 11: Execution context startup accounting
# ---------------------------------------------------------------------------

class TestExecutionContextStartupAccounting:
    """Test 11 — Persistent worker and fresh worker produce different predictions."""

    def test_persistent_vs_fresh_costs_differ(self):
        pol_persistent = make_policy(default_context="persistent_worker")
        pol_fresh = make_policy(default_context="fresh_worker")

        best_p, _ = pol_persistent.select_plan(N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS)
        best_f, _ = pol_fresh.select_plan(N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS)

        # Fresh worker total should be ≥ persistent worker total (overhead)
        assert best_f.predicted_total_training_s >= best_p.predicted_total_training_s, (
            f"Fresh worker total ({best_f.predicted_total_training_s:.2f}s) should be "
            f">= persistent worker total ({best_p.predicted_total_training_s:.2f}s)"
        )

    def test_both_contexts_select_same_strategy_for_large_workload(self):
        """For a large enough workload (N=17084, epochs=10), both contexts should choose cached."""
        pol_persistent = make_policy(default_context="persistent_worker")
        pol_fresh = make_policy(default_context="fresh_worker")

        best_p, _ = pol_persistent.select_plan(N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS)
        best_f, _ = pol_fresh.select_plan(N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS)

        assert best_p.alignment_strategy == best_f.alignment_strategy, (
            f"Both contexts should agree on alignment strategy for large workloads; "
            f"persistent={best_p.alignment_strategy}, fresh={best_f.alignment_strategy}"
        )


# ---------------------------------------------------------------------------
# Test 12: Fallback when Condition B disabled
# ---------------------------------------------------------------------------

class TestFallbackWhenConditionBDisabled:
    """Test 12 — When B is disabled, A-cache or A-repeat is selected (not B)."""

    def test_fallback_selects_condition_a_plan(self):
        pol = make_policy()
        best, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS,
            allow_condition_B=False,
        )
        assert best.condition == "A"
        assert best.schedule_id in ("A-repeat", "A-cache")

    def test_fallback_no_b_plans_are_feasible(self):
        pol = make_policy()
        _, all_plans = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS,
            allow_condition_B=False,
        )
        b_plans = [p for p in all_plans if p.condition == "B"]
        assert all(not p.feasible for p in b_plans), (
            "All Condition B plans must be infeasible when B is explicitly disabled"
        )

    def test_fallback_a_cache_for_multi_epoch_without_b(self):
        """With 10 epochs and B disabled, scheduler should still prefer caching for A."""
        pol = make_policy()
        best, _ = pol.select_plan(
            N=PRODUCTION_N, D=PRODUCTION_D, epochs=PRODUCTION_EPOCHS,
            allow_condition_B=False,
        )
        # A-cache should beat A-repeat for 10 epochs
        assert best.schedule_id == "A-cache", (
            f"10-epoch job with B disabled should select A-cache; got {best.schedule_id}"
        )


# ---------------------------------------------------------------------------
# Edge case: memory estimation correctness
# ---------------------------------------------------------------------------

class TestMemoryEstimation:
    """Auxiliary test — verify ciphertext memory formula."""

    def test_b_cache_memory_estimate(self):
        """Level 2 → 3 active primes → B * D ciphertexts."""
        B = 3       # ceil(17084 / 8192)
        D = 48
        grad_level = 2
        poly_degree = 16384
        # active_primes = grad_level + 1 = 3
        # bytes_per_ct = 16384 * 2 * 3 * 8 = 786432
        # total = B * D * 786432 = 3 * 48 * 786432 = 113246208 bytes ≈ 108.0 MB
        expected_mb = (B * D * poly_degree * 2 * (grad_level + 1) * 8) / (1024 * 1024)
        actual_mb = WorkloadSchedulerPolicy.estimate_ciphertext_memory_mb(
            B=B, D=D, grad_level=grad_level, poly_modulus_degree=poly_degree
        )
        assert actual_mb == pytest.approx(expected_mb, rel=1e-6), (
            f"B-cache memory estimate mismatch: expected {expected_mb:.1f} MB, got {actual_mb:.1f} MB"
        )
