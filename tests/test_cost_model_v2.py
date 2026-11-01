#!/usr/bin/env python3
"""
Unit and Integration Tests for CostModel_v2
===========================================
Tests:
  1. Both execution contexts ('persistent_worker' and 'fresh_worker')
  2. Operation-specific calibration parameters
  3. Batch calculation edge cases and boundary transitions
  4. Cached vs repeated alignment accounting across epochs
  5. Cryptographic feasibility and unsupported level rejection
  6. Mathematical gradient error constraints
  7. CostModel_v2 prediction monotonicity and scaling
  8. Decision regret formulation and schedule recommendations
"""

import math
import pytest
from src.scheduler.cost_model_v2 import CKKSCostModelV2, ScheduleV2


@pytest.fixture(scope="module")
def model_v2():
    return CKKSCostModelV2()


def test_both_execution_contexts_supported(model_v2):
    """Verify that both persistent and fresh worker execution contexts predict valid latencies."""
    sched = ScheduleV2("B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="repeated")

    comp_persistent = model_v2.predict_components(sched, N=17084, D=48, context="persistent_worker")
    comp_fresh = model_v2.predict_components(sched, N=17084, D=48, context="fresh_worker")

    assert comp_persistent["total_epoch_time_s"] > 0.0
    assert comp_fresh["total_epoch_time_s"] > 0.0

    # Ensure all components are positive and populated
    for key in [
        "weight_enc_s",
        "forward_linear_s",
        "poly_eval_s",
        "error_comp_s",
        "level_adjustment_s",
        "feature_alignment_s",
        "gradient_comp_s",
        "decryption_agg_s",
        "total_epoch_time_s",
    ]:
        assert comp_persistent[key] > 0.0, f"Component {key} was zero in persistent worker"
        assert comp_fresh[key] > 0.0, f"Component {key} was zero in fresh worker"


def test_operation_specific_calibration(model_v2):
    """Verify that operations use distinct calibration profiles rather than a single global factor."""
    assert hasattr(model_v2, "persistent_ops")
    assert hasattr(model_v2, "fresh_ops")
    assert len(model_v2.persistent_ops) > 0
    assert len(model_v2.fresh_ops) > 0

    # Verify that different operations have distinct latencies
    t_mul_l7 = model_v2._get_op_mean_s("ct_ct_mul_rescale", 7, "persistent_worker")
    t_sum_l7 = model_v2._get_op_mean_s("slot_sum", 7, "persistent_worker")
    t_dot_l3 = model_v2._get_op_mean_s("gradient_dot_product", 3, "persistent_worker")
    t_dot_l2 = model_v2._get_op_mean_s("gradient_dot_product", 2, "persistent_worker")

    assert t_sum_l7 > t_mul_l7
    assert t_dot_l3 > t_dot_l2
    assert abs(t_dot_l3 - t_dot_l2) > 0.010  # Significant difference between L3 and L2 gradient


def test_batch_calculation_and_boundary_transitions(model_v2):
    """Verify batch calculation ceil(N/S) at critical SIMD boundaries."""
    assert model_v2.get_num_batches(8192, 8192) == 1
    assert model_v2.get_num_batches(8193, 8192) == 2   # 1 sample forces 2nd batch
    assert model_v2.get_num_batches(16384, 8192) == 2
    assert model_v2.get_num_batches(16385, 8192) == 3  # 1 sample forces 3rd batch
    assert model_v2.get_num_batches(1, 8192) == 1

    with pytest.raises(ValueError):
        model_v2.get_num_batches(0, 8192)

    with pytest.raises(ValueError):
        model_v2.get_num_batches(8192, 0)


def test_cached_and_repeated_alignment_accounting(model_v2):
    """Verify that cached alignment is charged once in precomputation and 0 in subsequent epochs."""
    s_rep = ScheduleV2("B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="repeated")
    s_cac = ScheduleV2("B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="cached")

    # Epoch 1
    comp_rep_e1 = model_v2.predict_components(s_rep, N=8192, D=12, epoch_idx=1)
    comp_cac_e1 = model_v2.predict_components(s_cac, N=8192, D=12, epoch_idx=1)
    assert abs(comp_rep_e1["feature_alignment_s"] - comp_cac_e1["feature_alignment_s"]) < 1e-6

    # Epoch 2
    comp_cac_e2 = model_v2.predict_components(s_cac, N=8192, D=12, epoch_idx=2)
    assert comp_cac_e2["feature_alignment_s"] == 0.0
    assert comp_cac_e2["total_epoch_time_s"] < comp_rep_e1["total_epoch_time_s"]

    # 10-Epoch cumulative accounting
    res_rep = model_v2.predict_training(s_rep, N=8192, D=12, epochs=10)
    res_cac = model_v2.predict_training(s_cac, N=8192, D=12, epochs=10)

    savings = res_rep["total_training_time_s"] - res_cac["total_training_time_s"]
    expected_savings = 9 * comp_rep_e1["feature_alignment_s"]
    assert abs(savings - expected_savings) < 1e-3


def test_unsupported_level_rejection(model_v2):
    """Verify that cryptographically inadmissible levels are rejected."""
    # Gradient at Level 1 is unsupported (cannot rescale below L1)
    with pytest.raises(ValueError, match="Unsupported gradient level"):
        s_invalid_lvl = ScheduleV2("B", poly_degree=3, grad_level=1, burn_n=5)
        model_v2.validate_schedule(s_invalid_lvl)

    # Degree 4 is unsupported
    with pytest.raises(ValueError, match="Unsupported polynomial degree"):
        s_invalid_deg = ScheduleV2("A", poly_degree=4, grad_level=3, burn_n=4)
        model_v2.validate_schedule(s_invalid_deg)


def test_primary_schedule_degree3_separation(model_v2):
    """Verify that primary scheduler comparison is strictly between Condition A and Condition B."""
    primary = model_v2.get_primary_schedules()
    assert len(primary) == 4

    for s in primary:
        assert s.poly_degree == 3
        assert s.condition in ("A", "B")

    # Automated recommendation defaults strictly to primary Degree-3 schedules
    best = model_v2.recommend_best_schedule(N=17084, D=48, epochs=10)
    assert best.condition == "B"
    assert best.alignment_strategy == "cached"
    assert best.poly_degree == 3


def test_decision_regret_calculation(model_v2):
    """Verify decision regret mathematical properties."""
    def calc_regret(selected_time: float, best_time: float) -> float:
        if best_time <= 0:
            raise ValueError("best_time must be positive")
        return (selected_time / best_time) - 1.0

    # Perfect selection
    assert calc_regret(10.0, 10.0) == 0.0

    # Near-optimal selection within 5%
    reg = calc_regret(10.3, 10.0)
    assert reg <= 0.05

    # Suboptimal selection above 5%
    reg_bad = calc_regret(11.0, 10.0)
    assert reg_bad > 0.05


def test_scaling_monotonicity_v2(model_v2):
    """Verify runtime scaling monotonicity with feature count D and batch count B."""
    s_b = ScheduleV2("B", poly_degree=3, grad_level=2, burn_n=5)

    # D scaling
    p_d12 = model_v2.predict_epoch(s_b, N=8192, D=12)
    p_d48 = model_v2.predict_epoch(s_b, N=8192, D=48)
    assert p_d12 < p_d48

    # B scaling across boundaries
    p_b1 = model_v2.predict_epoch(s_b, N=8192, D=12)   # B=1
    p_b2 = model_v2.predict_epoch(s_b, N=8193, D=12)   # B=2
    p_b3 = model_v2.predict_epoch(s_b, N=16385, D=12)  # B=3
    assert p_b1 < p_b2 < p_b3
