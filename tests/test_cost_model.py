#!/usr/bin/env python3
"""
Unit and Integration Tests for CKKS Analytical Cost Model
=========================================================
Tests:
  1. Batch partition calculations and edge cases
  2. Schedule parameter enumeration and consistency
  3. Rejection of invalid and unsupported modulus states (e.g. Level 1 multiplication)
  4. Monotonicity and schedule ordering (Condition A > Condition B runtime)
  5. Gradient speedup ratio fidelity (target ~1.86x)
  6. Cost model prediction accuracy against Phase 4 (target MAPE <= 15.0%)
  7. Feature-alignment caching behavior across multi-epoch training
  8. Automated optimal schedule recommendation
"""

import pytest
import math
from src.scheduler.cost_model import CKKSCostModel, Schedule


@pytest.fixture(scope="module")
def model():
    return CKKSCostModel()


def test_batch_count_edge_cases(model):
    """Test batch partitioning ceil(N / S) across boundary values."""
    assert model.get_num_batches(8192, 8192) == 1
    assert model.get_num_batches(8193, 8192) == 2
    assert model.get_num_batches(16384, 8192) == 2
    assert model.get_num_batches(16385, 8192) == 3
    assert model.get_num_batches(17084, 8192) == 3
    assert model.get_num_batches(1, 8192) == 1

    with pytest.raises(ValueError):
        model.get_num_batches(0, 8192)

    with pytest.raises(ValueError):
        model.get_num_batches(100, 0)


def test_schedule_enumeration_and_attributes(model):
    """Ensure standard schedules exist and have correct attributes."""
    schedules = model.get_standard_schedules()
    assert len(schedules) == 6

    sched_map = {(s.condition, s.alignment_strategy): s for s in schedules}
    assert ("A", "repeated") in sched_map
    assert ("B", "repeated") in sched_map
    assert ("C", "repeated") in sched_map
    assert ("A", "cached") in sched_map
    assert ("B", "cached") in sched_map
    assert ("C", "cached") in sched_map

    s_a = sched_map[("A", "repeated")]
    assert s_a.poly_degree == 3
    assert s_a.grad_level == 3
    assert s_a.burn_n == 4

    s_b = sched_map[("B", "repeated")]
    assert s_b.poly_degree == 3
    assert s_b.grad_level == 2
    assert s_b.burn_n == 5

    s_c = sched_map[("C", "repeated")]
    assert s_c.poly_degree == 5
    assert s_c.grad_level == 2
    assert s_c.burn_n == 5


def test_unsupported_level_rejection(model):
    """Ensure invalid / cryptographically unsupported states are rejected."""
    # Level 1 multiplication is unsupported in RNS-CKKS
    with pytest.raises(ValueError, match="Unsupported gradient level"):
        s_invalid_lvl = Schedule(condition="B", poly_degree=3, grad_level=1, burn_n=5)
        model.validate_schedule(s_invalid_lvl)

    # Unsupported polynomial degree
    with pytest.raises(ValueError, match="Unsupported polynomial degree"):
        s_invalid_deg = Schedule(condition="A", poly_degree=4, grad_level=3, burn_n=4)
        model.validate_schedule(s_invalid_deg)

    # Condition mismatch
    with pytest.raises(ValueError, match="Condition A requires"):
        s_mismatch = Schedule(condition="A", poly_degree=3, grad_level=2, burn_n=4)
        model.validate_schedule(s_mismatch)


def test_condition_ordering(model):
    """Ensure predicted runtime reflects the causal hierarchy observed in Phase 4."""
    s_a = Schedule("A", poly_degree=3, grad_level=3, burn_n=4, alignment_strategy="repeated")
    s_b = Schedule("B", poly_degree=3, grad_level=2, burn_n=5, alignment_strategy="repeated")
    s_c = Schedule("C", poly_degree=5, grad_level=2, burn_n=5, alignment_strategy="repeated")

    comp_a = model.predict_components(s_a)
    comp_b = model.predict_components(s_b)
    comp_c = model.predict_components(s_c)

    # Condition A total runtime must exceed Condition B
    assert comp_a["total_epoch_time_s"] > comp_b["total_epoch_time_s"]
    # Condition A gradient runtime must exceed Condition B gradient runtime
    assert comp_a["gradient_comp_s"] > comp_b["gradient_comp_s"]
    # Condition B and C gradient runtime should be identical (both evaluated at Level 2)
    assert abs(comp_b["gradient_comp_s"] - comp_c["gradient_comp_s"]) < 1e-6
    # Condition C total epoch runtime is slightly larger than B due to degree-5 polynomial eval
    assert comp_c["total_epoch_time_s"] > comp_b["total_epoch_time_s"]


def test_gradient_speedup_ratio(model):
    """Ensure predicted gradient speedup matches the empirical benchmark (~1.86x)."""
    s_a = Schedule("A", poly_degree=3, grad_level=3, burn_n=4)
    s_b = Schedule("B", poly_degree=3, grad_level=2, burn_n=5)

    comp_a = model.predict_components(s_a)
    comp_b = model.predict_components(s_b)

    ratio = comp_a["gradient_comp_s"] / comp_b["gradient_comp_s"]
    assert 1.80 <= ratio <= 1.90, f"Predicted ratio {ratio:.4f} outside [1.80, 1.90]"


def test_mape_within_15_percent(model):
    """Ensure all conditions predict total epoch runtime within 15% MAPE vs Phase 4."""
    conditions = [
        ("A", Schedule("A", 3, 3, 4)),
        ("B", Schedule("B", 3, 2, 5)),
        ("C", Schedule("C", 5, 2, 5)),
    ]

    for cond_code, sched in conditions:
        actual_total = model.phase4_summary[cond_code]["total_epoch_mean_s"]
        pred_total = model.predict_epoch(sched)
        mape = abs(pred_total - actual_total) / actual_total * 100.0

        assert mape <= 15.0, f"Condition {cond_code} MAPE {mape:.2f}% exceeds 15.0% threshold"
        # In fact, our model is calibrated within 1.5%
        assert mape <= 2.0, f"Condition {cond_code} MAPE {mape:.2f}% unexpectedly high"


def test_cached_alignment_saves_time(model):
    """Verify that cached alignment eliminates feature alignment in subsequent epochs."""
    s_b_rep = Schedule("B", 3, 2, 5, alignment_strategy="repeated")
    s_b_cac = Schedule("B", 3, 2, 5, alignment_strategy="cached")

    # Epoch 1: both have alignment cost
    comp_ep1_rep = model.predict_components(s_b_rep, epoch_idx=1)
    comp_ep1_cac = model.predict_components(s_b_cac, epoch_idx=1)
    assert abs(comp_ep1_rep["feature_alignment_s"] - comp_ep1_cac["feature_alignment_s"]) < 1e-6

    # Epoch 2: cached has 0.0 alignment cost
    comp_ep2_cac = model.predict_components(s_b_cac, epoch_idx=2)
    assert comp_ep2_cac["feature_alignment_s"] == 0.0

    # 10-Epoch training total time
    res_rep = model.predict_training(s_b_rep, epochs=10)
    res_cac = model.predict_training(s_b_cac, epochs=10)
    assert res_cac["total_training_time_s"] < res_rep["total_training_time_s"]

    savings_s = res_rep["total_training_time_s"] - res_cac["total_training_time_s"]
    expected_savings = 9 * comp_ep1_rep["feature_alignment_s"]
    assert abs(savings_s - expected_savings) < 1e-3


def test_recommend_best_schedule(model):
    """Verify that the model automatically selects Condition B with caching for 10 epochs."""
    best = model.recommend_best_schedule(epochs=10)
    assert best.condition == "B"
    assert best.alignment_strategy == "cached"
    assert best.poly_degree == 3
    assert best.grad_level == 2
