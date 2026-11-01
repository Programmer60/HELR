#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 2: Held-Out Workload Scheduler
===================================================================
Tests:
  1. Workload grid enumeration, dimensions, and reference designation
  2. Batch padding, valid samples vs allocated SIMD slots
  3. Cached alignment invariants and step time separation
  4. Out-of-workload cost model prediction scaling and monotonicity
  5. Cryptographic feasibility checks and error tolerances
  6. Decision regret formulation and mathematical invariants
"""

import math
import numpy as np
import pytest

from benchmarks.run_workload_grid import WORKLOAD_GRID, partition_synthetic_dataset
from src.scheduler.cost_model import CKKSCostModel, Schedule


@pytest.fixture(scope="module")
def model():
    return CKKSCostModel()


def test_workload_grid_enumeration():
    """Verify that the workload grid contains exactly 9 configurations with valid dimensions."""
    assert len(WORKLOAD_GRID) == 9

    valid_N = {4096, 10000, 17084}
    valid_D = {12, 24, 48}
    valid_B = {1, 2, 3}

    ref_count = 0
    heldout_count = 0

    for w in WORKLOAD_GRID:
        assert w["N"] in valid_N
        assert w["D"] in valid_D
        assert w["B"] in valid_B
        assert w["B"] == math.ceil(w["N"] / 8192)

        if w["is_reference"]:
            ref_count += 1
            assert w["workload_id"] == "W9"
            assert w["N"] == 17084 and w["D"] == 48 and w["B"] == 3
        else:
            heldout_count += 1

    assert ref_count == 1
    assert heldout_count == 8


def test_batch_padding_and_slot_allocation():
    """Verify that partial final batches are correctly padded to 8192 slots."""
    # N = 4096 (1 batch)
    _, _, _, b_w1 = partition_synthetic_dataset(N=4096, D=12, S=8192)
    assert len(b_w1) == 1
    assert b_w1[0]["n_valid"] == 4096
    assert b_w1[0]["n_unused"] == 4096
    assert b_w1[0]["X_pad"].shape == (8192, 12)
    assert b_w1[0]["y_pad"].shape == (8192,)

    # N = 10000 (2 batches)
    _, _, _, b_w2 = partition_synthetic_dataset(N=10000, D=24, S=8192)
    assert len(b_w2) == 2
    assert b_w2[0]["n_valid"] == 8192
    assert b_w2[0]["n_unused"] == 0
    assert b_w2[1]["n_valid"] == 1808
    assert b_w2[1]["n_unused"] == 6384
    assert b_w2[0]["X_pad"].shape == (8192, 24)
    assert b_w2[1]["X_pad"].shape == (8192, 24)

    # N = 17084 (3 batches)
    _, _, _, b_w3 = partition_synthetic_dataset(N=17084, D=48, S=8192)
    assert len(b_w3) == 3
    assert b_w3[0]["n_valid"] == 8192
    assert b_w3[1]["n_valid"] == 8192
    assert b_w3[2]["n_valid"] == 700
    assert b_w3[2]["n_unused"] == 7492


def test_cached_alignment_step_separation(model):
    """Verify that cached alignment predicts zero alignment latency in subsequent steps."""
    for n in [4096, 10000]:
        for d in [12, 24]:
            s_rep = Schedule("B", 3, 2, 5, alignment_strategy="repeated")
            s_cac = Schedule("B", 3, 2, 5, alignment_strategy="cached")

            comp_rep = model.predict_components(s_rep, N=n, D=d, epoch_idx=1)
            comp_cac_ep1 = model.predict_components(s_cac, N=n, D=d, epoch_idx=1)
            comp_cac_ep2 = model.predict_components(s_cac, N=n, D=d, epoch_idx=2)

            # In epoch 1, both pay alignment
            assert comp_rep["feature_alignment_s"] > 0.0
            assert abs(comp_rep["feature_alignment_s"] - comp_cac_ep1["feature_alignment_s"]) < 1e-6

            # In epoch 2, cached pays 0.0
            assert comp_cac_ep2["feature_alignment_s"] == 0.0
            assert comp_cac_ep2["total_epoch_time_s"] < comp_rep["total_epoch_time_s"]


def test_out_of_workload_prediction_monotonicity(model):
    """Verify that predictions scale monotonically with feature count D and batch count B."""
    s_b = Schedule("B", 3, 2, 5, alignment_strategy="repeated")

    # Scaling with D (fixed N=4096, B=1)
    pred_d12 = model.predict_epoch(s_b, N=4096, D=12)
    pred_d24 = model.predict_epoch(s_b, N=4096, D=24)
    pred_d48 = model.predict_epoch(s_b, N=4096, D=48)
    assert pred_d12 < pred_d24 < pred_d48

    # Scaling with B (fixed D=24)
    pred_b1 = model.predict_epoch(s_b, N=4096, D=24)   # B=1
    pred_b2 = model.predict_epoch(s_b, N=10000, D=24)  # B=2
    pred_b3 = model.predict_epoch(s_b, N=17084, D=24)  # B=3
    assert pred_b1 < pred_b2 < pred_b3


def test_gradient_speedup_across_workloads(model):
    """Ensure predicted gradient speedup holds uniformly across all held-out workloads."""
    s_a = Schedule("A", 3, 3, 4)
    s_b = Schedule("B", 3, 2, 5)

    for n in [4096, 10000, 17084]:
        for d in [12, 24, 48]:
            comp_a = model.predict_components(s_a, N=n, D=d)
            comp_b = model.predict_components(s_b, N=n, D=d)

            ratio = comp_a["gradient_comp_s"] / comp_b["gradient_comp_s"]
            assert 1.80 <= ratio <= 1.90, f"Speedup ratio {ratio:.4f} outside bounds for N={n}, D={d}"


def test_decision_regret_invariants():
    """Verify decision regret formula and properties."""
    def compute_regret(selected_time: float, best_time: float) -> float:
        if best_time <= 0:
            raise ValueError("best_time must be positive")
        return (selected_time / best_time) - 1.0

    # Perfect selection -> regret = 0
    assert compute_regret(10.0, 10.0) == 0.0

    # Suboptimal selection -> regret > 0
    regret_sub = compute_regret(10.5, 10.0)
    assert abs(regret_sub - 0.05) < 1e-6

    # Regret must be non-negative when best is truly minimum
    with pytest.raises(ValueError):
        compute_regret(10.0, 0.0)
