#!/usr/bin/env python3
"""
PTB-XL Workload-Aware Encrypted Training Harness (Phase 3B)
==========================================================
Integrates CostModel_v2 and WorkloadSchedulerPolicy into the PTB-XL
CKKS Logistic Regression training pipeline.

Supports:
  - Policy-guided automated plan selection or manual condition/strategy specification
  - Condition A (Level 3 gradient entry, burn_X=4, no level adjustment)
  - Condition B (Level 2 gradient entry, burn_X=5, enc_err * 1.0 level adjustment)
  - Feature alignment strategies: 'repeated' (per-batch) vs 'cached' (one-time precompute)
  - Detailed stage profiling matching Phase 0-3 instrumentation
  - Numerical gradient validation against matched plaintext reference
  - Strict preservation of research boundaries (Fold 10 untouched, Folds 1-8 train, Fold 9 val)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tenseal as ts
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "ptbxl_processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ptbxl" / "scheduler" / "phase3b"

# Ensure src is on sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.cost_model_v2 import CKKSCostModelV2, ExecutionContext
from src.scheduler.workload_policy import ExecutionPlan, WorkloadSchedulerPolicy

DATA_MED_NPZ = DATA_DIR / "ptbxl_features_48_median_centered.npz"

SLOT_CAP = 8192
LR_DEFAULT = 0.05
L2_DEFAULT = 0.01
GRAD_CLIP_DEFAULT = 1.0
MAX_EPOCHS_DEFAULT = 10
EARLY_STOP_MIN_EPOCHS = 5
EARLY_STOP_PATIENCE = 2
EARLY_STOP_DELTA = 0.001
SAFETY_MAX_LOGIT = 6.0
TOLERANCE_GRAD_MAX_AE = 1e-4

POLY3 = {"c0": 0.5, "c1": 0.15012, "c3": -0.001593}


class StageProfiler:
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


prof = StageProfiler()


def build_ckks_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def get_levels(enc) -> int:
    return enc.ciphertext()[0].coeff_modulus_size()


def get_parms_id(enc) -> List[int]:
    return list(enc.ciphertext()[0].parms_id())


def get_scale(enc) -> float:
    return float(enc.ciphertext()[0].scale)


def burn(v, n: int):
    for _ in range(n):
        v = v * 1.0
    return v


def true_sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def poly_sigmoid(z: np.ndarray) -> np.ndarray:
    return POLY3["c0"] + POLY3["c1"] * z + POLY3["c3"] * (z**3)


def enc_poly_sigmoid(enc_z):
    z2 = enc_z * enc_z
    z3 = z2 * burn(enc_z, 1)
    t3 = z3 * POLY3["c3"]
    t1 = burn(enc_z, 2) * POLY3["c1"]
    return t3 + t1 + POLY3["c0"]


def partition_batches(X: np.ndarray, y: np.ndarray, slot_capacity: int = SLOT_CAP):
    N, D = X.shape
    num_batches = int(np.ceil(N / slot_capacity))
    batches = []
    for b in range(num_batches):
        start = b * slot_capacity
        end = min(start + slot_capacity, N)
        n_valid = end - start
        n_unused = slot_capacity - n_valid

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
    return batches


def csp_batch_step(
    enc_X_cols: List[Any],
    enc_y: Any,
    enc_w_cols: List[Any],
    plan: ExecutionPlan,
    D: int,
    enc_X_cached: Optional[List[Any]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Executes one CSP training batch step under the given ExecutionPlan:
      - Forward linear combination (drops to Level 6)
      - Degree-3 polynomial evaluation (drops to Level 3)
      - Encrypted error computation (Level 3)
      - If plan.level_adjustment_required (Condition B):
          controlled level adjustment via * 1.0 (drops to Level 2)
      - Feature alignment:
          if enc_X_cached provided: reuse precomputed Level L ciphertexts (0s latency)
          else: burn(enc_X_cols[j], plan.burn_X) on the fly
      - Gradient computation: D dot products at grad_level
    """
    meta = {}

    with prof.measure("forward_linear"):
        enc_z = enc_X_cols[0] * enc_w_cols[0]
        for j in range(1, D):
            enc_z = enc_z + (enc_X_cols[j] * enc_w_cols[j])
    meta["forward_level"] = get_levels(enc_z)

    with prof.measure("poly_eval"):
        enc_p = enc_poly_sigmoid(enc_z)
    meta["poly_level"] = get_levels(enc_p)

    with prof.measure("error_computation"):
        enc_y_aligned = burn(enc_y, plan.burn_y)
        enc_err = enc_p - enc_y_aligned
    meta["err_before_adj_level"] = get_levels(enc_err)

    # Controlled level adjustment (Condition B: enc_err * 1.0; Condition A: identity)
    if plan.level_adjustment_required:
        with prof.measure("level_adjustment"):
            enc_err_adj = enc_err * 1.0
    else:
        enc_err_adj = enc_err
    meta["err_after_adj_level"] = get_levels(enc_err_adj)
    meta["err_after_adj_parms_id"] = get_parms_id(enc_err_adj)
    meta["err_after_adj_scale"] = get_scale(enc_err_adj)

    # Feature alignment
    if enc_X_cached is not None:
        enc_X_aligned = enc_X_cached
    else:
        with prof.measure("feature_alignment"):
            enc_X_aligned = [burn(enc_X_cols[j], plan.burn_X) for j in range(D)]

    meta["feature_aligned_level"] = get_levels(enc_X_aligned[0])
    meta["feature_aligned_parms_id"] = get_parms_id(enc_X_aligned[0])
    meta["feature_aligned_scale"] = get_scale(enc_X_aligned[0])

    # Assert invariant levels and SEAL parms_id match
    assert meta["err_after_adj_level"] == plan.grad_level, (
        f"Expected Level {plan.grad_level} error, got {meta['err_after_adj_level']}"
    )
    assert meta["feature_aligned_level"] == plan.grad_level, (
        f"Expected Level {plan.grad_level} features, got {meta['feature_aligned_level']}"
    )
    assert meta["err_after_adj_parms_id"] == meta["feature_aligned_parms_id"], (
        "SEAL parms_id mismatch between error and features at gradient entry!"
    )

    # Gradient computation: D dot products
    enc_sums = []
    with prof.measure("gradient_computation"):
        for j in range(D):
            enc_sums.append((enc_err_adj * enc_X_aligned[j]).sum())
    meta["grad_end_level"] = get_levels(enc_sums[0])
    assert meta["grad_end_level"] == plan.grad_level - 1, (
        f"Expected Level {plan.grad_level - 1} gradient output, got {meta['grad_end_level']}"
    )

    return enc_sums, meta


def compute_plaintext_reference(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w: np.ndarray,
    l2: float = L2_DEFAULT,
    grad_clip: float = GRAD_CLIP_DEFAULT,
    lr: float = LR_DEFAULT,
) -> Dict[str, Any]:
    N, D = X_train.shape
    z = X_train @ w
    p = poly_sigmoid(z)
    err = p - y_train
    g_unreg = (X_train.T @ err) / N
    g_reg = g_unreg + l2 * w
    g_norm_raw = float(np.linalg.norm(g_reg))
    if g_norm_raw > grad_clip:
        g_clipped = g_reg * (grad_clip / g_norm_raw)
    else:
        g_clipped = g_reg.copy()
    g_norm_clipped = float(np.linalg.norm(g_clipped))
    w_update = lr * g_clipped
    w_next = w - w_update
    return {
        "g_reg": g_reg,
        "g_clipped": g_clipped,
        "g_norm_raw": g_norm_raw,
        "g_norm_clipped": g_norm_clipped,
        "w_update_norm": float(np.linalg.norm(w_update)),
        "w_next": w_next,
    }


def evaluate_model_val(X_val: np.ndarray, y_val: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    z_val = X_val @ w
    p_val_true = true_sigmoid(z_val)
    p_val_poly = poly_sigmoid(z_val)

    # 1. Exact sigmoid metrics (Clinical reference)
    y_pred_true = (p_val_true >= 0.5).astype(int)
    acc_true = float(accuracy_score(y_val, y_pred_true))
    roc_auc_true = float(roc_auc_score(y_val, p_val_true))
    prec_true = float(precision_score(y_val, y_pred_true, zero_division=0))
    rec_true = float(recall_score(y_val, y_pred_true, zero_division=0))
    f1_true = float(f1_score(y_val, y_pred_true, zero_division=0))
    cm_true = confusion_matrix(y_val, y_pred_true, labels=[0, 1]).tolist()

    # 2. Polynomial sigmoid metrics (Circuit activation behavior)
    y_pred_poly = (p_val_poly >= 0.5).astype(int)
    acc_poly = float(accuracy_score(y_val, y_pred_poly))
    roc_auc_poly = float(roc_auc_score(y_val, p_val_poly))
    prec_poly = float(precision_score(y_val, y_pred_poly, zero_division=0))
    rec_poly = float(recall_score(y_val, y_pred_poly, zero_division=0))
    f1_poly = float(f1_score(y_val, y_pred_poly, zero_division=0))
    cm_poly = confusion_matrix(y_val, y_pred_poly, labels=[0, 1]).tolist()

    poly_outside = int(np.sum((p_val_poly < 0.0) | (p_val_poly > 1.0)))

    quantiles = {
        "min": float(np.min(z_val)),
        "q05": float(np.quantile(z_val, 0.05)),
        "q25": float(np.quantile(z_val, 0.25)),
        "q50_median": float(np.quantile(z_val, 0.50)),
        "q75": float(np.quantile(z_val, 0.75)),
        "q95": float(np.quantile(z_val, 0.95)),
        "max": float(np.max(z_val)),
    }

    max_abs_logit = float(np.max(np.abs(z_val)))

    return {
        "exact_sigmoid": {
            "accuracy": acc_true,
            "roc_auc": roc_auc_true,
            "precision": prec_true,
            "recall": rec_true,
            "f1": f1_true,
            "confusion_matrix": cm_true,
        },
        "poly_sigmoid": {
            "accuracy": acc_poly,
            "roc_auc": roc_auc_poly,
            "precision": prec_poly,
            "recall": rec_poly,
            "f1": f1_poly,
            "confusion_matrix": cm_poly,
            "poly_outside_0_1": poly_outside,
        },
        "logit_quantiles": quantiles,
        "max_abs_logit": max_abs_logit,
        "all_finite": bool(
            np.all(np.isfinite(z_val))
            and np.all(np.isfinite(p_val_true))
            and np.all(np.isfinite(p_val_poly))
        ),
    }


def run_training_experiment(
    plan: ExecutionPlan,
    seed: int = 42,
    epochs: int = 10,
    lr: float = LR_DEFAULT,
    l2: float = L2_DEFAULT,
    grad_clip: float = GRAD_CLIP_DEFAULT,
    disable_early_stop: bool = False,
    log_fn=print,
) -> Dict[str, Any]:
    """
    Executes a complete training run under the specified ExecutionPlan.
    """
    log_fn("=" * 80)
    log_fn(f" TRAINING RUN: Plan={plan.schedule_id} | Seed={seed} | Epochs={epochs}")
    log_fn(f" GradLevel={plan.grad_level} | BurnX={plan.burn_X} | LevelAdj={plan.level_adjustment_required} | Strategy={plan.alignment_strategy}")
    log_fn("=" * 80)

    # 1. Load Data
    npz_med = np.load(DATA_MED_NPZ, allow_pickle=True)
    features_raw = npz_med["features"].astype(np.float64)
    strat_fold = npz_med["strat_fold"]
    labels_all = npz_med["label"].astype(np.float64)

    train_mask = np.isin(strat_fold, np.arange(1, 9))
    val_mask = strat_fold == 9
    test_mask = strat_fold == 10

    # Ensure Fold 10 is untouched
    assert not np.any(train_mask & test_mask), "CRITICAL: Fold 10 contamination in training set!"
    assert not np.any(val_mask & test_mask), "CRITICAL: Fold 10 contamination in validation set!"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_raw[train_mask])
    y_train = labels_all[train_mask]
    X_val = scaler.transform(features_raw[val_mask])
    y_val = labels_all[val_mask]

    N_train, D = X_train.shape
    N_val = len(y_val)

    batches = partition_batches(X_train, y_train, slot_capacity=SLOT_CAP)

    # 2. CKKS Context
    ctx = build_ckks_context()

    # 3. Hospital encrypts features and labels
    t0_enc = time.perf_counter()
    enc_batches = []
    for b in batches:
        enc_X_cols = [ts.ckks_vector(ctx, b["X_pad"][:, j].tolist()) for j in range(D)]
        enc_y = ts.ckks_vector(ctx, b["y_pad"].tolist())
        enc_batches.append({"enc_X": enc_X_cols, "enc_y": enc_y})
    t_enc = time.perf_counter() - t0_enc
    log_fn(f"  Feature & label encryption: {t_enc:.2f}s ({len(batches)} batches)")

    # 4. Feature Alignment Caching Setup (if applicable)
    cached_aligned_batches = None
    setup_alignment_time_s = 0.0
    if plan.alignment_strategy == "cached":
        t0_align = time.perf_counter()
        cached_aligned_batches = []
        for b in enc_batches:
            aligned_cols = [burn(b["enc_X"][j], plan.burn_X) for j in range(D)]
            cached_aligned_batches.append(aligned_cols)
        setup_alignment_time_s = time.perf_counter() - t0_align
        log_fn(f"  Precomputed feature alignment cache: {setup_alignment_time_s:.2f}s ({len(batches)*D} ciphertexts)")

    # 5. Deterministic Fresh Initialization
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.05, 0.05, size=D).astype(np.float64)
    w_init = w.copy()

    init_val = evaluate_model_val(X_val, y_val, w_init)
    log_fn(f"  Pre-training Val (Fold 9): Acc={init_val['exact_sigmoid']['accuracy']:.4f}, AUC={init_val['exact_sigmoid']['roc_auc']:.4f}")

    # 6. Training Loop
    epoch_records = []
    best_val_auc = -1.0
    best_epoch = 0
    best_w = w_init.copy()
    no_improve_count = 0
    stopped_early = False
    stop_reason = f"Completed all {epochs} planned epochs without early stopping"
    run_passed = True

    t0_all_epochs = time.perf_counter()

    for ep in range(1, epochs + 1):
        prof.clear()
        t0_epoch = time.perf_counter()

        pt_ref = compute_plaintext_reference(X_train, y_train, w, l2=l2, grad_clip=grad_clip, lr=lr)

        # Step 1: Encrypt weights
        with prof.measure("weight_encryption"):
            enc_w_cols = [ts.ckks_vector(ctx, [float(w[j])] * SLOT_CAP) for j in range(D)]

        # Step 2: CSP executes 3 batches under plan
        unreg_sum = np.zeros(D, dtype=np.float64)
        last_meta = None

        for b_idx, b in enumerate(enc_batches):
            cached_X_batch = cached_aligned_batches[b_idx] if cached_aligned_batches is not None else None
            enc_sums, b_meta = csp_batch_step(
                b["enc_X"], b["enc_y"], enc_w_cols, plan, D, enc_X_cached=cached_X_batch
            )
            last_meta = b_meta

            with prof.measure("decryption_aggregation"):
                dec_b_sums = np.array([g.decrypt()[0] for g in enc_sums], dtype=np.float64)
                unreg_sum += dec_b_sums

        # Step 3: Hospital normalization, regularization, clipping, and update
        with prof.measure("decryption_aggregation"):
            g_unreg = unreg_sum / N_train
            g_ckks = g_unreg + l2 * w

            g_norm_raw = float(np.linalg.norm(g_ckks))
            if g_norm_raw > grad_clip:
                g_ckks_clipped = g_ckks * (grad_clip / g_norm_raw)
            else:
                g_ckks_clipped = g_ckks.copy()
            g_norm_clipped = float(np.linalg.norm(g_ckks_clipped))

            w_update = lr * g_ckks_clipped
            w_update_norm = float(np.linalg.norm(w_update))
            w = w - w_update

        total_epoch_time = time.perf_counter() - t0_epoch

        # Finite checks
        finite_w = bool(np.all(np.isfinite(w)))
        finite_g = bool(np.all(np.isfinite(g_ckks)))
        if not (finite_w and finite_g):
            run_passed = False
            stop_reason = f"Safety stop: non-finite values detected in weights or gradients at Epoch {ep}!"
            log_fn(f"  CRITICAL FAILURE: {stop_reason}")
            break

        # Numerical gradient comparison vs matched plaintext reference
        grad_abs_err = np.abs(g_ckks - pt_ref["g_reg"])
        max_ae = float(grad_abs_err.max())
        mean_ae = float(grad_abs_err.mean())
        rel_err = float(np.linalg.norm(grad_abs_err) / np.linalg.norm(pt_ref["g_reg"]))
        tol_pass = max_ae <= TOLERANCE_GRAD_MAX_AE

        if not tol_pass:
            run_passed = False
            stop_reason = f"Safety stop: gradient error {max_ae:.2e} exceeded tolerance {TOLERANCE_GRAD_MAX_AE:.2e} at Epoch {ep}!"
            log_fn(f"  CRITICAL FAILURE: {stop_reason}")
            break

        w_diff_pt = float(np.linalg.norm(w - pt_ref["w_next"]))

        # Evaluate model on Fold 9 (Validation)
        val_res = evaluate_model_val(X_val, y_val, w)
        val_auc_true = val_res["exact_sigmoid"]["roc_auc"]
        val_acc_true = val_res["exact_sigmoid"]["accuracy"]
        val_f1_true = val_res["exact_sigmoid"]["f1"]
        val_prec_true = val_res["exact_sigmoid"]["precision"]
        val_rec_true = val_res["exact_sigmoid"]["recall"]

        val_auc_poly = val_res["poly_sigmoid"]["roc_auc"]
        val_acc_poly = val_res["poly_sigmoid"]["accuracy"]
        val_f1_poly = val_res["poly_sigmoid"]["f1"]
        val_prec_poly = val_res["poly_sigmoid"]["precision"]
        val_rec_poly = val_res["poly_sigmoid"]["recall"]

        max_abs_logit = val_res["max_abs_logit"]

        if max_abs_logit > SAFETY_MAX_LOGIT:
            run_passed = False
            stop_reason = f"Safety stop: validation max logit {max_abs_logit:.2f} exceeded safety limit {SAFETY_MAX_LOGIT:.1f} at Epoch {ep}!"
            log_fn(f"  CRITICAL FAILURE: {stop_reason}")
            break

        improved = (val_auc_true - best_val_auc) >= EARLY_STOP_DELTA
        if val_auc_true > best_val_auc:
            best_val_auc = val_auc_true
            best_epoch = ep
            best_w = w.copy()

        if ep > EARLY_STOP_MIN_EPOCHS and not disable_early_stop:
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= EARLY_STOP_PATIENCE:
                stopped_early = True
                stop_reason = (
                    f"Early stopping triggered at Epoch {ep}: validation ROC-AUC failed to improve by "
                    f">={EARLY_STOP_DELTA} for {EARLY_STOP_PATIENCE} consecutive epochs (best={best_val_auc:.4f} at Ep {best_epoch})"
                )

        ep_record = {
            "schedule_id": plan.schedule_id,
            "seed": seed,
            "epoch": ep,
            "total_epoch_time_s": total_epoch_time,
            "forward_linear_s": prof.get("forward_linear"),
            "poly_eval_s": prof.get("poly_eval"),
            "error_comp_s": prof.get("error_computation"),
            "level_adjustment_s": prof.get("level_adjustment"),
            "feature_alignment_s": prof.get("feature_alignment"),
            "gradient_comp_s": prof.get("gradient_computation"),
            "decryption_agg_s": prof.get("decryption_aggregation"),
            "weight_enc_s": prof.get("weight_encryption"),
            "grad_start_level": last_meta["err_after_adj_level"],
            "grad_end_level": last_meta["grad_end_level"],
            "grad_max_ae_vs_pt": max_ae,
            "grad_mean_ae_vs_pt": mean_ae,
            "grad_rel_err_vs_pt": rel_err,
            "tolerance_pass": tol_pass,
            "grad_norm_raw": g_norm_raw,
            "grad_norm_clipped": g_norm_clipped,
            "w_norm": float(np.linalg.norm(w)),
            "w_update_norm": w_update_norm,
            "w_diff_vs_pt_next": w_diff_pt,
            "val_accuracy_exact": val_acc_true,
            "val_roc_auc_exact": val_auc_true,
            "val_precision_exact": val_prec_true,
            "val_recall_exact": val_rec_true,
            "val_f1_exact": val_f1_true,
            "val_accuracy_poly": val_acc_poly,
            "val_roc_auc_poly": val_auc_poly,
            "val_precision_poly": val_prec_poly,
            "val_recall_poly": val_rec_poly,
            "val_f1_poly": val_f1_poly,
            "val_poly_outside_0_1": val_res["poly_sigmoid"]["poly_outside_0_1"],
            "val_max_abs_logit": max_abs_logit,
            "best_val_auc_so_far": best_val_auc,
            "best_epoch_so_far": best_epoch,
            "no_improve_counter": no_improve_count,
            "all_finite": val_res["all_finite"],
        }
        epoch_records.append(ep_record)

        log_fn(
            f"  Epoch {ep:2d}/{epochs:2d}: Time={total_epoch_time:.2f}s | "
            f"GradTime={ep_record['gradient_comp_s']:.2f}s | "
            f"MaxAE={max_ae:.2e} | "
            f"ValAUC={val_auc_true:.4f} | ValAcc={val_acc_true:.4f}"
        )

        if stopped_early:
            log_fn(f"  EARLY STOPPING: {stop_reason}")
            break

    total_training_wallclock_s = time.perf_counter() - t0_all_epochs
    cumulative_training_s = setup_alignment_time_s + sum(r["total_epoch_time_s"] for r in epoch_records)

    summary = {
        "schedule_id": plan.schedule_id,
        "condition": plan.condition,
        "alignment_strategy": plan.alignment_strategy,
        "grad_level": plan.grad_level,
        "burn_X": plan.burn_X,
        "level_adjustment_required": plan.level_adjustment_required,
        "seed": seed,
        "planned_epochs": epochs,
        "completed_epochs": len(epoch_records),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "run_passed": run_passed,
        "setup_alignment_time_s": setup_alignment_time_s,
        "total_training_wallclock_s": total_training_wallclock_s,
        "cumulative_training_s": cumulative_training_s,
        "mean_epoch_time_s": float(np.mean([r["total_epoch_time_s"] for r in epoch_records])),
        "mean_gradient_comp_s": float(np.mean([r["gradient_comp_s"] for r in epoch_records])),
        "max_grad_ae": float(max(r["grad_max_ae_vs_pt"] for r in epoch_records)),
        "best_epoch": best_epoch,
        "best_val_auc_exact": best_val_auc,
        "final_val_auc_exact": epoch_records[-1]["val_roc_auc_exact"] if epoch_records else 0.0,
        "final_val_accuracy_exact": epoch_records[-1]["val_accuracy_exact"] if epoch_records else 0.0,
        "final_val_f1_exact": epoch_records[-1]["val_f1_exact"] if epoch_records else 0.0,
        "predicted_total_training_s": plan.predicted_total_training_s,
        "predicted_subsequent_epoch_s": plan.predicted_subsequent_epoch_s,
        "prediction_error_s": cumulative_training_s - plan.predicted_total_training_s,
        "prediction_mape_percent": abs(cumulative_training_s - plan.predicted_total_training_s) / cumulative_training_s * 100.0,
        "epochs_data": epoch_records,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="PTB-XL Workload-Aware Encrypted Training Harness")
    parser.add_argument("--condition", choices=["A", "B", "auto"], default="auto",
                        help="Execution condition: 'A', 'B', or 'auto' (scheduler selected)")
    parser.add_argument("--alignment", choices=["repeated", "cached", "auto"], default="auto",
                        help="Feature alignment strategy: 'repeated', 'cached', or 'auto'")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS_DEFAULT,
                        help="Maximum training epochs (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic RNG seed (default: 42)")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="Learning rate (default: 0.05)")
    parser.add_argument("--l2", type=float, default=L2_DEFAULT, help="L2 regularization (default: 0.01)")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_DEFAULT, help="Gradient clipping (default: 1.0)")
    parser.add_argument("--memory-budget-mb", type=float, default=2048.0,
                        help="Ciphertext memory budget in MB (default: 2048.0)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Directory to save experimental results")
    parser.add_argument("--pilot", action="store_true",
                        help="Run quick pilot check (3 epochs, single seed 42)")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run matched evaluation across all 4 configurations (A-repeat, A-cache, B-repeat, B-cache)")
    parser.add_argument("--disable-early-stop", action="store_true",
                        help="Disable early stopping to force completion of all planned epochs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026],
                        help="Seeds for matched evaluation (default: 42 123 2026)")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pol = WorkloadSchedulerPolicy()

    if args.pilot:
        print("\n=======================================================")
        print(" EXECUTING PHASE 3B PILOT RUN (3 EPOCHS, SEED 42)")
        print("=======================================================\n")
        pilot_plan, all_plans = pol.select_plan(N=17084, D=48, epochs=3, memory_budget_mb=args.memory_budget_mb)
        print(f"Policy Selected Plan: {pilot_plan.schedule_id}")
        print(f"Predicted Total (3 ep): {pilot_plan.predicted_total_training_s:.2f}s")

        res = run_training_experiment(
            pilot_plan,
            seed=42,
            epochs=3,
            lr=args.lr,
            l2=args.l2,
            grad_clip=args.grad_clip,
            disable_early_stop=args.disable_early_stop,
            log_fn=print,
        )

        pilot_path = out_dir / "pilot_result.json"
        with open(pilot_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nPilot run complete. Results saved to {pilot_path}")
        print(f"Observed Cumulative Time: {res['cumulative_training_s']:.2f}s")
        print(f"Predicted Time:          {res['predicted_total_training_s']:.2f}s")
        print(f"MAPE:                    {res['prediction_mape_percent']:.2f}%")
        print(f"Max Gradient AE:         {res['max_grad_ae']:.2e} (Pass <= 1e-4: {res['run_passed']})")
        return

    if args.all_conditions:
        print("\n=======================================================")
        print(" EXECUTING FULL MATCHED EVALUATION (4 CONFIGS x 3 SEEDS)")
        print("=======================================================\n")
        configs = [
            ("A", "repeated"),
            ("A", "cached"),
            ("B", "repeated"),
            ("B", "cached"),
        ]

        all_summaries = []
        all_epoch_rows = []

        # First, record scheduler decisions for documentation
        decisions = {}
        for ep_cnt in [1, 3, 5, 10]:
            best_p, candidates = pol.select_plan(N=17084, D=48, epochs=ep_cnt, memory_budget_mb=args.memory_budget_mb)
            decisions[f"epochs_{ep_cnt}"] = {
                "selected_plan": best_p.schedule_id,
                "predicted_total_s": best_p.predicted_total_training_s,
                "candidates": {
                    c.schedule_id: {
                        "feasible": c.feasible,
                        "predicted_total_s": c.predicted_total_training_s if c.feasible else None,
                        "rejection_reason": c.rejection_reason,
                    }
                    for c in candidates
                },
            }

        with open(out_dir / "phase3b_scheduler_decisions.json", "w") as f:
            json.dump(decisions, f, indent=2)

        for seed in args.seeds:
            for cond, strat in configs:
                plan = pol.evaluate_candidate(
                    condition=cond,  # type: ignore
                    alignment_strategy=strat,  # type: ignore
                    N=17084,
                    D=48,
                    epochs=args.epochs,
                    memory_budget_mb=args.memory_budget_mb,
                )
                print(f"\n--- Running Config: {plan.schedule_id} | Seed: {seed} ---")
                res = run_training_experiment(
                    plan,
                    seed=seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    l2=args.l2,
                    grad_clip=args.grad_clip,
                    disable_early_stop=args.disable_early_stop,
                    log_fn=print,
                )
                all_summaries.append(res)
                for ep_d in res["epochs_data"]:
                    all_epoch_rows.append(ep_d)

        # Save main JSON results
        with open(out_dir / "phase3b_training_results.json", "w") as f:
            json.dump(all_summaries, f, indent=2)

        # Save timing breakdown CSV
        timing_df = pd.DataFrame([
            {
                "schedule_id": s["schedule_id"],
                "condition": s["condition"],
                "alignment_strategy": s["alignment_strategy"],
                "seed": s["seed"],
                "setup_alignment_time_s": s["setup_alignment_time_s"],
                "cumulative_training_s": s["cumulative_training_s"],
                "mean_epoch_time_s": s["mean_epoch_time_s"],
                "mean_gradient_comp_s": s["mean_gradient_comp_s"],
                "predicted_total_s": s["predicted_total_training_s"],
                "prediction_mape_percent": s["prediction_mape_percent"],
                "best_val_auc": s["best_val_auc_exact"],
                "final_val_auc": s["final_val_auc_exact"],
                "max_grad_ae": s["max_grad_ae"],
                "run_passed": s["run_passed"],
            }
            for s in all_summaries
        ])
        timing_df.to_csv(out_dir / "phase3b_training_results.csv", index=False)
        print(f"\nSaved results table to {out_dir / 'phase3b_training_results.csv'}")

        # Save epoch-level CSV
        epochs_df = pd.DataFrame(all_epoch_rows)
        epochs_df.to_csv(out_dir / "phase3b_timing_breakdown.csv", index=False)
        print(f"Saved epoch breakdown to {out_dir / 'phase3b_timing_breakdown.csv'}")

        # Save validation metrics summary CSV
        val_df = timing_df[[
            "schedule_id", "condition", "alignment_strategy", "seed",
            "best_val_auc", "final_val_auc", "max_grad_ae", "cumulative_training_s", "mean_epoch_time_s"
        ]]
        val_df.to_csv(out_dir / "phase3b_validation_metrics.csv", index=False)
        print(f"Saved validation metrics to {out_dir / 'phase3b_validation_metrics.csv'}")

        print("\nAll matched experiments completed successfully!")
        return

    # Single configuration execution
    if args.condition == "auto" or args.alignment == "auto":
        plan, _ = pol.select_plan(
            N=17084, D=48, epochs=args.epochs, memory_budget_mb=args.memory_budget_mb
        )
    else:
        plan = pol.evaluate_candidate(
            condition=args.condition,  # type: ignore
            alignment_strategy=args.alignment,  # type: ignore
            N=17084,
            D=48,
            epochs=args.epochs,
            memory_budget_mb=args.memory_budget_mb,
        )

    res = run_training_experiment(
        plan,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        grad_clip=args.grad_clip,
        disable_early_stop=args.disable_early_stop,
        log_fn=print,
    )
    res_path = out_dir / f"training_{plan.schedule_id}_seed{args.seed}.json"
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nRun complete. Results saved to {res_path}")


if __name__ == "__main__":
    main()
