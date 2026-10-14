#!/usr/bin/env python3
"""Generate PTBXL_SIGMOID_CALIBRATION.md from the completed JSON results."""
import json
from pathlib import Path

OUT_DIR = Path("results/ptbxl/sigmoid_calibration")
DOCS_DIR = Path("docs/ptbxl")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "ptbxl_sigmoid_calibration.json", encoding="utf-8") as f:
    data = json.load(f)

def fmt(x, d=4):
    if x is None: return "-"
    if isinstance(x, bool): return "Yes" if x else "No"
    if isinstance(x, float): return f"{x:.{d}f}"
    return str(x)

lines = []
def L(s=""): lines.append(str(s))

L("# PTB-XL Sigmoid Calibration and Objective-Consistency Audit")
L()
L(f"**Generated**: {data['timestamp']}")
L()
L("> All numerical values are measured empirically. Test fold (Fold 10) is never used in any analysis.")
L()
L("---")
L()

# ── 1. Objective Consistency
L("## 1. Objective Consistency Audit")
L()
ba = data["baseline_model_audit"]
oa = data["objective_audit"]
L("| Property | Saved Baseline (`baseline_logistic_regression.joblib`) | CKKS Convention |")
L("| :--- | :--- | :--- |")
L(f"| Intercept | Yes (b = {ba['intercept_value']:.4f}) | **No** |")
L(f"| L2 regularization | sklearn C=1.0 (total objective norm) | lambda=0.01 (per-sample gradient) |")
L(f"| Objective | sklearn cross-entropy + (1/2C)||w||^2 | (1/N)*X^T*(p-y) + lambda*w |")
L(f"| Objectives consistent? | **No** |  |")
L()
L("> [!IMPORTANT]")
L("> The saved baseline was fitted with `fit_intercept=True` and `C=1.0` (sklearn convention).")
L("> The CKKS encrypted training circuit carries no intercept ciphertext.")
L("> These are different objectives. The saved baseline is preserved unchanged.")
L("> A no-intercept plaintext reference model is used for calibration comparisons.")
L()
L("---")
L()

# ── 2. Logit distributions
L("## 2. PTB-XL Logit Distribution")
L()
L("Computed on training and validation splits only. Test fold excluded.")
L()
L("| Split | N | Min | Max | Mean | Std | p1% | p50% | p99% |")
L("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for key in ["train_with_intercept", "val_with_intercept", "train_no_intercept", "val_no_intercept"]:
    s = data["logit_distributions"][key]
    q = s["quantiles"]
    q1 = q.get("q0100", 0.0)
    q50 = q.get("q0500", 0.0)
    q99 = q.get("q0990", 0.0)
    L(f"| {key} | {s['n']} | {s['min']:.2f} | {s['max']:.2f} | "
      f"{s['mean']:.3f} | {s['std']:.3f} | "
      f"{q1:.3f} | {q50:.3f} | {q99:.3f} |")

L()
L("**Fraction of training records (with intercept) outside candidate polynomial intervals:**")
L()
L("| Interval | Fraction | Percentage |")
L("| :--- | :--- | :--- |")
frac = data["logit_distributions"]["train_with_intercept"]["fraction_outside_interval"]
for k, v in frac.items():
    L(f"| {k} | {v:.4f} | {v*100:.2f}% |")

L()
L("> [!WARNING]")
L("> Validation set contains an outlier at z ~= 112. This record produces extreme polynomial outputs")
L("> for degree-3 and degree-5 polynomials fitted on [-3, +10] ranges and will require robust handling.")
L()
L("---")
L()

# ── 3. MHEALTH polynomials evaluated on PTB-XL
L("## 3. MHEALTH Polynomial Evaluation on PTB-XL Logits")
L()
L("Polynomials are unchanged from the MHEALTH experiments (frozen).")
L()
L("| Degree | Split | Valid Interval | Prob MaxAE | Prob MeanAE | Out-of-[0,1] Count | Grad Factor MaxAE |")
L("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
mhe = data["mhealth_poly_evaluation_on_ptbxl"]
for deg in [1, 3, 5]:
    entry = mhe[f"deg{deg}"]
    vi = entry["mhealth_valid_interval"]
    for split in ["train_full", "val"]:
        r = entry[split]
        L(f"| {deg} | {split} | [{vi[0]}, {vi[1]}] | "
          f"{r['prob_max_ae']:.4f} | {r['prob_mean_ae']:.4f} | "
          f"{r['prob_out_of_01']} | {r['grad_factor_max_ae']:.4f} |")

L()
L("> [!NOTE]")
L("> The degree-3 MHEALTH polynomial (`c1=0.15012, c3=-0.001593`) produces out-of-[0,1]")
L("> probability outputs for 573 training records and 60 validation records when evaluated with")
L("> the baseline model's weights. The degree-5 polynomial is severely unstable beyond [-6, 6].")
L()
L("---")
L()

# ── 4. PTB-XL fitted candidates
L("## 4. PTB-XL-Fitted Candidate Polynomials")
L()
L("Coefficients fitted by least squares on **training logits only** (CKKS convention: no intercept).")
L("Candidate comparison uses validation data. Test fold excluded.")
L()
L("| Degree | c0 | c1 | c3 | c5 | Fit Interval | Train Prob MaxAE | Val Prob MaxAE | Val Out-of-[0,1] |")
L("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
cands = data["ptbxl_candidate_polynomials"]
for deg in [1, 3, 5]:
    c = cands[f"deg{deg}"]
    cc = c["coefficients"]
    te = c["train_eval"]
    ve = c["val_eval"]
    fi = c["fit_interval"]
    L(f"| {deg} | {cc['c0']:.4f} | {cc['c1']:.6f} | {cc['c3']:.6f} | {cc['c5']:.6f} | "
      f"[{fi[0]}, {fi[1]}] | {te['prob_max_ae']:.4f} | {ve['prob_max_ae']:.4f} | {ve['prob_out_of_01']} |")

L()
L("> [!WARNING]")
L("> All three candidate polynomials show catastrophically large val prob MaxAE (>8 for degree 1,")
L("> >1498 for degree 3, >379,914 for degree 5). This is caused by the val outlier at z~=112.")
L("> The polynomial approximation error on the 99.9th percentile bounded subset is much smaller.")
L()
L("---")
L()

# ── 5. CKKS depth feasibility
L("## 5. CKKS Depth Feasibility")
L()
L("CKKS: N=16384, coeff_mod_bit_sizes=[60,40,40,40,40,40,40,60], total usable levels=7.")
L()
L("| Degree | Fwd Dot (Ct*Ct) | Poly Eval Levels | Bwd Grad (Ct*Ct) | Total Consumed | Budget Remaining | Feasible |")
L("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
feas = data["ckks_depth_feasibility"]
poly_levels_map = {1: 1, 3: 3, 5: 4}
for deg in [1, 3, 5]:
    dc = feas[f"deg{deg}"]
    pL = poly_levels_map[deg]
    L(f"| {deg} | 1 | {pL} | 1 | {dc['total_needed']} | {dc['budget_margin']} | "
      f"{'FEASIBLE' if dc['feasible'] else 'INFEASIBLE'} |")

L()
L("> [!NOTE]")
L("> All three degrees fit within the 7-level budget for the forward+backward pass.")
L("> The constraint is numerical, not depth.")
L()
L("---")
L()

# ── 6. Gradient comparison
L("## 6. Gradient Comparison (Polynomial vs Exact Sigmoid)")
L()
L("Comparison uses training data only with CKKS convention (no intercept, L2=0.01).")
L("Shows max absolute error between gradient computed with polynomial vs exact sigmoid.")
L()
L("| Polynomial | Weight Case | Grad MaxAE | Grad MeanAE |")
L("| :--- | :--- | :--- | :--- |")
for r in data["gradient_comparison"]:
    L(f"| {r['polynomial']} | {r['weight_case']} | {r['grad_max_ae']:.4e} | {r['grad_mean_ae']:.4e} |")

L()
L("> [!NOTE]")
L("> The gradient error at `w_noic_ref` (no-intercept plaintext reference) is large for degrees 3 and 5,")
L("> indicating the polynomial approximation breaks down at the trained weight scale.")
L("> At small weights (w_rand1-5 with norm ~0.04-0.09), gradient errors are modest (<0.1).")
L()
L("---")
L()

# ── 7. Plaintext training sanity check
L("## 7. Plaintext Training Sanity Check (50 epochs)")
L()
L("CKKS convention: no intercept, lr=0.05, L2=0.01, gradient clip=1.0.")
L("**Validation metrics only. Test fold excluded.**")
L()
L("| Polynomial | Val Acc (ep50) | Val Loss | W Norm | Grad Norm | Stable |")
L("| :--- | :--- | :--- | :--- | :--- | :--- |")
ts = data["plaintext_training_sanity"]
for poly, v in ts.items():
    L(f"| {poly} | {fmt(v['final_val_acc'])} | {fmt(v['final_val_loss'])} | "
      f"{fmt(v['final_w_norm'])} | {fmt(v['final_grad_norm'])} | "
      f"{'Yes' if v['stable'] else 'No'} |")

L()
L("> All polynomials achieve ~66-67% val accuracy at 50 epochs. This is below the plaintext baseline")
L("> (~70% with intercept), confirming that no-intercept CKKS training starts with a handicap.")
L()
L("---")
L()

# ── 8. Final candidate assessment
L("## 8. Final Candidate Assessment and Recommendation")
L()
cs = data["candidate_selection"]
L(f"Thresholds applied: epsilon_poly = {cs['epsilon_poly']}, epsilon_grad = {cs['epsilon_grad']}")
L()
L("| Degree | Val Prob MaxAE | Grad MaxAE | CKKS Feasible | Training Stable | Val Acc@50 | Result |")
L("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for r in cs["results"]:
    status = "PASS" if r["passes_all_checks"] else "FAIL"
    L(f"| {r['degree']} | {r['val_prob_max_ae']:.4f} | {r['grad_max_ae']:.4e} | "
      f"{'Yes' if r['depth_feasible'] else 'No'} | "
      f"{'Yes' if r['training_stable'] else 'No'} | "
      f"{fmt(r['val_acc_epoch50'])} | {status} |")

L()
L(f"**Recommendation: `{cs['recommendation']}`**")
L()
L("---")
L()

# ── 9. Analysis and Conclusions
L("## 9. Analysis and Conclusions")
L()
L("### Root Cause of Calibration Failure")
L()
L("The primary issue is not polynomial degree or CKKS depth. It is **logit range and outliers**.")
L()
L("The PTB-XL validation set contains at least one record with an extreme logit z ~= 112")
L("under the saved baseline model weights. When any bounded polynomial is evaluated at z=112:")
L()
L("- Degree 1: p(112) = 0.5 + 0.077 * 112 = 9.12 (far outside [0,1])")
L("- Degree 3: p(112) = 0.5 + 0.138*112 - 0.00109*112^3 = catastrophically large")
L("- Degree 5: similarly divergent")
L()
L("The val MaxAE numbers (8.1, 1498, 379914) are dominated by this single outlier.")
L()
L("### What the Bounded-Range Results Tell Us")
L()
L("On the p99.9 bounded subset (|z| <= ~10 on training):")
L()
L("| Degree | Train Bounded Prob MaxAE | Notes |")
L("| :--- | :--- | :--- |")
for deg in [1, 3, 5]:
    r = data["mhealth_poly_evaluation_on_ptbxl"][f"deg{deg}"]["train_bounded_p999"]
    L(f"| {deg} (MHEALTH) | {r['prob_max_ae']:.4f} | out_of_01={r['prob_out_of_01']} |")

L()
L("### Required Actions Before Phase 4 Can Proceed")
L()
L("1. **Outlier investigation**: Identify the validation record with z~=112 and determine")
L("   whether it is a data quality issue (mislabeled, corrupted waveform) or a genuine")
L("   extreme ECG morphology. If it is a data anomaly, document and handle it consistently.")
L()
L("2. **Logit clipping / gradient clipping**: The encrypted training circuit must handle")
L("   extreme logits gracefully. Options:")
L("   - Clip gradients in the trusted Hospital step (already done at clip=1.0, but")
L("     the polynomial probability explosion happens before decryption)")
L("   - Use a wider-range polynomial fitted on [-15, +15] with higher stability requirements")
L("   - Investigate whether the extreme logit arises only during inference (saved baseline")
L("     with intercept) vs. during CKKS training (no intercept, different weight trajectory)")
L()
L("3. **CKKS-convention logit range**: The no-intercept CKKS training logits (z_train_ckks)")
L("   are zero-mean and have a much lower max (~19.7) than the val outlier (~111.5).")
L("   If the extreme outlier arises primarily from the intercept b=0.5846 pushing boundary")
L("   records over a cliff, then the CKKS training circuit (no intercept) may not encounter it.")
L()
L("4. **No-intercept CKKS training feasibility**: Degree 3 PTB-XL candidate gives val_acc=0.6631")
L("   at 50 epochs with gradient clipping. If the outlier is in a specific validation record,")
L("   the no-intercept encrypted training may converge adequately without encountering z~=112.")
L()
L("### Conclusion")
L()
L("No candidate polynomial passes all three thresholds (val_prob_MaxAE<=0.10, grad_MaxAE<=0.20,")
L("CKKS depth feasibility) when evaluated on the **full validation set** including the z~=112 outlier.")
L()
L("**The experiment correctly reports NO_FEASIBLE_CANDIDATE rather than selecting an invalid fallback.**")
L()
L("The experiment did NOT find that polynomial approximation is fundamentally unsuitable for PTB-XL.")
L("It found that there is one or more extreme-logit records in the validation fold that must be")
L("investigated before a candidate can be certified as ready for Phase 4 encrypted training.")
L()
L("---")
L()
L("## 10. Reproducibility")
L()
L("```bash")
L("# Run calibration experiment (from project root)")
L("python results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.py")
L("# Regenerate this document from completed JSON")
L("python results/ptbxl/sigmoid_calibration/generate_calibration_report.py")
L("```")
L()
L("Outputs:")
L("- `results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.json` - all measured values")
L("- `results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.csv`  - polynomial evaluation table")
L("- `results/ptbxl/sigmoid_calibration/ptbxl_sigmoid_calibration.log`  - execution log")
L("- `docs/ptbxl/PTBXL_SIGMOID_CALIBRATION.md`                          - this document")

md_path = DOCS_DIR / "PTBXL_SIGMOID_CALIBRATION.md"
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {md_path}")
print(f"Lines: {len(lines)}")
