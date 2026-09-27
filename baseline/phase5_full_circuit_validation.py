import numpy as np
import tenseal as ts
import time

# MIMIC-specific polynomial coefficients
c0 = 0.5
c1_1 = 0.197
c1_3 = 0.180597
c3_3 = -0.003091
c1_5 = 0.217101
c3_5 = -0.007823
c5_5 = 0.000118

def get_levels(enc):
    return enc.ciphertext()[0].coeff_modulus_size()

def get_scale(enc):
    import math
    scale_val = enc.ciphertext()[0].scale
    return math.log2(scale_val)

def burn_levels(vec, n):
    for _ in range(n):
        vec = vec * 1.0
    return vec

print("Initializing CKKS context (practical profile)...")
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60])
ctx.global_scale = 2 ** 40
ctx.generate_galois_keys()
ctx.generate_relin_keys()

# Dummy data from X_test, top 100 rows
X_full = np.load(r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready\X_test.npy")
y_full = np.load(r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready\y_test.npy")
w_full = np.load(r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready\weights.npy")
b_full = np.load(r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready\intercept.npy")[0]

BATCH_SIZE = 100
X = X_full[:BATCH_SIZE]
y = y_full[:BATCH_SIZE]
# for simplicity, we absorb the bias into the first step or ignore it for gradient validation?
# gradient of bias is mean(err). Let's just track weights for simplicity, or include bias.
# We will use exactly w_full. We can skip bias for this precise trace or just add it to z.
w = w_full.copy()
b = b_full

results = []

def pt_sigmoid(z, degree):
    if degree == 1:
        return c0 + c1_1 * z
    elif degree == 3:
        return c0 + c1_3 * z + c3_3 * (z**3)
    elif degree == 5:
        return c0 + c1_5 * z + c3_5 * (z**3) + c5_5 * (z**5)

def pt_gradient(X, y, w, b, degree, l2=0.0):
    P, D = X.shape
    z = X @ w + b
    p = pt_sigmoid(z, degree)
    err = p - y
    grad = (X.T @ err) / P + l2 * w
    return grad, p

def track_full_circuit(degree):
    print(f"\n=======================================================")
    print(f"--- DEGREE-{degree} FULL FORWARD + BACKWARD CIRCUIT ---")
    print(f"=======================================================")
    
    t0 = time.time()
    enc_X = [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(10)]
    enc_y = ts.ckks_vector(ctx, y.tolist())
    
    logs = []
    
    def log_state(step, enc, note="intrinsic"):
        logs.append({
            "step": step,
            "levels": get_levels(enc),
            "scale": get_scale(enc),
            "note": note
        })
        print(f"{step:<35} | Levels: {get_levels(enc):<2} | Scale: 2^{get_scale(enc):.1f} | Type: {note}")

    log_state("1. Fresh encrypted feature", enc_X[0], "start")
    
    # 3. feature * weight
    z = enc_X[0] * w[0]
    log_state("3. feature * weight", z, "intrinsic")
    
    for j in range(1, 10):
        z = z + (enc_X[j] * w[j])
        
    z = z + b # add bias
    log_state("4. accumulated z (inc. bias)", z, "intrinsic")
    
    z_start_level = get_levels(z)
    
    # Forward pass
    alignment_consumed = 0
    fwd_intrinsic = 0
    
    if degree == 1:
        p = z * c1_1 + c0
        log_state("6. final prediction p", p, "intrinsic")
        fwd_intrinsic = 1
        
    elif degree == 3:
        z2 = z * z
        log_state("5. z2 = z * z", z2, "intrinsic")
        fwd_intrinsic += 1
        
        z_match1 = burn_levels(z, 1)
        z3 = z2 * z_match1
        alignment_consumed += 1
        log_state("5. z3 = z2 * z", z3, "intrinsic")
        fwd_intrinsic += 1
        
        term3 = z3 * c3_3
        log_state("5. term3 = z3 * c3", term3, "intrinsic")
        fwd_intrinsic += 1
        
        term1 = burn_levels(z, 2) * c1_3
        alignment_consumed += 2
        
        p = term3 + term1 + c0
        log_state("6. final prediction p", p, "intrinsic")
        
    elif degree == 5:
        z2 = z * z
        fwd_intrinsic += 1
        
        z_match1 = burn_levels(z, 1)
        alignment_consumed += 1
        z3 = z2 * z_match1
        log_state("5. z3 = z2 * z", z3, "intrinsic")
        fwd_intrinsic += 1
        
        z_match2 = burn_levels(z2, 1)
        alignment_consumed += 1
        z5 = z3 * z_match2
        log_state("5. z5 = z3 * z2", z5, "intrinsic")
        fwd_intrinsic += 1
        
        term5 = z5 * c5_5
        log_state("5. term5 = z5 * c5", term5, "intrinsic")
        fwd_intrinsic += 1
        
        term3 = burn_levels(z3, 1) * c3_5
        alignment_consumed += 1
        
        term1 = burn_levels(z, 3) * c1_5
        alignment_consumed += 3
        
        p = term5 + term3 + term1 + c0
        log_state("6. final prediction p", p, "intrinsic")
        
    p_level = get_levels(p)
    
    # BACKWARD PASS
    print("\n--- BACKWARD PASS ---")
    depth_to_p = 7 - p_level
    
    # error = p - y
    # align y to match p
    enc_y_aligned = burn_levels(enc_y, depth_to_p)
    # the burn_levels calls consume alignment levels, but they don't count towards depth of p
    # enc_y_aligned drops from 7 down to p_level
    
    err = p - enc_y_aligned
    log_state("7. error = p - y", err, "intrinsic (no drop)")
    
    # gradient multiplication
    grads_enc = []
    # align X_j to match err
    enc_X_aligned = [burn_levels(enc_X[j], depth_to_p) for j in range(10)]
    
    err_times_x0 = err * enc_X_aligned[0]
    log_state("8. error * X_0", err_times_x0, "intrinsic")
    
    bwd_intrinsic = 1
    
    for j in range(10):
        grad_sum = (err * enc_X_aligned[j]).sum()
        grads_enc.append(grad_sum)
        
    log_state("9. packed sum (grad_0)", grads_enc[0], "intrinsic (no drop)")
    log_state("10. final encrypted gradient", grads_enc[0], "intrinsic (no drop)")
    
    final_level = get_levels(grads_enc[0])
    rt = time.time() - t0
    
    print("\n--- NUMERICAL VALIDATION ---")
    grad_pt, p_pt = pt_gradient(X, y, w, b, degree, l2=0.0)
    
    decrypted_grad = np.array([g.decrypt()[0] / BATCH_SIZE for g in grads_enc])
    
    grad_err = np.abs(grad_pt - decrypted_grad)
    max_err = grad_err.max()
    mean_err = grad_err.mean()
    
    print(f"Max absolute gradient error : {max_err:.8f}")
    print(f"Mean absolute gradient error: {mean_err:.8f}")
    
    results.append({
        "degree": degree,
        "fwd_depth": fwd_intrinsic,
        "bwd_depth": bwd_intrinsic,
        "align_levels": alignment_consumed,
        "total_consumed": 7 - final_level,
        "final_remaining": final_level,
        "max_err": max_err,
        "mean_err": mean_err,
        "runtime": rt
    })

for deg in [1, 3, 5]:
    try:
        track_full_circuit(deg)
    except Exception as e:
        print(f"\n[ERROR] Degree-{deg} tracking failed:", e)

print("\n--- TASK 4: COMPARISON TABLE ---")
print(f"{'Degree':<8} | {'Fwd Depth':<10} | {'Bwd Depth':<10} | {'Align Levels':<14} | {'Total Consumed':<15} | {'Final Remaining':<16} | {'Max Err':<10} | {'Runtime (s)':<10}")
for r in results:
    print(f"{r['degree']:<8} | {r['fwd_depth']:<10} | {r['bwd_depth']:<10} | {r['align_levels']:<14} | {r['total_consumed']:<15} | {r['final_remaining']:<16} | {r['max_err']:<10.6f} | {r['runtime']:<10.2f}")
