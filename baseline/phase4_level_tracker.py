import numpy as np
import tenseal as ts
import time

def get_levels(enc):
    """Returns the remaining modulus size (levels remaining)."""
    return enc.ciphertext()[0].coeff_modulus_size()

def get_scale(enc):
    import math
    scale_val = enc.ciphertext()[0].scale
    return math.log2(scale_val)

def burn_levels(vec, n):
    for _ in range(n):
        vec = vec * 1.0
    return vec

# Mimic-specific coefficients from Phase 3
# Deg 3: 0.5 + 0.180597 * z - 0.003091 * z^3
c0 = 0.5
c1_3 = 0.180597
c3_3 = -0.003091

# Deg 5: 0.5 + 0.217101 * z - 0.007823 * z^3 + 0.000118 * z^5
c1_5 = 0.217101
c3_5 = -0.007823
c5_5 = 0.000118

print("Initializing CKKS context (practical profile)...")
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60])
ctx.global_scale = 2 ** 40
ctx.generate_galois_keys()
ctx.generate_relin_keys()

# Dummy data for validation (first 100 rows, 10 features)
np.random.seed(42)
X = np.random.randn(100, 10)
w = np.random.randn(10)

def track_linear_forward():
    print("\n--- DEGREE-1 POLYNOMIAL TRACKING ---")
    enc_X = [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(10)]
    print(f"1. Encrypted feature       | Depth (remaining): {get_levels(enc_X[0])} | Scale: 2^{get_scale(enc_X[0]):.1f}")
    
    # Feature * Weight (pt)
    z = enc_X[0] * w[0]
    print(f"2. Feature * Weight (pt)   | Depth (remaining): {get_levels(z)} | Scale: 2^{get_scale(z):.1f}")
    
    for j in range(1, 10):
        z = z + (enc_X[j] * w[j])
        
    print(f"3. Accumulated z           | Depth (remaining): {get_levels(z)} | Scale: 2^{get_scale(z):.1f}")
    
    p = z * c1_3 + c0
    print(f"4. Final p (linear approx) | Depth (remaining): {get_levels(p)} | Scale: 2^{get_scale(p):.1f}")

def track_degree3_forward():
    print("\n--- DEGREE-3 POLYNOMIAL TRACKING ---")
    enc_X = [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(10)]
    
    z = enc_X[0] * w[0]
    for j in range(1, 10):
        z = z + (enc_X[j] * w[j])
    
    start_level = get_levels(z)     # Track the starting level before any multiplications
    print(f"1. Accumulated z           | Depth (remaining): {start_level} | Scale: 2^{get_scale(z):.1f}")
    
    z2 = z * z
    print(f"2. z2 = z * z              | Depth (remaining): {get_levels(z2)} | Scale: 2^{get_scale(z2):.1f}")
    
    z_match1 = burn_levels(z, 1)        # Burn 1 level to align with z2 for multiplication
    z3 = z2 * z_match1
    print(f"3. z3 = z2 * z             | Depth (remaining): {get_levels(z3)} | Scale: 2^{get_scale(z3):.1f}")
    
    term3 = z3 * c3_3
    print(f"4. term3 = z3 * c3         | Depth (remaining): {get_levels(term3)} | Scale: 2^{get_scale(term3):.1f}")
    
    term1 = burn_levels(z, 2) * c1_3
    print(f"5. term1 = z * c1 (aligned)| Depth (remaining): {get_levels(term1)} | Scale: 2^{get_scale(term1):.1f}")
    
    p = term3 + term1 + c0          # Final polynomial evaluation
    print(f"6. Final p (deg-3 approx)  | Depth (remaining): {get_levels(p)} | Scale: 2^{get_scale(p):.1f}")
    print(f"-> Levels consumed: {start_level - get_levels(p)}")

def track_degree5_forward():
    print("\n--- DEGREE-5 POLYNOMIAL TRACKING ---")
    enc_X = [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(10)]
    
    z = enc_X[0] * w[0]
    for j in range(1, 10):
        z = z + (enc_X[j] * w[j])           # Accumulate weighted sum of features
        
    start_level = get_levels(z)
    print(f"1. Accumulated z           | Depth (remaining): {start_level} | Scale: 2^{get_scale(z):.1f}")
    
    z2 = z * z      
    z_match1 = burn_levels(z, 1)        # Burn 1 level to align with z2 for multiplication
    z3 = z2 * z_match1          
    print(f"2. z3 = z2 * z             | Depth (remaining): {get_levels(z3)} | Scale: 2^{get_scale(z3):.1f}")
    
    z_match2 = burn_levels(z2, 1)   # Burn 1 level to align with z3 for multiplication
    z5 = z3 * z_match2
    print(f"3. z5 = z3 * z2            | Depth (remaining): {get_levels(z5)} | Scale: 2^{get_scale(z5):.1f}")
    
    term5 = z5 * c5_5
    print(f"4. term5 = z5 * c5         | Depth (remaining): {get_levels(term5)} | Scale: 2^{get_scale(term5):.1f}")
    
    term3 = burn_levels(z3, 1) * c3_5
    print(f"5. term3 = z3 * c3 (aligned)| Depth (remaining): {get_levels(term3)} | Scale: 2^{get_scale(term3):.1f}")
    
    term1 = burn_levels(z, 3) * c1_5
    print(f"6. term1 = z * c1 (aligned)| Depth (remaining): {get_levels(term1)} | Scale: 2^{get_scale(term1):.1f}")
    
    p = term5 + term3 + term1 + c0
    print(f"7. Final p (deg-5 approx)  | Depth (remaining): {get_levels(p)} | Scale: 2^{get_scale(p):.1f}")
    print(f"-> Levels consumed: {start_level - get_levels(p)}")

track_linear_forward()
track_degree3_forward()

try:
    track_degree5_forward()
except Exception as e:
    print("\n[ERROR] Degree-5 tracking failed:", e)
    print("This means the current context depth (7 primes) is insufficient for degree-5!")

