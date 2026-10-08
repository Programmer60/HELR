import numpy as np

# We reuse the polynomial definitions from the ablation study.
# We also include degree-7 for completeness if depth budget allows it.
POLY_COEFFS = {
    1: {"c0": 0.5, "c1": 0.197, "c3": 0.0, "c5": 0.0},
    3: {"c0": 0.5, "c1": 0.15012, "c3": -0.001593, "c5": 0.0},
    5: {"c0": 0.5, "c1": 0.217101, "c3": -0.007823, "c5": 0.000118},
    # Note: If we needed a degree-7, we'd add it here. For now we use 1, 3, 5.
}

# The intrinsic levels the SIGMOID part of the circuit consumes.
_SIG_LEVELS = {1: 1, 3: 3, 5: 4}

def true_sigmoid(z):
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def eval_poly(z, degree, coeffs):
    p = coeffs
    out = p["c0"] + p["c1"] * z
    if degree >= 3:
        out = out + p["c3"] * (z**3)
    if degree >= 5:
        out = out + p["c5"] * (z**5)
    return out

def get_circuit_depth(degree):
    """
    Returns total levels consumed for one training epoch.
    dot_product (1) + sigmoid (depends on degree) + backward (1)
    """
    return 1 + _SIG_LEVELS[degree] + 1

def select_polynomial_degree(
    logit_range=(-6.0, 6.0),
    candidate_degrees=[1, 3, 5],
    max_approx_error=0.1,
    budget_levels=7
):
    """
    Selects the lowest-degree polynomial that meets the approximation error
    requirement and fits within the given CKKS modulus depth budget.
    
    Inputs:
    - logit_range: tuple (min, max) defining the interval of interest
    - candidate_degrees: list of degrees to consider (e.g. [1, 3, 5])
    - max_approx_error: threshold for maximum absolute error on the interval
    - budget_levels: usable levels in the CKKS context (e.g. 7)
    
    Returns: dict with selection results.
    """
    z_eval = np.linspace(logit_range[0], logit_range[1], 2000)
    true_sig = true_sigmoid(z_eval)
    
    results = {}
    feasible = []
    
    for deg in sorted(candidate_degrees):
        if deg not in POLY_COEFFS:
            continue
            
        coeffs = POLY_COEFFS[deg]
        poly_sig = eval_poly(z_eval, deg, coeffs)
        max_err = float(np.abs(true_sig - poly_sig).max())
        
        depth_consumed = get_circuit_depth(deg)
        depth_fits = (depth_consumed < budget_levels) # Need at least 1 remaining
        
        err_fits = (max_err <= max_approx_error)
        
        res = {
            "degree": deg,
            "max_error": max_err,
            "depth_consumed": depth_consumed,
            "depth_fits": depth_fits,
            "error_fits": err_fits
        }
        results[deg] = res
        
        if depth_fits and err_fits:
            feasible.append(deg)
            
    # Selection logic
    if not feasible:
        # If none fit both, what do we do? We should prioritize depth fitting so it at least runs.
        depth_fitting = [d for d in candidate_degrees if d in results and results[d]["depth_fits"]]
        if not depth_fitting:
            selected_deg = None
            reason = "No candidates fit the depth budget."
            coeffs = None
        else:
            # Pick the one with the lowest error among those that fit depth
            best_deg = min(depth_fitting, key=lambda d: results[d]["max_error"])
            selected_deg = best_deg
            reason = f"No candidate met max_error <= {max_approx_error}. Selected degree {best_deg} as best available."
            coeffs = POLY_COEFFS[best_deg]
    else:
        # Lowest degree among feasible candidates
        selected_deg = min(feasible)
        reason = f"Degree {selected_deg} is the lowest degree meeting error <= {max_approx_error} and fitting depth budget."
        coeffs = POLY_COEFFS[selected_deg]
        
    return {
        "selected_degree": selected_deg,
        "selected_interval": list(logit_range),
        "polynomial_coefficients": coeffs,
        "approximation_error": results[selected_deg]["max_error"] if selected_deg else None,
        "estimated_depth_consumed": results[selected_deg]["depth_consumed"] if selected_deg else None,
        "reason": reason,
        "candidates": results
    }

if __name__ == "__main__":
    # Test the selector
    print("Testing polynomial selector...")
    res = select_polynomial_degree(logit_range=(-5.0, 5.0), max_approx_error=0.05, budget_levels=7)
    import json
    print(json.dumps(res, indent=2))
