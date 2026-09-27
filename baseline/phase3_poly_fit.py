import numpy as np
import pandas as pd
import os, json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score, roc_auc_score

results_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results"
he_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\results\he_ready"

# Load the saved logits and labels
y_test = np.load(os.path.join(he_dir, "y_test.npy"))
logits_test = np.load(os.path.join(he_dir, "logits_test.npy"))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def poly3(x, c1, c3):
    return 0.5 + c1 * x + c3 * (x**3)

def poly5(x, c1, c3, c5):
    return 0.5 + c1 * x + c3 * (x**3) + c5 * (x**5)

# Fit over [-6.0, 6.0] to safely cover min (-4.73) and max (+2.10)
x_fit = np.linspace(-6.0, 6.0, 10000)
y_fit = sigmoid(x_fit)

popt3, _ = curve_fit(poly3, x_fit, y_fit)
popt5, _ = curve_fit(poly5, x_fit, y_fit)

c1_3, c3_3 = popt3
c1_5, c3_5, c5_5 = popt5

print("\n--- POLYNOMIAL SIGMOID DESIGN ---")
print("Fitting Interval: [-6.0, 6.0]")

for degree, func, coefs in [(3, poly3, popt3), (5, poly5, popt5)]:
    print(f"\n--- Degree {degree} Polynomial ---")
    if degree == 3:
        print(f"Coefficients: c0=0.5, c1={coefs[0]:.6f}, c3={coefs[1]:.6f}")
    else:
        print(f"Coefficients: c0=0.5, c1={coefs[0]:.6f}, c3={coefs[1]:.6f}, c5={coefs[2]:.6f}")
    
    # Approx error on fitting interval
    y_approx_fit = func(x_fit, *coefs)
    err_fit = np.abs(y_fit - y_approx_fit)
    print(f"Max error on [-6.0, 6.0]: {err_fit.max():.6f}")
    print(f"Mean error on [-6.0, 6.0]: {err_fit.mean():.6f}")
    
    # Approx error on actual test logits
    y_true_test = sigmoid(logits_test)
    y_approx_test = func(logits_test, *coefs)
    err_test = np.abs(y_true_test - y_approx_test)
    print(f"Max error on test logits: {err_test.max():.6f}")
    print(f"Mean error on test logits: {err_test.mean():.6f}")
    
    # Evaluate plaintext classification with this approx
    y_pred = (y_approx_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_approx_test)
    print(f"Plaintext Accuracy (Poly{degree}): {acc:.4f}")
    print(f"Plaintext ROC-AUC  (Poly{degree}): {auc:.4f}")

# Save the MIMIC-specific coefficients
poly_config = {
    "fitting_interval": [-6.0, 6.0],
    "degree_3": {
        "c0": 0.5,
        "c1": float(c1_3),
        "c3": float(c3_3)
    },
    "degree_5": {
        "c0": 0.5,
        "c1": float(c1_5),
        "c3": float(c3_5),
        "c5": float(c5_5)
    }
}

with open(os.path.join(he_dir, "mimic_polynomials.json"), "w") as f:
    json.dump(poly_config, f, indent=4)

print("\nSaved mimic-specific coefficients to results/he_ready/mimic_polynomials.json")
