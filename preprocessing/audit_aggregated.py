"""
PHASE 1 — Post-Aggregation Audit
Reads mimic3_aggregated_features.csv and produces a full data-quality report.
"""
import pandas as pd
import numpy as np
import os

csv_path = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\preprocessing\mimic3_aggregated_features.csv"
out_dir  = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\preprocessing"

df = pd.read_csv(csv_path, low_memory=False)

# ---------- 1. SCHEMA ---------------------------------------------------
print("=" * 70)
print("1. SCHEMA & DTYPES")
print("=" * 70)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print()
for col in df.columns:
    print(f"  {col:<30s}  dtype={str(df[col].dtype):<12s}  nulls={df[col].isna().sum()}")

# ---------- 2. KEY INTEGRITY -------------------------------------------
print()
print("=" * 70)
print("2. KEY INTEGRITY")
print("=" * 70)
print(f"Total rows              : {len(df)}")
print(f"Unique HADM_ID          : {df['HADM_ID'].nunique()}")
print(f"Duplicate HADM_ID rows  : {df.duplicated(subset=['HADM_ID']).sum()}")
print(f"Unique SUBJECT_ID       : {df['SUBJECT_ID'].nunique()}")
multi = df.groupby('SUBJECT_ID')['HADM_ID'].nunique()
print(f"Subjects with >1 admission: {(multi > 1).sum()}")
print()
print("Target distribution:")
vc = df['is_target'].value_counts().sort_index()
for k, v in vc.items():
    print(f"  target={k}  count={v}  ({v/len(df)*100:.2f}%)")

# ---------- 3 & 4. SUSPICIOUS MINS — detailed percentiles -------------
print()
print("=" * 70)
print("3-4. SUSPICIOUS MINIMUM VALUES — PERCENTILE ANALYSIS")
print("=" * 70)

suspicious = {
    "Creatinine":    {"warn_below": 0.2,  "pctiles": [1, 5, 25, 50, 75, 95, 99]},
    "Systolic BP":   {"warn_below": 50,   "pctiles": [1, 5, 25, 50, 75, 95, 99]},
    "Diastolic BP":  {"warn_below": 20,   "pctiles": [1, 5, 25, 50, 75, 95, 99]},
    "Mean Arterial Pressure": {"warn_below": 30, "pctiles": [1, 5, 25, 50, 75, 95, 99]},
    "Temperature":   {"warn_below": 34,   "pctiles": [1, 5, 25, 50, 75, 95, 99]},
}

for feat, cfg in suspicious.items():
    col = feat
    if col not in df.columns:
        print(f"\n  [MISSING column: {col}]")
        continue
    s = df[col].dropna()
    threshold = cfg["warn_below"]
    below = (s < threshold).sum()
    print(f"\n  Feature   : {feat}")
    print(f"  n (non-null): {len(s)}")
    print(f"  Min       : {s.min():.4f}")
    print(f"  Max       : {s.max():.4f}")
    pcts = np.percentile(s, cfg["pctiles"])
    for pct, val in zip(cfg["pctiles"], pcts):
        print(f"  P{pct:<2d}       : {val:.4f}")
    print(f"  Below {threshold} (warning threshold): {below} admissions ({below/len(s)*100:.3f}%)")

# ---------- 5. EXTREME HIGH VALUES -----------------------------------
print()
print("=" * 70)
print("5. EXTREME HIGH LABORATORY VALUES — FREQUENCY & PERCENTILE POSITION")
print("=" * 70)

extremes = {
    "WBC":      {"warn_above": 100,  "pctiles": [95, 99, 99.5, 99.9]},
    "Platelets":{"warn_above": 1000, "pctiles": [95, 99, 99.5, 99.9]},
    "Creatinine":{"warn_above": 10,  "pctiles": [95, 99, 99.5, 99.9]},
    "Glucose":  {"warn_above": 500,  "pctiles": [95, 99, 99.5, 99.9]},
}

for feat, cfg in extremes.items():
    if feat not in df.columns:
        print(f"\n  [MISSING column: {feat}]")
        continue
    s = df[feat].dropna()
    threshold = cfg["warn_above"]
    above = (s > threshold).sum()
    pcts = np.percentile(s, cfg["pctiles"])
    print(f"\n  Feature   : {feat}")
    print(f"  n (non-null): {len(s)}")
    print(f"  Min: {s.min():.2f}  |  Max: {s.max():.2f}")
    for pct, val in zip(cfg["pctiles"], pcts):
        print(f"  P{pct:<4}    : {val:.2f}")
    print(f"  Above {threshold} (extreme threshold): {above} admissions ({above/len(s)*100:.3f}%)")

# ---------- 6. NaN / INF / DTYPE CHECKS ------------------------------
print()
print("=" * 70)
print("6. NaN / INF / UNEXPECTED DTYPE CHECKS")
print("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {len(numeric_cols)}")
print(f"Non-numeric columns: {[c for c in df.columns if c not in numeric_cols]}")
print()

inf_found = []
for col in numeric_cols:
    n_inf = np.isinf(df[col].replace([None], np.nan).astype(float)).sum()
    if n_inf > 0:
        inf_found.append((col, n_inf))

if inf_found:
    print("Inf / -Inf values found:")
    for col, n in inf_found:
        print(f"  {col}: {n}")
else:
    print("No inf or -inf values found. [OK]")

dup_cols = [c for c in df.columns if df.columns.tolist().count(c) > 1]
if dup_cols:
    print(f"\nDuplicated column names: {dup_cols}")
else:
    print("No duplicated column names. [OK]")

print()
total_nan = df[numeric_cols].isna().sum().sum()
print(f"Total NaN cells across all numeric columns: {total_nan}")
print()

# Missingness table for report
miss = df.isna().sum()
miss_pct = (df.isna().mean() * 100).round(2)
miss_df = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
miss_df = miss_df[miss_df["missing_count"] > 0].sort_values("missing_pct")
print("Full missingness table:")
print(miss_df.to_string())

# ---------- SAVE AUDIT REPORT ----------------------------------------
audit_path = os.path.join(out_dir, "audit_report.txt")
import io, sys
# already printed above; just save csv summary
miss_df.to_csv(os.path.join(out_dir, "audit_missingness.csv"))
print(f"\nAudit missingness saved to audit_missingness.csv")
print("\nPHASE 1 AUDIT COMPLETE.")
