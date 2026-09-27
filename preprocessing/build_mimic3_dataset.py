import pandas as pd
import numpy as np
import gzip
import os
import json

data_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4"
output_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\preprocessing"

# Approved ITEMIDs
ITEMIDS = {
    # Labs
    50931: "Glucose", 50809: "Glucose",
    50912: "Creatinine",
    50983: "Sodium", 50824: "Sodium",
    50971: "Potassium", 50822: "Potassium",
    51222: "Hemoglobin", 50811: "Hemoglobin",
    51301: "WBC",
    51265: "Platelets",
    
    # Vitals
    211: "Heart Rate", 220045: "Heart Rate",
    51: "Systolic BP", 220050: "Systolic BP", 455: "Systolic BP", 220179: "Systolic BP",
    8368: "Diastolic BP", 220051: "Diastolic BP", 8441: "Diastolic BP", 220180: "Diastolic BP",
    52: "Mean Arterial Pressure", 220052: "Mean Arterial Pressure", 456: "Mean Arterial Pressure", 220181: "Mean Arterial Pressure",
    618: "Respiratory Rate", 220210: "Respiratory Rate", 224689: "Respiratory Rate", 224690: "Respiratory Rate",
    676: "Temperature", 223762: "Temperature", 678: "Temperature", 223761: "Temperature",
    646: "Oxygen Saturation", 220277: "Oxygen Saturation"
}

# Plausibility Filters
def check_plausibility(row):
    feat = row['Feature']
    v = row['VALUENUM']
    if feat == "Heart Rate" and not (0 < v <= 300): return False
    if feat == "Systolic BP" and not (0 < v <= 350): return False
    if feat == "Diastolic BP" and not (0 < v <= 250): return False
    if feat == "Mean Arterial Pressure" and not (0 < v <= 300): return False
    if feat == "Respiratory Rate" and not (0 < v <= 120): return False
    if feat == "Temperature" and not (25 <= v <= 45): return False
    if feat == "Oxygen Saturation" and not (30 <= v <= 100): return False
    if feat == "Sodium" and not (80 <= v <= 200): return False
    if feat == "Potassium" and not (0.5 <= v <= 15): return False
    if feat == "Hemoglobin" and not (0 < v <= 30): return False
    return True

stats = {
    "total_rows_read": 0,
    "removed_by_itemid": 0,
    "removed_by_missing_valuenum": 0,
    "removed_by_time_window": 0,
    "removed_by_plausibility": 0,
    "feature_plausibility_drops": {},
    "feature_total_counts": {}
}

def analyze_cohort():
    print("Building cohort...")
    admissions = pd.read_csv(os.path.join(data_dir, "ADMISSIONS.csv.gz"), compression='gzip', 
                             usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'HAS_CHARTEVENTS_DATA'])
    patients = pd.read_csv(os.path.join(data_dir, "PATIENTS.csv.gz"), compression='gzip', 
                           usecols=['SUBJECT_ID', 'GENDER', 'DOB'])
    diagnoses = pd.read_csv(os.path.join(data_dir, "DIAGNOSES_ICD.csv.gz"), compression='gzip', 
                            usecols=['HADM_ID', 'ICD9_CODE'])
                            
    # Target Construction
    diagnoses = diagnoses.dropna(subset=['ICD9_CODE'])
    diagnoses['is_target'] = diagnoses['ICD9_CODE'].astype(str).str.startswith(('410', '411', '412', '413', '414')).astype(int)
    target_labels = diagnoses.groupby('HADM_ID')['is_target'].max().reset_index()
    
    # Cohort join
    df = admissions[admissions['HAS_CHARTEVENTS_DATA'] == 1].copy()
    df = df.merge(patients, on='SUBJECT_ID', how='inner')
    df = df.merge(target_labels, on='HADM_ID', how='left')
    df['is_target'] = df['is_target'].fillna(0).astype(int)
    
    # Calculate Age safely
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['DOB'] = pd.to_datetime(df['DOB'])
    years = df['ADMITTIME'].dt.year - df['DOB'].dt.year
    days = df['ADMITTIME'].dt.dayofyear - df['DOB'].dt.dayofyear
    df['AGE'] = years + (days / 365.242)
    
    num_over_89 = (df['AGE'] > 89).sum()
    print(f"Admissions with calculated Age > 89: {num_over_89}")
    df.loc[df['AGE'] > 89, 'AGE'] = 90
    
    df = df[df['AGE'] >= 18]
    print(f"Final Adult Cohort (Age >= 18): {len(df)}")
    return df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'AGE', 'GENDER', 'is_target']]

def process_events(filename, cohort_df, chunksize=1000000):
    print(f"\nProcessing {filename} ...")
    path = os.path.join(data_dir, filename)
    
    valid_hadm = set(cohort_df['HADM_ID'])
    cohort_times = cohort_df.set_index('HADM_ID')['ADMITTIME']
    
    all_retained = []
    
    reader = pd.read_csv(path, compression='gzip', chunksize=chunksize, 
                         usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
                         dtype={'HADM_ID': 'Int64', 'ITEMID': 'Int64', 'VALUENUM': float})
    
    for chunk in reader:
        stats["total_rows_read"] += len(chunk)
        
        # 1. Filter by ITEMID
        c = chunk[chunk['ITEMID'].isin(ITEMIDS.keys())].copy()
        c = c[c['HADM_ID'].isin(valid_hadm)]
        dropped_itemid = len(chunk) - len(c)
        stats["removed_by_itemid"] += dropped_itemid
        
        if len(c) == 0: continue
        
        # 2. Missing VALUENUM
        c_valid = c.dropna(subset=['VALUENUM']).copy()
        stats["removed_by_missing_valuenum"] += (len(c) - len(c_valid))
        
        if len(c_valid) == 0: continue
            
        # 3. Time Window
        c_valid['CHARTTIME'] = pd.to_datetime(c_valid['CHARTTIME'])
        c_valid = c_valid.merge(cohort_times, left_on='HADM_ID', right_index=True)
        time_diff = (c_valid['CHARTTIME'] - c_valid['ADMITTIME']).dt.total_seconds() / 3600
        
        c_time = c_valid[(time_diff >= 0) & (time_diff <= 24)].copy()
        stats["removed_by_time_window"] += (len(c_valid) - len(c_time))
        
        if len(c_time) > 0:
            c_time['Feature'] = c_time['ITEMID'].map(ITEMIDS)
            
            # Temp conversion
            mask_f = c_time['ITEMID'].isin([678, 223761])
            if mask_f.any():
                c_time.loc[mask_f, 'VALUENUM'] = (c_time.loc[mask_f, 'VALUENUM'] - 32) * 5/9
            
            all_retained.append(c_time[['HADM_ID', 'ITEMID', 'Feature', 'VALUENUM']])
            
    if all_retained:
        return pd.concat(all_retained, ignore_index=True)
    return pd.DataFrame()

def apply_plausibility(events):
    print("Applying plausibility filters...")
    mask = events.apply(check_plausibility, axis=1)
    
    dropped = events[~mask]
    kept = events[mask]
    
    stats["removed_by_plausibility"] = len(dropped)
    
    # Stats per feature
    for feat in events['Feature'].unique():
        f_all = len(events[events['Feature'] == feat])
        f_drop = len(dropped[dropped['Feature'] == feat])
        stats["feature_total_counts"][feat] = f_all
        stats["feature_plausibility_drops"][feat] = f_drop
        
    return kept

if __name__ == "__main__":
    cohort = analyze_cohort()
    
    labs = process_events("LABEVENTS.csv.gz", cohort)
    vitals = process_events("CHARTEVENTS.csv.gz", cohort)
    
    all_events = pd.concat([labs, vitals], ignore_index=True)
    all_events.to_csv(os.path.join(output_dir, "mimic3_intermediate_events.csv"), index=False)
    
    cleaned_events = apply_plausibility(all_events)
    cleaned_events.to_csv(os.path.join(output_dir, "mimic3_intermediate_cleaned_events.csv"), index=False)
    
    # Aggregation (Mean)
    print("Aggregating features...")
    agg = cleaned_events.groupby(['HADM_ID', 'Feature'])['VALUENUM'].mean().unstack('Feature')
    
    # Merge with cohort
    final = cohort.merge(agg, on='HADM_ID', how='left')
    
    # Encode Gender
    final['GENDER'] = (final['GENDER'] == 'M').astype(int)
    
    final.to_csv(os.path.join(output_dir, "mimic3_aggregated_features.csv"), index=False)
    
    with open(os.path.join(output_dir, "processing_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    # Print Final Report
    print("\n--- FINAL DATASET-QUALITY REPORT ---")
    print(f"Total Admissions: {len(final)}")
    print(f"Target Positive (1): {(final['is_target'] == 1).sum()}")
    print(f"Target Negative (0): {(final['is_target'] == 0).sum()}")
    print(f"Duplicate HADM_ID rows: {final['HADM_ID'].duplicated().sum()}")
    
    print("\nMissing Values & Percentages:")
    missing = final.isna().sum()
    pct = final.isna().mean() * 100
    for col in final.columns:
        if missing[col] > 0 or col in ITEMIDS.values():
            print(f"{col:25s} | Missing: {missing[col]:5d} | {pct[col]:.1f}%")
            
    print("\nFeature Distributions (Mean/Std/Min/Max):")
    features = ['AGE'] + list(set(ITEMIDS.values()))
    for f in features:
        if f in final.columns:
            s = final[f].dropna()
            print(f"{f:25s} | Mean: {s.mean():.2f} | Std: {s.std():.2f} | Min: {s.min():.2f} | Max: {s.max():.2f}")
    
    print("\nFiltering Statistics:")
    print(f"Total rows read: {stats['total_rows_read']}")
    print(f"Removed by ITEMID/Cohort: {stats['removed_by_itemid']}")
    print(f"Removed by missing VALUENUM: {stats['removed_by_missing_valuenum']}")
    print(f"Removed by Time Window: {stats['removed_by_time_window']}")
    print(f"Removed by Plausibility Filters: {stats['removed_by_plausibility']}")
    
    print("\nPlausibility Drop Rates by Feature:")
    for feat, total in stats["feature_total_counts"].items():
        if total > 0:
            drops = stats["feature_plausibility_drops"].get(feat, 0)
            rate = (drops / total) * 100
            print(f"{feat:25s} | Drops: {drops:6d} out of {total:6d} | {rate:.2f}%")
