import pandas as pd
import numpy as np
import gzip
import os

data_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4"

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

def analyze_cohort():
    print("--- Cohort Analysis ---")
    admissions = pd.read_csv(os.path.join(data_dir, "ADMISSIONS.csv.gz"), compression='gzip', 
                             usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'HAS_CHARTEVENTS_DATA'])
    patients = pd.read_csv(os.path.join(data_dir, "PATIENTS.csv.gz"), compression='gzip', 
                           usecols=['SUBJECT_ID', 'GENDER', 'DOB'])
    
    # Keep only those with chartevents
    df = admissions[admissions['HAS_CHARTEVENTS_DATA'] == 1].copy()
    df = df.merge(patients, on='SUBJECT_ID', how='inner')
    
    # Calculate Age safely (avoiding timedelta64[ns] 292-year overflow)
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['DOB'] = pd.to_datetime(df['DOB'])
    
    # Calculate approximate fractional years
    years = df['ADMITTIME'].dt.year - df['DOB'].dt.year
    days = df['ADMITTIME'].dt.dayofyear - df['DOB'].dt.dayofyear
    df['AGE'] = years + (days / 365.242)
    
    num_over_89 = (df['AGE'] > 89).sum()
    print(f"Total admissions with chartevents: {len(df)}")
    print(f"Age Distribution (All):\n{df['AGE'].describe()}")
    print(f"\nNumber of admissions with Age > 89 (de-identified as ~300yo): {num_over_89}")
    
    # For now, we will drop the >89 ones just to keep the dataframe clean, 
    # but the rule will be to cap at 90.
    df.loc[df['AGE'] > 89, 'AGE'] = 90
    
    df = df[df['AGE'] >= 18]
    print(f"\nAdult admissions after capping Age>89 to 90: {len(df)}")
    return df[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME']]

def dry_run_events(filename, cohort_df, num_chunks=3, chunksize=500000):
    print(f"\n--- Dry Run: {filename} ---")
    path = os.path.join(data_dir, filename)
    
    total_read = 0
    total_kept_itemid = 0
    total_kept_time = 0
    
    valid_hadm = set(cohort_df['HADM_ID'])
    cohort_times = cohort_df.set_index('HADM_ID')['ADMITTIME']
    
    # Trackers for stats
    all_retained = []
    
    reader = pd.read_csv(path, compression='gzip', chunksize=chunksize, 
                         usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM'],
                         dtype={'HADM_ID': 'Int64', 'ITEMID': 'Int64', 'VALUENUM': float, 'VALUEUOM': str})
    
    for i, chunk in enumerate(reader):
        if i >= num_chunks: break
        
        total_read += len(chunk)
        
        # 1. Filter by ITEMID
        c = chunk[chunk['ITEMID'].isin(ITEMIDS.keys())].copy()
        c = c[c['HADM_ID'].isin(valid_hadm)]
        total_kept_itemid += len(c)
        
        if len(c) == 0: continue
            
        # 2. Filter by Time Window
        c['CHARTTIME'] = pd.to_datetime(c['CHARTTIME'])
        c = c.merge(cohort_times, left_on='HADM_ID', right_index=True)
        time_diff = (c['CHARTTIME'] - c['ADMITTIME']).dt.total_seconds() / 3600
        
        # 0 to +24 hours
        c_time = c[(time_diff >= 0) & (time_diff <= 24)].copy()
        total_kept_time += len(c_time)
        
        all_retained.append(c_time)

    print(f"Rows Read: {total_read}")
    print(f"Rows Retained (ITEMID + valid HADM): {total_kept_itemid}")
    print(f"Rows Retained (24-hour window): {total_kept_time}")
    
    if not all_retained:
        print("No rows retained in dry run.")
        return
        
    res = pd.concat(all_retained, ignore_index=True)
    print(f"Unique Admissions in retained data: {res['HADM_ID'].nunique()}")
    print(f"Missing VALUENUM: {res['VALUENUM'].isna().sum()} out of {len(res)}")
    
    # Drop missing valuenum for stats
    res = res.dropna(subset=['VALUENUM'])
    
    # Map feature names
    res['Feature'] = res['ITEMID'].map(ITEMIDS)
    
    # Convert Temp F to C
    mask_f = res['ITEMID'].isin([678, 223761])
    if mask_f.any():
        res.loc[mask_f, 'VALUENUM'] = (res.loc[mask_f, 'VALUENUM'] - 32) * 5/9
        # Update UOM for clarity
        res.loc[mask_f, 'VALUEUOM'] = '?C'
    
    # Group and report stats
    print("\n--- VALUEUOM Distributions ---")
    for feature in res['Feature'].unique():
        f_data = res[res['Feature'] == feature]
        print(f"\n{feature}:")
        print(f_data['VALUEUOM'].value_counts(dropna=False).to_string())
        
        # Stats
        print(f"  Min : {f_data['VALUENUM'].min():.2f}")
        print(f"  Max : {f_data['VALUENUM'].max():.2f}")
        print(f"  Mean: {f_data['VALUENUM'].mean():.2f}")

if __name__ == "__main__":
    cohort = analyze_cohort()
    dry_run_events("LABEVENTS.csv.gz", cohort, num_chunks=2, chunksize=1000000)
    dry_run_events("CHARTEVENTS.csv.gz", cohort, num_chunks=2, chunksize=1000000)
