import pandas as pd
import os
data_dir=r'C:\Users\mishr\Desktop\ML\venv\MinorProject\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4'
df = pd.read_csv(os.path.join(data_dir, 'D_ITEMS.csv.gz'), compression='gzip')
df.columns = df.columns.str.lower()
df = df[df['linksto'] == 'chartevents']

vital_vars = {
    "heart rate": ["heart rate"],
    "systolic bp": ["systolic", "sbp"],
    "diastolic bp": ["diastolic", "dbp"],
    "mean arterial pressure": ["mean arterial", "map"],
    "respiratory rate": ["respiratory rate"],
    "temperature": ["temperature c", "temperature f", "temperature"],
    "oxygen saturation": ["spo2", "o2 saturation"]
}

for var, keywords in vital_vars.items():
    print(f"\n--- {var.upper()} ---")
    matches = df[df['label'].str.contains('|'.join(keywords), case=False, na=False)]
    for _, row in matches.iterrows():
        # exclude some obvious noise
        if "alarm" in str(row['label']).lower() or "score" in str(row['label']).lower():
            continue
        print(f"{row['itemid']} | {row['label']} | {row['category']} | {row.get('unitname', 'N/A')}")
