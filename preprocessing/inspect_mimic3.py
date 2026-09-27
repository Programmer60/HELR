import pandas as pd
import gzip
import os

data_dir = r"C:\Users\mishr\Desktop\ML\venv\MinorProject\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4"

def inspect_schema():
    files = ["PATIENTS.csv.gz", "ADMISSIONS.csv.gz", "DIAGNOSES_ICD.csv.gz", 
             "ICUSTAYS.csv.gz", "D_LABITEMS.csv.gz", "D_ITEMS.csv.gz"]
    
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=5, compression='gzip')
            print(f"\n--- Schema for {f} ---")
            print(df.columns.tolist())
            print(df.dtypes)
        else:
            print(f"File {f} not found!")

def search_dictionaries():
    print("\n\n--- DICTIONARY SEARCH ---")
    
    lab_vars = {
        "glucose": ["glucose"],
        "creatinine": ["creatinine"],
        "sodium": ["sodium"],
        "potassium": ["potassium"],
        "hemoglobin": ["hemoglobin"],
        "wbc": ["white blood cell", "wbc"],
        "platelets": ["platelet"]
    }
    
    vital_vars = {
        "heart rate": ["heart rate"],
        "systolic bp": ["systolic", "sbp"],
        "diastolic bp": ["diastolic", "dbp"],
        "mean arterial pressure": ["mean arterial", "map"],
        "respiratory rate": ["respiratory rate"],
        "temperature": ["temperature f", "temperature c"],
        "oxygen saturation": ["spo2", "o2 saturation", "oxygen saturation"]
    }

    # Search D_LABITEMS
    d_lab_path = os.path.join(data_dir, "D_LABITEMS.csv.gz")
    d_lab = pd.read_csv(d_lab_path, compression='gzip')
    d_lab.columns = d_lab.columns.str.lower()
    
    print("\n--- D_LABITEMS Matches ---")
    for var, keywords in lab_vars.items():
        matches = d_lab[d_lab['label'].str.contains('|'.join(keywords), case=False, na=False)]
        # Filter mostly for Blood/fluid
        matches = matches[matches['fluid'].str.contains('blood', case=False, na=False)]
        for _, row in matches.iterrows():
            print(f"LAB | {var} | {row['itemid']} | {row['label']} | {row['fluid']} | {row['category']}")

    # Search D_ITEMS
    d_items_path = os.path.join(data_dir, "D_ITEMS.csv.gz")
    d_items = pd.read_csv(d_items_path, compression='gzip')
    d_items.columns = d_items.columns.str.lower()
    
    print("\n--- D_ITEMS Matches ---")
    for var, keywords in vital_vars.items():
        matches = d_items[d_items['label'].str.contains('|'.join(keywords), case=False, na=False)]
        # Filter to linksto == chartevents
        matches = matches[matches['linksto'] == 'chartevents']
        for _, row in matches.iterrows():
            print(f"VITAL | {var} | {row['itemid']} | {row['label']} | {row['category']} | {row.get('unitname', 'N/A')}")

if __name__ == "__main__":
    inspect_schema()
    search_dictionaries()
