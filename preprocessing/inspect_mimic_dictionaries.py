import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab_dict", default=r"C:\Users\mishr\Desktop\ML\venv\MinorProject\mimic-iv 3.1\mimic-iv\hosp\d_labitems.csv.gz")
    parser.add_argument("--vital_dict", default=r"C:\Users\mishr\Desktop\ML\venv\MinorProject\mimic-iv 3.1\mimic-iv\icu\d_items.csv.gz")
    args = parser.parse_args()

    if not os.path.exists(args.lab_dict) or not os.path.exists(args.vital_dict):
        print(f"Error: Dictionary files not found.")
        print(f"Looked for: \n- {args.lab_dict}\n- {args.vital_dict}")
        print("Please download them from PhysioNet and place them in the correct directories.")
        return

    # Define candidate variables
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
        "oxygen saturation": ["o2 saturation", "spo2"]
    }

    results = []

    # Process Labs
    print(f"Loading {args.lab_dict}...")
    try:
        df_lab = pd.read_csv(args.lab_dict, compression='gzip')
        # d_labitems usually has: itemid, label, fluid, category, loinc_code
        for var, keywords in lab_vars.items():
            for kw in keywords:
                matches = df_lab[df_lab['label'].str.contains(kw, case=False, na=False)]
                for _, row in matches.iterrows():
                    results.append({
                        "clinical variable": var,
                        "itemid": row['itemid'],
                        "label": row['label'],
                        "category": row.get('category', ''),
                        "unit/valueuom": "N/A (check labevents)",
                        "source dictionary": "d_labitems",
                        "recommendation": "?" # To be filled by manual review
                    })
    except Exception as e:
        print(f"Error reading lab dict: {e}")

    # Process Vitals
    print(f"Loading {args.vital_dict}...")
    try:
        df_vital = pd.read_csv(args.vital_dict, compression='gzip')
        # d_items usually has: itemid, label, abbreviation, linksto, category, unitname, param_type, lownormalvalue, highnormalvalue
        for var, keywords in vital_vars.items():
            for kw in keywords:
                matches = df_vital[df_vital['label'].str.contains(kw, case=False, na=False)]
                for _, row in matches.iterrows():
                    # Filter to only chartevents if possible, to avoid huge noise from other tables
                    if 'linksto' in row and row['linksto'] != 'chartevents':
                        continue
                    results.append({
                        "clinical variable": var,
                        "itemid": row['itemid'],
                        "label": row['label'],
                        "category": row.get('category', ''),
                        "unit/valueuom": row.get('unitname', ''),
                        "source dictionary": "d_items",
                        "recommendation": "?"
                    })
    except Exception as e:
        print(f"Error reading vital dict: {e}")

    # Convert to Markdown table
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Just save it to a CSV for now so we can inspect it easily
        res_df.to_csv("dict_inspection_results.csv", index=False)
        print("\nSaved initial matches to dict_inspection_results.csv.")
        print("Please review and filter the best ITEMIDs, then we will create the final markdown table.")
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()
