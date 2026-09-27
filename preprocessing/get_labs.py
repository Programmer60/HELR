import pandas as pd
import os
data_dir=r'C:\Users\mishr\Desktop\ML\venv\MinorProject\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4'
df = pd.read_csv(os.path.join(data_dir, 'D_LABITEMS.csv.gz'), compression='gzip')
df.columns = df.columns.str.lower()
keywords = ['glucose', 'creatinine', 'sodium', 'potassium', 'hemoglobin', 'white blood cell', 'wbc', 'platelet']
matches = df[df['label'].str.contains('|'.join(keywords), case=False, na=False)]
matches = matches[matches['fluid'].str.contains('blood', case=False, na=False)]
for _, row in matches.iterrows():
    print(f"{row['itemid']} | {row['label']} | {row['fluid']} | {row['category']}")
