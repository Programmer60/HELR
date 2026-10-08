import os
import glob
import json
import numpy as np
import pandas as pd

def main():
    data_dir = "MHEALTHDATASET"
    log_files = glob.glob(os.path.join(data_dir, "mHealth_subject*.log"))
    
    col_names = [
        "chest_acc_x", "chest_acc_y", "chest_acc_z",
        "ecg_lead_1", "ecg_lead_2",
        "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
        "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
        "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
        "arm_acc_x", "arm_acc_y", "arm_acc_z",
        "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
        "arm_mag_x", "arm_mag_y", "arm_mag_z",
        "label"
    ]
    
    all_data = []
    subject_sample_counts = {}
    subject_label_distributions = {}
    
    for file in sorted(log_files):
        # Extract subject ID
        subject_id = int(os.path.basename(file).split('subject')[1].split('.log')[0])
        
        # Read the file
        df = pd.read_csv(file, sep='\t', header=None, names=col_names)
        
        # Add subject column for convenience
        df['subject_id'] = subject_id
        
        subject_sample_counts[subject_id] = len(df)
        subject_label_distributions[subject_id] = df['label'].value_counts().to_dict()
        
        all_data.append(df)
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    stats = {
        "num_subjects": len(log_files),
        "total_raw_samples": len(full_df),
        "samples_per_subject": subject_sample_counts,
        "num_sensor_channels": len(col_names) - 1, # excluding label
        "label_distribution_overall": full_df['label'].value_counts().to_dict(),
        "label_distribution_per_subject": subject_label_distributions,
        "missing_value_counts": full_df.isnull().sum().to_dict(),
        "duplicate_rows": int(full_df.duplicated().sum()),
    }
    
    # Sensor stats (min, max, mean, std)
    sensor_cols = col_names[:-1]
    sensor_stats = {}
    for col in sensor_cols:
        sensor_stats[col] = {
            "min": float(full_df[col].min()),
            "max": float(full_df[col].max()),
            "mean": float(full_df[col].mean()),
            "std": float(full_df[col].std())
        }
    stats["sensor_statistics"] = sensor_stats
    
    os.makedirs("results", exist_ok=True)
    with open("results/mhealth_dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Dataset statistics saved to results/mhealth_dataset_statistics.json")
    
if __name__ == "__main__":
    main()
