import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_col_names():
    return [
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

def extract_features(window_data):
    # window_data shape: (window_size, num_sensor_channels)
    # Extracts basic statistical features
    features = []
    
    # mean
    features.extend(np.mean(window_data, axis=0))
    # std
    features.extend(np.std(window_data, axis=0))
    # min
    features.extend(np.min(window_data, axis=0))
    # max
    features.extend(np.max(window_data, axis=0))
    
    return np.array(features)

def create_windows(df, subject_id, window_size=128, stride=64):
    windows = []
    labels = []
    subjects = []
    
    # Drop null class (0) for cleaner windows. Or we can just window over everything and drop mixed/null windows.
    # It's better to window continuous segments of the SAME activity.
    # We find contiguous segments of the same label.
    
    # Identify segment boundaries
    df['segment'] = (df['label'] != df['label'].shift()).cumsum()
    
    for _, segment_df in df.groupby('segment'):
        seg_label = segment_df['label'].iloc[0]
        if seg_label == 0:
            continue # Skip null class
            
        data_matrix = segment_df.drop(columns=['label', 'segment', 'subject_id'], errors='ignore').values
        num_samples = len(data_matrix)
        
        for start in range(0, num_samples - window_size + 1, stride):
            end = start + window_size
            window_data = data_matrix[start:end]
            
            features = extract_features(window_data)
            windows.append(features)
            labels.append(seg_label)
            subjects.append(subject_id)
            
    return windows, labels, subjects

def main():
    data_dir = "MHEALTHDATASET"
    out_dir = "data/processed/mhealth"
    os.makedirs(out_dir, exist_ok=True)
    
    # Configuration
    train_subjects = [1, 2, 3, 4, 5, 6, 7]
    test_subjects = [8, 9, 10]
    window_size = 128  # ~2.56 seconds at 50Hz
    stride = 64        # 50% overlap
    
    col_names = get_col_names()
    
    log_files = glob.glob(os.path.join(data_dir, "mHealth_subject*.log"))
    
    all_features = []
    all_labels = []
    all_subjects = []
    
    print("Windowing and extracting features...")
    for file in sorted(log_files):
        subject_id = int(os.path.basename(file).split('subject')[1].split('.log')[0])
        
        df = pd.read_csv(file, sep='\t', header=None, names=col_names)
        df['subject_id'] = subject_id
        
        w, l, s = create_windows(df, subject_id, window_size, stride)
        all_features.extend(w)
        all_labels.extend(l)
        all_subjects.extend(s)
        print(f"  Subject {subject_id}: {len(w)} windows generated.")
        
    X = np.array(all_features)
    y = np.array(all_labels)
    subj = np.array(all_subjects)
    
    # Create mask for train and test
    train_mask = np.isin(subj, train_subjects)
    test_mask = np.isin(subj, test_subjects)
    
    X_train, y_train, subj_train = X[train_mask], y[train_mask], subj[train_mask]
    X_test, y_test, subj_test = X[test_mask], y[test_mask], subj[test_mask]
    
    print(f"\nTrain set: {X_train.shape[0]} windows from subjects {train_subjects}")
    print(f"Test set: {X_test.shape[0]} windows from subjects {test_subjects}")
    
    # Simplify label mapping to a binary classification task for the first HE test.
    # Let's say: Static/Low-intensity (1,2,3) vs High-intensity/Dynamic (4-12)
    # Alternatively, just pick two classes, e.g., L1 (Standing still = 1) vs L11 (Running = 11).
    # Let's do binary: Class 1 (Standing still) -> 0, Class 11 (Running) -> 1.
    # Wait, the prompt says: "For the first encrypted Logistic Regression experiment, we will probably use a simple binary classification problem, but the exact class mapping should be chosen after inspecting the actual label distribution."
    # Let's map Sitting/Lying down (2,3) -> 0, Walking/Running/Jogging (4, 10, 11) -> 1.
    # Filter the dataset to just these classes.
    valid_classes = [2, 3, 4, 10, 11]
    
    train_valid_mask = np.isin(y_train, valid_classes)
    test_valid_mask = np.isin(y_test, valid_classes)
    
    X_train = X_train[train_valid_mask]
    y_train = y_train[train_valid_mask]
    subj_train = subj_train[train_valid_mask]
    
    X_test = X_test[test_valid_mask]
    y_test = y_test[test_valid_mask]
    subj_test = subj_test[test_valid_mask]
    
    # Map labels: 2,3 -> 0 (Inactive), 4,10,11 -> 1 (Active)
    y_train_binary = np.where(np.isin(y_train, [2,3]), 0, 1)
    y_test_binary = np.where(np.isin(y_test, [2,3]), 0, 1)
    
    print(f"\nAfter binary mapping (Inactive vs Active):")
    print(f"Train set: {X_train.shape[0]} windows")
    print(f"Test set: {X_test.shape[0]} windows")
    
    # Normalization
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    np.save(os.path.join(out_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train_binary)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test_binary)
    np.save(os.path.join(out_dir, "subject_train.npy"), subj_train)
    np.save(os.path.join(out_dir, "subject_test.npy"), subj_test)
    
    # Save metadata
    feature_names = []
    sensor_cols = get_col_names()[:-1]
    for stat in ["mean", "std", "min", "max"]:
        for col in sensor_cols:
            feature_names.append(f"{col}_{stat}")
            
    metadata = {
        "source_dataset": "MHEALTH",
        "window_size": window_size,
        "stride": stride,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "original_classes_used": valid_classes,
        "label_mapping": {"Inactive (Sitting/Lying)": 0, "Active (Walking/Jogging/Running)": 1},
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "train_samples": len(X_train_scaled),
        "test_samples": len(X_test_scaled)
    }
    
    import joblib
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Saved all processed data to {out_dir}")

if __name__ == "__main__":
    main()
