# MHEALTH Dataset Preprocessing and Plaintext Baseline

## 1. What is the MHEALTH Dataset?
The MHEALTH (Mobile HEALTH) dataset is a public dataset comprising body motion and vital signs recordings for 10 volunteers of diverse profiles while performing 12 physical activities. It captures multimodal sensor data in an out-of-lab environment, providing a robust testbed for human activity recognition and physiological monitoring tasks.

## 2. Why We Selected It
For our "End-to-End Privacy-Preserving Machine Learning for CIED-Oriented Cardiac/Physiological Remote Monitoring" project, we need a dataset that allows us to build and validate our Homomorphic Encryption (HE) pipeline on multimodal time-series data. MHEALTH includes 2-lead ECG and 3D accelerometers/gyroscopes/magnetometers, mimicking the type of multisensor telemetry that next-generation CIEDs (Cardiac Implantable Electronic Devices) or wearable monitors might transmit. It acts as an authentic, manageable **Small Development Benchmark**.

## 3. Why It Is a Benchmark and Not Actual CIED Data
MHEALTH was collected via external Shimmer2 wearable sensors placed on the chest, right wrist, and left ankle—not via an implanted pacemaker, defibrillator, or internal loop recorder (CIED). While it contains real ECG and physiological motion, it lacks the specific noise profiles, hardware constraints, and clinical context of true CIED telemetry. We are using it strictly to validate our *cryptographic architecture* before progressing to larger or more authentic medical datasets.

## 4. Raw Data Structure
The dataset contains 10 `.log` files, one for each subject. 
- **Total Samples:** 1,215,745
- **Samples per Subject:** Varies from ~98k to ~161k.
- **Missing Values:** None found in the raw logs.
- **Labels:** 0 is the "null" (unlabelled/transition) class. Classes 1 through 12 represent specific activities (e.g., standing, walking, cycling, running). Each active class has approximately 30,000 samples overall.

## 5. Sensor Modalities
There are 23 continuous sensor channels collected at 50 Hz:
- **Chest:** Accelerometer (X, Y, Z), ECG (Lead 1, Lead 2)
- **Left Ankle:** Accelerometer, Gyroscope, Magnetometer (each X, Y, Z)
- **Right Lower Arm:** Accelerometer, Gyroscope, Magnetometer (each X, Y, Z)

## 6. Windowing
To convert continuous time-series data into discrete samples suitable for Logistic Regression, we applied fixed-length sliding windows:
- **Window Size:** 128 samples (~2.56 seconds at 50 Hz)
- **Stride:** 64 samples (50% overlap)
- **Rule:** Windows were only extracted within contiguous blocks of the *same* activity class. We completely ignored the "null" class (0) to ensure clean training examples. Windows never crossed subject boundaries.

## 7. Feature Extraction
For this initial benchmark, we extracted simple, compact statistical features from each window to keep the dimensionality manageable for CKKS:
- `mean`
- `std` (standard deviation)
- `min`
- `max`
With 23 sensor channels and 4 statistics, this produces **92 numerical features** per window. We deliberately avoided complex frequency-domain or RR-interval feature engineering at this stage to prioritize establishing the HE pipeline.

## 8. Subject-Level Split
To prevent data leakage, we enforced a strict **subject-wise split**:
- **Train Subjects:** [1, 2, 3, 4, 5, 6, 7]
- **Test Subjects:** [8, 9, 10]
This ensures that the model is evaluated on completely unseen individuals, mimicking a real-world scenario where a new patient's data is sent to the cloud.

## 9. Normalization
Cryptographic stability in CKKS requires bounded input magnitudes. We applied `StandardScaler` (zero mean, unit variance):
- The scaler was **fitted strictly on the training subjects**.
- The learned transform was then applied to both the training and testing sets.

## 10. Resulting Dataset Shapes
For our first experiment, we simplified the 12-class problem into a **Binary Classification Task**:
- **Class 0 (Inactive):** Sitting and relaxing (L2), Lying down (L3)
- **Class 1 (Active):** Walking (L4), Jogging (L10), Running (L11)

Final processed dataset dimensions:
- **X_train:** (1645, 92)
- **X_test:** (705, 92)

## 11. Plaintext Baseline
We trained a standard scikit-learn `LogisticRegression` model on the extracted features.
- **Accuracy:** 1.0000
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1 Score:** 1.0000
- **ROC-AUC:** 1.0000

*Note:* The perfect separation is expected because distinguishing lying/sitting from walking/running using 92 statistical features across 3 body sensors is a trivially separable task. This is highly beneficial for our first HE test, as any drop in accuracy during encrypted training will be purely attributable to cryptographic noise or polynomial approximation errors, not inherent data ambiguity.

## 12. Issues/Limitations
- **Extreme Separability:** The binary task is too easy for a robust ML challenge, though ideal for a cryptographic proof-of-concept.
- **Feature Sparsity:** We are generating 92 features. In HE, more features mean a wider dot-product (more multiplications). We may need to use feature selection (e.g., PCA or L1 regularization) if the encrypted training becomes too slow or consumes too much depth.
- **Simplistic ECG Use:** We are just taking the mean/std/min/max of the raw ECG signal, which ignores the actual morphological (QRS complex) and temporal (Heart Rate) value of the ECG. 

## 13. Next Step Toward CKKS Encrypted Training
Now that the plaintext baseline is established, our next step is to implement the **End-to-End Encrypted Training Pipeline**. We will:
1. Encrypt `X_train` and `y_train` using CKKS.
2. Initialize and encrypt the weight vector `w`.
3. Implement the encrypted forward pass (using a polynomial sigmoid approximation).
4. Implement the encrypted gradient computation.
5. Execute multiple training epochs entirely in the ciphertext domain.
6. Decrypt the final model and evaluate its performance against this 100% accurate plaintext baseline.
