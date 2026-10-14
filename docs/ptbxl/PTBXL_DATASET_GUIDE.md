# PTB-XL Dataset Guide: Comprehensive Reference for Privacy-Preserving Cardiac Evaluation

## 1. Overview and Project Role

The **PTB-XL** dataset (v1.0.3) is a large, clinically rich, publicly accessible database of standard 10-second 12-lead electrocardiograms (ECGs) collected from 18,869 patients over nearly seven years (October 1989 – June 1996) at the Physikalisch-Technische Bundesanstalt (PTB) and the Department of Cardiology and Angiology at Charité – Universitätsmedizin Berlin.

### Role in the Privacy-Preserving Machine Learning (PPML) Project
Our primary research objective is developing **end-to-end privacy-preserving machine learning for CIED-oriented physiological and cardiac telemetry** using CKKS homomorphic encryption (TenSEAL/SEAL). 

Prior project phases (Phase 0 through Phase 4) validated:
1. End-to-end encrypted training with encrypted feature matrices $\text{Enc}(X)$, labels $\text{Enc}(y)$, and model parameters $\text{Enc}(W)$.
2. Polynomial sigmoid approximations (Degrees 1, 3, and 5).
3. The causal discovery of modulus-level-aware circuit scheduling (showing that lower modulus levels accelerate the downstream $O(D)$ gradient workload).

However, those experiments were conducted on the **MHEALTH** wearable sensor dataset (body kinematics and 2-lead ECG monitoring). PTB-XL serves as an **independent, large-scale cardiac benchmark** to rigorously validate whether our modulus-level-aware PPML training architecture generalizes to real-world multi-lead clinical cardiac electrophysiology.

---

## 2. Dataset Scale: Official Metadata vs. Processed Cohort

It is essential to distinguish between the raw, dataset-wide figures reported in the official PTB-XL documentation and the filtered subset used in our binary classification task:

| Entity | Official Dataset Metadata | Processed PPML Cohort | Notes / Reason for Difference |
|:---|:---:|:---:|:---|
| **Total ECG Records** | **21,799** | **21,388** | 411 records excluded because their SCP-ECG statements did not map to any of the 5 official diagnostic superclasses. |
| **Unique Patients** | **18,869** | **18,617** | 252 patients had all their records fall into the unmapped category and were excluded. |
| **Normal Records (NORM)** | 9,514 (raw NORM statement) | 9,069 (label 0) | Strict NORM-only policy (records with co-occurring non-NORM diagnoses are labeled abnormal). |
| **Abnormal Records** | 12,285 | 12,319 (label 1) | Any mapped non-NORM class (MI, STTC, CD, HYP). |
| **Sampling Rates Available** | 100 Hz and 500 Hz | 100 Hz (`records100`) | Standard 100 Hz low-resolution signals provide 1,000 samples per lead. |
| **Total Leads** | 12 | 12 | Standard clinical configuration: I, II, III, aVR, aVL, aVF, V1–V6. |

---

## 3. Directory Layout and File Formats

An extracted PTB-XL repository contains the following core components:

```text
ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
├── ptbxl_database.csv           # Master tabular metadata (21,799 rows)
├── scp_statements.csv           # Dictionary mapping SCP-ECG codes to diagnostic classes
├── RECORDS                      # Plaintext list of all relative waveform file paths
├── SHA256SUMS.txt               # Checksums for integrity verification
├── records100/                  # 100 Hz low-resolution waveform records (100 Hz sampling)
│   ├── 00000/                   # Subfolders partitioned by ECG ID ranges
│   │   ├── 00001_lr.hea         # WFDB header file for record 1 (100 Hz)
│   │   ├── 00001_lr.dat         # Binary waveform data for record 1 (100 Hz, 16-bit signed)
│   │   └── ...
│   └── 21000/
└── records500/                  # 500 Hz high-resolution waveform records (optional/uncompressed)
    ├── 00000/
    │   ├── 00001_hr.hea
    │   ├── 00001_hr.dat
    │   └── ...
    └── 21000/
```

### File Types Explained
- `.hea` (WFDB Header File): An ASCII text file defining signal metadata, including recording name, number of channels (12), sampling frequency (100 or 500 Hz), ADC gain (typically 200 ADC units per mV), baseline zero offset, signal resolution, and lead names.
- `.dat` (WFDB Binary Signal File): Raw 16-bit signed integer binary data storing the discrete voltage samples sequentially across all 12 channels.
- `ptbxl_database.csv`: Metadata table indexed by `ecg_id`, containing demographic, technical, administrative, and clinical annotation fields.
- `scp_statements.csv`: Annotation dictionary mapping the 71 clinical SCP-ECG statements to 5 diagnostic superclasses, diagnostic likelihood weights, and form/rhythm categories.

---

## 4. Key Metadata Columns Defined

The master table `ptbxl_database.csv` provides 28 columns. The most critical for machine learning are:

1. `ecg_id`: Unique integer identifier for each 10-second ECG recording ($1 \le \text{ecg\_id} \le 21837$).
2. `patient_id`: Unique patient identifier (float/int). Crucial for patient-level grouping; multiple ECGs can belong to the same patient over time.
3. `strat_fold`: Assigned validation fold ($1 \dots 10$). Patient-aware partition designed by the original PTB-XL authors: all records for any given patient reside strictly within a single fold to prevent data leakage.
4. `filename_lr`: Relative path to the 100 Hz waveform record (e.g., `records100/00000/00001_lr`).
5. `filename_hr`: Relative path to the 500 Hz waveform record (e.g., `records500/00000/00001_hr`).
6. `scp_codes`: Dictionary string formatted as `{'CODE': likelihood}`, where `CODE` matches an entry in `scp_statements.csv` and `likelihood` is the clinical confidence score (0 to 100, typically 100 or 0).

---

## 5. Why the 100 Hz Signal is the Starting Point

PTB-XL provides two sampling resolutions:
- **Low Resolution (`records100`)**: 100 Hz sampling rate $\to 1,000$ discrete time steps per 10-second recording per lead.
- **High Resolution (`records500`)**: 500 Hz sampling rate $\to 5,000$ discrete time steps per 10-second recording per lead.

### Scientific Rationale for Selecting 100 Hz:
1. **Clinical Frequency Content**: The diagnostic power of statistical summary features (mean, standard deviation, minimum, maximum) is governed by macro-morphological deflections (P-wave, QRS complex, T-wave). Under the Nyquist theorem, a 100 Hz sampling rate captures frequency components up to 50 Hz, which encompasses the vast majority of spectral energy in diagnostic ECG (0.05–40 Hz standard clinical diagnostic band).
2. **Computational and Memory Footprint**: Standard 100 Hz files consume $24\text{ KB}$ per record ($1000 \text{ samples} \times 12 \text{ channels} \times 2 \text{ bytes}$), allowing the entire 21,388-record corpus to be read and processed in under 3 minutes.
3. **Telemetry Realism**: Implantable and wearable cardiac monitors operate under severe battery constraints and typically transmit low-frequency, pre-sampled, or feature-extracted telemetry rather than raw high-frequency multi-lead waveforms.

---

## 6. Standard 12-Lead Structure and Recording Topology

Every recording is a synchronous 10-second acquisition across 12 standard clinical leads:
- **Limb Leads (Frontal Plane)**:
  - Bipolar: Lead I, Lead II, Lead III
  - Augmented unipolar: aVR, aVL, aVF
- **Precordial Leads (Horizontal Plane)**:
  - Chest leads: V1, V2, V3, V4, V5, V6

Each lead is stored in units of millivolts ($\text{mV}$). The consistent lead ordering verified across all records is:
$$\text{Leads} = [\text{I}, \text{II}, \text{III}, \text{aVR}, \text{aVL}, \text{aVF}, \text{V1}, \text{V2}, \text{V3}, \text{V4}, \text{V5}, \text{V6}]$$

For each lead, four summary statistics are extracted:
- **Mean**: Baseline potential and overall DC level.
- **Standard Deviation (ddof=0)**: Signal amplitude dispersion and aggregate power.
- **Minimum**: Peak negative deflection (e.g., S-wave or Q-wave nadir).
- **Maximum**: Peak positive deflection (e.g., R-wave apex).

Across 12 leads, this produces an exact feature vector of dimension **$D = 48$** ($12 \times 4$).

---

## 7. Diagnostic Annotation Structure and Project Target Policy

PTB-XL annotations follow the Standard Communications Protocol for Computer-Assisted Electrocardiography (SCP-ECG). The statements are organized into 5 hierarchical superclasses:

1. **`NORM`**: Normal ECG.
2. **`MI`**: Myocardial Infarction.
3. **`STTC`**: ST/T Change (ischemia, repolarization abnormalities).
4. **`CD`**: Conduction Disturbance (bundle branch blocks, AV blocks).
5. **`HYP`**: Hypertrophy (left/right ventricular hypertrophy, atrial enlargement).

### The Binary Target Policy
In clinical practice, ECGs often present multiple co-occurring findings. For our baseline binary classification task, we establish the following explicit policy:

- **Label 0 (Normal)**: All mapped SCP-ECG statements belong exclusively to the `NORM` superclass.
- **Label 1 (Abnormal)**: At least one mapped SCP-ECG statement belongs to a non-NORM superclass (`MI`, `STTC`, `CD`, or `HYP`).
- **Mixed Cases**: If a record contains both `NORM` and a non-NORM statement (e.g., `NORM` with minor `STTC`), it is explicitly assigned to **Label 1 (Abnormal)**.
- **Exclusion**: Records whose SCP codes contain no mapped diagnostic superclasses (e.g., rhythm-only codes or unmapped technical codes) are excluded from the dataset.

> [!IMPORTANT]
> This binary labeling scheme is an **explicit project policy** designed to benchmark normal vs. abnormal cardiac screening. It is not an inherent property of the raw PTB-XL database, which is natively a multi-label classification dataset.

---

## 8. Clinical Reality Check: PTB-XL vs. CIED Telemetry

To ensure scientific integrity in our research reports, we must explicitly articulate what PTB-XL is and what it is not:

- **What PTB-XL Is**: A standardized database of **surface 12-lead electrocardiograms** recorded with external electrodes placed on the chest and limbs of resting patients in clinical hospital settings.
- **What PTB-XL Is Not**: It is **not** raw telemetry from Cardiac Implantable Electronic Devices (CIEDs), such as pacemakers, implantable cardioverter-defibrillators (ICDs), cardiac resynchronization therapy (CRT) units, or implantable loop recorders (ILRs).
- **Key Differences**:
  1. *Lead Count*: CIEDs measure 1 to 3 intracardiac electrogram (EGM) channels (e.g., right ventricular tip-to-ring, right atrial, subcutaneous shock vector), not 12 surface leads.
  2. *Signal Morphology*: Intracardiac EGMs measure near-field local depolarization wavefronts (sharp spike waveforms), whereas surface ECGs measure far-field integrated cardiac potentials.
  3. *Transmission Constraints*: CIED telemetry transmits periodically over low-power inductive or RF links (e.g., Medtronic CareLink, Abbott Merlin, Boston Scientific LATITUDE) in heavily packetized and quantized bursts.
- **Justification for Use**: PTB-XL provides an authentic, high-dimensional, standardized cardiac signal benchmark. Proving that our CKKS modulus-level scheduling architecture operates efficiently on 48-feature cardiac morphology confirms the algorithmic scalability of privacy-preserving logistic regression for remote cardiac monitoring pipelines.
