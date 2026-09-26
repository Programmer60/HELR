"""
MIMIC-IV -> flat feature table extraction, matching the shape of heart.csv
so the existing HELR pipeline (helr_end_to_end.py) can consume it unchanged.

WHY THIS EXISTS: MIMIC has no ready-made "heart disease" label and its
labs/vitals live in enormous per-event tables (labevents.csv.gz is 25GB+
compressed), not a tidy per-patient CSV. This script does the one-time
plaintext data-engineering work of turning that into a heart.csv-shaped
table BEFORE any homomorphic encryption code ever touches the data.

USAGE:
    python mimic_extract.py \
        --hosp_dir "path/to/mimic-iv/hosp" \
        --icu_dir  "path/to/mimic-iv/icu" \
        --out mimic_heart_features.csv

Requires only pandas (reads .csv.gz directly, no need to unzip first).

-----------------------------------------------------------------------
DESIGN NOTES

1. TARGET LABEL: derived from hosp/diagnoses_icd.csv.gz using standard,
   well-documented ICD code ranges for ischemic heart disease:
   ICD-9 prefixes 410-414, ICD-10 prefixes I20-I25. This keeps the
   project's task identical to the paper's (heart disease prediction)
   rather than switching to an unrelated MIMIC task.

2. LAB/VITAL SELECTION BY NAME, NOT HARDCODED ITEMID: itemid values are
   internal MIMIC identifiers that differ in ways that are easy to get
   wrong from memory. This script instead looks itemids up by matching
   human-readable names in d_labitems.csv.gz / d_items.csv.gz, which is
   robust regardless of the exact numeric IDs in your copy of MIMIC-IV.
   PRINT THE MATCHED ITEMIDS AND VERIFY THEM YOURSELF before trusting
   the extracted features -- name-matching can occasionally pull in an
   unintended sibling test (e.g. "Creatinine" vs "Creatinine, Urine").

3. CHUNKED STREAMING: labevents.csv.gz and chartevents.csv.gz are too
   large to load into memory. Both are read in chunks via pandas'
   chunksize parameter, filtering to only the itemids we care about and
   only the admissions in our cohort, discarding everything else as we
   go.

4. AGGREGATION: for each selected lab/vital, we take the MEDIAN value
   recorded during the admission (labs) or ICU stay (vitals). Median is
   more robust to the occasional erroneous extreme value than mean, and
   simpler than a time-windowed approach for a first pass.
"""

import argparse
import gzip
import numpy as np
import pandas as pd


# ICD code prefixes for ischemic heart disease (standard clinical ranges)
ICD9_HEART_PREFIXES = tuple(str(c) for c in range(410, 415))   # 410-414
ICD10_HEART_PREFIXES = tuple(f"I2{d}" for d in range(0, 6))     # I20-I25

# Lab test names to look for in d_labitems.csv.gz (case-insensitive
# substring match against the 'label' column). Edit this list to taste.
LAB_NAMES = [
    "Creatinine",
    "Glucose",
    "Potassium",
    "Sodium",
    "Cholesterol, Total",
    "Troponin T",
    "Hemoglobin",
    "White Blood Cells",
    "Urea Nitrogen",
]

# Vital sign names to look for in icu/d_items.csv.gz
VITAL_NAMES = [
    "Heart Rate",
    "Non Invasive Blood Pressure systolic",
    "Non Invasive Blood Pressure diastolic",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
    "Temperature Fahrenheit",
]


def build_target_label(hosp_dir):
    """Returns a DataFrame [hadm_id, heart_disease] with one row per
    admission that appears in diagnoses_icd.csv.gz."""
    print("Loading diagnoses_icd.csv.gz ...")
    diag = pd.read_csv(f"{hosp_dir}/diagnoses_icd.csv.gz",
                        compression="gzip",
                        usecols=["hadm_id", "icd_code", "icd_version"],
                        dtype={"hadm_id": "Int64", "icd_code": str,
                               "icd_version": "Int64"})

    def is_heart_code(row):
        code = row["icd_code"]
        if row["icd_version"] == 9:
            return code.startswith(ICD9_HEART_PREFIXES)
        else:
            return code.startswith(ICD10_HEART_PREFIXES)

    diag["is_heart"] = diag.apply(is_heart_code, axis=1)
    label = diag.groupby("hadm_id")["is_heart"].any().astype(int)
    label.name = "heart_disease"
    print(f"  {len(label)} admissions with diagnosis records; "
          f"{label.sum()} ({label.mean():.1%}) flagged heart_disease=1")
    return label.reset_index()


def load_demographics(hosp_dir):
    """Age and gender per subject, from patients.csv.gz. Also loads
    admissions.csv.gz to map hadm_id -> subject_id."""
    print("Loading patients.csv.gz and admissions.csv.gz ...")
    patients = pd.read_csv(f"{hosp_dir}/patients.csv.gz", compression="gzip",
                            usecols=["subject_id", "anchor_age", "gender"])
    admissions = pd.read_csv(f"{hosp_dir}/admissions.csv.gz", compression="gzip",
                              usecols=["subject_id", "hadm_id"])
    demo = admissions.merge(patients, on="subject_id", how="left")
    demo["gender"] = (demo["gender"] == "M").astype(int)  # 1=male, 0=female
    demo = demo.rename(columns={"anchor_age": "age"})
    return demo[["hadm_id", "age", "gender"]]


def lookup_itemids(dict_path, names, label_col="label"):
    """Looks up itemids in a d_labitems/d_items style dictionary file by
    case-insensitive substring match on the label column. Returns a dict
    {matched_name: [itemid, ...]} -- PRINT AND VERIFY THIS before trusting
    it; a name might match more items than intended."""
    d = pd.read_csv(dict_path, compression="gzip")
    matches = {}
    for name in names:
        hit = d[d[label_col].str.contains(name, case=False, na=False)]
        matches[name] = hit["itemid"].tolist()
        print(f"  '{name}' -> itemids {matches[name]} "
              f"({', '.join(hit[label_col].tolist())})")
    return matches


def extract_events_chunked(events_path, itemid_to_name, cohort_hadm_ids,
                            id_col="hadm_id", chunksize=1_000_000):
    """
    Streams a large MIMIC event table (labevents.csv.gz or
    chartevents.csv.gz) in chunks, keeping only rows whose itemid is one
    we care about and whose hadm_id is in our cohort, then returns the
    per-admission MEDIAN value for each selected item.

    itemid_to_name: dict {itemid: feature_name} (inverse of the lookup
    above, flattened) so the output columns have readable names.
    """
    wanted_itemids = set(itemid_to_name.keys())
    cohort_hadm_ids = set(cohort_hadm_ids)
    rows = []

    reader = pd.read_csv(events_path, compression="gzip", chunksize=chunksize,
                          usecols=[id_col, "itemid", "valuenum"],
                          dtype={id_col: "Int64", "itemid": "Int64"})
    n_chunks = 0
    for chunk in reader:
        n_chunks += 1
        chunk = chunk[chunk["itemid"].isin(wanted_itemids) &
                      chunk[id_col].isin(cohort_hadm_ids) &
                      chunk["valuenum"].notna()]
        if len(chunk):
            rows.append(chunk)
        if n_chunks % 20 == 0:
            print(f"    ... processed {n_chunks} chunks "
                  f"({n_chunks * chunksize:,} rows scanned)")

    if not rows:
        print("  WARNING: no matching rows found at all -- check itemids "
              "and column names for this MIMIC-IV version.")
        return pd.DataFrame(columns=[id_col])

    events = pd.concat(rows, ignore_index=True)
    events["feature_name"] = events["itemid"].map(itemid_to_name)
    pivoted = events.groupby([id_col, "feature_name"])["valuenum"] \
                     .median().unstack("feature_name")
    return pivoted.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosp_dir", required=True)
    ap.add_argument("--icu_dir", required=True)
    ap.add_argument("--out", default="mimic_heart_features.csv")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    # 1. target label
    label = build_target_label(args.hosp_dir)

    # 2. demographics
    demo = load_demographics(args.hosp_dir)

    # 3. lab itemid lookup + extraction
    print("\nLooking up lab itemids in d_labitems.csv.gz ...")
    lab_matches = lookup_itemids(f"{args.hosp_dir}/d_labitems.csv.gz", LAB_NAMES)
    lab_itemid_to_name = {iid: name for name, ids in lab_matches.items()
                           for iid in ids}
    print("\nExtracting lab values from labevents.csv.gz "
          "(this will take a while -- it's a very large file)...")
    labs = extract_events_chunked(
        f"{args.hosp_dir}/labevents.csv.gz", lab_itemid_to_name,
        cohort_hadm_ids=label["hadm_id"].tolist(),
        id_col="hadm_id", chunksize=args.chunksize,
    )

    # 4. vital itemid lookup + extraction
    print("\nLooking up vital itemids in d_items.csv.gz ...")
    vital_matches = lookup_itemids(f"{args.icu_dir}/d_items.csv.gz", VITAL_NAMES)
    vital_itemid_to_name = {iid: name for name, ids in vital_matches.items()
                             for iid in ids}
    print("\nExtracting vitals from chartevents.csv.gz "
          "(this will take a while -- it's a very large file)...")
    # chartevents is keyed by stay_id, not hadm_id directly; icustays.csv.gz
    # maps stay_id -> hadm_id, so we join through it.
    icustays = pd.read_csv(f"{args.icu_dir}/icustays.csv.gz", compression="gzip",
                            usecols=["stay_id", "hadm_id"])
    vitals_by_stay = extract_events_chunked(
        f"{args.icu_dir}/chartevents.csv.gz", vital_itemid_to_name,
        cohort_hadm_ids=icustays["stay_id"].tolist(),
        id_col="stay_id", chunksize=args.chunksize,
    )
    vitals = vitals_by_stay.merge(icustays, on="stay_id", how="left") \
                            .drop(columns=["stay_id"]) \
                            .groupby("hadm_id").median().reset_index()

    # 5. merge everything into one flat table
    print("\nMerging into flat feature table ...")
    df = label.merge(demo, on="hadm_id", how="left") \
               .merge(labs, on="hadm_id", how="left") \
               .merge(vitals, on="hadm_id", how="left")

    print(f"\nFinal table: {df.shape[0]} admissions x {df.shape[1]} columns")
    print("Missingness per column (fraction NaN):")
    print(df.isna().mean().sort_values(ascending=False).to_string())

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print("\nNEXT STEPS (do these before touching any HE code):")
    print("  1. Handle missingness explicitly (impute or drop columns/rows).")
    print("  2. Run feature selection (Random Forest importance / PCA) down")
    print("     to ~15-20 features, matching heart.csv's scale.")
    print("  3. Check row count against the 8192-slot budget -- if exceeded,")
    print("     the packing scheme needs patient-batching before HE training.")
    print("  4. THEN feed the resulting CSV into helr_end_to_end.py unchanged.")


if __name__ == "__main__":
    main()