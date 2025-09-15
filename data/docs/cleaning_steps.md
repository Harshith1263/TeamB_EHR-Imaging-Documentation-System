# Cleaning Steps – Team B: EHR Imaging Documentation System

The datasets included in this project come from multiple Kaggle sources and include both imaging and EHR data. Cleaning ensures consistency, usability, and readiness for integration.

## Imaging Datasets

1. Remove corrupted, duplicate, or unreadable image files.
2. Standardize image formats (e.g., convert to PNG/JPEG).
3. Normalize image resolution and dimensions to ensure model compatibility.
4. Organize images into structured directories by modality (X-ray, MRI, CT, Ultrasound).
5. Validate image-label mappings (ensure fracture, tumor, pneumonia labels are correct).
6. Balance datasets where class distributions are skewed (e.g., normal vs abnormal).

## EHR Datasets

1. Handle missing values in patient demographics, diagnoses, and procedures.
2. Standardize column names across files (patient ID, age, sex).
3. Normalize categorical values (e.g., M/F vs Male/Female).
4. Remove duplicate patient entries across `200k_patients_EHR.csv` and `leukemia_ehr_full.csv`.
5. Map diagnosis/procedure codes using ICD resources (`ICD10codes.csv`, `icd9to10dictionary.txt`).
6. Ensure patient IDs in `cxr_df.csv.zip` align with corresponding imaging files.

## ICD Code Resources

1. Validate code descriptions across multiple files (`ICD10codes.csv`, `ICDCodeSet.csv`).
2. Remove duplicates between ICD-9 and ICD-10 mappings.
3. Ensure formatting consistency (all codes uppercase, no extra whitespace).
