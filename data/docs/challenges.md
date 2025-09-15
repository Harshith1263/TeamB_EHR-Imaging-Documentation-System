# Challenges – Team B: EHR Imaging Documentation System

Integrating multimodal datasets from various sources introduces multiple challenges in data quality, consistency, and usability.

## Imaging Datasets

1. Variability in image quality and resolution across datasets (X-ray, CT, MRI, Ultrasound).
2. Class imbalance (e.g., fewer abnormal cases than normal ones).
3. Non-standard naming conventions and folder structures in Kaggle datasets.
4. Labeling inconsistencies (different datasets may use slightly different terms).

## EHR Datasets

1. Missing or incomplete patient demographic and clinical data.
2. Different schema structures between `200k_patients_EHR.csv` and `leukemia_ehr_full.csv`.
3. Ensuring that `cxr_df.csv.zip` patient IDs correctly map to chest X-ray files.
4. Handling sensitive health data responsibly and ensuring compliance.

## ICD Code Resources

1. Multiple files containing overlapping ICD-10 codes and descriptions.
2. Ambiguities in ICD-9 to ICD-10 mapping (one ICD-9 code can map to multiple ICD-10 codes).
3. Maintaining updated versions (e.g., ICD-10-CM 2023 vs older ICD-10).

## General Challenges

1. Combining structured (EHR) and unstructured (imaging) datasets for multimodal AI models.
2. High storage requirements for large imaging datasets.
3. Preprocessing and cleaning are time-intensive due to dataset diversity.
