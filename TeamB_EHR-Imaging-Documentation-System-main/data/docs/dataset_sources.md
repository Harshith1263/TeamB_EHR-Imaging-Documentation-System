# Dataset Sources – Team B: EHR Imaging Documentation System

This project integrates multiple medical imaging modalities and electronic health record (EHR) datasets to build an EHR-Imaging Documentation System. The datasets uploaded in this repository provide both structured clinical records and unstructured imaging data that can be combined for multimodal analysis.

## Imaging Datasets

1. Bone Fracture X-rays

   - Description: Binary classification dataset for detecting fractures in bone X-rays.
   - Applications: Used in orthopedic diagnostic AI and fracture detection tasks.
   - Source: https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
   - Format: PNG/JPEG images.

2. Brain MRI

   - Description: MRI scans of the brain for tumor detection and segmentation tasks.
   - Applications: Used in oncology for brain tumor localization and classification.
   - Source: https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset
   - Format: PNG/JPEG slices extracted from MRIs.

3. Chest CT Scans

   - Description: Thoracic CT images useful for lung abnormality and cancer detection.
   - Applications: Lung nodule classification, early-stage lung cancer detection.
   - Source: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

4. Chest X-rays

   - Description: Standard dataset for pneumonia and normal classification.
   - Applications: Widely used for pulmonary disease detection and AI benchmarking.
   - Sources:
     - https://www.kaggle.com/datasets/wasifnafee/mimic-cxr
     - https://www.kaggle.com/datasets/nih-chest-xrays/data

5. Thyroid Ultrasound

   - Description: Ultrasound images of thyroid gland to identify nodules and growths.
   - Applications: Nodule classification (benign vs malignant).
   - Source: https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images

6. Fetus Ultrasound

   - Description: Prenatal ultrasound images for gestational monitoring.
   - Applications: Fetal health monitoring, AI-assisted pregnancy screening.
   - Source: https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset

7. Kidney Stone Ultrasound

   - Description: Ultrasound scans for kidney stone detection.
   - Applications: Automated detection of kidney stones and urological research.
   - Source: https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection

8. Breast Ultrasound

   - Description: Ultrasound images of breast tissue for tumor detection.
   - Applications: Breast cancer diagnosis and tumor segmentation.
   - Source: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

9. Normal Images
   - Description: Control group images without abnormalities.
   - Applications: Used for balancing datasets across modalities and providing baseline controls.

---

## EHR Datasets

1. 200k_patients_EHR.csv

   - Description: Structured dataset containing demographic, clinical, and diagnostic records of ~200,000 patients.
   - Applications: Used for predictive modeling, risk stratification, and comorbidity studies.
   - Source: https://www.kaggle.com/datasets/enriquemas/200k-patients-ehr-dataset-demo-10-en-es-ru
   - Fields: Patient ID, age, sex, comorbidities, diagnoses, procedures.

2. leukemia_ehr_full.csv

   - Description: Patient EHR data specific to leukemia cases.
   - Applications: Hematology research, treatment outcome prediction, biomarker identification.
   - Source: https://www.kaggle.com/datasets/vipulshahi/ehr-data
   - Fields: Blood markers, treatment history, survival outcomes.

3. cxr_df.csv.zip
   - Description: Tabular dataset linking chest X-ray images with metadata.
   - Applications: Supports multimodal learning (image + tabular metadata).
   - Sources:
     - https://www.kaggle.com/datasets/wasifnafee/mimic-cxr
     - https://www.kaggle.com/datasets/nih-chest-xrays/data
   - Fields: Labels (pneumonia vs. normal), patient ID, image paths.

---

## ICD Code Resources

1. ICD10codes.csv

   - Description: List of ICD-10 codes and descriptions for disease classification.
   - Source: https://www.kaggle.com/datasets/arashnic/icd10-codes-and-descriptions

2. ICDCodeSet.csv

   - Description: Expanded set of ICD codes, possibly structured into disease categories.
   - Source: https://www.kaggle.com/datasets/arashnic/icd10-codes-and-descriptions

3. icd10cm_codes_2023.txt

   - Description: Latest ICD-10-CM codes (2023 edition).
   - Applications: Standardized coding for billing, diagnosis, and interoperability.
   - Source: https://www.kaggle.com/datasets/arashnic/icd10-codes-and-descriptions

4. icd9to10dictionary.txt
   - Description: Mapping dictionary to convert ICD-9 codes to ICD-10.
   - Applications: Ensures compatibility of legacy datasets with ICD-10 standards.
   - Source: https://www.kaggle.com/datasets/arashnic/icd10-codes-and-descriptions
