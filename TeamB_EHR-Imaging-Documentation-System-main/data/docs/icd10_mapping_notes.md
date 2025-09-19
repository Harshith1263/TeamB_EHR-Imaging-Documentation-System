# ICD-10 Mapping Notes

This document explains the rationale for selecting ICD-10 codes in the dataset mapping.

---

## Codes Used  

| ICD-10 Code | Definition (WHO ICD-10)                                            | Example Diagnoses in Dataset           |
|-------------|--------------------------------------------------------------------|----------------------------------------|
| D33.0       | Benign neoplasm of brain and other parts of central nervous system | Benign brain tumour (MRI, Ultrasound) |
| C71.9       | Malignant neoplasm of brain, unspecified                           | Malignant brain tumour (MRI)          |
| Z00.0       | General medical examination                                        | Normal scans (MRI, X-ray)             |
| C34.1       | Malignant neoplasm of upper lobe, bronchus or lung                 | Adenocarcinoma (CT Chest)             |
| J18.9       | Pneumonia, unspecified organism                                    | Pneumonia (Chest X-ray)               |

---

## Notes  
- Codes were selected according to **WHO ICD-10 official definitions**.  
- **Benign and malignant brain tumours** were mapped separately to D33.0 and C71.9 to distinguish between cases.  
- **Normal findings** were mapped to Z00.0 to ensure consistency across MRI and X-ray modalities.  
- **Chest CT adenocarcinoma cases** were mapped to C34.1 (lung cancer code).  
- **Pneumonia** was mapped to J18.9, the standard code for unspecified-organism pneumonia.  
- This ensures **standardization, interoperability, and readiness** for AI model training and clinical workflows.  

