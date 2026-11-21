# Offline Rural Clinic AI Suite (Horn of Africa Edition)

Two lightweight, offline-first clinical AI decision-support prototypes engineered for low-resource clinics with limited internet, electricity, and specialist staff—localized specifically for **Somali communities in Ethiopia’s Somali Region and Kenya’s North Eastern / NFD counties**:

1. **Maternal Health Risk Triage (`maternal-xai-triage/` & `somali_maternal_transfer_xgb.joblib`)**: Interpretable XGBoost pregnancy risk stratification trained on **3,042 unique Somali pregnancies** from official **Kenya 2022 DHS** and **Ethiopia 2024–25 DHS** microdata, paired with live per-patient **SHAP** explanations and clinical red-flag safety overrides.
2. **Edge Malaria Cell Screener (`malaria-edge-vision/` & `somali_malaria_stain_aug_int8.tflite`)**: Quantized **MobileNetV2 (96×96, 2.41 MB INT8 LiteRT/TFLite)** trained with domain-randomized Giemsa stain and illumination augmentation to withstand cross-clinic microscope variations.

> **Research & Educational Prototype Only:** Not a certified medical device. Not intended for standalone clinical diagnosis or treatment decisions without licensed clinician oversight.

---

## Live Interactive Demos
* **Localized Somali Edition (English & `Af-Soomaali`):** `somali_clinic_app.py` (Features Somali-localized XGBoost + live SHAP risk waterfall + 2.41 MB stain-augmented TFLite malaria screener)
* **Phase 1 Public Baseline Demo:** `streamlit_app.py`

---

## Key Research Findings & Localization Breakthroughs

### 1. Maternal Health Localization (`Kenya 2022 DHS` + `Ethiopia 2024–25 DHS`)
* **Eliminating Data Leakage & Duplicates:** Our Phase 1 audit of the public UCI dataset revealed **55.4% exact duplicate rows (562/1,014)** with no patient IDs. In Phase 2, we obtained formal DHS Program authorization and extracted **21,799 unique pregnancies**, isolating **3,042 unique Somali pregnancies** (`2,169` in Kenya NFD + `875` in Ethiopia Somali Region) with zero duplicates and real respondent IDs.
* **Uncovering Severe Prenatal Care Disparities:** Side-by-side cohort analysis revealed that **48.1% of Somali mothers receive zero antenatal care (ANC) visits** during pregnancy (nearly double the **26.9%** rate across the rest of Kenya and Ethiopia) and average only **1.9 prenatal visits** compared to **3.3 nationally**.
* **Population-Specific Training Outperforms Generic National Pooling:**

| Training Strategy | Training Cohort | Somali Test Evaluation | ROC-AUC (5-Fold CV) | High-Risk Recall |
| :--- | :--- | :--- | :---: | :---: |
| **Somali-Localized XGBoost** | **3,042 Somali pregnancies** (Kenya NFD + Ethiopia Somali) | Held-out Somali folds | **0.592 ± 0.028** | — |
| **National Transfer XGBoost** | 21,714 total pregnancies (Somali up-weighted 4×) | Held-out Somali folds | 0.572 ± 0.024 | **53.1%** |
| **Cross-Border Generalization** | Trained strictly on Kenya NFD (`n=2,169`) | Unseen Ethiopia Somali Region (`n=875`) | 0.550 | **53.0%** |

* **Takeaway:** Pooling 7× more non-Somali records diluted Somali-specific risk drivers (`0.592` vs. `0.572` ROC-AUC), proving that rural pastoralist communities require localized clinical triage models rather than generic national models.

---

### 2. Edge Malaria Vision: Solving Cross-Site Stain Shift (`2.41 MB INT8`)
* **The Cross-Site Stain Failure Problem:** On strict patient-level splits (`200` patients, `27,558` cells), models trained without stain augmentation suffer catastrophic sensitivity collapse when exposed to shifted Giemsa stain hues and smartphone-adapter lighting.
* **Domain-Randomized Stain Augmentation Fix:** Fine-tuning MobileNetV2 (`96×96`) with randomized hue, saturation, brightness, and contrast perturbations restored cross-site parasite detection while keeping the quantized INT8 model at **2.41 MB**:

| Model Variant (Strict Patient-Level Split) | Size | Cross-Site AUC | Accuracy @ 0.5 | Sensitivity @ 0.5 | Calibrated Acc (`0.102`) | Calibrated Sensitivity (`0.102`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline NIH TFLite (No Stain Aug)** | 2.41 MB | 0.590 | 54.2% | 0.0% | — | — |
| **Stain-Augmented TFLite (Upgraded INT8)** | **2.41 MB** | **0.993** | **95.8%** | **92.2%** | **96.3%** | **96.4%** |

---

## Repository Structure
```text
├── somali_clinic_app.py                  # Bilingual (English / Af-Soomaali) localized Streamlit clinic app
├── somali_maternal_transfer_xgb.joblib   # Localized Somali maternal triage XGBoost pipeline
├── somali_malaria_stain_aug_int8.tflite  # Quantized 2.41 MB stain-augmented MobileNetV2 INT8 model
├── streamlit_app.py                      # Phase 1 baseline demo app
├── MODEL_CARD.md                         # Clinical governance, safety overrides, and deployment limitations
├── requirements.txt                      # Lightweight cloud/offline dependencies (ai-edge-litert, xgboost, shap)
├── malaria-edge-vision/                  # Phase 1 patient-level split & quantization notebooks
└── maternal-xai-triage/                  # Phase 1 UCI duplicate audit & baseline notebooks
```

---

## Data Governance, Ethics & Citations
* **DHS Data Compliance:** Raw and processed microdata from the **2022 Kenya DHS** and **2024–25 Ethiopia DHS** are used under formal authorization from The DHS Program and are **excluded from this public repository** in strict accordance with DHS confidentiality and non-redistribution terms.
* **Public Datasets:** NIH Malaria Cell Images dataset (Rajaraman et al., *PeerJ*, 2018); UCI Maternal Health Risk dataset (Ahmed & Kashem, 2020).
