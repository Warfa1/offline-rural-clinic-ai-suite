# Model Card — Offline Rural Clinic AI Suite (Horn of Africa Edition)

## 1. Model Details & Intended Use
* **Developer:** Yusuf Warfa (Karabük University)
* **Models Included:**
  1. `somali_maternal_transfer_xgb.joblib` — Gradient-boosted decision tree (`XGBoost`) with `SHAP` feature attribution for antenatal pregnancy risk screening.
  2. `somali_malaria_stain_aug_int8.tflite` — Quantized `MobileNetV2` (`96×96` RGB, `2.41 MB` INT8 LiteRT/TFLite) binary classifier (`Parasitized` vs. `Uninfected` red blood cells).
* **Intended Users:** Community health workers (CHWs), nurses, and clinical researchers operating in low-bandwidth or offline rural health posts in Ethiopia's Somali Region and Kenya's North Eastern / NFD counties.
* **Out-of-Scope / Prohibited Use:** Neither model is a certified medical device. They must **never** replace licensed physician judgment, emergency obstetric care, or confirmatory thick-smear microscopy / Rapid Diagnostic Tests (RDTs).

---

## 2. Training Data & Localization Scope
* **Maternal Triage Cohort:** 3,042 unique, de-duplicated pregnancy records extracted under formal authorization from the **2022 Kenya DHS (`KEIR8C`/`KENR8C`)** (`n = 2,169`) and **2024–25 Ethiopia DHS (`ETIR8A`/`ETNR8A`)** (`n = 875`), benchmarked against 18,672 non-Somali Kenyan and Ethiopian pregnancies.
* **Malaria Vision Cohort:** 27,558 single-cell thin-smear images across 200 patients (strict patient-level 70/15/15 split), augmented with domain-randomized hue, saturation, brightness, and contrast shifts to simulate cross-clinic Giemsa staining variation.

---

## 3. Clinical Safety Guardrails & Threshold Calibration
* **Hard Clinical Red-Flag Override (Maternal Triage):** In `somali_clinic_app.py`, any report of **active vaginal bleeding** automatically triggers an urgent hospital referral alert (`🚨 HIGH RISK`), overriding the ML probability score so acute obstetric emergencies are never missed.
* **Site-Specific Calibration (Malaria Vision):** Because microscopy staining varies across clinics, `somali_clinic_app.py` exposes a tunable decision threshold calibrated via Youden's J (`0.102` default, achieving `96.4%` sensitivity on cross-site stain shifts).

---

## 4. Known Limitations & Next Steps
1. **Survey-Based Maternal Outcomes:** DHS pregnancy outcomes rely on retrospective maternal recall and household survey biomarkers rather than continuous hospital vital signs (such as live systolic/diastolic blood pressure cuffs or blood glucose meters).
2. **Single-Cell Cropping:** The malaria classifier evaluates pre-cropped single red blood cell patches (`96×96`) rather than raw whole-slide field-of-view images.
3. **Prospective Clinical Validation:** Field validation with de-identified patient records and local Giemsa smears at regional referral hospitals (such as Jigjiga or Garissa) is required before clinical deployment.
