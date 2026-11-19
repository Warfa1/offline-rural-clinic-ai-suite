# Offline Point-of-Care AI Triage Suite for Low-Resource Rural Clinics

A dual-modality clinical decision-support suite engineered to run completely offline on low-cost smartphones and single-board computers in off-grid healthcare facilities.

## Modules

### 1. Edge Computer Vision for Malaria Screening (`malaria-edge-vision/`)
* **Architecture:** Lightweight MobileNetV2 (96×96) compressed via TensorFlow Lite (LiteRT) post-training quantization.
* **Dataset:** 27,558 labeled red blood cell images (NIH Malaria Dataset).
* **Performance:** **93.49% Accuracy** | **92.31% Sensitivity** | **94.63% Specificity**.
* **Edge Efficiency:** **2.41 MB** model size | **4.07 ms/cell** single-threaded CPU inference latency.

### 2. Explainable Maternal Health Risk Triage (`maternal-xai-triage/`)
* **Architecture:** Random Forest & XGBoost classifiers integrated with SHAP clinical feature attribution.
* **Dataset:** 1,014 rural maternal clinic vital-sign records (UCI Maternal Health Risk Dataset).
* **Performance:** **84.73% Accuracy** | **96.36% High-Risk Recall** | **85.17% Macro F1**.
* **Edge Efficiency:** **700.2 KB** deployable model | **1.63–10.14 ms** CPU inference latency.
