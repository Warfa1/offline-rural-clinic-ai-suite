# Offline Health AI Prototypes for Low-Resource Clinics

Two early-stage research prototypes that run without an internet connection.
Built for education and research. Not a medical device. Not for diagnosis or
treatment decisions.

## 1. Malaria cell classification (`malaria-edge-vision/`)
- Model: MobileNetV2 (96x96 input), quantized to TensorFlow Lite
- Data: NIH Malaria Dataset, 27,558 cell images, 80/20 split
- Validation results (5,511 images): 93.49% accuracy, 92.31% sensitivity,
  94.63% specificity
- Size and speed: 2.41 MB model, about 4 ms per image on a Colab CPU

## 2. Maternal health risk triage with SHAP (`maternal-xai-triage/`)
- Models: Random Forest and XGBoost, explained with SHAP
- Data: UCI Maternal Health Risk dataset, 1,014 records (203 test)
- Test results: Random Forest 84.73% accuracy, 96.36% high-risk recall;
  XGBoost 81.28% accuracy, 94.55% high-risk recall
- Saved model size: about 700 KB

## Limitations
- Trained and tested only on public datasets, with one split each
- Not validated with local data, clinicians, or in a real clinic

## Next steps
Validation with local clinical data and health workers in the Somali Region
of Ethiopia, and testing on low-cost hardware.
