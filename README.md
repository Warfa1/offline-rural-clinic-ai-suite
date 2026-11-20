Offline Health AI Prototypes for Low-Resource Clinics

Two early-stage research prototypes for settings with limited internet, electricity and specialist staff: malaria cell classification and maternal health risk triage.

Research project only. Not a medical device. Not for diagnosis or treatment decisions. Both models were trained on public datasets and have not been validated with local data, clinicians, or in any clinic.

Key findings
Random splits overstate malaria performance. A random image-level split put cells from all 200 patients in both training and test sets. On unseen patients, accuracy fell from 93.5% to 68.8% at the default threshold.
The drop was mostly a threshold problem. On unseen patients AUC stayed at 0.97. After choosing the decision threshold on validation patients, test accuracy rose to about 91% with 90 to 93% sensitivity. This suggests the model needs site-specific calibration.
Quantization is cheap. Converting from FP32 to INT8 cut the model from 8.48 MB to 2.41 MB with under 1 point of accuracy loss.
The maternal dataset contains many duplicates. 562 of 1,014 rows (55.4%) are exact duplicates. After removing them, Random Forest accuracy fell from 84.8% to 65.8% and high-risk recall from 90.4% to 73.0%.
A simple threshold rule caught almost as many high-risk cases as the ML models (69.6% vs about 73% high-risk recall), so the models add limited value on this dataset.
1. Malaria cell classification (malaria-edge-vision/)
Task: classify NIH thin-smear cell images as Parasitized or Uninfected (27,558 images)
Model: MobileNetV2 (ImageNet weights), 96x96 input, converted to TensorFlow Lite
Patient IDs: read from the image file names (200 patients). Patients were split 70 / 15 / 15 into train, validation and test sets (19,564 / 3,846 / 4,148 images).
Positive class: Parasitized
Random split vs patient-level split
Setting	Accuracy	Sensitivity	Specificity	AUC
Random split, frozen base	93.54%	90.61%	96.41%	0.98
Patient-level split, frozen base	68.80%	37.87%	99.86%	0.97
Patient-level split, fine-tuned	64.97%	30.13%	99.95%	0.97

All rows use the default threshold of 0.5. Fine-tuning the last 30 layers did not help on unseen patients.

Decision threshold chosen on validation patients (fine-tuned model, test set)
Rule	Threshold	Accuracy	Sensitivity	Specificity
Default	0.5	64.97%	30.13%	99.95%
Youden's J (validation)	0.0037	91.15%	93.46%	88.84%
Sensitivity >= 90% (validation)	0.0057	91.10%	90.38%	91.84%

Test AUC: 0.968. The very small thresholds mean the model gives low "parasitized" scores to infected cells from new patients, so it would need calibration on a labelled sample from each new site. A likely cause is differences between slides (for example staining), but this has not been tested.

TensorFlow Lite formats (fine-tuned model, default threshold)
Format	Size	Accuracy	Sensitivity	Mean latency
FP32	8.48 MB	64.97%	30.13%	2.05 ms
FP16	4.30 MB	64.54%	29.26%	2.07 ms
INT8 (dynamic range)	2.41 MB	64.42%	29.07%	4.28 ms
INT8 (full integer)	2.61 MB	64.25%	28.92%	2.76 ms

Latency is the mean per image on a Colab CPU with default settings. It has not been measured on a phone or single-board computer.

2. Maternal health risk triage (maternal-xai-triage/)
Task: predict low / mid / high maternal risk from age, blood pressure, blood sugar, body temperature and heart rate
Data: UCI Maternal Health Risk dataset, 1,014 records
Models: Logistic Regression, Random Forest, XGBoost, explained with SHAP
Evaluation: repeated stratified 5-fold cross-validation (5 repeats), mean +/- standard deviation
Duplicate audit
562 exact duplicate rows (55.4%), leaving 452 unique rows
35 feature combinations appear with conflicting labels
The data has no patient IDs, so it is not possible to tell whether duplicates are the same woman or different women with identical rounded vitals. Scores on the raw data are therefore likely, not certainly, optimistic.
Results
Data	Model	Accuracy	Macro F1	High-risk recall
Raw (1,014 rows)	Logistic Regression	61.5 +/- 2.9	60.4 +/- 2.9	72.2 +/- 4.5
Raw (1,014 rows)	Random Forest	84.8 +/- 2.3	85.2 +/- 2.3	90.4 +/- 3.9
Raw (1,014 rows)	XGBoost	79.8 +/- 2.1	80.1 +/- 2.1	86.6 +/- 4.4
De-duplicated (452 rows)	Logistic Regression	66.9 +/- 3.3	53.0 +/- 4.6	63.7 +/- 10.8
De-duplicated (452 rows)	Random Forest	65.8 +/- 3.9	59.5 +/- 4.8	73.0 +/- 6.7
De-duplicated (452 rows)	XGBoost	70.3 +/- 2.7	62.3 +/- 4.7	72.7 +/- 6.6
De-duplicated (452 rows)	Simple threshold rule	56.0	51.2	69.6

Notes

Class balance changes after de-duplication (low risk rises from 40% to 52% of rows), so accuracy is not directly comparable across the two data settings. Macro F1 and high-risk recall are more informative.
The threshold rule uses illustrative blood pressure and glucose cut-offs, was not cross-validated, and was evaluated on all 452 rows.
On de-duplicated data, differences between Random Forest and XGBoost are small compared with the spread.
With Random Forest on de-duplicated data (one 5-fold pass), 78 of 112 high-risk cases were found, 16 were labelled low risk, and only 26 of 106 mid-risk cases were identified.
The earlier single-split results (Random Forest 84.73%, XGBoost 81.28% on 203 test records) were obtained on the raw data and should be treated as optimistic. The saved model and the SHAP chart in this repository were also trained on the raw data.
Limitations
Public datasets only, with no local data, clinicians, or clinical validation
Malaria: one patient-level split (about 30 test patients), so results may change with another split; no external dataset yet; the threshold needs calibration per site
The malaria models classify single, cropped cells, not whole slide images
Maternal: small dataset from one population, many duplicates, no patient IDs
Latency measured on a Colab CPU only, not on low-cost hardware
The demo app uses the original malaria model, which was trained with a random split and whose cut-offs have not been calibrated on unseen patients
Next steps
Repeat the malaria patient-level evaluation over several seeds
Test colour and brightness augmentation against staining differences
External test on African datasets (for example Tanzanian or Ugandan thin and thick smear images)
Validation with local data and health workers in the Somali Region of Ethiopia
Measure speed on low-cost phones and single-board computers
Repository contents
Path	Description
malaria-edge-vision/	Malaria notebooks, TFLite model, evaluation notebook
maternal-xai-triage/	Maternal notebooks, saved model, SHAP chart, evaluation notebook
streamlit_app.py	Research demo (not for clinical use)
Data sources
NIH Malaria Cell Images dataset (Rajaraman et al., PeerJ, 2018)
UCI Maternal Health Risk dataset (Ahmed and Kashem, 2020)
