import os
import time
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

st.set_page_config(page_title="Offline Rural Clinic AI Suite (Baseline)", page_icon="🏥", layout="wide")

st.title("🏥 Offline Health AI Prototypes for Low-Resource Clinics (Phase 1 Baseline)")
st.warning("Research project only. Not a medical device. Trained on public datasets (UCI & NIH).")

tab1, tab2 = st.tabs(["🤰 Maternal Health Risk Triage (UCI Baseline)", "🔬 Malaria Cell Classification (NIH Baseline)"])

with tab1:
    st.subheader("Maternal Health Risk Stratification (UCI Maternal Health Risk Dataset)")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 10, 70, 25)
        sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", 70, 200, 120)
        dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 140, 80)
    with col2:
        bs = st.number_input("Blood Sugar / Glucose (mmol/L)", 5.0, 25.0, 7.5)
        temp = st.number_input("Body Temperature (°F)", 95.0, 105.0, 98.0)
        hr = st.number_input("Heart Rate (bpm)", 50, 140, 76)

    if st.button("Predict Maternal Risk Level", type="primary"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mat_models = glob.glob(os.path.join(base_dir, "maternal-xai-triage", "*.joblib")) + \
                     glob.glob(os.path.join(base_dir, "maternal-xai-triage", "*.pkl"))

        row = pd.DataFrame([{
            "Age": age,
            "SystolicBP": sys_bp,
            "DiastolicBP": dia_bp,
            "BS": bs,
            "BodyTemp": temp,
            "HeartRate": hr
        }])

        if mat_models:
            model = joblib.load(mat_models)
            pred = model.predict(row)
            label_map = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk", "low risk": "Low Risk", "mid risk": "Mid Risk", "high risk": "High Risk"}
            risk_label = label_map.get(pred, str(pred))
            if "High" in risk_label:
                st.error(f"🚨 Predicted Triage Tier: **{risk_label}**")
            elif "Mid" in risk_label:
                st.warning(f"⚠️ Predicted Triage Tier: **{risk_label}**")
            else:
                st.success(f"✅ Predicted Triage Tier: **{risk_label}**")
        else:
            if sys_bp >= 140 or dia_bp >= 90 or bs >= 11.0:
                st.error("🚨 High Risk (Clinical Threshold Rule)")
            elif sys_bp >= 130 or dia_bp >= 85 or bs >= 8.0:
                st.warning("⚠️ Mid Risk (Clinical Threshold Rule)")
            else:
                st.success("✅ Low Risk (Clinical Threshold Rule)")

with tab2:
    st.subheader("Malaria Thin-Smear Cell Classifier (MobileNetV2 96x96 TFLite)")
    thresh = st.slider("Decision Threshold (Default = 0.50 | Validated Youden's J = 0.0037)", 0.001, 0.90, 0.50, step=0.001, format="%.4f")
    uploaded = st.file_uploader("Upload Thin-Smear Cell Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=180, caption="Uploaded Cell Patch (96x96)")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        tflite_path = os.path.join(base_dir, "malaria-edge-vision", "malaria_edge_quantized.tflite")

        interp = Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        in_idx = next(iter(interp.get_input_details()))["index"]
        out_idx = next(iter(interp.get_output_details()))["index"]

        arr = np.expand_dims(np.array(img.resize((96, 96)), dtype=np.float32) / 255.0, axis=0)
        t0 = time.perf_counter()
        interp.set_tensor(in_idx, arr)
        interp.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        score = float(interp.get_tensor(out_idx).reshape(-1).item(0))

        st.metric("Parasitized Score", f"{score:.4f}", f"Latency: {latency_ms:.2f} ms")
        if score >= thresh:
            st.error("⚠️ Parasitized (Malaria Infected)")
        else:
            st.success("✅ Uninfected")
