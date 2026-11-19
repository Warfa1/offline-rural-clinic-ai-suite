import threading
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

st.set_page_config(page_title="Offline Health AI Prototypes (Research Demo)", page_icon="🔬")

HERE = Path(__file__).resolve().parent


def find_file(names, folders):
    for folder in folders:
        for name in names:
            p = HERE / folder / name if folder else HERE / name
            if p.exists():
                return str(p)
    return None


# ---------------------------------------------------------------- malaria
@st.cache_resource
def load_malaria(path):
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, threading.Lock()


def predict_malaria(interpreter, lock, image):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    img = image.convert("RGB").resize((96, 96), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)[None, ...]  # 0-255, rescaling is inside the model
    with lock:
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        p_uninfected = float(interpreter.get_tensor(out["index"])[0][0])
    return 1.0 - p_uninfected  # label 0 = Parasitized, label 1 = Uninfected


# --------------------------------------------------------------- maternal
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
RISK_NAMES = ["Low risk", "Mid risk", "High risk"]


@st.cache_resource
def load_maternal(path):
    import xgboost as xgb
    if path.endswith(".json"):
        model = xgb.XGBClassifier()
        model.load_model(path)
    else:
        import joblib
        model = joblib.load(path)
    return model.get_booster()


def predict_maternal(booster, values):
    import xgboost as xgb
    df = pd.DataFrame([values], columns=FEATURES)
    dm = xgb.DMatrix(df)
    probs = np.asarray(booster.predict(dm))[0]
    contribs = None
    try:
        c = booster.predict(dm, pred_contribs=True)  # (1, classes, features + 1)
        contribs = np.asarray(c)[0][int(np.argmax(probs))][:-1]
    except Exception:
        pass
    return probs, contribs


# --------------------------------------------------------------------- UI
st.title("Offline Health AI Prototypes (Research Demo)")
st.warning(
    "Research demo only. Not a medical device. Not for diagnosis or treatment decisions. "
    "Both models were trained on small public datasets and have not been validated in any "
    "clinic. Do not enter or upload real patient information."
)

tab_malaria, tab_maternal = st.tabs(["Malaria cell classifier", "Maternal risk triage"])

with tab_malaria:
    st.subheader("Malaria cell classifier")
    st.write(
        "Upload a close-up image of ONE red blood cell, like the images in the NIH malaria "
        "dataset. Whole-slide or wide microscope photos will not give meaningful results."
    )
    malaria_path = find_file(["malaria_edge_quantized.tflite"], ["", "malaria-edge-vision"])
    if malaria_path is None:
        st.error("Model file malaria_edge_quantized.tflite was not found in the repository.")
    else:
        interpreter, lock = load_malaria(malaria_path)
        uploaded = st.file_uploader("Single blood cell image", type=["png", "jpg", "jpeg"])
        with st.expander("Or use the camera"):
            photo = st.camera_input("Take a photo of a single cell image")
        source = uploaded or photo
        if source is not None:
            image = Image.open(source)
            st.image(image, caption="Image received", width=240)
            p_par = predict_malaria(interpreter, lock, image)
            if p_par >= 0.7:
                st.error("Possible parasitized cell (model suggests infection).")
            elif p_par <= 0.3:
                st.success("Likely uninfected cell.")
            else:
                st.info("Uncertain. The model is not confident about this image.")
            st.write(f"Parasitized probability: **{p_par * 100:.1f}%**")
            st.progress(min(max(p_par, 0.0), 1.0))
            st.caption("Screening aid only. Confirm with microscopy or a rapid test. "
                       "Images are processed in memory and not saved by this app.")

with tab_maternal:
    st.subheader("Maternal health risk triage")
    st.write(
        "Enter example vital signs to see how the model estimates risk (low, mid, high) and "
        "which values pushed the estimate. This is an illustration of the method, not "
        "medical advice."
    )
    st.error(
        "A 'low risk' result does NOT mean a pregnancy is safe. If someone has severe "
        "headache, vaginal bleeding, blurred vision, fits, severe abdominal pain or reduced "
        "baby movement, they need a health worker or hospital immediately."
    )
    maternal_path = find_file(
        ["maternal_xgb_triage.json", "maternal_xgb_triage.joblib"],
        ["", "maternal-xai-triage"],
    )
    if maternal_path is None:
        st.error("Maternal model file (maternal_xgb_triage.json or .joblib) was not found "
                 "in the repository.")
    else:
        booster = load_maternal(maternal_path)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 10, 60, 25)
        sys_bp = c1.number_input("Systolic blood pressure (mmHg)", 60, 220, 120)
        dia_bp = c1.number_input("Diastolic blood pressure (mmHg)", 40, 140, 80)
        bs = c2.number_input("Blood sugar (mmol/L)", 3.0, 25.0, 7.0, step=0.1)
        temp = c2.number_input("Body temperature (°F)", 95.0, 106.0, 98.6, step=0.1)
        hr = c2.number_input("Heart rate (beats per minute)", 40, 160, 76)
        st.caption("Units follow the public UCI dataset the model was trained on "
                   "(blood sugar in mmol/L, temperature in °F). Values far outside the "
                   "usual range may give unreliable results.")

        if st.button("Estimate risk"):
            probs, contribs = predict_maternal(booster, [age, sys_bp, dia_bp, bs, temp, hr])
            top = int(np.argmax(probs))
            message = f"Model estimate: {RISK_NAMES[top]}"
            if top == 2:
                st.error(message)
            elif top == 1:
                st.warning(message)
            else:
                st.success(message)
            st.write("Estimated probabilities:")
            st.dataframe(
                pd.DataFrame({"Risk level": RISK_NAMES,
                              "Probability (%)": np.round(probs * 100, 1)}),
                hide_index=True,
            )
            if contribs is not None:
                st.write(f"What pushed the estimate toward '{RISK_NAMES[top]}' "
                         "(longer bar = stronger effect):")
                st.bar_chart(pd.Series(contribs, index=FEATURES))
            st.caption("Trained on 1,014 public records; tested on 203. Not clinically "
                       "validated. Entered values are not saved by this app.")

        shap_path = find_file(["shap_feature_importance.png"], ["", "maternal-xai-triage"])
        if shap_path:
            with st.expander("Overall feature importance (SHAP, from the notebook)"):
                st.image(shap_path)
