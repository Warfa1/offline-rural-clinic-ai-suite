import threading
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

st.set_page_config(page_title="Malaria Cell Classifier (Research Demo)", page_icon="🔬")

MODEL_NAME = "malaria_edge_quantized.tflite"


def find_model():
    here = Path(__file__).resolve().parent
    for p in [here / MODEL_NAME, here / "malaria-edge-vision" / MODEL_NAME]:
        if p.exists():
            return str(p)
    return None


@st.cache_resource
def load_model(path):
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, threading.Lock()


def predict(interpreter, lock, image):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    img = image.convert("RGB").resize((96, 96), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)[None, ...]  # 0-255, rescaling is inside the model
    with lock:
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        p_uninfected = float(interpreter.get_tensor(out["index"])[0][0])
    return 1.0 - p_uninfected, p_uninfected  # label 0 = Parasitized, label 1 = Uninfected


st.title("Malaria Cell Classifier (Research Demo)")
st.warning(
    "Research demo. Not a medical device. Not for diagnosis or treatment decisions. "
    "Trained only on public images of single, cropped red blood cells and not validated "
    "in any clinic. Always confirm malaria with microscopy or a rapid diagnostic test."
)
st.write(
    "Upload a close-up image of ONE red blood cell, like the images in the NIH malaria "
    "dataset. Whole-slide or wide microscope photos will not give meaningful results. "
    "Please do not upload images of real patients or anything identifiable."
)

model_path = find_model()
if model_path is None:
    st.error(f"Model file {MODEL_NAME} was not found in the repository.")
    st.stop()

interpreter, lock = load_model(model_path)

uploaded = st.file_uploader("Single blood cell image", type=["png", "jpg", "jpeg"])
with st.expander("Or use the camera"):
    photo = st.camera_input("Take a photo of a single cell image")

source = uploaded or photo
if source is not None:
    image = Image.open(source)
    st.image(image, caption="Image received", width=240)
    p_par, p_unin = predict(interpreter, lock, image)

    if p_par >= 0.7:
        st.error("Possible parasitized cell (model suggests infection).")
    elif p_par <= 0.3:
        st.success("Likely uninfected cell.")
    else:
        st.info("Uncertain. The model is not confident about this image.")

    st.write(f"Parasitized probability: **{p_par * 100:.1f}%**")
    st.progress(min(max(p_par, 0.0), 1.0))
    st.caption("Screening aid only. Confirm with microscopy or a rapid test. Images are processed in memory and not saved by this app.")
