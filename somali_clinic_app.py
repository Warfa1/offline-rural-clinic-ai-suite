import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from PIL import Image

# Use lightweight LiteRT runtime on Streamlit Cloud (falls back to tf.lite locally if needed)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

st.set_page_config(page_title="Offline Rural Clinic AI Suite", page_icon="🏥", layout="wide")

def find_model_path(filename):
    """Finds model whether placed in 'models/' subfolder or repository root."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [os.path.join(base_dir, "models", filename), os.path.join(base_dir, filename)]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in models/ or root directory.")

TEXT = {
    "en": {
        "title": "🏥 Offline Rural Clinic AI Suite (Horn of Africa Edition)",
        "subtitle": "Research Prototype Only — Decision Support for Low-Resource Clinics (Somali Region & NFD)",
        "tab_mat": "🤰 Maternal Health Risk Triage (XAI)",
        "tab_mal": "🔬 Edge Malaria Cell Screener (2.41 MB TFLite)",
        "mat_header": "Antenatal Risk Stratification (Somali DHS-Localized XGBoost)",
        "age": "Maternal Age (Years)",
        "gest_week": "Current Gestational Age (Weeks)",
        "parity": "Previous Births (Parity)",
        "prior_loss": "History of Prior Pregnancy Loss / Stillbirth",
        "bleeding": "⚠️ Active Vaginal Bleeding Today (Red-Flag Symptom)",
        "bmi": "Body Mass Index (BMI kg/m²)",
        "interval": "Months Since Last Birth (36 if first pregnancy)",
        "anc_visits": "Number of Antenatal (ANC) Visits So Far",
        "first_anc": "Month of First ANC Visit (9 if none)",
        "tetanus": "Tetanus Toxoid Injections Received",
        "iron": "Taking Iron / Folic Acid Supplements",
        "told_comp": "Counseled on Pregnancy Danger Signs",
        "screen_score": "Routine ANC Tests Done (0=None to 3=BP+Urine+Blood)",
        "edu": "Education Level (0=None, 1=Primary, 2=Secondary, 3=Higher)",
        "wealth": "Household Resource Index (1=Lowest to 5=Highest)",
        "rural": "Rural / Pastoralist Residence",
        "dist": "Distance to Health Facility is a Major Barrier",
        "btn_mat": "Assess Maternal Risk",
        "high_risk": "🚨 HIGH RISK — Urgent Hospital Referral Recommended",
        "low_risk": "✅ STANDARD / LOW RISK — Continue Routine Scheduled ANC",
        "red_flag_msg": "CLINICAL SAFETY OVERRIDE: Active vaginal bleeding requires immediate emergency obstetric evaluation regardless of model score.",
        "mal_header": "Offline Malaria Parasite Screener (Stain-Augmented MobileNetV2 INT8)",
        "upload_img": "Upload Giemsa-Stained Blood Smear Cell Image (PNG/JPG)",
        "thresh": "Site Calibration Threshold (Validated Youden's J = 0.102)",
        "infected": "⚠️ PARASITIZED (Malaria Detected — Confirm via Thick Smear / RDT)",
        "uninfected": "✅ UNINFECTED (No Parasite Detected in Cell)"
    },
    "so": {
        "title": "🏥 Nidaamka AI ee Xarumaha Caafimaadka Miyiga (Geeska Afrika)",
        "subtitle": "Tijaabo Cilmi-baaris Kaliya — Caawinta Shaqaalaha Caafimaadka (Deegaanka Soomaalida & NFD)",
        "tab_mat": "🤰 Qiimeynta Khatarta Hooyada Uurka Leh",
        "tab_mal": "🔬 Baaritaanka Duumada / Kaneecada (2.41 MB Offline)",
        "mat_header": "Qiimeynta Khatarta Uurka (Model lagu tababaray Xogta Hooyooyinka Soomaaliyeed)",
        "age": "Da'da Hooyada (Sano)",
        "gest_week": "Toddobaadka Uurka (1 ilaa 42 Toddobaad)",
        "parity": "Tirada Dhalmada Hore",
        "prior_loss": "Hore ma u dhacday Ilmo-dil / Dhicis?",
        "bleeding": "⚠️ Dhiig-bax firfircoon ma jiraa maanta? (Calaamad Degdeg ah)",
        "bmi": "Miisaanka iyo Dhererka (BMI kg/m²)",
        "interval": "Bilaha u dhexeeya labada dhalmo (36 haddii uurkii 1aad yahay)",
        "anc_visits": "Tirada Booqashooyinka Xarunta Caafimaadka (ANC)",
        "first_anc": "Bishii ugu horreysay ee la booqday xarunta (9 haddii aan la tagin)",
        "tetanus": "Tirada Tallaalka Teetanada ee la qaatay",
        "iron": "Ma qaadataa Kiniiniga Dhiig-kordhinta (Iron/Folic Acid)?",
        "told_comp": "Ma lagala taliyay calaamadaha khatarta uurka?",
        "screen_score": "Baaritaannada la sameeyay (0=Waxba ilaa 3=Cadaadis+Kaadi+Dhiig)",
        "edu": "Heerka Waxbarashada (0=Waxba, 1=Hoose, 2=Sare, 3=Jaamacad)",
        "wealth": "Heerka Dhaqaalaha Qoyska (1=Ugu hooseeya ilaa 5=Ugu sarreeya)",
        "rural": "Deggan Miyi / Reer Guuraa",
        "dist": "Fogaanta Xarunta Caafimaadka waa Caqabad Weyn",
        "btn_mat": "Qiimee Khatarta Hooyada",
        "high_risk": "🚨 KHATAR SARE — U Gudbi Isbitaal Degdeg ah",
        "low_risk": "✅ KHATAR HOOSE — Sii wad Booqashooyinka Caadiga ah ee ANC",
        "red_flag_msg": "DIGNIIN CAAFIMAAD: Dhiig-baxa xilliga uurka wuxuu u baahan yahay gurmad isbitaal oo degdeg ah.",
        "mal_header": "Baaraha Unugyada Duumada (MobileNetV2 INT8 Offline)",
        "upload_img": "Soo geli Sawirka Unugga Dhiigga (PNG/JPG)",
        "thresh": "Xadka Kalibraynta Shaybaarka (Youden's J = 0.102)",
        "infected": "⚠️ DUUMO AYAA LAGA HELAY (Parasitized — Xaqiiji)",
        "uninfected": "✅ NADIIF / DUUMO MA LEH (Uninfected)"
    }
}

lang_choice = st.sidebar.radio("Language / Luuqadda", ["English", "Af-Soomaali"])
L = TEXT["en"] if lang_choice == "English" else TEXT["so"]

st.title(L["title"])
st.warning(L["subtitle"])

tab1, tab2 = st.tabs([L["tab_mat"], L["tab_mal"]])

with tab1:
    st.subheader(L["mat_header"])
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(L["age"], 14, 50, 26)
        gest_week = st.slider(L["gest_week"], 1, 42, 24)
        parity = st.number_input(L["parity"], 0, 15, 3)
        prior_loss = st.checkbox(L["prior_loss"])
        bleeding = st.checkbox(L["bleeding"])
    with col2:
        bmi = st.number_input(L["bmi"], 13.0, 45.0, 21.5)
        interval = st.number_input(L["interval"], 6, 120, 24)
        anc_visits = st.number_input(L["anc_visits"], 0, 15, 1)
        first_anc = st.number_input(L["first_anc"], 1, 9, 5)
        tetanus = st.number_input(L["tetanus"], 0, 5, 1)
    with col3:
        iron = st.checkbox(L["iron"], value=True)
        told_comp = st.checkbox(L["told_comp"])
        screen_score = st.slider(L["screen_score"], 0, 3, 1)
        edu = st.selectbox(L["edu"], list((0, 1, 2, 3)), index=0)
        wealth = st.slider(L["wealth"], 1, 5, 2)
        rural = st.checkbox(L["rural"], value=True)
        dist = st.checkbox(L["dist"], value=True)

    if st.button(L["btn_mat"], type="primary"):
        row = pd.DataFrame([{
            "age": age,
            "is_teen_or_advanced_age": int(age < 18 or age >= 38),
            "parity": parity,
            "grand_multipara": int(parity >= 5),
            "prior_pregnancy_loss": int(prior_loss),
            "bmi": bmi,
            "birth_interval_months": interval,
            "anc_visits": anc_visits,
            "first_anc_month": first_anc,
            "tetanus_injections": tetanus,
            "iron_supplements": int(iron),
            "told_complications": int(told_comp),
            "anc_screening_score": screen_score,
            "education_level": edu,
            "wealth_quintile": wealth,
            "rural_residence": int(rural),
            "facility_distance_barrier": int(dist)
        }])

        model = joblib.load(find_model_path("somali_maternal_transfer_xgb.joblib"))
        prob = float(model.predict_proba(row)[:, 1].item(0))

        if bleeding:
            st.error(L["high_risk"])
            st.error(L["red_flag_msg"])
        elif prob >= 0.45:
            st.error(f"{L['high_risk']} (Risk Score: {prob:.2f})")
        else:
            st.success(f"{L['low_risk']} (Risk Score: {prob:.2f})")

        imp = model.named_steps["imp"]
        clf = model.named_steps["clf"]
        X_imp = pd.DataFrame(imp.transform(row), columns=row.columns)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_imp).reshape(-1)
        shap_df = pd.DataFrame({"Feature": row.columns, "Impact": sv}).sort_values("Impact", key=abs, ascending=True).tail(8)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        colors = ["#d95f02" if v > 0 else "#1b9e77" for v in shap_df["Impact"]]
        ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("SHAP Risk Drivers (Orange = Increases Risk, Green = Lowers Risk)")
        st.pyplot(fig)

with tab2:
    st.subheader(L["mal_header"])
    thresh = st.slider(L["thresh"], 0.01, 0.90, 0.102, step=0.005)
    uploaded = st.file_uploader(L["upload_img"], type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=180, caption="Input Cell Patch (Resized to 96x96)")

        interp = Interpreter(model_path=find_model_path("somali_malaria_stain_aug_int8.tflite"))
        interp.allocate_tensors()
        in_idx = next(iter(interp.get_input_details()))["index"]
        out_idx = next(iter(interp.get_output_details()))["index"]

        arr = np.expand_dims(np.array(img.resize((96, 96)), dtype=np.float32) / 255.0, axis=0)
        t0 = time.perf_counter()
        interp.set_tensor(in_idx, arr)
        interp.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        score = float(interp.get_tensor(out_idx).reshape(-1).item(0))

        st.metric("Parasite Probability Score", f"{score:.4f}", f"Latency: {latency_ms:.2f} ms")
        if score >= thresh:
            st.error(L["infected"])
        else:
            st.success(L["uninfected"])
