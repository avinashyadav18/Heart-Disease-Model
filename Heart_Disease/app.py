import os

import joblib
import pandas as pd
import streamlit as st




st.set_page_config(
    page_title="Heart Disease Risk Estimator",
    page_icon="❤️",
    layout="wide",
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_files():
    model = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    return model, scaler, list(columns)


model, scaler, expected_columns = load_files()




st.markdown(
    """
<style>
.main-title { font-size: 40px; font-weight: 700; text-align: center; margin-bottom: 4px; }
.subtitle   { text-align: center; font-size: 17px; color: gray; margin-bottom: 28px; }
.section-title { font-size: 21px; font-weight: 600; margin: 18px 0 6px 0; }
.result-box { padding: 22px; border-radius: 14px; text-align: center; margin-top: 12px; }
.result-low      { background: rgba(46, 160, 67, 0.12);  border: 1px solid rgba(46, 160, 67, 0.5); }
.result-moderate { background: rgba(227, 160, 8, 0.12);  border: 1px solid rgba(227, 160, 8, 0.5); }
.result-high     { background: rgba(218, 54, 51, 0.12);  border: 1px solid rgba(218, 54, 51, 0.5); }
.result-box h2 { margin: 0 0 6px 0; }
.result-box p  { margin: 0; color: gray; }
</style>
""",
    unsafe_allow_html=True,
)




st.markdown('<div class="main-title">❤️ Heart Disease Risk Estimator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter a few clinical values and see how similar patients '
    "in the dataset were classified.</div>",
    unsafe_allow_html=True,
)




with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a **K-Nearest Neighbors** model. It compares your input "
        "with the most similar patients from the training data and checks "
        "how many of them had heart disease."
    )
    st.divider()
    st.subheader("Model")
    st.write("Algorithm: KNN")
    st.write("Preprocessing: feature scaling")
    k_value = getattr(model, "n_neighbors", None)
    if k_value:
        st.write(f"Neighbors compared (k): {k_value}")
    st.divider()
    st.caption("Educational project only. Not a medical diagnosis.")



with st.form("patient_form"):

    # ---- Patient ----
    st.markdown('<div class="section-title">Patient</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age (years)", 18, 100, 50)
    with c2:
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

    # ---- Vitals and lab values ----
    st.markdown('<div class="section-title">Vitals and lab values</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        resting_bp = st.slider(
            "Resting blood pressure (mm Hg)", 80, 200, 120,
            help="Blood pressure measured while at rest.",
        )
        cholesterol = st.slider(
            "Serum cholesterol (mg/dl)", 100, 600, 200,
            help="Total cholesterol from a blood test.",
        )
    with c2:
        max_heart_rate = st.slider(
            "Maximum heart rate achieved (bpm)", 60, 220, 150,
            help="Highest heart rate reached during the exercise test.",
        )
        fasting_bs = st.radio(
            "Fasting blood sugar above 120 mg/dl?", ["No", "Yes"], horizontal=True
        )

    # ---- Symptoms and ECG ----
    st.markdown('<div class="section-title">Symptoms and ECG</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        chest_pain = st.selectbox(
            "Chest pain type",
            ["Asymptomatic", "Non-Anginal Pain", "Atypical Angina", "Typical Angina"],
            help="Asymptomatic means no chest pain symptoms were reported.",
        )
    with c2:
        resting_ecg = st.selectbox(
            "Resting ECG result",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
            help="ECG recorded at rest, before the exercise test.",
        )

    # ---- Exercise test ----
    st.markdown('<div class="section-title">Exercise test</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        exercise_angina = st.radio(
            "Chest pain during exercise?", ["No", "Yes"], horizontal=True
        )
    with c2:
        old_peak = st.slider(
            "ST depression during exercise", 0.0, 6.0, 0.0, step=0.1,
            help="How much the ST segment of the ECG drops during exercise compared to rest (Oldpeak).",
        )
    with c3:
        slope = st.selectbox(
            "ST segment slope at peak exercise",
            ["Upsloping", "Flat", "Downsloping"],
            help="Shape of the ST segment at the peak of exercise.",
        )

    submitted = st.form_submit_button("Estimate risk", use_container_width=True)


if submitted:

    # Every possible feature, using the usual one-hot names from pd.get_dummies.
    # Only the ones present in columns.pkl are used.
    candidates = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": int(fasting_bs == "Yes"),
        "MaxHR": max_heart_rate,
        "Max_HR": max_heart_rate,
        "Oldpeak": old_peak,
        # binary columns (works if you label-encoded these)
        "Sex": int(sex == "Male"),
        "ExerciseAngina": int(exercise_angina == "Yes"),
        # one-hot columns
        "Sex_M": int(sex == "Male"),
        "Sex_F": int(sex == "Female"),
        "ExerciseAngina_Y": int(exercise_angina == "Yes"),
        "ExerciseAngina_N": int(exercise_angina == "No"),
        "ChestPainType_TA": int(chest_pain == "Typical Angina"),
        "ChestPainType_ATA": int(chest_pain == "Atypical Angina"),
        "ChestPainType_NAP": int(chest_pain == "Non-Anginal Pain"),
        "ChestPainType_ASY": int(chest_pain == "Asymptomatic"),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST": int(resting_ecg == "ST-T Wave Abnormality"),
        "RestingECG_LVH": int(resting_ecg == "Left Ventricular Hypertrophy"),
        "ST_Slope_Up": int(slope == "Upsloping"),
        "ST_Slope_Flat": int(slope == "Flat"),
        "ST_Slope_Down": int(slope == "Downsloping"),
    }

    # Fail loudly if the model expects a column we cannot build.
    missing = [col for col in expected_columns if col not in candidates]
    if missing:
        st.error(
            "Input mismatch: the model expects columns this app does not create: "
            f"{missing}. Compare them with the names in columns.pkl."
        )
        st.stop()

    input_df = pd.DataFrame([{col: candidates[col] for col in expected_columns}])
    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    probability = (
        float(model.predict_proba(input_scaled)[0][1])
        if hasattr(model, "predict_proba")
        else float(prediction)
    )

    # Risk tier
    if probability < 0.30:
        tier, css, heading = "low", "result-low", "Lower likelihood of heart disease"
    elif probability < 0.60:
        tier, css, heading = "moderate", "result-moderate", "Moderate likelihood of heart disease"
    else:
        tier, css, heading = "high", "result-high", "Higher likelihood of heart disease"

    # Explain the number in plain words (KNN probability = neighbor vote)
    k_value = getattr(model, "n_neighbors", None)
    if k_value:
        votes = round(probability * k_value)
        detail = (
            f"{votes} of the {k_value} most similar patients in the training "
            "data had heart disease."
        )
    else:
        detail = "Based on patterns learned from the training data."

    st.divider()
    st.subheader("Result")

    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            f'<div class="result-box {css}"><h2>{heading}</h2><p>{detail}</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.metric("Model score", f"{probability * 100:.0f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    if tier != "low":
        st.warning(
            "This is a statistical estimate from a small dataset, not a diagnosis. "
            "If you are worried about your heart health, please consult a doctor."
        )
    else:
        st.info(
            "A low score does not rule out heart disease. This tool is for learning, "
            "not for medical decisions."
        )



st.divider()
st.caption("Heart Disease Risk Estimator | scikit-learn + Streamlit | Educational project")
