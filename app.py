```python
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD FILES ----------------

model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- HEADER ----------------

st.title("❤️ Heart Disease Risk Predictor")

st.markdown("""
This application uses a **K-Nearest Neighbors (KNN)** Machine Learning model
to estimate the likelihood of heart disease based on patient health information.
""")

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("ℹ️ About Project")

    st.markdown("""
**Model:** K-Nearest Neighbors (KNN)

**Tools Used**
- Python
- Pandas
- Scikit-Learn
- Streamlit

**Purpose**
Predict heart disease risk using clinical patient data.
""")

# ---------------- INPUT FORM ----------------

st.subheader("🩺 Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    chestpain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_BP = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=50,
        max_value=700,
        value=200
    )

with col2:

    FastingBS = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    Resting_ECG = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    raw_input = {
        "Age": age,
        "RestingBP": resting_BP,
        "Cholesterol": cholesterol,
        "FastingBS": FastingBS,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        f"Sex_{sex}": 1,
        f"ChestPainType_{chestpain}": 1,
        f"RestingECG_{Resting_ECG}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    with st.spinner("Analyzing patient data..."):

        prediction = model.predict(scaled_input)[0]

        try:
            probability = model.predict_proba(
                scaled_input
            )[0][1]
        except:
            probability = None

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ High Risk of Heart Disease Detected"
        )

    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )

    # ---------------- RISK SCORE ----------------

    if probability is not None:

        st.subheader("Risk Score")

        st.progress(
            min(int(probability * 100), 100)
        )

        st.metric(
            "Predicted Risk",
            f"{probability * 100:.2f}%"
        )

    # ---------------- PATIENT SUMMARY ----------------

    st.subheader("Patient Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Age", age)

    with c2:
        st.metric("Cholesterol", cholesterol)

    with c3:
        st.metric("Max Heart Rate", max_hr)

# ---------------- MODEL INFO ----------------

with st.expander("📚 Model Information"):

    st.markdown("""
### Machine Learning Pipeline

- Algorithm: **K-Nearest Neighbors (KNN)**
- Feature Scaling: **StandardScaler**
- Features:
    - Age
    - Blood Pressure
    - Cholesterol
    - ECG Results
    - Chest Pain Type
    - Heart Rate
    - Exercise Angina
    - ST Slope

### Note

This application is intended for educational purposes only and should not be used as a substitute for professional medical diagnosis.
""")

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Built using Python, Scikit-Learn, Pandas, Joblib and Streamlit"
)
```
