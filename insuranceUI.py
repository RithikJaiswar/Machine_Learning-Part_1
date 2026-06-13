import streamlit as st
import pandas as pd
import joblib

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("insurance_model.pkl")
expected_columns = joblib.load("columns.pkl")

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main-header{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#1E88E5;
}

.sub-header{
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:25px;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================

st.markdown(
    '<div class="main-header">💰 Insurance Premium Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">Machine Learning Powered Insurance Risk Assessment Dashboard</div>',
    unsafe_allow_html=True
)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.header("📊 Project Information")

    st.info("""
Model Type: Machine Learning

Use Case:
Predict Insurance Premium

Built Using:
• Python
• Pandas
• Scikit-Learn
• Streamlit
""")

    st.success("Deployment Ready")

# ==========================
# INPUT SECTION
# ==========================

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        100,
        30
    )

    bmi = st.slider(
        "BMI",
        15.0,
        50.0,
        25.0
    )

    children = st.slider(
        "Children",
        0,
        10,
        0
    )

with col2:

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "northwest",
            "northeast",
            "southwest",
            "southeast"
        ]
    )

# ==========================
# PREDICTION BUTTON
# ==========================

if st.button(
    "🚀 Predict Insurance Premium",
    use_container_width=True
):

    # -------------------------
    # INPUT DATA
    # -------------------------

    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        f"sex_{sex}": 1,
        f"smoker_{smoker}": 1,
        f"region_{region}": 1
    }

    input_df = pd.DataFrame([input_data])

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # -------------------------
    # PREDICTION
    # -------------------------

    with st.spinner(
        "Analyzing customer profile..."
    ):

        prediction = model.predict(
            input_df
        )[0]

    st.balloons()

    # ==========================
    # RESULTS
    # ==========================

    st.divider()

    st.subheader("📈 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Estimated Premium",
            f"₹{prediction:,.0f}"
        )

    with col2:

        st.metric(
            "Customer Age",
            age
        )

    with col3:

        st.metric(
            "BMI",
            bmi
        )

    # ==========================
    # RISK CATEGORY
    # ==========================

    if prediction < 10000:

        st.success(
            "🟢 Low Premium Category"
        )

    elif prediction < 30000:

        st.warning(
            "🟡 Medium Premium Category"
        )

    else:

        st.error(
            "🔴 High Premium Category"
        )

    # ==========================
    # PROGRESS BAR
    # ==========================

    premium_score = min(
        int(prediction / 500),
        100
    )

    st.subheader(
        "Premium Risk Indicator"
    )

    st.progress(
        premium_score
    )

    st.caption(
        f"Risk Score: {premium_score}%"
    )

    # ==========================
    # CUSTOMER SUMMARY
    # ==========================

    st.subheader(
        "Customer Summary"
    )

    summary = {
        "Age": age,
        "BMI": bmi,
        "Children": children,
        "Smoker": smoker,
        "Region": region
    }

    st.table(
        pd.DataFrame(
            summary.items(),
            columns=["Feature", "Value"]
        )
    )

    # ==========================
    # DOWNLOAD REPORT
    # ==========================

    report = f"""
INSURANCE PREMIUM REPORT

Predicted Premium:
₹{prediction:,.2f}

Age: {age}
BMI: {bmi}
Children: {children}
Smoker: {smoker}
Region: {region}

Generated By:
Insurance Premium Prediction System
"""

    st.download_button(
        "📥 Download Report",
        report,
        file_name="insurance_report.txt"
    )

# ==========================
# MODEL DETAILS
# ==========================

with st.expander(
    "📚 Model Information"
):

    st.markdown("""
### Machine Learning Pipeline

- Data Cleaning
- Feature Engineering
- Encoding
- Model Training
- Prediction

### Business Value

Helps insurance companies estimate
customer premiums based on
demographic and health factors.
""")

# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "Built by Ritheek | Python • Scikit-Learn • Streamlit"
)