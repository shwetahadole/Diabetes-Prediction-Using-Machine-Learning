import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
/* Overall background */
.stApp {
    background: linear-gradient(135deg, #eef5fb 0%, #f8fbff 55%, #e6f7f4 100%);
    color: #18324b;
}

.block-container {
    max-width: 1120px;
    padding-top: 0.5rem;
    padding-bottom: 3rem;
}

/* Premium header */
.hero {
    background: linear-gradient(120deg, #102b4e 0%, #145b78 55%, #159b98 100%);
    padding: 38px;
    border-radius: 24px;
    color: white;
    margin-top: 0;
    margin-bottom: 28px;
    box-shadow: 0 14px 35px rgba(16, 43, 78, 0.18);
    position: relative;
    overflow: hidden;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 12px;
}

.hero p {
    color: #e0f5fa !important;
    font-size: 17px;
    line-height: 1.7;
}

/* Headings */
h2, h3 {
    color: #123c5a !important;
    font-weight: 750 !important;
}

/* Form card */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #dce8f1;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 28px rgba(26, 64, 94, 0.07);
}

/* Input labels */
.stNumberInput label {
    color: #254965 !important;
    font-weight: 650 !important;
    font-size: 14px !important;
}

/* Input boxes */
.stNumberInput input {
    background-color: #f9fcff !important;
    border: 1px solid #d4e2ed !important;
    border-radius: 10px !important;
    color: #173650 !important;
    min-height: 42px;
}

.stNumberInput input:focus {
    border-color: #159b98 !important;
    box-shadow: 0 0 0 2px rgba(21, 155, 152, 0.12) !important;
}

/* Predict button */
div.stFormSubmitButton > button {
    background: linear-gradient(110deg, #146c85, #159b98);
    color: white !important;
    border: none;
    border-radius: 12px;
    min-height: 50px;
    font-size: 16px;
    font-weight: 750;
    box-shadow: 0 6px 16px rgba(21, 119, 139, 0.2);
    transition: all 0.2s ease;
}

div.stFormSubmitButton > button:hover {
    background: linear-gradient(110deg, #104f69, #117f80);
    transform: translateY(-2px);
    box-shadow: 0 9px 20px rgba(21, 119, 139, 0.25);
}

/* Prediction metric */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dce8f1;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 7px 24px rgba(26, 64, 94, 0.07);
}

[data-testid="stMetricLabel"] {
    color: #526b80 !important;
}

[data-testid="stMetricValue"] {
    color: #147f8a !important;
    font-weight: 800;
}

/* Probability progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #147f8a, #24b5a5);
    border-radius: 20px;
}

/* Alerts and information panels */
[data-testid="stAlert"] {
    border-radius: 13px;
}

/* Input help text */
.stNumberInput [data-testid="stTooltipIcon"] {
    color: #148b96;
}

/* Footer */
.disclaimer {
    color: #607487;
    font-size: 13px;
    border-top: 1px solid #d6e4ed;
    padding-top: 20px;
    margin-top: 30px;
    line-height: 1.8;
}

/* Mobile layout */
@media (max-width: 640px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 24px;
        border-radius: 18px;
    }

    .hero h1 {
        font-size: 1.7rem;
        letter-spacing: -0.5px;
    }

    [data-testid="stForm"] {
        padding: 15px;
    }
}

/* Reduce empty space above the page title */
[data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* Remove extra top margin from the header */
.hero {
    margin-top: -1rem !important;
}

/* Reduce Streamlit header height */
[data-testid="stHeader"] {
    height: 0rem !important;
    background: transparent !important;
}

/* Keep the main content close to the top */
[data-testid="stMain"] {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")


try:
    model_data = load_model()
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>Machine learning for educational health-risk exploration.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the measurements below to see how the trained "
    "machine-learning model classifies the supplied values."
)

st.info(
    "Educational project only. This model is not a medical "
    "diagnostic tool and has not been clinically validated."
)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("👤 Patient Measurements")

with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", 0, 20, 0, step=1
        )

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=1,
            max_value=300,
            value=120,
            step=1,
            help="Enter the glucose measurement in mg/dL. The appropriate interpretation depends on how and when it was measured."
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)",
            min_value=1,
            max_value=200,
            value=70,
            step=1,
            help="Enter blood pressure in mmHg. The dataset uses a single blood-pressure measurement; this model is for educational use only."
        )

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
            help="Enter skin thickness in millimeters. A value of zero is treated as missing by this project's data-cleaning process."
        )

    with col2:
        insulin = st.number_input(
            "Insulin (mu U/mL)",
            min_value=0,
            max_value=900,
            value=80,
            step=1,
            help="Enter the insulin measurement in the dataset's units. A value of zero is treated as missing and replaced with the dataset median."
        )

        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=1.0,
            max_value=80.0,
            value=25.0,
            step=0.1,
            help="Enter Body Mass Index (BMI) in kg/m². A value of zero is treated as missing by this project's data-cleaning process."
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="This dataset feature represents a diabetes-related family-history measure. Enter a value from 0 to 3."
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
            help="Enter the person's age in completed years."
        )

    submitted = st.form_submit_button(
        "🔍 Predict", use_container_width=True
    )


st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    patient_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=feature_names
    )

    # Handle zero values treated as missing in the training dataset
    median_values = {
        "Glucose": 117.0,
        "BloodPressure": 72.0,
        "SkinThickness": 29.0,
        "Insulin": 125.0,
        "BMI": 32.3
    }

    missing_fields = []

    for field, median in median_values.items():
        if patient_data.loc[0, field] == 0:
            patient_data.loc[0, field] = median
            missing_fields.append(field)

    if missing_fields:
        st.warning(
            "Zero values were entered for "
            + ", ".join(missing_fields)
            + ". The app substituted median values "
            "from the dataset for this prediction."
        )

    # Scale the input using the scaler saved during model training
    scaled_data = scaler.transform(patient_data)

    # Predict class: 0 or 1
    prediction = model.predict(scaled_data)[0]

    # Calculate probability for the positive class (class 1)
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_data)[0]
        probability = probabilities[
            list(model.classes_).index(1)
        ]

    st.subheader("📊 Prediction Result")

    # Display both the predicted class and estimated probability
    if prediction == 1:
        st.error("Prediction: 1 - Diabetes Detected")
    else:
        st.success("Prediction: 0 - No Diabetes Detected")

    if probability is not None:
        st.metric(
            "Model-estimated diabetes probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.caption(
            "This percentage represents the model's estimated "
            "probability for its positive class. It is not a "
            "validated measure of personal medical risk."
        )

    st.caption(
        "This result is not a diagnosis. Consult a qualified "
        "healthcare professional for medical assessment."
    )


st.markdown("""
<div class="disclaimer">
    <b>About this project:</b> Diabetes Prediction using
    Machine Learning | Built with Python, Scikit-learn and Streamlit.
    <br><br>
    <b>Disclaimer:</b> Predictions are educational only and should
    not be used to make medical decisions.
</div>
""", unsafe_allow_html=True)
