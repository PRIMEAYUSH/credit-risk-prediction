"""
Streamlit Credit Risk Prediction App

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
MODEL_PATH = Path("models/credit_risk_model.joblib")

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # First-run fallback: train the same pipeline used by credit_risk_project.py.
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = load_data()
    df["default"] = (df["credit_risk"] == 0).astype(int)
    df = df.drop(columns="credit_risk")

    X = df.drop(columns="default")
    y = df["default"]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X, y)
    return model

df = load_data()
model = load_model()

st.title("💳 Credit Risk Predictor")
st.caption(
    "Machine-learning demonstration using the German Credit dataset. "
    "The prediction estimates default risk and is not financial advice."
)

with st.sidebar:
    st.header("Applicant Information")

    status = st.selectbox("Checking account status", sorted(df["status"].unique()))
    duration = st.slider("Loan duration (months)", 4, 72, 24)
    credit_history = st.selectbox(
        "Credit history", sorted(df["credit_history"].unique())
    )
    purpose = st.selectbox("Loan purpose", sorted(df["purpose"].unique()))
    amount = st.number_input(
        "Loan amount",
        min_value=250,
        max_value=20000,
        value=3000,
        step=250,
    )
    savings = st.selectbox("Savings account", sorted(df["savings"].unique()))
    employment_duration = st.selectbox(
        "Employment duration", sorted(df["employment_duration"].unique())
    )
    installment_rate = st.slider("Installment rate (%)", 1, 4, 2)
    personal_status_sex = st.selectbox(
        "Personal status / sex", sorted(df["personal_status_sex"].unique())
    )
    other_debtors = st.selectbox(
        "Other debtors", sorted(df["other_debtors"].unique())
    )
    present_residence = st.slider("Present residence (years)", 1, 4, 2)
    property_value = st.selectbox("Property", sorted(df["property"].unique()))
    age = st.slider("Age", 18, 75, 30)
    other_installment_plans = st.selectbox(
        "Other installment plans",
        sorted(df["other_installment_plans"].unique()),
    )
    housing = st.selectbox("Housing", sorted(df["housing"].unique()))
    number_credits = st.slider("Number of existing credits", 1, 4, 1)
    job = st.selectbox("Job", sorted(df["job"].unique()))
    people_liable = st.slider("People liable for maintenance", 1, 2, 1)
    telephone = st.selectbox("Telephone", sorted(df["telephone"].unique()))
    foreign_worker = st.selectbox(
        "Foreign worker", sorted(df["foreign_worker"].unique())
    )

input_data = pd.DataFrame([{
    "status": status,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "amount": amount,
    "savings": savings,
    "employment_duration": employment_duration,
    "installment_rate": installment_rate,
    "personal_status_sex": personal_status_sex,
    "other_debtors": other_debtors,
    "present_residence": present_residence,
    "property": property_value,
    "age": age,
    "other_installment_plans": other_installment_plans,
    "housing": housing,
    "number_credits": number_credits,
    "job": job,
    "people_liable": people_liable,
    "telephone": telephone,
    "foreign_worker": foreign_worker,
}])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Applicant Profile")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("Prediction")

    if st.button("Predict Credit Risk", type="primary", use_container_width=True):
        probability = float(model.predict_proba(input_data)[0, 1])
        prediction = int(probability >= 0.50)

        if prediction == 1:
            st.error("⚠️ HIGHER DEFAULT RISK")
            st.metric("Estimated default probability", f"{probability:.1%}")
        else:
            st.success("✅ LOWER DEFAULT RISK")
            st.metric("Estimated default probability", f"{probability:.1%}")

        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "Threshold used by this demo: 50%. "
            "A production credit system should calibrate thresholds using business costs and validation data."
        )

st.divider()

st.subheader("About this project")
m1, m2, m3 = st.columns(3)
m1.metric("Applicants", f"{len(df):,}")
m2.metric("Features", f"{df.shape[1] - 1}")
m3.metric("Observed default rate", f"{(df['credit_risk'].eq(0).mean()):.1%}")

st.info(
    "This application is an educational machine-learning project. "
    "It should not be used as the sole basis for real lending decisions."
)
