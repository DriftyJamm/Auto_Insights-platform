import streamlit as st
import pandas as pd

from src.eda import run_eda
from src.model import train_model
from src.insights import generate_insights
from src.report import generate_pdf
from src.api import get_crypto_price
from src.auth import login, check_auth

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AutoInsights", page_icon="📊", layout="wide")

# ---------------- LOGIN ----------------
login()

if not check_auth():
    st.warning("Please login to continue")
    st.stop()

# ---------------- STATUS BADGE ----------------
st.markdown("""
<div style="text-align:right; color:lightgreen;">
🟢 System Active
</div>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align: center;'>🚀 AutoInsights Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Turn Raw Data into Business Decisions</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

problem = st.sidebar.selectbox("Select Problem", [
    "Customer Churn",
    "Fraud Detection",
    "Sales Prediction",
    "Student Performance"
])

section = st.sidebar.radio("", [
    "🏠 Overview",
    "📂 Upload Data",
    "📊 EDA",
    "🤖 Model",
    "🔮 Prediction",
    "📈 Insights"
])

# ---------------- SESSION ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- OVERVIEW ----------------
if section == "🏠 Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Problems Supported", "4+")
    col2.metric("Models", "ML")
    col3.metric("Mode", "Real-Time")

    st.markdown("---")

    st.subheader("Real-Time Data")

    try:
        price = get_crypto_price()
        if price:
            st.metric("Bitcoin Price (USD)", f"${price:,.2f}")
        else:
            st.warning("Real-time data unavailable")
    except:
        st.warning("API not responding")

# ---------------- UPLOAD ----------------
elif section == "📂 Upload Data":

    st.markdown("## 📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.success("Dataset uploaded successfully!")
        st.dataframe(df.head())

# ---------------- EDA ----------------
elif section == "📊 EDA":

    st.markdown("## 📊 Exploratory Data Analysis")

    if st.session_state.df is not None:
        run_eda(st.session_state.df)
    else:
        st.warning("Upload dataset first")

# ---------------- MODEL ----------------
elif section == "🤖 Model":

    st.markdown("## 🤖 Model Training")

    if st.session_state.df is not None:
        train_model(st.session_state.df)
    else:
        st.warning("Upload dataset first")

# ---------------- PREDICTION ----------------
elif section == "🔮 Prediction":

    st.markdown("## 🔮 Smart Prediction")

    if "model" not in st.session_state:
        st.warning("Train model first")
        st.stop()

    model = st.session_state.model
    columns = st.session_state.columns
    df = st.session_state.original_df

    st.write("### Enter Input Data")

    input_data = {}

    cols = st.columns(3)

    i = 0

    for col in df.columns:

        if col == df.columns[-1]:  # skip target column
            continue

        unique_vals = df[col].dropna().unique()

        # ---------- NUMERIC ----------
        if pd.api.types.is_numeric_dtype(df[col]):

            val = cols[i % 3].number_input(
                f"{col}",
                float(df[col].mean())
            )

        # ---------- CATEGORICAL ----------
        else:

            val = cols[i % 3].selectbox(
                f"{col}",
                unique_vals
            )

        input_data[col] = val
        i += 1

    # ---------- PREDICT ----------
    if st.button("Predict"):

        df_input = pd.DataFrame([input_data])

        # Convert to dummies
        df_input = pd.get_dummies(df_input)

        # Match training columns
        for col in columns:
            if col not in df_input:
                df_input[col] = 0

        df_input = df_input[columns]

        pred = model.predict(df_input)[0]

        st.markdown("---")

        if pred in [1, "Yes"]:
            st.error("⚠️ High Risk / Negative Outcome")
        else:
            st.success("✅ Low Risk / Positive Outcome")

        # Probability (if available)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df_input)[0][1]

            st.metric("Confidence", f"{prob*100:.2f}%")
            st.progress(float(prob))

# ---------------- INSIGHTS ----------------
elif section == "📈 Insights":

    st.markdown("## 📈 Insights")

    if st.session_state.df is not None:
        generate_insights(st.session_state.df, problem)

        if st.button("Generate Report"):
            summary = [
                f"Rows: {st.session_state.df.shape[0]}",
                f"Columns: {st.session_state.df.shape[1]}",
                f"Problem: {problem}"
            ]

            file = generate_pdf(summary)

            with open(file, "rb") as f:
                st.download_button("Download Report", f)

    else:
        st.warning("Upload dataset first")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>Built with ❤️ | AutoInsights Platform</center>
""", unsafe_allow_html=True)
