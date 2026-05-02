import streamlit as st
import pandas as pd

from src.eda import run_eda
from src.model import train_model
from src.insights import generate_insights
from src.report import generate_pdf
from src.api import get_crypto_price
from src.auth import login, check_auth

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AutoInsights", layout="wide")
#st.write("App started")

# ---------------- LOGIN ----------------
login()

if not check_auth():
    st.warning("Please login to continue")
    st.stop()

# ---------------- LOAD CSS ----------------
def load_css():
    try:
        with open("assets/style.css") as f:  # make sure name matches
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

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

    with col1:
        st.markdown("<div class='card'><div class='metric'>4+</div>Problems Supported</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><div class='metric'>ML</div>Automated Models</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><div class='metric'>Real-Time</div>Predictions</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>This platform allows users to upload datasets, analyze them, train models, and generate business insights automatically.</div>", unsafe_allow_html=True)

    # -------- REAL-TIME DATA --------
    st.subheader("Real-Time Data")
    try:
        price = get_crypto_price()
        st.write("DEBUG:", price)
        if price:
            st.metric("Bitcoin Price (USD)", f"${price:,.2f}")
        else:
            st.warning("Real-time data unavailable")
    except:
        st.warning("API not responding")

# ---------------- UPLOAD ----------------
elif section == "📂 Upload Data":
    st.markdown("<div class='section-title'>Upload Dataset</div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.success("Dataset uploaded successfully!")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- EDA ----------------
elif section == "📊 EDA":
    st.markdown("<div class='section-title'>Exploratory Analysis</div>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("Dataset Shape:", st.session_state.df.shape)
            st.write("Missing Values:")
            st.write(st.session_state.df.isnull().sum())
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            run_eda(st.session_state.df)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Upload dataset first")

# ---------------- MODEL ----------------
elif section == "Model":
    st.markdown("<div class='section-title'>Model Training</div>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        try:
            st.write("Running model...")  # DEBUG

            model, X = train_model(st.session_state.df)

            st.write("Model trained")  # DEBUG

            if model is not None:
                st.session_state.model = model
                st.session_state.columns = X.columns

        except Exception as e:
            st.error(f"Model Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Upload dataset first")
    #st.markdown("<div class='section-title'>Model Training</div>", unsafe_allow_html=True)

    #if st.session_state.df is not None:
        #st.markdown("<div class='card'>", unsafe_allow_html=True)

        #try:
            #model, X = train_model(st.session_state.df)

            #if model is not None:
                #st.session_state.model = model
                #st.session_state.columns = X.columns

        #except Exception as e:
            #st.error(f"Model Error: {e}")

        #st.markdown("</div>", unsafe_allow_html=True)
    #else:
        #st.warning("Upload dataset first")

# ---------------- PREDICTION ----------------
elif section == "🔮 Prediction":
    st.markdown("<div class='section-title'>Prediction Interface</div>", unsafe_allow_html=True)

    if "model" in st.session_state:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        input_data = {}
        cols = st.columns(3)

        for i, col in enumerate(st.session_state.columns):
            input_data[col] = cols[i % 3].number_input(f"{col}")

        if st.button("Predict"):
            try:
                df_input = pd.DataFrame([input_data])
                pred = st.session_state.model.predict(df_input)
                st.success(f"Prediction Result: {pred[0]}")
            except:
                st.error("Input mismatch. Please retrain model.")

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Train model first")

# ---------------- INSIGHTS ----------------
elif section == "📈 Insights":
    st.markdown("<div class='section-title'>Business Insights</div>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        try:
            generate_insights(st.session_state.df, problem)
        except Exception as e:
            st.error(f"Insights Error: {e}")

        # -------- PDF DOWNLOAD --------
        if st.button("Generate Report"):
            summary = [
                f"Rows: {st.session_state.df.shape[0]}",
                f"Columns: {st.session_state.df.shape[1]}",
                f"Problem Type: {problem}",
                "Insights generated successfully"
            ]

            file = generate_pdf(summary)

            with open(file, "rb") as f:
                st.download_button("Download PDF", f, file_name="report.pdf")

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Upload dataset first")
