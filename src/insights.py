import streamlit as st
import pandas as pd
import numpy as np

def generate_insights(df, problem_type):
    st.subheader("📈 Smart Business Insights")

    # ---------------- BASIC INFO ----------------
    st.write(f"Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**")

    missing = df.isnull().sum().sum()
    st.write(f"Total Missing Values: **{missing}**")

    numeric_cols = df.select_dtypes(include=np.number).columns

    # ---------------- PROBLEM-SPECIFIC INSIGHTS ----------------

    # -------- CUSTOMER CHURN --------
    if problem_type == "Customer Churn":
        st.markdown("### 🔍 Churn Analysis Insights")

        if "tenure" in df.columns:
            avg_tenure = df["tenure"].mean()
            st.write(f"✔ Average tenure is **{avg_tenure:.2f} months**")

        if "monthly_charges" in df.columns:
            high_charge = df["monthly_charges"].mean()
            st.write(f"✔ Customers paying above **{high_charge:.2f}** are high-value users")

        st.markdown("### 💡 Business Recommendation")
        st.write("""
        - Target low-tenure customers with retention offers  
        - Provide discounts for high monthly charge users  
        - Improve onboarding experience  
        """)

    # -------- FRAUD DETECTION --------
    elif problem_type == "Fraud Detection":
        st.markdown("### 🔍 Fraud Detection Insights")

        if "amount" in df.columns:
            avg_amt = df["amount"].mean()
            st.write(f"✔ Average transaction amount: **{avg_amt:.2f}**")

        st.markdown("### 💡 Business Recommendation")
        st.write("""
        - Monitor high-value transactions closely  
        - Implement real-time alerts for unusual spikes  
        - Use stricter verification for large payments  
        """)

    # -------- SALES --------
    elif problem_type == "Sales Prediction":
        st.markdown("### 🔍 Sales Insights")

        if len(numeric_cols) > 0:
            top_feature = df[numeric_cols].corr().sum().idxmax()
            st.write(f"✔ Feature most influencing sales: **{top_feature}**")

        st.markdown("### 💡 Business Recommendation")
        st.write("""
        - Focus on high-performing products  
        - Optimize pricing strategy  
        - Increase marketing during peak demand  
        """)

    # -------- STUDENT --------
    elif problem_type == "Student Performance":
        st.markdown("### 🔍 Student Insights")

        if len(numeric_cols) > 0:
            st.write("✔ Performance varies significantly across features")

        st.markdown("### 💡 Recommendation")
        st.write("""
        - Provide extra support to low-performing students  
        - Focus on attendance and study time  
        - Personalized learning plans  
        """)

    # ---------------- GENERIC INSIGHTS ----------------
    st.markdown("### 📊 General Observations")

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()

        strong_corr = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )

        if len(strong_corr) > 0:
            top_pair = strong_corr.index[0]
            st.write(f"✔ Strong correlation between **{top_pair[0]}** and **{top_pair[1]}**")

    st.markdown("### 🚀 Final Insight")

    st.success("""
    This dataset shows clear patterns that can be leveraged using machine learning 
    to improve decision-making and optimize outcomes.
    """)