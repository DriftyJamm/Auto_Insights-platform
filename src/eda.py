import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    st.subheader("Dataset Overview")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_cols) > 0:
        col = st.selectbox("Select column", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
        st.pyplot(fig)