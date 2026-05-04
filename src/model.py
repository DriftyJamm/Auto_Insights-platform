import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.express as px
import plotly.figure_factory as ff


def train_model(df):

    st.subheader("📊 Model Training & Comparison")

    df = df.dropna()

    # Remove useless ID column
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Target selection
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    # Convert categorical → numeric
    X = pd.get_dummies(X)

    # Limit size for Streamlit cloud
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Models"):

        with st.spinner("Training models..."):

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=50)
            }

            results = []
            best_model = None
            best_score = 0

            for name, model in models.items():

                cv_score = cross_val_score(model, X_train, y_train, cv=3).mean()

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "CV Score": cv_score
                })

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_preds = preds
                    best_name = name

            df_res = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

        st.session_state.model = best_model
        st.session_state.columns = X.columns
        st.session_state.original_df = df   # 👈 VERY IMPORTANT

        # ---------------- RESULTS ----------------
        st.success("✅ Models trained successfully!")

        col1, col2 = st.columns(2)
        col1.metric("🏆 Best Model", best_name)
        col2.metric("🎯 Accuracy", f"{best_score:.2f}")

        st.dataframe(df_res)

        # ---------------- BAR CHART ----------------
        fig = px.bar(
            df_res,
            x="Accuracy",
            y="Model",
            orientation="h",
            text="Accuracy",
            title="📊 Model Performance Comparison"
        )

        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- SCATTER ----------------
        fig2 = px.scatter(
            df_res,
            x="CV Score",
            y="Accuracy",
            text="Model",
            size="Accuracy",
            title="📈 Accuracy vs Cross-Validation"
        )

        fig2.update_layout(template="plotly_dark", height=350)

        st.plotly_chart(fig2, use_container_width=True)

        # ---------------- CONFUSION MATRIX ----------------
        cm = confusion_matrix(y_test, best_preds)

        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues"
        )

        fig_cm.update_layout(
            title="Confusion Matrix",
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(fig_cm, use_container_width=True)

        # ---------------- FEATURE IMPORTANCE ----------------
        if hasattr(best_model, "feature_importances_"):

            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            fig3 = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="🔥 Top Feature Importance"
            )

            fig3.update_layout(template="plotly_dark", height=400)

            st.plotly_chart(fig3, use_container_width=True)
            # Save top 8 important features
     if hasattr(best_model, "feature_importances_"):
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        top_features = feat_df["Feature"].head(8).tolist()
        st.session_state.top_features = top_features
    else:
        st.session_state.top_features = X.columns[:8].tolist()

        

        # ---------------- SAVE MODEL ----------------
        st.session_state.model = best_model
        st.session_state.columns = X.columns

        with open("model.pkl", "wb") as f:
            pickle.dump(best_model, f)

    return None, X
