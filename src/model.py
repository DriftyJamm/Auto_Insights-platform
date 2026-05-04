import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)


def train_model(df):

    st.subheader("🚀 Advanced Model Training & Evaluation")

    df = df.dropna()

    if df.shape[0] < 20:
        st.error("Dataset too small")
        return None, None

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # -------- TARGET --------
    target = st.selectbox("🎯 Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    X = pd.get_dummies(X)

    if len(X) > 3000:
        X = X.sample(3000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Models"):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100)
        }

        results = []
        best_model = None
        best_score = 0

        for name, model in models.items():

            # Cross-validation (industry level)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1 Score": f1,
                "CV Score": np.mean(cv_scores)
            })

            if acc > best_score:
                best_score = acc
                best_model = model
                best_preds = preds
                best_model_name = name

        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

        st.success("✅ Training Complete")

        # -------- TABS --------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📈 Performance",
            "🔍 Explainability",
            "⬇️ Export"
        ])

        # ================= TAB 1 =================
        with tab1:

            col1, col2, col3 = st.columns(3)

            col1.metric("🏆 Best Model", best_model_name)
            col2.metric("📊 Accuracy", f"{best_score:.2f}")
            col3.metric("🔁 Models Tested", len(results_df))

            st.markdown("---")

            st.write("### Model Ranking")
            st.dataframe(results_df.reset_index(drop=True))

        # ================= TAB 2 =================
        with tab2:

            st.write("### 📈 Accuracy Comparison")

            fig, ax = plt.subplots()
            ax.barh(results_df["Model"], results_df["Accuracy"])
            ax.set_title("Model Accuracy")
            st.pyplot(fig)

            # Confusion Matrix
            st.write("### 🔢 Confusion Matrix")

            cm = confusion_matrix(y_test, best_preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

            # ROC Curve (if binary)
            if len(np.unique(y)) == 2:

                st.write("### 📉 ROC Curve")

                probs = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.legend()
                ax.set_title("ROC Curve")

                st.pyplot(fig)

        # ================= TAB 3 =================
        with tab3:

            if best_model_name in ["Decision Tree", "Random Forest"]:

                st.write("### 🔍 Feature Importance")

                importance = best_model.feature_importances_

                feat_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False).head(10)

                fig, ax = plt.subplots()
                ax.barh(feat_df["Feature"], feat_df["Importance"])
                st.pyplot(fig)

            st.write("### 🧠 Insights")

            st.markdown(f"""
            - Best model: **{best_model_name}**
            - Accuracy: **{best_score:.2f}**
            - Cross-validation improves reliability
            - Tree models capture complex patterns
            - Model performance depends on feature engineering
            """)

        # ================= TAB 4 =================
        with tab4:

            st.write("### ⬇️ Download Trained Model")

            model_file = "trained_model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(best_model, f)

            with open(model_file, "rb") as f:
                st.download_button("Download Model", f, file_name=model_file)

        # -------- SAVE --------
        st.session_state.model = best_model
        st.session_state.columns = X.columns

        st.success("✅ Model ready for prediction")

    return None, X
