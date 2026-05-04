import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_model(df):

    st.subheader("🤖 Model Training & Comparison")

    # -------- CLEAN DATA --------
    df = df.dropna()

    if df.shape[0] < 10:
        st.error("Dataset too small to train model")
        return None, None

    # Remove ID column if exists
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # -------- TARGET SELECTION --------
    target = st.selectbox("🎯 Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    # -------- ENCODING --------
    X = pd.get_dummies(X)

    # Reduce size (for Streamlit Cloud performance)
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)
        y = y.loc[X.index]

    # -------- TRAIN TEST SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- TRAIN BUTTON --------
    if st.button("🚀 Train Models"):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=50)
        }

        results = []
        best_model = None
        best_score = 0
        best_preds = None

        st.write("### 🔄 Training in progress...")

        # -------- TRAIN ALL MODELS --------
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

            if acc > best_score:
                best_score = acc
                best_model = model
                best_preds = preds
                best_model_name = name

        results_df = pd.DataFrame(results)

        # -------- SORT MODELS --------
        results_df = results_df.sort_values(by="Accuracy", ascending=False)

        st.success("✅ Models trained successfully!")

        # -------- KPI CARDS --------
        st.write("### 📌 Key Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🏆 Best Model", best_model_name)

        with col2:
            st.metric("📊 Best Accuracy", f"{best_score:.2f}")

        with col3:
            st.metric("🤖 Models Tested", len(results_df))

        st.markdown("---")

        # -------- MODEL TABLE --------
        st.write("### 🏆 Model Ranking")
        st.dataframe(results_df.reset_index(drop=True))

        st.markdown("---")

        # -------- BAR CHART --------
        st.write("### 📈 Model Performance")

        fig, ax = plt.subplots()
        ax.barh(results_df["Model"], results_df["Accuracy"])
        ax.set_xlabel("Accuracy")
        ax.set_title("Model Comparison")

        st.pyplot(fig)

        st.markdown("---")

        # -------- CONFUSION MATRIX --------
        st.write("### 🔍 Confusion Matrix (Best Model)")

        cm = confusion_matrix(y_test, best_preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        st.markdown("---")

        # -------- FEATURE IMPORTANCE --------
        if best_model_name in ["Decision Tree", "Random Forest"]:
            st.write("### 🔍 Feature Importance")

            importance = best_model.feature_importances_

            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False).head(10)

            fig, ax = plt.subplots()
            ax.barh(feat_df["Feature"], feat_df["Importance"])
            ax.set_title("Top 10 Important Features")

            st.pyplot(fig)

            st.markdown("---")

        # -------- MODEL INSIGHTS --------
        st.write("### 🧠 Model Insights")

        st.markdown(f"""
        - The best performing model is **{best_model_name}**
        - Achieved accuracy of **{best_score:.2f}**
        - Tree-based models capture complex patterns better
        - Logistic Regression works well for linear relationships
        - Performance depends on feature quality and dataset size
        """)

        st.markdown("---")

        # -------- SAVE MODEL --------
        st.session_state.model = best_model
        st.session_state.columns = X.columns

        st.success("✅ Model is ready for prediction!")

    return None, X
