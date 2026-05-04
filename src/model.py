import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def train_model(df):

    st.subheader("🤖 Model Training & Comparison")

    df = df.dropna()

    # Remove ID column if exists
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Select target
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    # Encode categorical
    X = pd.get_dummies(X)

    # Reduce size for performance
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Models"):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=50)
        }

        results = []
        best_model = None
        best_score = 0

        st.write("### 🔍 Training Models...")

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

        results_df = pd.DataFrame(results)

        # -------- SHOW TABLE --------
        st.success("✅ Models trained successfully!")
        st.write("### 📊 Model Comparison Table")
        st.dataframe(results_df.style.highlight_max(axis=0))

        # -------- BAR CHART --------
        st.write("### 📈 Accuracy Comparison")
        st.bar_chart(results_df.set_index("Model")["Accuracy"])

        # -------- BEST MODEL --------
        best_model_name = results_df.iloc[results_df['Accuracy'].idxmax()]['Model']
        st.markdown(f"### 🏆 Best Model: **{best_model_name}**")

        # -------- CONFUSION MATRIX --------
        st.write("### 🔢 Confusion Matrix (Best Model)")

        cm = confusion_matrix(y_test, best_preds)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # -------- FEATURE IMPORTANCE --------
        if best_model_name in ["Decision Tree", "Random Forest"]:
            st.write("### 🔍 Feature Importance")

            importance = best_model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False).head(10)

            st.bar_chart(feat_df.set_index("Feature"))

        # -------- INTERPRETATION TEXT --------
        st.write("### 🧠 Model Insights")
        st.markdown(f"""
        - The best performing model is **{best_model_name}**
        - Accuracy achieved: **{best_score:.2f}**
        - Model performance depends on feature quality and data distribution
        - Tree-based models capture non-linear patterns better
        """)

        # -------- SAVE MODEL --------
        st.session_state.model = best_model
        st.session_state.columns = X.columns

    return None, X
