import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_model(df):
    st.subheader("Model Selection & Comparison")

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical to numeric
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("Train & Compare Models"):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        results = []

        best_model = None
        best_score = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted')

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec
            })

            if acc > best_score:
                best_score = acc
                best_model = model

        results_df = pd.DataFrame(results)

        st.success("Models trained successfully!")

        st.write("### 📊 Model Comparison")
        st.dataframe(results_df)

        st.write(f"🏆 Best Model: **{results_df.iloc[results_df['Accuracy'].idxmax()]['Model']}**")

        # Bar Chart
        st.write("### 📈 Accuracy Comparison")
        st.bar_chart(results_df.set_index("Model")["Accuracy"])

        return best_model, X

    return None, X