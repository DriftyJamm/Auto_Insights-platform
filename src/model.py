import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

def train_model(df):

    st.subheader("Model Training & Comparison")

    df = df.dropna()

    # Select target
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    # Convert categorical
    X = pd.get_dummies(X)

    # Reduce size for cloud performance
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("Train Models"):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=50)
        }

        results = []
        best_model = None
        best_score = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            results.append({
                "Model": name,
                "Accuracy": acc
            })

            if acc > best_score:
                best_score = acc
                best_model = model

        results_df = pd.DataFrame(results)

        st.success("Models trained successfully!")
        st.dataframe(results_df)
        st.bar_chart(results_df.set_index("Model")["Accuracy"])

        # ✅ SAVE MODEL (VERY IMPORTANT)
        st.session_state.model = best_model
        st.session_state.columns = X.columns

    return None, X
