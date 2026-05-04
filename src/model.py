import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix


def train_model(df):

    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    X = pd.get_dummies(X)

    if len(X) > 2000:
        X = X.sample(2000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    if st.button("Train Models"):

        with st.spinner("Training..."):

            models = {
                "Logistic": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier()
            }

            results = []
            best_model = None
            best_score = 0

            for name, model in models.items():

                cv = cross_val_score(model, X_train, y_train, cv=3).mean()

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "CV": cv
                })

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_preds = preds
                    best_name = name

            df_res = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

            st.success("Model trained!")

            st.metric("Best Model", best_name)
            st.metric("Accuracy", round(best_score, 2))

            st.dataframe(df_res)

            fig, ax = plt.subplots()
            ax.barh(df_res["Model"], df_res["Accuracy"])
            st.pyplot(fig)

            cm = confusion_matrix(y_test, best_preds)

            fig, ax = plt.subplots()
            ax.imshow(cm)
            st.pyplot(fig)

            st.session_state.model = best_model
            st.session_state.columns = X.columns

            # Save model
            with open("model.pkl", "wb") as f:
                pickle.dump(best_model, f)

    return None, X
