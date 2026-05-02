import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(df):

    df = df.dropna()

    # Assume last column is target
    target = df.columns[-1]

    X = df.drop(target, axis=1)
    y = df[target]

    # Convert categorical to numeric
    X = pd.get_dummies(X)

    # Limit data size (VERY IMPORTANT)
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X
