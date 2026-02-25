import numpy as np
from sklearn.linear_model import LogisticRegression


class DemographicAgentIntelligent:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.X_train_ = None

    # --------------------------------------------------
    # Feature builder
    # --------------------------------------------------

    def _build_features(self, df):

        age = df["Age"].astype(float)

        if "Pregnancies" in df.columns:
            preg = df["Pregnancies"].astype(float)
        else:
            preg = np.zeros(len(df))

        return np.column_stack([age, preg])

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def fit(self, df, y):

        X = self._build_features(df)

        self.model.fit(X, y)

        self.X_train_ = X.copy()

        return self

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(self, df):

        X = self._build_features(df)

        prob = self.model.predict_proba(X)[0][1]

        return {
            "risk": round(float(prob), 4)
        }
