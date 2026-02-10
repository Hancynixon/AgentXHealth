import pandas as pd
from sklearn.linear_model import LogisticRegression


class DemographicAgentIntelligent:
    """
    DemographicAgentIntelligent
    ---------------------------
    Estimates baseline diabetes risk using demographic features.
    Designed to provide population-level context, not dominance.
    """

    def __init__(self):
        self.features = ["Age", "Pregnancies"]
        self.model = LogisticRegression(solver="liblinear")
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y: pd.Series):
        X_demo = df[self.features].copy()
        self.model.fit(X_demo, y)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame) -> dict:
        if not self.is_fitted:
            raise RuntimeError("DemographicAgentIntelligent is not fitted.")

        X_demo = df[self.features].copy()
        risk = self.model.predict_proba(X_demo)[0, 1]

        explanations = dict(
            zip(self.features, self.model.coef_[0])
        )

        return {
            "demographic_risk_score": float(risk),
            "demographic_explanations": explanations
        }
