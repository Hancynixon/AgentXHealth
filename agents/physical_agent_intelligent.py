import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PhysicalAgentIntelligent:
    """
    PhysicalAgentIntelligent
    ------------------------
    Learns diabetes risk from physical measurements
    (BMI, BloodPressure) using an interpretable model.
    """

    def __init__(self):
        self.model = LogisticRegression(solver="liblinear")
        self.features = ["BMI", "BloodPressure"]
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_phys = X[self.features].copy()
        self.model.fit(X_phys, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict:
        if not self.is_fitted:
            raise RuntimeError("PhysicalAgentIntelligent is not fitted.")

        X_phys = X[self.features].copy()

        risk = self.model.predict_proba(X_phys)[0, 1]

        explanations = dict(
            zip(self.features, self.model.coef_[0])
        )

        counterfactuals = {
            "BMI": "reduce" if explanations["BMI"] > 0 else "maintain",
            "BloodPressure": "reduce" if explanations["BloodPressure"] > 0 else "maintain"
        }

        return {
            "physical_risk_score": float(risk),
            "physical_explanations": explanations,
            "physical_counterfactuals": counterfactuals
        }
