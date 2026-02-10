import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier


class LabAgentIntelligent:
    """
    Intelligent Lab Agent
    ---------------------
    Learns lab-specific diabetes risk using an explainable-by-design model
    and produces risk, explanations, and counterfactual insights.
    """

    def __init__(self):
        self.features = ["Glucose", "Insulin"]
        self.model = ExplainableBoostingClassifier(
            interactions=0,
            random_state=42
        )
        self.fitted = False

    def fit(self, df: pd.DataFrame, y: pd.Series):
        """
        Train the lab-specific risk model.
        """
        X = df[self.features].copy()
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict lab risk and provide explanations.
        """
        if not self.fitted:
            raise RuntimeError("LabAgentIntelligent must be fitted first.")

        X = df[self.features].copy()

        # Risk probability
        prob = self.model.predict_proba(X)[0, 1]

        # Feature contributions (local explanation)
        contributions = self.model.explain_local(X).data(0)["scores"]
        explanation = dict(zip(self.features, contributions))

        # Counterfactual reasoning (simple, actionable)
        counterfactuals = self._counterfactual_suggestions(explanation)

        return {
            "lab_risk_score": float(prob),
            "lab_explanations": explanation,
            "lab_counterfactuals": counterfactuals
        }

    def _counterfactual_suggestions(self, explanation: dict) -> dict:
        """
        Generate simple counterfactual guidance based on feature effects.
        """
        suggestions = {}

        for feature, effect in explanation.items():
            if effect > 0:
                suggestions[feature] = "reduce"
            else:
                suggestions[feature] = "maintain"

        return suggestions
