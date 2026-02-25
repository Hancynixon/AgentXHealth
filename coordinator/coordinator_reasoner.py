import numpy as np
from sklearn.linear_model import LogisticRegression


class CoordinatorReasoner:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.is_fitted = False

        # Baseline means (estimated from dataset)
        self.lab_baseline = 0.20
        self.physical_baseline = 0.20
        self.demo_baseline = 0.20

    # ======================================================
    # Train stacking layer
    # ======================================================
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    # ======================================================
    # Predict final risk
    # ======================================================
    def reason(self, outputs):

        lab_risk = outputs["lab"]["risk"]
        phys_risk = outputs["physical"]["risk"]
        demo_risk = outputs["demographic"]["risk"]

        features = np.array([[lab_risk, phys_risk, demo_risk]])

        # If stacking model trained
        if self.is_fitted:
            final_risk = self.model.predict_proba(features)[0][1]
        else:
            final_risk = (lab_risk + phys_risk + demo_risk) / 3

        # Binary prediction using 0.5 threshold
        final_prediction = 1 if final_risk >= 0.5 else 0

        # ==========================================
        # Dominance Logic
        # ==========================================

        lab_adj = lab_risk - self.lab_baseline
        phys_adj = phys_risk - self.physical_baseline
        demo_adj = demo_risk - self.demo_baseline

        adjusted_scores = {
            "Lab Agent": lab_adj,
            "Physical Agent": phys_adj,
            "Demographic Agent": demo_adj
        }

        dominant_agent = max(adjusted_scores, key=adjusted_scores.get)

        return {
            "final_risk": float(final_risk),
            "final_prediction": int(final_prediction),
            "dominant_agent": dominant_agent,
            "confidence": float(max(lab_risk, phys_risk, demo_risk))
        }