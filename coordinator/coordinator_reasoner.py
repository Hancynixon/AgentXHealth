import numpy as np


class CoordinatorReasoner:
    def __init__(self):
        self.model = None
        self.is_fitted = False

        # Baseline means for dominance scoring (can be overridden)
        self.lab_baseline = 0.20
        self.physical_baseline = 0.20
        self.demo_baseline = 0.20

    # ======================================================
    # Optional: train stacking layer
    # ======================================================
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    # ======================================================
    # Optionally set baselines from training data
    # ======================================================
    def set_baselines(self, lab_probs, phys_probs, demo_probs):
        self.lab_baseline = float(np.mean(lab_probs))
        self.physical_baseline = float(np.mean(phys_probs))
        self.demo_baseline = float(np.mean(demo_probs))

    # ======================================================
    # Safe scalar extractor (handles float, dict, list, ndarray)
    # ======================================================
    @staticmethod
    def _to_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("risk", "probability", "prob", "final_risk", "score"):
                if key in value:
                    return float(value[key])
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return float(value[0])
        if isinstance(value, np.ndarray):
            return float(value.flat[0])
        raise ValueError(f"CoordinatorReasoner: cannot convert to float: {value!r}")

    # ======================================================
    # Predict final fused risk — always returns a clean dict
    # ======================================================
    def reason(self, outputs):
        # Accept both plain floats AND dicts from agents
        lab_risk = self._to_float(outputs["lab"])
        phys_risk = self._to_float(outputs["physical"])
        demo_risk = self._to_float(outputs["demographic"])

        features = np.array([[lab_risk, phys_risk, demo_risk]])

        if self.is_fitted and self.model is not None:
            final_risk = float(self.model.predict_proba(features)[0][1])
        else:
            # Default: simple unweighted average (paper uses weighted in eval script)
            final_risk = (lab_risk + phys_risk + demo_risk) / 3.0

        final_prediction = 1 if final_risk >= 0.5 else 0

        # Dominance logic — which agent contributed most above baseline
        adjusted_scores = {
            "Lab Agent":         lab_risk - self.lab_baseline,
            "Physical Agent":    phys_risk - self.physical_baseline,
            "Demographic Agent": demo_risk - self.demo_baseline,
        }
        dominant_agent = max(adjusted_scores, key=adjusted_scores.get)

        return {
            "final_risk":       round(final_risk, 4),
            "final_prediction": int(final_prediction),
            "dominant_agent":   dominant_agent,
            "agent_risks": {
                "lab":         round(lab_risk, 4),
                "physical":    round(phys_risk, 4),
                "demographic": round(demo_risk, 4),
            },
        }
