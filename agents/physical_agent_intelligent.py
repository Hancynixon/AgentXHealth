import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class PhysicalAgentIntelligent:

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.X_train_ = None

    # --------------------------------------------------
    # Clinical BMI staging (WHO inspired)
    # --------------------------------------------------
    def _bmi_stage(self, bmi):
        if bmi < 18.5:
            return 0  # underweight
        elif bmi < 25:
            return 1  # normal
        elif bmi < 30:
            return 2  # overweight
        else:
            return 3  # obese

    # --------------------------------------------------
    # Blood pressure staging (simplified risk levels)
    # --------------------------------------------------
    def _bp_stage(self, bp):
        if bp < 80:
            return 0
        elif bp < 90:
            return 1
        elif bp < 100:
            return 2
        else:
            return 3

    # --------------------------------------------------
    # Feature builder
    # --------------------------------------------------
    def _build_features(self, df):

        bmi = df["BMI"].astype(float)
        bp = df["BloodPressure"].astype(float)

        bmi_stage = bmi.apply(self._bmi_stage)
        bp_stage = bp.apply(self._bp_stage)

        features = np.column_stack([
            bmi,
            bp,
            bmi_stage,
            bp_stage
        ])

        return features

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