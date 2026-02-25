import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LabAgentIntelligent:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        self.is_fitted = False
        self.X_train_ = None  # <-- important for SHAP

    # --------------------------------------------------
    # Clinical indices
    # --------------------------------------------------

    def _compute_homa_ir(self, glucose, insulin):
        if glucose <= 0 or insulin <= 0:
            return 0.0
        return (glucose * insulin) / 405.0

    def _compute_quicki(self, glucose, insulin):
        if glucose <= 0 or insulin <= 0:
            return 0.0
        return 1 / (np.log(insulin) + np.log(glucose))

    # --------------------------------------------------
    # Feature builder (IMPORTANT)
    # --------------------------------------------------

    def _build_features(self, df):

        features = []

        for _, row in df.iterrows():

            g = float(row["Glucose"])
            ins = float(row["Insulin"])
            dpf = float(row["DiabetesPedigreeFunction"])

            homa = self._compute_homa_ir(g, ins)
            quicki = self._compute_quicki(g, ins)

            features.append([
                g,
                g**2,
                ins,
                g * ins,
                homa,
                quicki,
                dpf
            ])

        return np.array(features)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def fit(self, df, y):

        X = self._build_features(df)

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)

        self.X_train_ = X_scaled.copy()  # store scaled training matrix
        self.is_fitted = True

        return self

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(self, row_df):

        X = self._build_features(row_df)
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0][1]

        return {"risk": float(prob)}
