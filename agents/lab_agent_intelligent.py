import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class LabAgentIntelligent:
    def __init__(self):
        self.model = None
        self.X_train_ = None

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
        try:
            return 1 / (np.log(insulin) + np.log(glucose))
        except Exception:
            return 0.0

    # --------------------------------------------------
    # Feature builder
    # --------------------------------------------------
    def _build_features(self, df):
        features = []
        for _, row in df.iterrows():
            g = float(row["Glucose"])
            ins = float(row["Insulin"])
            dpf = float(row["DiabetesPedigreeFunction"])
            age = float(row["Age"]) if "Age" in row else 0.0
            preg = float(row["Pregnancies"]) if "Pregnancies" in row else 0.0

            homa = self._compute_homa_ir(g, ins)
            quicki = self._compute_quicki(g, ins)

            features.append([
                g,
                g ** 2,
                ins,
                g * ins,
                homa,
                quicki,
                dpf,
                dpf * age,           # family history × age
                dpf * preg,          # family history × pregnancies
                float(g >= 126),     # diabetic threshold flag
                float(ins > 25),     # hyperinsulinemia flag
                float(homa >= 2.5),  # insulin resistance flag
            ])
        return np.array(features)

    # --------------------------------------------------
    # Training — GridSearchCV, winner between LR and GB
    # --------------------------------------------------
    def fit(self, df, y):
        X = self._build_features(df)

        pipe_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=42
                       )),
        ])
        param_grid_lr = {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        }

        pipe_gb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(random_state=42)),
        ])
        param_grid_gb = {
            "clf__n_estimators":  [100, 200],
            "clf__learning_rate": [0.05, 0.10],
            "clf__max_depth":     [3, 4],
        }

        grid_lr = GridSearchCV(
            pipe_lr, param_grid_lr,
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        grid_gb = GridSearchCV(
            pipe_gb, param_grid_gb,
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
        )

        grid_lr.fit(X, y)
        grid_gb.fit(X, y)

        if grid_lr.best_score_ >= grid_gb.best_score_:
            self.model = grid_lr.best_estimator_
            winner = "LogisticRegression"
            best_auc = grid_lr.best_score_
            best_params = grid_lr.best_params_
        else:
            self.model = grid_gb.best_estimator_
            winner = "GradientBoosting"
            best_auc = grid_gb.best_score_
            best_params = grid_gb.best_params_

        self.X_train_ = X.copy()
        print(f"  Lab Agent winner       : {winner}")
        print(f"  Lab Agent best params  : {best_params}")
        print(f"  Lab Agent CV AUC       : {best_auc:.4f}")
        return self

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, row_df):
        X = self._build_features(row_df)
        assert X.shape[0] == 1, "LabAgentIntelligent.predict expects a single-row DataFrame"
        prob = self.model.predict_proba(X)[0][1]
        return {"risk": float(prob)}
