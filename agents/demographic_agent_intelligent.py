import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class DemographicAgentIntelligent:
    def __init__(self):
        self.model = None
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

        age_risk_flag = (age >= 45).astype(int)
        high_parity_flag = (preg >= 4).astype(int)
        age_x_preg = age * preg
        age_squared = age ** 2
        midlife_flag = ((age >= 35) & (age < 60)).astype(int)

        return np.column_stack([
            age,
            preg,
            age_risk_flag,
            high_parity_flag,
            age_x_preg,
            age_squared,
            midlife_flag,
        ])

    # --------------------------------------------------
    # Training — LR vs GB, pick winner by CV AUC
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
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        }

        pipe_gb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(random_state=42)),
        ])
        param_grid_gb = {
            "clf__n_estimators":  [100, 200],
            "clf__learning_rate": [0.05, 0.10],
            "clf__max_depth":     [2, 3],
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
        print(f"  Demographic Agent winner       : {winner}")
        print(f"  Demographic Agent best params  : {best_params}")
        print(f"  Demographic Agent CV AUC       : {best_auc:.4f}")
        return self

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, df):
        X = self._build_features(df)
        assert X.shape[0] == 1, "DemographicAgentIntelligent.predict expects a single-row DataFrame"
        prob = self.model.predict_proba(X)[0][1]
        return {"risk": round(float(prob), 4)}
