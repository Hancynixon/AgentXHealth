import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class PhysicalAgentIntelligent:
    def __init__(self):
        self.model = None
        self.X_train_ = None

    # --------------------------------------------------
    # Clinical staging
    # --------------------------------------------------
    def _bmi_stage(self, bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 25:
            return 1
        elif bmi < 30:
            return 2
        else:
            return 3

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
        skin = df["SkinThickness"].astype(float) \
            if "SkinThickness" in df.columns \
            else np.zeros(len(df))

        bmi_stage = bmi.apply(self._bmi_stage)
        bp_stage = bp.apply(self._bp_stage)

        bmi_bp_ratio = bmi / (bp + 1)
        bmi_squared = bmi ** 2
        hypertension_flag = (bp >= 130).astype(int)
        obesity_flag = (bmi >= 30).astype(int)
        obesity_htn_combo = obesity_flag * hypertension_flag
        skin_bmi_ratio = skin / (bmi + 1)
        high_skin_flag = (skin >= 30).astype(int)

        return np.column_stack([
            bmi,
            bp,
            skin,
            bmi_stage,
            bp_stage,
            bmi_bp_ratio,
            bmi_squared,
            hypertension_flag,
            obesity_flag,
            obesity_htn_combo,
            skin_bmi_ratio,
            high_skin_flag,
        ])

    # --------------------------------------------------
    # Training — sample_weight balanced + GridSearchCV
    # --------------------------------------------------
    def fit(self, df, y):
        X = self._build_features(df)

        classes, counts = np.unique(y, return_counts=True)
        weight_map = {
            c: len(y) / (len(classes) * cnt)
            for c, cnt in zip(classes, counts)
        }
        sample_weights = np.array([weight_map[yi] for yi in y])

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(random_state=42)),
        ])
        param_grid = {
            "clf__n_estimators":  [100, 200, 300],
            "clf__learning_rate": [0.05, 0.10],
            "clf__max_depth":     [3, 4],
            "clf__subsample":     [0.8, 1.0],
        }
        grid = GridSearchCV(
            pipeline, param_grid,
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        grid.fit(X, y, clf__sample_weight=sample_weights)

        self.model = grid.best_estimator_
        self.X_train_ = X.copy()
        print(f"  Physical Agent best params : {grid.best_params_}")
        print(f"  Physical Agent CV AUC      : {grid.best_score_:.4f}")
        return self

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, df):
        X = self._build_features(df)
        assert X.shape[0] == 1, "PhysicalAgentIntelligent.predict expects a single-row DataFrame"
        prob = self.model.predict_proba(X)[0][1]
        return {"risk": round(float(prob), 4)}
