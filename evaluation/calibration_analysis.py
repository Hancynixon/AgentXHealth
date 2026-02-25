import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from coordinator.coordinator_reasoner import CoordinatorReasoner


# ======================================================
# Strong Base Agent (NHANES)
# ======================================================
class StrongAgent:

    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()
        self.model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

    def fit(self, df, y):
        X = df[self.features].astype(float).fillna(0)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def predict_proba(self, df):
        X = df[self.features].astype(float).fillna(0)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]


# ======================================================
# Expected Calibration Error (ECE)
# ======================================================
def expected_calibration_error(y_true, y_prob, n_bins=10):

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0

    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            avg_confidence = np.mean(y_prob[mask])
            avg_accuracy = np.mean(y_true[mask])
            ece += (
                np.abs(avg_confidence - avg_accuracy)
                * np.sum(mask)
                / len(y_true)
            )

    return ece


# ======================================================
# MAIN
# ======================================================
def run():

    print("\nLoading NHANES processed dataset...")

    df = pd.read_csv("data/NHNES/nhanes_diabetes_processed.csv")

    y = df["Outcome"]

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    print("Train size:", len(train_df))
    print("Test size :", len(test_df))

    # --------------------------------------------------
    # Feature Groups
    # --------------------------------------------------
    lab_features = ["Glucose", "Insulin"]
    physical_features = ["BMI", "BloodPressure"]
    demo_features = ["Age"]

    print("\nTraining base agents...")

    lab = StrongAgent(lab_features)
    phys = StrongAgent(physical_features)
    demo = StrongAgent(demo_features)

    lab.fit(train_df, train_df["Outcome"])
    phys.fit(train_df, train_df["Outcome"])
    demo.fit(train_df, train_df["Outcome"])

    # --------------------------------------------------
    # STACK TRAIN DATA
    # --------------------------------------------------
    lab_train = lab.predict_proba(train_df)
    phys_train = phys.predict_proba(train_df)
    demo_train = demo.predict_proba(train_df)

    X_stack_train = np.column_stack(
        [lab_train, phys_train, demo_train]
    )

    # --------------------------------------------------
    # FIT COORDINATOR ON TRAIN
    # --------------------------------------------------
    coordinator = CoordinatorReasoner()
    coordinator.fit(X_stack_train, train_df["Outcome"])

    # --------------------------------------------------
    # STACK TEST DATA
    # --------------------------------------------------
    lab_test = lab.predict_proba(test_df)
    phys_test = phys.predict_proba(test_df)
    demo_test = demo.predict_proba(test_df)

    X_stack_test = np.column_stack(
        [lab_test, phys_test, demo_test]
    )

    final_probs = coordinator.model.predict_proba(X_stack_test)[:, 1]
    y_test = test_df["Outcome"].values

    # --------------------------------------------------
    # Brier Score
    # --------------------------------------------------
    brier = brier_score_loss(y_test, final_probs)
    print("\nBrier Score:", round(brier, 4))

    # --------------------------------------------------
    # Expected Calibration Error
    # --------------------------------------------------
    ece = expected_calibration_error(y_test, final_probs)
    print("Expected Calibration Error (ECE):", round(ece, 4))

    # --------------------------------------------------
    # Reliability Diagram
    # --------------------------------------------------
    prob_true, prob_pred = calibration_curve(
        y_test,
        final_probs,
        n_bins=10
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.show()


if __name__ == "__main__":
    run()
