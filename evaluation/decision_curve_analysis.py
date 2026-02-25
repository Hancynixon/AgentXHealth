import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from coordinator.coordinator_reasoner import CoordinatorReasoner


# ======================================================
# Strong Agent
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
# Net Benefit Calculation
# ======================================================
def net_benefit(y_true, probs, threshold):

    preds = (probs >= threshold).astype(int)

    TP = np.sum((preds == 1) & (y_true == 1))
    FP = np.sum((preds == 1) & (y_true == 0))

    N = len(y_true)

    if threshold == 1:
        return 0

    return (TP / N) - (FP / N) * (threshold / (1 - threshold))


# ======================================================
# MAIN
# ======================================================
def run():

    print("\nLoading NHANES dataset...")

    df = pd.read_csv("data/NHNES/nhanes_diabetes_processed.csv")
    y = df["Outcome"]

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # Feature groups
    lab_features = ["Glucose", "Insulin"]
    physical_features = ["BMI", "BloodPressure"]
    demo_features = ["Age"]

    print("Training base agents...")

    lab = StrongAgent(lab_features)
    phys = StrongAgent(physical_features)
    demo = StrongAgent(demo_features)

    lab.fit(train_df, train_df["Outcome"])
    phys.fit(train_df, train_df["Outcome"])
    demo.fit(train_df, train_df["Outcome"])

    # Stack train
    lab_train = lab.predict_proba(train_df)
    phys_train = phys.predict_proba(train_df)
    demo_train = demo.predict_proba(train_df)

    X_stack_train = np.column_stack([lab_train, phys_train, demo_train])

    coordinator = CoordinatorReasoner()
    coordinator.fit(X_stack_train, train_df["Outcome"])

    # Stack test
    lab_test = lab.predict_proba(test_df)
    phys_test = phys.predict_proba(test_df)
    demo_test = demo.predict_proba(test_df)

    X_stack_test = np.column_stack([lab_test, phys_test, demo_test])

    # 🔥 Use Platt calibrated model (best version)
    calibrated = CalibratedClassifierCV(
        coordinator.model,
        method="sigmoid",
        cv="prefit"
    )

    calibrated.fit(X_stack_train, train_df["Outcome"])

    probs = calibrated.predict_proba(X_stack_test)[:, 1]
    y_test = test_df["Outcome"].values

    # --------------------------------------------------
    # Decision Curve
    # --------------------------------------------------
    thresholds = np.linspace(0.01, 0.99, 100)

    model_nb = []
    treat_all_nb = []
    treat_none_nb = []

    prevalence = np.mean(y_test)

    for t in thresholds:
        model_nb.append(net_benefit(y_test, probs, t))
        treat_all_nb.append(
            prevalence - (1 - prevalence) * (t / (1 - t))
        )
        treat_none_nb.append(0)

    # Plot
    plt.figure()
    plt.plot(thresholds, model_nb, label="AgentXHealth Model")
    plt.plot(thresholds, treat_all_nb, linestyle="--", label="Treat All")
    plt.plot(thresholds, treat_none_nb, linestyle="--", label="Treat None")

    plt.xlabel("Risk Threshold")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.show()

    print("\nDecision Curve Analysis Completed.")


if __name__ == "__main__":
    run()
