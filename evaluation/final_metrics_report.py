import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner


# -------------------------------------------------------
# Bootstrap CI
# -------------------------------------------------------

def bootstrap_auc_ci(y_true, probs, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(probs), len(probs))

        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], probs[indices])
        scores.append(score)

    sorted_scores = np.sort(scores)

    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return lower, upper


# -------------------------------------------------------
# Internal Evaluation
# -------------------------------------------------------

def run():

    print("\n==============================")
    print("FINAL MODEL EVALUATION REPORT")
    print("==============================\n")

    df = pd.read_csv("data/raw/diabetes.csv")

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["Outcome"],
        random_state=42
    )

    lab = LabAgentIntelligent()
    phys = PhysicalAgentIntelligent()
    demo = DemographicAgentIntelligent()
    coord = CoordinatorReasoner()

    lab.fit(train_df, train_df["Outcome"])
    phys.fit(train_df, train_df["Outcome"])
    demo.fit(train_df, train_df["Outcome"])

    X_train, y_train = [], []
    X_test, y_test = [], []

    # Build stacking features
    for _, row in train_df.iterrows():
        row_df = pd.DataFrame([row])

        lab_risk = lab.predict(row_df)["risk"]
        phys_risk = phys.predict(row_df)["risk"]
        demo_risk = demo.predict(row_df)["risk"]

        X_train.append([lab_risk, phys_risk, demo_risk])
        y_train.append(row["Outcome"])

    for _, row in test_df.iterrows():
        row_df = pd.DataFrame([row])

        lab_risk = lab.predict(row_df)["risk"]
        phys_risk = phys.predict(row_df)["risk"]
        demo_risk = demo.predict(row_df)["risk"]

        X_test.append([lab_risk, phys_risk, demo_risk])
        y_test.append(row["Outcome"])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    coord.fit(X_train, y_train)

    probs = coord.model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    ci_lower, ci_upper = bootstrap_auc_ci(y_test, probs)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print(f"95% CI    : [{ci_lower:.4f} – {ci_upper:.4f}]")

    print("\nConfusion Matrix:")
    print(cm)

    # -------------------------------------------------------
    # Save Confusion Matrix Image
    # -------------------------------------------------------

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Internal Validation")
    plt.savefig("confusion_matrix_internal.png", dpi=300)
    plt.close()

    print("✔ confusion_matrix_internal.png saved")

    # -------------------------------------------------------
    # Save ROC Curve
    # -------------------------------------------------------

    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Internal Validation")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_internal.png", dpi=300)
    plt.close()

    print("✔ roc_curve_internal.png saved\n")


if __name__ == "__main__":
    run()