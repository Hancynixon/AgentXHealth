import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner


def run():

    df = pd.read_csv("data/raw/diabetes.csv")
    y = df["Outcome"]

    lab = LabAgentIntelligent()
    phys = PhysicalAgentIntelligent()
    demo = DemographicAgentIntelligent()

    lab.fit(df, y)
    phys.fit(df, y)
    demo.fit(df, y)

    coord = CoordinatorReasoner()

    risks = []

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])

        outputs = {
            "lab": lab.predict(row_df),
            "physical": phys.predict(row_df),
            "demographic": demo.predict(row_df)
        }

        decision = coord.reason(outputs)
        risks.append(decision["final_risk"])

    df["final_risk"] = risks

    y_true = df["Outcome"]
    y_pred = (df["final_risk"] >= 0.40).astype(int)

    print("\n==============================")
    print("FULL MODEL EVALUATION")
    print("==============================")

    print("Accuracy  :", accuracy_score(y_true, y_pred))
    print("Precision :", precision_score(y_true, y_pred))
    print("Recall    :", recall_score(y_true, y_pred))
    print("F1 Score  :", f1_score(y_true, y_pred))
    print("ROC AUC   :", roc_auc_score(y_true, df["final_risk"]))
    print("Brier     :", brier_score_loss(y_true, df["final_risk"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    run()
