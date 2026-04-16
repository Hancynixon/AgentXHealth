import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

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

    probs = []

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])

        outputs = {
            "lab": lab.predict(row_df),
            "physical": phys.predict(row_df),
            "demographic": demo.predict(row_df)
        }

        decision = coord.reason(outputs)
        probs.append(decision["final_risk"])

    df["prob"] = probs

    print("\nCalibration Metrics")
    print("-------------------")
    print("ROC AUC :", roc_auc_score(y, df["prob"]))
    print("Brier   :", brier_score_loss(y, df["prob"]))

    prob_true, prob_pred = calibration_curve(y, df["prob"], n_bins=10)

    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Reliability Diagram")
    plt.show()


if __name__ == "__main__":
    run()
