import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner


# --------------------------------------------------
# Expected Calibration Error (ECE)
# --------------------------------------------------

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            avg_conf = np.mean(y_prob[mask])
            avg_acc = np.mean(y_true[mask])
            ece += np.abs(avg_conf - avg_acc) * np.sum(mask) / len(y_true)

    return ece


# --------------------------------------------------
# Evaluate configuration on NHANES
# --------------------------------------------------

def evaluate_external(df, use_lab, use_phys, use_demo):

    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df["Outcome"], random_state=42
    )

    lab = LabAgentIntelligent()
    phys = PhysicalAgentIntelligent()
    demo = DemographicAgentIntelligent()
    coord = CoordinatorReasoner()

    if use_lab:
        lab.fit(train_df, train_df["Outcome"])
    if use_phys:
        phys.fit(train_df, train_df["Outcome"])
    if use_demo:
        demo.fit(train_df, train_df["Outcome"])

    X_train, y_train = [], []
    X_test, y_test = [], []

    # Build stacking features
    for _, row in train_df.iterrows():
        row_df = pd.DataFrame([row])

        lab_risk = lab.predict(row_df)["risk"] if use_lab else 0
        phys_risk = phys.predict(row_df)["risk"] if use_phys else 0
        demo_risk = demo.predict(row_df)["risk"] if use_demo else 0

        X_train.append([lab_risk, phys_risk, demo_risk])
        y_train.append(row["Outcome"])

    for _, row in test_df.iterrows():
        row_df = pd.DataFrame([row])

        lab_risk = lab.predict(row_df)["risk"] if use_lab else 0
        phys_risk = phys.predict(row_df)["risk"] if use_phys else 0
        demo_risk = demo.predict(row_df)["risk"] if use_demo else 0

        X_test.append([lab_risk, phys_risk, demo_risk])
        y_test.append(row["Outcome"])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    coord.fit(X_train, y_train)

    probs = coord.model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    ece = compute_ece(y_test, probs)

    return auc, brier, ece


# --------------------------------------------------
# Main
# --------------------------------------------------

def run():

    print("\n==============================")
    print("EXTERNAL ABLATION ANALYSIS (NHANES)")
    print("==============================\n")

    df = pd.read_csv("data/NHNES/nhanes_diabetes_processed.csv")

    configs = {
        "Full Model": (True, True, True),
        "Lab Only": (True, False, False),
        "Physical Only": (False, True, False),
        "Demo Only": (False, False, True),
        "No Lab": (False, True, True),
        "No Physical": (True, False, True),
        "No Demo": (True, True, False),
    }

    results = []

    for name, (lab_flag, phys_flag, demo_flag) in configs.items():
        auc, brier, ece = evaluate_external(df, lab_flag, phys_flag, demo_flag)

        print(f"{name}")
        print(f"   AUC    : {auc:.4f}")
        print(f"   Brier  : {brier:.4f}")
        print(f"   ECE    : {ece:.4f}\n")

        results.append([name, auc, brier, ece])

    print("External Ablation Complete.\n")


if __name__ == "__main__":
    run()