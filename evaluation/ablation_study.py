import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner


# --------------------------------------------------
# Evaluate One Configuration
# --------------------------------------------------

def evaluate_configuration(df, use_lab, use_phys, use_demo):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, test_idx in skf.split(df, df["Outcome"]):

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

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
        aucs.append(auc)

    return np.array(aucs)


# --------------------------------------------------
# Significance Marker
# --------------------------------------------------

def significance_marker(p):
    if p < 0.001:
        return "*** (p < 0.001)"
    elif p < 0.01:
        return "** (p < 0.01)"
    elif p < 0.05:
        return "* (p < 0.05)"
    else:
        return "ns (not significant)"


# --------------------------------------------------
# Run Ablation Study
# --------------------------------------------------

def run():

    print("\n==============================")
    print("ABLATION STUDY ANALYSIS")
    print("==============================\n")

    df = pd.read_csv("data/raw/diabetes.csv")

    configs = {
        "Full Model": (True, True, True),
        "Lab Only": (True, False, False),
        "Physical Only": (False, True, False),
        "Demo Only": (False, False, True),
        "No Lab": (False, True, True),
        "No Physical": (True, False, True),
        "No Demo": (True, True, False),
    }

    results = {}

    # Evaluate all configurations
    for name, (lab_flag, phys_flag, demo_flag) in configs.items():
        aucs = evaluate_configuration(df, lab_flag, phys_flag, demo_flag)
        results[name] = aucs
        print(f"{name:20s} -> Mean AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")

    print("\n==============================")
    print("STATISTICAL COMPARISON vs FULL MODEL")
    print("==============================\n")

    full_aucs = results["Full Model"]

    for name, aucs in results.items():
        if name == "Full Model":
            continue

        stat, p = ttest_rel(full_aucs, aucs)

        print(f"Full vs {name:15s}")
        print(f"   t-statistic = {stat:.4f}")
        print(f"   p-value     = {p:.6e}")
        print(f"   Significance: {significance_marker(p)}\n")


if __name__ == "__main__":
    run()