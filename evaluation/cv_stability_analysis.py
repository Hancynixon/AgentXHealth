import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner


def run():

    print("\nLoading Pima dataset...")

    df = pd.read_csv("data/raw/diabetes.csv")
    y = df["Outcome"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, y)):

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Initialize agents
        lab = LabAgentIntelligent()
        phys = PhysicalAgentIntelligent()
        demo = DemographicAgentIntelligent()
        coord = CoordinatorReasoner()

        # Train agents
        lab.fit(train_df, train_df["Outcome"])
        phys.fit(train_df, train_df["Outcome"])
        demo.fit(train_df, train_df["Outcome"])

        # Collect stacking features
        X_stack_train = []
        y_train = train_df["Outcome"].values

        for _, row in train_df.iterrows():
            row_df = pd.DataFrame([row])

            lab_out = lab.predict(row_df)
            phys_out = phys.predict(row_df)
            demo_out = demo.predict(row_df)

            X_stack_train.append([
                lab_out["risk"],
                phys_out["risk"],
                demo_out["risk"]
            ])

        X_stack_train = np.array(X_stack_train)

        coord.fit(X_stack_train, y_train)

        # Evaluate on test set
        final_probs = []
        y_test = test_df["Outcome"].values

        for _, row in test_df.iterrows():
            row_df = pd.DataFrame([row])

            lab_out = lab.predict(row_df)
            phys_out = phys.predict(row_df)
            demo_out = demo.predict(row_df)

            outputs = {
                "lab": lab_out,
                "physical": phys_out,
                "demographic": demo_out
            }

            final = coord.reason(outputs)
            final_probs.append(final["final_risk"])

        final_probs = np.array(final_probs)

        auc = roc_auc_score(y_test, final_probs)
        aucs.append(auc)

        print(f"Fold {fold+1} AUC = {auc:.4f}")

    print("\n==============================")
    print("Cross-Validation Stability")
    print("==============================")
    print("Mean AUC :", round(np.mean(aucs), 4))
    print("Std  AUC :", round(np.std(aucs), 4))


if __name__ == "__main__":
    run()
