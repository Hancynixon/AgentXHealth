import pandas as pd
import numpy as np

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner
from evaluation.fairness_evaluator import FairnessEvaluator
from evaluation.visualization import RiskVisualizer
from evaluation.model_evaluator import ModelEvaluator


def main():

    print("=== Starting AgentXHealth ===")

    # LOAD DATA
    df = pd.read_csv("data/raw/diabetes.csv")
    y = df["Outcome"]

    # TRAIN DOMAIN AGENTS
    lab = LabAgentIntelligent()
    lab.fit(df, y)

    physical = PhysicalAgentIntelligent()
    physical.fit(df, y)

    demo = DemographicAgentIntelligent()
    demo.fit(df, y)

    # BUILD STACKING FEATURES
    stack_features = []

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])

        lab_risk = lab.predict(row_df)["risk"]
        phys_risk = physical.predict(row_df)["risk"]
        demo_risk = demo.predict(row_df)["risk"]

        stack_features.append([lab_risk, phys_risk, demo_risk])

    X_stack = np.array(stack_features)

    # TRAIN COORDINATOR
    coord = CoordinatorReasoner()
    coord.fit(X_stack, y)

    # FINAL PREDICTION LOOP
    results = []

    for _, row in df.iterrows():

        row_df = pd.DataFrame([row])

        outputs = {
            "lab": lab.predict(row_df),
            "physical": physical.predict(row_df),
            "demographic": demo.predict(row_df)
        }

        decision = coord.reason(outputs)
        results.append(decision)

    # MERGE RESULTS
    results_df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1
    )

    results_df.to_csv("results_with_risk.csv", index=False)

    # FAIRNESS
    fe = FairnessEvaluator()
    print("Fairness Report:", fe.evaluate(results_df))

    # VISUALIZATION
    viz = RiskVisualizer()
    viz.generate_all(results_df)

    # EVALUATION
    evaluator = ModelEvaluator()
    evaluator.evaluate(results_df)

    print("=== Finished AgentXHealth ===")


if __name__ == "__main__":
    main()