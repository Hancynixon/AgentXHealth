import os
import sys
import numpy as np
import pandas as pd

# Path setup
sys.path.append(os.path.abspath("."))

# Agent imports
from agents.lab_agent import LabAgent
from agents.physical_agent import PhysicalAgent
from agents.demographic_agent import DemographicAgent
from coordinator.coordinator_agent import CoordinatorAgent

# Risk engine
from explainability.risk_engine import RiskEngine

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def main():
    print("🚀 Starting AgentXHealth System...\n")

    # 1. Load dataset
    df = pd.read_csv("data/raw/diabetes.csv")
    print(f"Dataset loaded: {df.shape}\n")

    # 2. Initialize agents
    lab_agent = LabAgent()
    physical_agent = PhysicalAgent()
    demographic_agent = DemographicAgent()

    # 3. Coordinator
    coordinator = CoordinatorAgent(
        lab_agent=lab_agent,
        physical_agent=physical_agent,
        demographic_agent=demographic_agent
    )

    X = coordinator.run(df)
    y = df["Outcome"]

    print("Agents + Coordinator completed")
    print("Final feature shape:", X.shape, "\n")

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        base_score=0.5,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("XGBoost training completed\n")

    # 6. Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    FINAL_THRESHOLD = 0.35
    y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred), "\n")

    # 7. Risk Engine
    risk_engine = RiskEngine(low=0.3, high=0.6)

    # 8. Final system output
    final_output = X_test.copy()
    final_output["True_Label"] = y_test.values
    final_output["Predicted_Probability"] = y_proba
    final_output["Predicted_Label"] = y_pred

    final_output["Risk_Level"] = final_output["Predicted_Probability"].apply(
        lambda p: risk_engine.assign_risk(p)["risk_level"]
    )

    final_output["Recommendation"] = final_output["Predicted_Probability"].apply(
        lambda p: risk_engine.assign_risk(p)["recommendation"]
    )

    print("🧠 Sample System Decisions:\n")
    print(final_output.head())

    print("\n✅ AgentXHealth system run completed successfully.")


if __name__ == "__main__":
    main()
