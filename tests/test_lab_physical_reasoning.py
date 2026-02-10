import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner
import pandas as pd

# Load data
df = pd.read_csv("data/raw/diabetes.csv")

# Fit agents
lab_agent = LabAgentIntelligent()
lab_agent.fit(df, df["Outcome"])

phys_agent = PhysicalAgentIntelligent()
phys_agent.fit(df, df["Outcome"])

# Single patient
sample = df.iloc[[0]]

lab_output = lab_agent.predict(sample)
phys_output = phys_agent.predict(sample)

# Combine for reasoner
agent_outputs = {
    "lab": lab_output,
    "physical": phys_output,
    "demographic": {"demographic_risk_score": 0.0}  # placeholder
}

# Reasoning
reasoner = CoordinatorReasoner()
final_decision = reasoner.reason(agent_outputs)

print(final_decision)
