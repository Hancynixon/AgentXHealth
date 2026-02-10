import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner
from coordinator.counterfactual_reasoner import CounterfactualReasoner
import pandas as pd

df = pd.read_csv("data/raw/diabetes.csv")

lab = LabAgentIntelligent()
lab.fit(df, df["Outcome"])

phys = PhysicalAgentIntelligent()
phys.fit(df, df["Outcome"])

demo = DemographicAgentIntelligent()
demo.fit(df, df["Outcome"])

sample = df.iloc[[0]]

agent_outputs = {
    "lab": lab.predict(sample),
    "physical": phys.predict(sample),
    "demographic": demo.predict(sample)
}

coordinator = CoordinatorReasoner()
counterfactual = CounterfactualReasoner()

print(counterfactual.rank_interventions(agent_outputs, coordinator))
