import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.demographic_agent_intelligent import DemographicAgentIntelligent
import pandas as pd

df = pd.read_csv("data/raw/diabetes.csv")

agent = DemographicAgentIntelligent()
agent.fit(df, df["Outcome"])

sample = df.iloc[[0]]
print(agent.predict(sample))
