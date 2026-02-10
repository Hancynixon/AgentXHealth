import sys
import os

# ✅ Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.physical_agent_intelligent import PhysicalAgentIntelligent
import pandas as pd

df = pd.read_csv("data/raw/diabetes.csv")

agent = PhysicalAgentIntelligent()
agent.fit(df, df["Outcome"])

sample = df.iloc[[0]]
print(agent.predict(sample))
