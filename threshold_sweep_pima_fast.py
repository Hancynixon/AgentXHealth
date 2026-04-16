import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading Pima dataset...")
df = pd.read_csv("data/raw/diabetes.csv")

# SAMPLE 200 ROWS FOR SPEED
df_sample = df.sample(n=200, random_state=42)
print(f"Using sample of {len(df_sample)} rows")

print("Training agents...")
from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

lab = LabAgentIntelligent().fit(df_sample, df_sample["Outcome"])
physical = PhysicalAgentIntelligent().fit(df_sample, df_sample["Outcome"])
demo = DemographicAgentIntelligent().fit(df_sample, df_sample["Outcome"])

print("Getting predictions (200 rows)...")
lab_risks = []
phys_risks = []
demo_risks = []

for i in range(len(df_sample)):
    row = df_sample.iloc[[i]]
    
    lab_risk = lab.predict(row)["risk"]
    phys_risk = physical.predict(row)["risk"]
    demo_risk
