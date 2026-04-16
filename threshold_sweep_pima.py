import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Pima dataset
print("Loading Pima dataset...")
df = pd.read_csv("data/raw/diabetes.csv")
print(f"Loaded {len(df)} rows")

# 2. Train the three agents (copy-pasted from your working code)
print("Training agents...")
from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

lab = LabAgentIntelligent().fit(df, df["Outcome"])
physical = PhysicalAgentIntelligent().fit(df, df["Outcome"])
demo = DemographicAgentIntelligent().fit(df, df["Outcome"])

# 3. Get predictions row by row
print("Getting predictions (row by row)...")
lab_risks = []
phys_risks = []
demo_risks = []

for i in range(len(df)):
    row = df.iloc[[i]]
    
    lab_risk = lab.predict(row)["risk"]
    phys_risk = physical.predict(row)["risk"]
    demo_risk = demo.predict(row)["risk"]
    
    lab_risks.append(lab_risk)
    phys_risks.append(phys_risk)
    demo_risks.append(demo_risk)

# 4. Ensemble (paper weights)
final_risk = 0.5 * np.array(lab_risks) + 0.3
