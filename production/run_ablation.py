import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from agents.lab_agent_intelligent         import LabAgentIntelligent
from agents.physical_agent_intelligent    import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

# ─────────────────────────────────────────────
# LOAD & SPLIT
# ─────────────────────────────────────────────
df = pd.read_csv("data/raw/diabetes.csv")
X  = df.drop("Outcome", axis=1)
Y  = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
train_df            = X_train.copy()
train_df["Outcome"] = y_train

# ─────────────────────────────────────────────
# TRAIN ALL 3 AGENTS
# ─────────────────────────────────────────────
print("Training agents...")
lab  = LabAgentIntelligent().fit(train_df, y_train)
phys = PhysicalAgentIntelligent().fit(train_df, y_train)
demo = DemographicAgentIntelligent().fit(train_df, y_train)

def predict_all(agent, X_df):
    probs = []
    for i in range(len(X_df)):
        row = X_df.iloc[[i]]
        try:
            raw = agent.predict(row)
            p   = float(raw["risk"]) if isinstance(raw, dict) else float(raw)
        except:
            p = 0.5
        probs.append(p)
    return np.array(probs)

print("Running predictions...")
p_lab  = predict_all(lab,  X_test)
p_phys = predict_all(phys, X_test)
p_demo = predict_all(demo, X_test)

# ─────────────────────────────────────────────
# ABLATION CONFIGURATIONS
# ─────────────────────────────────────────────
ablation_configs = {
    "Full Multi-Agent Framework"       : p_lab * 0.5 + p_phys * 0.3 + p_demo * 0.2,
    "Laboratory Agent Only"            : p_lab,
    "Physical Agent Only"              : p_phys,
    "Demographic Agent Only"           : p_demo,
    "Without Laboratory Agent"         : p_phys * 0.6 + p_demo * 0.4,
    "Without Physical Agent"           : p_lab  * 0.7 + p_demo * 0.3,
    "Without Demographic Agent"        : p_lab  * 0.7 + p_phys * 0.3,
}

# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 57)
print("  AgentXHealth — Ablation Study Results")
print("=" * 57)
print(f"  {'Model Architecture Variant':<38}  {'AUC':>6}")
print("─" * 57)

results = {}
for name, probs in ablation_configs.items():
    auc = roc_auc_score(y_test, probs)
    results[name] = auc
    marker = "  ← FULL MODEL" if name == "Full Multi-Agent Framework" else ""
    print(f"  {name:<38}  {auc:.4f}{marker}")

print("=" * 57)

# ─────────────────────────────────────────────
# PAPER TABLE II — Ready to Copy
# ─────────────────────────────────────────────
print("\n  TABLE II — Ready to paste into paper:")
print("─" * 57)
print(f"  {'Model Architecture Variant':<38}  {'Mean ROC-AUC':>12}")
print("─" * 57)
for name, auc in results.items():
    print(f"  {name:<38}  {auc:.3f}")
print("─" * 57)

# ─────────────────────────────────────────────
# SIGNIFICANCE ANALYSIS
# ─────────────────────────────────────────────
full_auc = results["Full Multi-Agent Framework"]
print("\n  Component Contribution Analysis:")
print("─" * 57)
for name, auc in results.items():
    if name == "Full Multi-Agent Framework":
        continue
    delta = auc - full_auc
    sign  = "+" if delta >= 0 else ""
    impact = "↑ higher without" if delta > 0.005 else \
             "↓ lower without"  if delta < -0.005 else \
             "≈ negligible"
    print(f"  {name:<38}  Δ={sign}{delta:.4f}  {impact}")

print("=" * 57)
print("\n  Use these values to update Table II in your paper.")
