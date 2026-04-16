import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

from agents.lab_agent_intelligent         import LabAgentIntelligent
from agents.physical_agent_intelligent    import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

# ─────────────────────────────────────────────
# LOAD & SPLIT — same seed as always
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
# TRAIN ALL AGENTS
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

# Ensemble probabilities
p_ens  = p_lab * 0.5 + p_phys * 0.3 + p_demo * 0.2
y_test_arr = np.array(y_test)

# ─────────────────────────────────────────────
# POINT ESTIMATE
# ─────────────────────────────────────────────
point_auc = roc_auc_score(y_test_arr, p_ens)
print(f"\n  Ensemble AUC (point estimate) : {point_auc:.4f}")

# ─────────────────────────────────────────────
# BOOTSTRAP 95% CI — 1000 resamples
# ─────────────────────────────────────────────
print("  Computing 95% CI via bootstrap (1000 resamples)...")
np.random.seed(42)
boot_aucs = []

for _ in range(1000):
    idx = resample(range(len(y_test_arr)), random_state=None)
    y_boot = y_test_arr[idx]
    p_boot = p_ens[idx]

    # Skip if only one class present in resample
    if len(np.unique(y_boot)) < 2:
        continue

    boot_aucs.append(roc_auc_score(y_boot, p_boot))

ci_lower = np.percentile(boot_aucs, 2.5)
ci_upper = np.percentile(boot_aucs, 97.5)

# ─────────────────────────────────────────────
# PRINT RESULT
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("  CONFIDENCE INTERVAL RESULT")
print("=" * 50)
print(f"  AUC         : {point_auc:.4f}")
print(f"  95% CI      : {ci_lower:.4f} – {ci_upper:.4f}")
print(f"  Bootstrap n : {len(boot_aucs)} valid resamples")
print("=" * 50)
print(f"\n  ✅ Update Section V-E in paper:")
print(f"  'Internal validation produced an AUC of {point_auc:.3f}")
print(f"   (95% CI: {ci_lower:.4f}–{ci_upper:.4f})'")
