import sys, os, warnings
sys.path.insert(0, os.path.abspath("."))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from agents.lab_agent_intelligent         import LabAgentIntelligent
from agents.physical_agent_intelligent    import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# ── Load Pima ──
print("\n[1/3] Loading Pima...")
pima = pd.read_csv("data/raw/diabetes.csv")
print(f"  Pima shape: {pima.shape}  |  Diabetic: {pima['Outcome'].mean():.1%}")

# ── Load NHANES ──
print("\n[2/3] Loading NHANES...")
nhanes_full = pd.read_csv("data/NHNES/nhanes_diabetes_processed.csv")

# Map HbA1c → DiabetesPedigreeFunction for NHANES
# (Pima has no HbA1c; NHANES has no real DPF — swap gives agents HbA1c signal)
if "HbA1c" in nhanes_full.columns:
    nhanes_full["DiabetesPedigreeFunction"] = (
        pd.to_numeric(nhanes_full["HbA1c"], errors="coerce").fillna(0)
    )
    print("  [INFO] HbA1c mapped → DiabetesPedigreeFunction for NHANES rows")

for c in PIMA_COLS:
    if c not in nhanes_full.columns:
        nhanes_full[c] = 0
nhanes = nhanes_full[PIMA_COLS].copy()
for c in PIMA_COLS:
    nhanes[c] = pd.to_numeric(nhanes[c], errors="coerce").fillna(0)

print(f"  NHANES shape: {nhanes.shape}  |  Diabetic: {nhanes['Outcome'].mean():.1%}")

# ── Combine ──
combined = pd.concat([pima, nhanes], axis=0).reset_index(drop=True)
X = combined.drop("Outcome", axis=1)
Y = combined["Outcome"].astype(int)
print(f"\n[3/3] Combined: {combined.shape}  |  Diabetic: {Y.mean():.1%}")

# ── Train/Test split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)
train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)
print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")

# ── Train agents ──
print("\nTraining agents on combined dataset...")
lab  = LabAgentIntelligent().fit(train_df, y_train)
phys = PhysicalAgentIntelligent().fit(train_df, y_train)
demo = DemographicAgentIntelligent().fit(train_df, y_train)

# ── Quick sanity check ──
PROB_KEYS = ("risk","probability","prob","final_risk","score",
             "diabetes_risk","prediction","pred_prob","risk_score")

def _extract_one(v):
    if isinstance(v, dict):
        for key in PROB_KEYS:
            if key in v:
                try: return float(v[key])
                except: continue
        for val in v.values():
            try: return float(val)
            except: continue
        return 0.0
    elif isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    elif isinstance(v, (list, np.ndarray)):
        arr = np.array(v).flatten()
        return float(arr[1]) if len(arr) == 2 else float(arr[0])
    else:
        try: return float(v)
        except: return 0.0

def get_probs_quick(agent, df, n=200):
    """Quick check on first n rows."""
    subset = df.iloc[:n]
    out = []
    for i in range(len(subset)):
        row = subset.iloc[[i]]
        r = agent.predict(row)
        if isinstance(r, dict):
            out.append(_extract_one(r))
        elif isinstance(r, (list, np.ndarray)):
            arr = list(r)
            out.append(_extract_one(arr[0]) if arr else 0.0)
        else:
            out.append(_extract_one(r))
    return np.clip(np.array(out, dtype=float), 0.0, 1.0)

print("\nSanity check on first 200 test rows...")
p_lab  = get_probs_quick(lab,  test_df)
p_phys = get_probs_quick(phys, test_df)
p_demo = get_probs_quick(demo, test_df)
ensemble = 0.5*p_lab + 0.3*p_phys + 0.2*p_demo
y_check  = y_test.iloc[:200].values
auc_check = roc_auc_score(y_check, ensemble)
print(f"  Quick ensemble AUC (200 rows): {auc_check:.4f}")

# ── Save models ──
joblib.dump(lab,  "models/lab_combined.pkl")
joblib.dump(phys, "models/phys_combined.pkl")
joblib.dump(demo, "models/demo_combined.pkl")
print("\nSaved:")
print("  models/lab_combined.pkl")
print("  models/phys_combined.pkl")
print("  models/demo_combined.pkl")
print("\n" + "="*55)
print("  Done. Next: python run_results.py")
print("="*55)
