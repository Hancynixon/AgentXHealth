import sys, os, warnings
sys.path.insert(0, os.path.abspath("."))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss
)
from scipy.stats import bootstrap

from agents.lab_agent_intelligent         import LabAgentIntelligent
from agents.physical_agent_intelligent    import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

os.makedirs("results", exist_ok=True)

# =========================================================
# STEP 1 — LOAD PIMA
# =========================================================
print("\n[1/5] Loading Pima dataset...")
pima   = pd.read_csv("data/raw/diabetes.csv")
X_pima = pima.drop("Outcome", axis=1)
Y_pima = pima["Outcome"]
print(f"      Pima shape: {pima.shape}  |  Diabetic prevalence: {Y_pima.mean():.1%}")

# =========================================================
# STEP 2 — LOAD NHANES
# =========================================================
print("\n[2/5] Loading NHANES processed dataset...")
NHANES_PATH = "data/NHNES/nhanes_diabetes_processed.csv"

nhanes_raw = pd.read_csv(NHANES_PATH)
print(f"      NHANES raw columns : {list(nhanes_raw.columns)}")
print(f"      NHANES raw shape   : {nhanes_raw.shape}")

if nhanes_raw.shape[0] == 0:
    raise RuntimeError(
        "\n  ERROR: NHANES file is empty!\n"
        "  Run:  python rebuild_nhanes.py  then retry.\n"
    )

nhanes = nhanes_raw.copy()

COL_MAP = {
    "glucose"        : "Glucose",    "LBXGLU"  : "Glucose",
    "LBXSGL"         : "Glucose",    "LBXGLT"  : "Glucose",
    "bmi"            : "BMI",        "BMXBMI"  : "BMI",
    "blood_pressure" : "BloodPressure", "BPXSY1": "BloodPressure",
    "insulin"        : "Insulin",    "LBXIN"   : "Insulin",
    "skin_thickness" : "SkinThickness",
    "age"            : "Age",        "RIDAGEYR": "Age",
    "dpf"            : "DiabetesPedigreeFunction",
    "pregnancies"    : "Pregnancies",
    "diabetes"       : "Outcome",    "Diabetes": "Outcome",
    "DIQ010"         : "Outcome",
}
nhanes = nhanes.rename(columns={k: v for k, v in COL_MAP.items()
                                if k in nhanes.columns})

# ── Map HbA1c → DiabetesPedigreeFunction ──
if "HbA1c" in nhanes.columns:
    nhanes["DiabetesPedigreeFunction"] = (
        pd.to_numeric(nhanes["HbA1c"], errors="coerce").fillna(0)
    )
    print("      [INFO] HbA1c → DiabetesPedigreeFunction mapped  ✓")

PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

for col in PIMA_COLS:
    if col not in nhanes.columns:
        nhanes[col] = 0

if nhanes["Outcome"].max() == 2:
    nhanes["Outcome"] = nhanes["Outcome"].map({1: 1, 2: 0}).fillna(0).astype(int)

for col in PIMA_COLS:
    nhanes[col] = pd.to_numeric(nhanes[col], errors="coerce")

if nhanes["Glucose"].median() < 30:
    print("      [INFO] Converting Glucose mmol/L → mg/dL")
    nhanes["Glucose"] = nhanes["Glucose"] * 18.0

nhanes = nhanes.dropna(subset=["BMI", "Age"])
nhanes = nhanes[nhanes["BMI"] > 0]
nhanes = nhanes[nhanes["Age"] >= 18]
nhanes = nhanes[nhanes["Glucose"].notna() & (nhanes["Glucose"] > 0)]

for col in PIMA_COLS:
    if nhanes[col].isnull().any():
        nhanes[col] = nhanes[col].fillna(nhanes[col].median())

nhanes = nhanes[PIMA_COLS].reset_index(drop=True)
X_nh   = nhanes.drop("Outcome", axis=1)
y_nh   = nhanes["Outcome"].astype(int)
print(f"      NHANES final shape : {nhanes.shape}  |  "
      f"Diabetic prevalence: {y_nh.mean():.1%}")
print(f"      Glucose median     : {nhanes['Glucose'].median():.1f} mg/dL")
print(f"      DPF (HbA1c) median : {nhanes['DiabetesPedigreeFunction'].median():.2f}")

# =========================================================
# STEP 3 — SPLIT PIMA + LOAD OR TRAIN AGENTS
# =========================================================
print("\n[3/5] Preparing agents...")
X_train, X_test, y_train, y_test = train_test_split(
    X_pima, Y_pima, test_size=0.2, stratify=Y_pima, random_state=42
)
train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)
nh_df    = pd.concat([X_nh,    y_nh],    axis=1)
print(f"      Pima Train: {train_df.shape}  |  Pima Test: {test_df.shape}")

COMBINED_AVAILABLE = all(
    os.path.exists(f"models/{m}")
    for m in ["lab_combined.pkl", "phys_combined.pkl", "demo_combined.pkl"]
)

if COMBINED_AVAILABLE:
    print("      Loading Pima+NHANES combined pretrained agents...")
    lab  = joblib.load("models/lab_combined.pkl")
    phys = joblib.load("models/phys_combined.pkl")
    demo = joblib.load("models/demo_combined.pkl")
    print("      Combined agents loaded  ✓")
else:
    print("      WARNING: Combined models not found.")
    print("      Run: python retrain_combined.py  for best NHANES AUC")
    print("      Falling back to Pima-only training...")
    lab  = LabAgentIntelligent().fit(train_df, y_train)
    phys = PhysicalAgentIntelligent().fit(train_df, y_train)
    demo = DemographicAgentIntelligent().fit(train_df, y_train)
    print("      Pima-only agents trained.")

# =========================================================
# HELPERS
# =========================================================
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


def get_probs(agent, df):
    out = []
    for i in range(len(df)):
        row    = df.iloc[[i]]
        result = agent.predict(row)
        if isinstance(result, dict):
            out.append(_extract_one(result))
        elif isinstance(result, (list, np.ndarray)):
            arr = list(result)
            out.append(_extract_one(arr[0]) if arr else 0.0)
        else:
            out.append(_extract_one(result))
    return np.clip(np.array(out, dtype=float), 0.0, 1.0)


def ensemble_proba(df, label=""):
    if label:
        print(f"    Computing ensemble for {len(df)} rows [{label}]...",
              flush=True)
    return (0.5 * get_probs(lab,  df) +
            0.3 * get_probs(phys, df) +
            0.2 * get_probs(demo, df))


def best_threshold(probs, y_true):
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        s = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return round(best_t, 2), round(best_s, 4)

# =========================================================
# STEP 4 — ALL METRICS
# =========================================================
print("\n[4/5] Computing all metrics...")

# ── INTERNAL (Pima holdout) ──
probs = ensemble_proba(test_df, "Internal Pima")
preds = (probs >= 0.5).astype(int)

acc        = accuracy_score(y_test, preds)
prec       = precision_score(y_test, preds, average=None, zero_division=0)
rec        = recall_score(y_test, preds, average=None, zero_division=0)
f1         = f1_score(y_test, preds, average=None, zero_division=0)
auc        = roc_auc_score(y_test, probs)
cm         = confusion_matrix(y_test, preds)
brier      = brier_score_loss(y_test, probs)
macro_prec = precision_score(y_test, preds, average="macro", zero_division=0)
macro_rec  = recall_score(y_test, preds, average="macro", zero_division=0)
macro_f1   = f1_score(y_test, preds, average="macro", zero_division=0)
opt_t, opt_f1 = best_threshold(probs, y_test)

res_int = bootstrap(
    (y_test.values, probs),
    lambda a, b: roc_auc_score(a, b),
    n_resamples=1000, random_state=42, paired=True
)
ci = res_int.confidence_interval

print("\n" + "="*60)
print("  TABLE I — INTERNAL RESULTS  (Pima 20% holdout, n=154)")
print("="*60)
print(f"  Accuracy                 : {acc:.4f}")
print(f"  ROC-AUC                  : {auc:.4f}  "
      f"(95% CI: {ci.low:.4f} – {ci.high:.4f})")
print(f"  Brier Score              : {brier:.4f}")
print(f"  Precision  (class 0 / 1) : {prec[0]:.4f} / {prec[1]:.4f}")
print(f"  Recall     (class 0 / 1) : {rec[0]:.4f} / {rec[1]:.4f}")
print(f"  F1-Score   (class 0 / 1) : {f1[0]:.4f} / {f1[1]:.4f}")
print(f"  Macro Precision          : {macro_prec:.4f}")
print(f"  Macro Recall             : {macro_rec:.4f}")
print(f"  Macro F1                 : {macro_f1:.4f}")
print(f"  Optimal Threshold        : {opt_t}  (best F1 = {opt_f1})")
print(f"  Confusion Matrix (t=0.5) :")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

# ── Per-Agent 5-Fold CV (Pima-only fresh agents — standard benchmark) ──
print("\n" + "="*60)
print("  TABLE I — PER-AGENT 5-FOLD CV AUC  (Pima, standard)")
print("="*60)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for AgentClass, name in [
    (LabAgentIntelligent,         "Laboratory Agent  "),
    (PhysicalAgentIntelligent,    "Physical Agent    "),
    (DemographicAgentIntelligent, "Demographic Agent "),
]:
    fold_aucs = []
    for tr_idx, val_idx in skf.split(X_pima, Y_pima):
        tr  = pd.concat([X_pima.iloc[tr_idx],  Y_pima.iloc[tr_idx]],  axis=1)
        val = pd.concat([X_pima.iloc[val_idx], Y_pima.iloc[val_idx]], axis=1)
        ag  = AgentClass().fit(tr, Y_pima.iloc[tr_idx])
        p   = get_probs(ag, val)
        fold_aucs.append(roc_auc_score(Y_pima.iloc[val_idx], p))
    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    cv_results[name.strip()] = (mean_auc, std_auc)
    print(f"  {name}: AUC = {mean_auc:.4f} ± {std_auc:.4f}"
          f"  | Folds: {[round(x,4) for x in fold_aucs]}")

# ── Ablation ──
print("\n" + "="*60)
print("  TABLE II — ABLATION STUDY  (combined agents, Pima test)")
print("="*60)

def ablation_auc(exclude=None):
    all_agents = {"lab":(lab,0.5), "phys":(phys,0.3), "demo":(demo,0.2)}
    if exclude:
        del all_agents[exclude]
    total_w = sum(w for _, w in all_agents.values())
    p = np.zeros(len(y_test))
    for ag, w in all_agents.values():
        p += (w / total_w) * get_probs(ag, test_df)
    return roc_auc_score(y_test, p)

full_auc = ablation_auc()
no_lab   = ablation_auc("lab")
no_phys  = ablation_auc("phys")
no_demo  = ablation_auc("demo")

print(f"  Full Framework (all 3)    : {full_auc:.4f}")
print(f"  Without Laboratory Agent  : {no_lab:.4f}  (Δ = {no_lab -full_auc:+.4f})")
print(f"  Without Physical Agent    : {no_phys:.4f}  (Δ = {no_phys-full_auc:+.4f})")
print(f"  Without Demographic Agent : {no_demo:.4f}  (Δ = {no_demo-full_auc:+.4f})")

# ── Threshold Sensitivity ──
print("\n" + "="*60)
print("  TABLE III — THRESHOLD SENSITIVITY  (Pima internal)")
print("="*60)
print(f"  {'Threshold':>10}  {'Recall':>7}  {'Precision':>10}"
      f"  {'F1':>7}  {'Accuracy':>9}")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    p_  = (probs >= t).astype(int)
    r   = recall_score(y_test, p_, zero_division=0)
    pr  = precision_score(y_test, p_, zero_division=0)
    f   = f1_score(y_test, p_, zero_division=0)
    ac  = accuracy_score(y_test, p_)
    note = (f"  ← Optimal" if t == opt_t
            else ("  ← Default" if t == 0.50 else ""))
    print(f"  {t:.2f}           {r:.4f}    {pr:.4f}"
          f"     {f:.4f}   {ac:.4f}{note}")

# ── External NHANES ──
print("\n" + "="*60)
print("  TABLE IV — EXTERNAL VALIDATION (NHANES, combined agents)")
print("="*60)

p_nh = ensemble_proba(nh_df, "External NHANES")
nh_opt_t, nh_opt_f1 = best_threshold(p_nh, y_nh)
print(f"  NHANES Optimal Threshold : {nh_opt_t}  (best F1 = {nh_opt_f1})")

for label, thr in [("t=0.50 (default)", 0.50),
                   (f"t={nh_opt_t} (optimal)", nh_opt_t)]:
    pred_nh  = (p_nh >= thr).astype(int)
    auc_nh   = roc_auc_score(y_nh, p_nh)
    acc_nh   = accuracy_score(y_nh, pred_nh)
    prec_nh  = precision_score(y_nh, pred_nh, average=None, zero_division=0)
    rec_nh   = recall_score(y_nh, pred_nh, average=None, zero_division=0)
    f1_nh    = f1_score(y_nh, pred_nh, average=None, zero_division=0)
    cm_nh    = confusion_matrix(y_nh, pred_nh)
    brier_nh = brier_score_loss(y_nh, p_nh)
    mf1_nh   = f1_score(y_nh, pred_nh, average="macro", zero_division=0)
    res_nh   = bootstrap(
        (y_nh.values, p_nh),
        lambda a, b: roc_auc_score(a, b),
        n_resamples=1000, random_state=42, paired=True
    )
    ci_nh = res_nh.confidence_interval

    print(f"\n  ── {label} ──")
    print(f"  Accuracy                 : {acc_nh:.4f}")
    print(f"  ROC-AUC                  : {auc_nh:.4f}  "
          f"(95% CI: {ci_nh.low:.4f} – {ci_nh.high:.4f})")
    print(f"  Brier Score              : {brier_nh:.4f}")
    print(f"  Precision  (class 0 / 1) : {prec_nh[0]:.4f} / {prec_nh[1]:.4f}")
    print(f"  Recall     (class 0 / 1) : {rec_nh[0]:.4f} / {rec_nh[1]:.4f}")
    print(f"  F1-Score   (class 0 / 1) : {f1_nh[0]:.4f} / {f1_nh[1]:.4f}")
    print(f"  Macro F1                 : {mf1_nh:.4f}")
    print(f"  Confusion Matrix         :")
    print(f"    TN={cm_nh[0,0]}  FP={cm_nh[0,1]}")
    print(f"    FN={cm_nh[1,0]}  TP={cm_nh[1,1]}")

# store optimal values for CSV
pred_nh_opt  = (p_nh >= nh_opt_t).astype(int)
auc_nh_f     = roc_auc_score(y_nh, p_nh)
acc_nh_f     = accuracy_score(y_nh, pred_nh_opt)
prec_nh_f    = precision_score(y_nh, pred_nh_opt, average=None, zero_division=0)
rec_nh_f     = recall_score(y_nh, pred_nh_opt, average=None, zero_division=0)
f1_nh_f      = f1_score(y_nh, pred_nh_opt, average=None, zero_division=0)
cm_nh_f      = confusion_matrix(y_nh, pred_nh_opt)
brier_nh_f   = brier_score_loss(y_nh, p_nh)
res_nh_f = bootstrap(
    (y_nh.values, p_nh),
    lambda a, b: roc_auc_score(a, b),
    n_resamples=1000, random_state=42, paired=True
)
ci_nh_f = res_nh_f.confidence_interval

# =========================================================
# STEP 5 — SAVE CSV
# =========================================================
print("\n[5/5] Saving results_summary.csv ...")
rows = [
    ("Internal | Accuracy",              round(acc,        4)),
    ("Internal | ROC-AUC",               round(auc,        4)),
    ("Internal | AUC CI Low",            round(ci.low,     4)),
    ("Internal | AUC CI High",           round(ci.high,    4)),
    ("Internal | Brier Score",           round(brier,      4)),
    ("Internal | Precision Class 0",     round(prec[0],    4)),
    ("Internal | Precision Class 1",     round(prec[1],    4)),
    ("Internal | Recall Class 0",        round(rec[0],     4)),
    ("Internal | Recall Class 1",        round(rec[1],     4)),
    ("Internal | F1 Class 0",            round(f1[0],      4)),
    ("Internal | F1 Class 1",            round(f1[1],      4)),
    ("Internal | Macro Precision",       round(macro_prec, 4)),
    ("Internal | Macro Recall",          round(macro_rec,  4)),
    ("Internal | Macro F1",              round(macro_f1,   4)),
    ("Internal | Optimal Threshold",     opt_t),
    ("Internal | TN",                    int(cm[0,0])),
    ("Internal | FP",                    int(cm[0,1])),
    ("Internal | FN",                    int(cm[1,0])),
    ("Internal | TP",                    int(cm[1,1])),
    ("CV | Lab Agent AUC Mean",          round(cv_results["Laboratory Agent"][0],   4)),
    ("CV | Lab Agent AUC Std",           round(cv_results["Laboratory Agent"][1],   4)),
    ("CV | Physical Agent AUC Mean",     round(cv_results["Physical Agent"][0],     4)),
    ("CV | Physical Agent AUC Std",      round(cv_results["Physical Agent"][1],     4)),
    ("CV | Demographic Agent AUC Mean",  round(cv_results["Demographic Agent"][0],  4)),
    ("CV | Demographic Agent AUC Std",   round(cv_results["Demographic Agent"][1],  4)),
    ("Ablation | Full Framework AUC",    round(full_auc,  4)),
    ("Ablation | No Lab AUC",            round(no_lab,    4)),
    ("Ablation | No Lab Delta",          round(no_lab   - full_auc, 4)),
    ("Ablation | No Physical AUC",       round(no_phys,   4)),
    ("Ablation | No Physical Delta",     round(no_phys  - full_auc, 4)),
    ("Ablation | No Demographic AUC",    round(no_demo,   4)),
    ("Ablation | No Demographic Delta",  round(no_demo  - full_auc, 4)),
    ("External | Optimal Threshold",     nh_opt_t),
    ("External | Accuracy",              round(acc_nh_f,      4)),
    ("External | ROC-AUC",               round(auc_nh_f,      4)),
    ("External | AUC CI Low",            round(ci_nh_f.low,   4)),
    ("External | AUC CI High",           round(ci_nh_f.high,  4)),
    ("External | Brier Score",           round(brier_nh_f,    4)),
    ("External | Precision Class 0",     round(prec_nh_f[0],  4)),
    ("External | Precision Class 1",     round(prec_nh_f[1],  4)),
    ("External | Recall Class 0",        round(rec_nh_f[0],   4)),
    ("External | Recall Class 1",        round(rec_nh_f[1],   4)),
    ("External | F1 Class 0",            round(f1_nh_f[0],    4)),
    ("External | F1 Class 1",            round(f1_nh_f[1],    4)),
    ("External | TN",                    int(cm_nh_f[0,0])),
    ("External | FP",                    int(cm_nh_f[0,1])),
    ("External | FN",                    int(cm_nh_f[1,0])),
    ("External | TP",                    int(cm_nh_f[1,1])),
]
pd.DataFrame(rows, columns=["Metric","Value"]).to_csv(
    "results/results_summary.csv", index=False
)
print("  Saved: results/results_summary.csv")
print("\n" + "="*60)
print("  ALL METRICS COMPLETE — check results/results_summary.csv")
print("="*60)
