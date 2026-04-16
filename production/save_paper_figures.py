import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
)
from sklearn.calibration import calibration_curve

from agents.lab_agent_intelligent         import LabAgentIntelligent
from agents.physical_agent_intelligent    import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

os.makedirs("paper_figures", exist_ok=True)

# ─────────────────────────────────────────────
# HELPER — pick correct SHAP explainer
# ─────────────────────────────────────────────
def get_shap_values(clf, X_scaled):
    """
    Automatically selects TreeExplainer or LinearExplainer
    based on the actual winning classifier type.
    Returns a 2D shap values array (n_samples, n_features).
    """
    if isinstance(clf, (GradientBoostingClassifier,)):
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X_scaled)
        # GradientBoostingClassifier returns single array
        if isinstance(shap_vals, list):
            return shap_vals[1]
        return shap_vals
    elif isinstance(clf, LogisticRegression):
        explainer  = shap.LinearExplainer(clf, X_scaled)
        shap_vals  = explainer.shap_values(X_scaled)
        # LinearExplainer returns single array for binary
        if isinstance(shap_vals, list):
            return shap_vals[1]
        return shap_vals
    else:
        # Fallback: KernelExplainer (slow but universal)
        print(f"  Using KernelExplainer for {type(clf).__name__} (slow)...")
        bg         = shap.kmeans(X_scaled, 50)
        explainer  = shap.KernelExplainer(clf.predict_proba, bg)
        shap_vals  = explainer.shap_values(X_scaled[:100])
        return shap_vals[1]


# ─────────────────────────────────────────────
# 1. LOAD & SPLIT DATA
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
# 2. TRAIN ALL AGENTS
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

p_lab  = predict_all(lab,  X_test)
p_phys = predict_all(phys, X_test)
p_demo = predict_all(demo, X_test)
p_ens  = p_lab * 0.5 + p_phys * 0.3 + p_demo * 0.2
b_ens  = (p_ens >= 0.5).astype(int)

print("All predictions done.")

# ─────────────────────────────────────────────
# FIG 2 — Confusion Matrix (Internal, Ensemble)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm   = confusion_matrix(y_test, b_ens)
disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Diabetic", "Diabetic"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(
    "Confusion Matrix – Internal Validation (Pima Dataset)\nDecision threshold = 0.5",
    fontsize=10
)
plt.tight_layout()
plt.savefig("paper_figures/fig2_confusion_matrix_internal.png", dpi=200)
plt.close()
print("Saved: fig2_confusion_matrix_internal.png")

# ─────────────────────────────────────────────
# FIG 3 — ROC Curves (All Agents + Ensemble)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fpr_l, tpr_l, _ = roc_curve(y_test, p_lab)
fpr_p, tpr_p, _ = roc_curve(y_test, p_phys)
fpr_d, tpr_d, _ = roc_curve(y_test, p_demo)
fpr_e, tpr_e, _ = roc_curve(y_test, p_ens)

ax.plot(fpr_l, tpr_l, label=f"Lab Agent         (AUC={roc_auc_score(y_test, p_lab):.3f})")
ax.plot(fpr_p, tpr_p, label=f"Physical Agent    (AUC={roc_auc_score(y_test, p_phys):.3f})")
ax.plot(fpr_d, tpr_d, label=f"Demographic Agent (AUC={roc_auc_score(y_test, p_demo):.3f})")
ax.plot(fpr_e, tpr_e, "k--", linewidth=2,
        label=f"Ensemble          (AUC={roc_auc_score(y_test, p_ens):.3f})")
ax.plot([0, 1], [0, 1], "gray", linestyle="dotted", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves – Internal Validation (Pima Dataset)", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("paper_figures/fig3_roc_curve_internal.png", dpi=200)
plt.close()
print("Saved: fig3_roc_curve_internal.png")

# ─────────────────────────────────────────────
# FIG 5 — Calibration Plot
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
prob_true, prob_pred = calibration_curve(y_test, p_ens, n_bins=10)
ax.plot(prob_pred, prob_true, "s-", color="steelblue",
        label="AgentXHealth (Ensemble)")
ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Observed Outcome Proportion", fontsize=11)
ax.set_title("Calibration Plot – Internal Validation (Pima Dataset)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("paper_figures/fig5_calibration_plot.png", dpi=200)
plt.close()
print("Saved: fig5_calibration_plot.png")

# ─────────────────────────────────────────────
# SHAP SETUP — scale features
# ─────────────────────────────────────────────
lab_feature_names = [
    "Glucose", "Glucose²", "Insulin", "Glucose×Insulin",
    "HOMA-IR", "QUICKI", "DPF", "DPF×Age",
    "DPF×Preg", "Glucose≥126", "Insulin>25", "HOMA-IR≥2.5",
]
phys_feature_names = [
    "BMI", "BloodPressure", "SkinThickness",
    "BMI_stage", "BP_stage", "BMI/BP_ratio",
    "BMI²", "Hypertension_flag", "Obesity_flag",
    "Obesity×HTN", "Skin/BMI_ratio", "HighSkin_flag",
]
demo_feature_names = [
    "Age", "Pregnancies", "AgeRisk_flag",
    "HighParity_flag", "Age×Preg", "Age²", "Midlife_flag",
]

lab_clf     = lab.model.named_steps["clf"]
lab_scaler  = lab.model.named_steps["scaler"]
X_lab_test  = lab._build_features(X_test)
X_lab_sc    = lab_scaler.transform(X_lab_test)

phys_clf    = phys.model.named_steps["clf"]
phys_scaler = phys.model.named_steps["scaler"]
X_phys_test = phys._build_features(X_test)
X_phys_sc   = phys_scaler.transform(X_phys_test)

demo_clf    = demo.model.named_steps["clf"]
demo_scaler = demo.model.named_steps["scaler"]
X_demo_test = demo._build_features(X_test)
X_demo_sc   = demo_scaler.transform(X_demo_test)

# ─────────────────────────────────────────────
# FIG 6 — SHAP Lab Agent
# ─────────────────────────────────────────────
print("Computing SHAP for Lab Agent...")
sv_lab = get_shap_values(lab_clf, X_lab_sc)
shap.summary_plot(
    sv_lab, X_lab_sc,
    feature_names=lab_feature_names,
    show=False, plot_size=(8, 5)
)
plt.title("SHAP Summary – Laboratory Agent", fontsize=12)
plt.tight_layout()
plt.savefig("paper_figures/fig6_shap_lab_agent.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig6_shap_lab_agent.png")

# ─────────────────────────────────────────────
# FIG 7 — SHAP Physical Agent
# ─────────────────────────────────────────────
print("Computing SHAP for Physical Agent...")
sv_phys = get_shap_values(phys_clf, X_phys_sc)
shap.summary_plot(
    sv_phys, X_phys_sc,
    feature_names=phys_feature_names,
    show=False, plot_size=(8, 5)
)
plt.title("SHAP Summary – Physical Agent", fontsize=12)
plt.tight_layout()
plt.savefig("paper_figures/fig7_shap_physical_agent.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig7_shap_physical_agent.png")

# ─────────────────────────────────────────────
# FIG 8 — SHAP Demographic Agent
# ─────────────────────────────────────────────
print("Computing SHAP for Demographic Agent...")
sv_demo = get_shap_values(demo_clf, X_demo_sc)
shap.summary_plot(
    sv_demo, X_demo_sc,
    feature_names=demo_feature_names,
    show=False, plot_size=(8, 5)
)
plt.title("SHAP Summary – Demographic Agent", fontsize=12)
plt.tight_layout()
plt.savefig("paper_figures/fig8_shap_demographic_agent.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig8_shap_demographic_agent.png")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n✅ All paper figures saved to: paper_figures/")
print("   fig2_confusion_matrix_internal.png  → Replace Fig 2 in paper")
print("   fig3_roc_curve_internal.png         → Replace Fig 3 in paper")
print("   fig5_calibration_plot.png           → Replace Fig 5 in paper")
print("   fig6_shap_lab_agent.png             → Replace Fig 6 in paper")
print("   fig7_shap_physical_agent.png        → Replace Fig 7 in paper")
print("   fig8_shap_demographic_agent.png     → Replace Fig 8 in paper")
