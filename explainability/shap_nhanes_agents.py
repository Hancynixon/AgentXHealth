import warnings

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent

# ==========================================================
# WARNING FILTERS (CLEANER OUTPUT)
# ==========================================================
# Suppress "X has feature names..." from sklearn
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but StandardScaler was fitted without feature names"
)

# Suppress noisy joblib resource_tracker warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib"
)

# ==========================================================
# GLOBAL PLOT SETTINGS (JOURNAL QUALITY)
# ==========================================================
plt.style.use("default")
plt.rcParams.update({
    "font.size": 12
})

# ==========================================================
# DATASET LOADER (Pima dataset)
# ==========================================================
def load_dataset():
    print("Loading Pima Indians Diabetes dataset...")
    df = pd.read_csv("data/raw/diabetes.csv")
    y = df["Outcome"]
    return df, y

# ==========================================================
# GENERIC SHAP RUNNER — WRAP PIPELINE IN CALLABLE
# ==========================================================
def run_shap(model, X_df, title, filename, max_samples=1000):
    """
    model     : fitted sklearn Pipeline (scaler + clf)
    X_df      : pandas DataFrame of engineered features
    title     : plot title suffix
    filename  : output PNG filename
    max_samples : subsample size for speed
    """
    print(f"\nGenerating SHAP for {title}...")

    # Downsample for speed if dataset is large
    if len(X_df) > max_samples:
        X_shap = X_df.sample(n=max_samples, random_state=42)
    else:
        X_shap = X_df

    # SHAP expects a callable model; wrap predict_proba[:, 1]
    explainer = shap.Explainer(
        lambda X: model.predict_proba(X)[:, 1],
        X_shap
    )
    shap_values = explainer(X_shap)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )
    plt.title(f"SHAP Summary – {title}")
    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )
    print(f"{filename} saved successfully.")
    plt.close()

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def run():
    df, y = load_dataset()

    # ------------------------------------------------------
    # Train agents
    # ------------------------------------------------------
    print("Training Lab Agent...")
    lab = LabAgentIntelligent()
    lab.fit(df, y)

    print("Training Physical Agent...")
    physical = PhysicalAgentIntelligent()
    physical.fit(df, y)

    print("Training Demographic Agent...")
    demo = DemographicAgentIntelligent()
    demo.fit(df, y)

    # ------------------------------------------------------
    # Build feature matrices EXACTLY as in agents
    # ------------------------------------------------------
    # Lab Agent features
    X_lab = pd.DataFrame(
        lab._build_features(df),
        columns=[
            "Glucose",
            "Glucose^2",
            "Insulin",
            "Glucose*Insulin",
            "HOMA-IR",
            "QUICKI",
            "DPF",
            "DPF*Age",
            "DPF*Pregnancies",
            "Glucose>=126_Flag",
            "Hyperinsulinemia_Flag",
            "HOMA>=2.5_Flag",
        ]
    )

    # Physical Agent features
    X_phys = pd.DataFrame(
        physical._build_features(df),
        columns=[
            "BMI",
            "BloodPressure",
            "SkinThickness",
            "BMI_Stage",
            "BP_Stage",
            "BMI_BP_Ratio",
            "BMI^2",
            "Hypertension_Flag",
            "Obesity_Flag",
            "Obesity_Htn_Combo",
            "Skin_BMI_Ratio",
            "High_Skin_Flag",
        ]
    )

    # Demographic Agent features
    X_demo = pd.DataFrame(
        demo._build_features(df),
        columns=[
            "Age",
            "Pregnancies",
            "Age_Risk_Flag",
            "High_Parity_Flag",
            "Age*Pregnancies",
            "Age^2",
            "Midlife_Flag",
        ]
    )

    # ------------------------------------------------------
    # Generate SHAP Plots (use full pipelines)
    # ------------------------------------------------------
    run_shap(
        lab.model,          # pipeline: scaler + clf
        X_lab,
        "Lab Agent",
        "Figure_4_SHAP_Lab_Agent.png"
    )

    run_shap(
        physical.model,
        X_phys,
        "Physical Agent",
        "Figure_5_SHAP_Physical_Agent.png"
    )

    run_shap(
        demo.model,
        X_demo,
        "Demographic Agent",
        "Figure_6_SHAP_Demographic_Agent.png"
    )

if __name__ == "__main__":
    run()
