import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent


# ==========================================================
# GLOBAL PLOT SETTINGS (JOURNAL QUALITY)
# ==========================================================

plt.style.use("default")
plt.rcParams.update({
    "font.size": 12
})


# ==========================================================
# DATASET LOADER
# ==========================================================

def load_dataset():
    print("Loading dataset...")
    df = pd.read_csv("data/raw/diabetes.csv")
    y = df["Outcome"]
    return df, y


# ==========================================================
# LAB FEATURE BUILDER
# ==========================================================

def build_lab_features(df, lab_agent):

    rows = []

    for _, row in df.iterrows():

        g = float(row["Glucose"])
        ins = float(row["Insulin"])
        dpf = float(row["DiabetesPedigreeFunction"])

        homa = lab_agent._compute_homa_ir(g, ins)
        quicki = lab_agent._compute_quicki(g, ins)

        rows.append([
            g,
            g**2,
            ins,
            g * ins,
            homa,
            quicki,
            dpf
        ])

    columns = [
        "Glucose",
        "Glucose^2",
        "Insulin",
        "Glucose*Insulin",
        "HOMA-IR",
        "QUICKI",
        "DPF"
    ]

    return pd.DataFrame(rows, columns=columns)


# ==========================================================
# PHYSICAL FEATURE BUILDER
# ==========================================================

def build_physical_features(df, phys_agent):

    bmi = df["BMI"].astype(float)
    bp = df["BloodPressure"].astype(float)

    bmi_stage = bmi.apply(phys_agent._bmi_stage)
    bp_stage = bp.apply(phys_agent._bp_stage)

    features = pd.DataFrame({
        "BMI": bmi,
        "BloodPressure": bp,
        "BMI_stage": bmi_stage,
        "BP_stage": bp_stage
    })

    return features


# ==========================================================
# DEMOGRAPHIC FEATURE BUILDER
# ==========================================================

def build_demo_features(df):

    features = pd.DataFrame({
        "Age": df["Age"].astype(float),
        "Pregnancies": df["Pregnancies"].astype(float)
    })

    return features


# ==========================================================
# GENERIC SHAP RUNNER (UPDATED FOR PUBLICATION)
# ==========================================================

def run_shap(model, X_df, title, filename):

    print(f"\nGenerating SHAP for {title}...")

    explainer = shap.Explainer(model, X_df)
    shap_values = explainer(X_df)

    plt.figure(figsize=(8, 6))

    shap.summary_plot(
        shap_values,
        X_df,
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

    plt.show()
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
    # Build feature matrices
    # ------------------------------------------------------

    X_lab = build_lab_features(df, lab)
    X_lab_scaled = pd.DataFrame(
        lab.scaler.transform(X_lab),
        columns=X_lab.columns
    )

    X_phys = build_physical_features(df, physical)
    X_demo = build_demo_features(df)

    # ------------------------------------------------------
    # Generate SHAP Plots (Publication Named)
    # ------------------------------------------------------

    run_shap(
        lab.model,
        X_lab_scaled,
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