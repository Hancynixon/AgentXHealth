import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent


RESULTS_DIR = "results/strict_publication_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

PIMA_PATH = "data/raw/diabetes.csv"
NHANES_PATH = "data/NHNES/nhanes_diabetes_processed.csv"

PIMA_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def load_pima():
    pima = pd.read_csv(PIMA_PATH)
    return pima[PIMA_COLS].copy()


def load_nhanes_harmonized():
    nh = pd.read_csv(NHANES_PATH).copy()

    col_map = {
        "glucose": "Glucose",
        "LBXGLU": "Glucose",
        "LBXSGL": "Glucose",
        "LBXGLT": "Glucose",
        "bmi": "BMI",
        "BMXBMI": "BMI",
        "blood_pressure": "BloodPressure",
        "BPXSY1": "BloodPressure",
        "insulin": "Insulin",
        "LBXIN": "Insulin",
        "skin_thickness": "SkinThickness",
        "age": "Age",
        "RIDAGEYR": "Age",
        "dpf": "DiabetesPedigreeFunction",
        "pregnancies": "Pregnancies",
        "diabetes": "Outcome",
        "Diabetes": "Outcome",
        "DIQ010": "Outcome",
    }
    nh = nh.rename(columns={k: v for k, v in col_map.items() if k in nh.columns})

    if "HbA1c" in nh.columns:
        nh["DiabetesPedigreeFunction"] = pd.to_numeric(
            nh["HbA1c"], errors="coerce"
        ).fillna(0)

    for col in PIMA_COLS:
        if col not in nh.columns:
            nh[col] = 0

    if nh["Outcome"].max() == 2:
        nh["Outcome"] = nh["Outcome"].map({1: 1, 2: 0}).fillna(0).astype(int)

    for col in PIMA_COLS:
        nh[col] = pd.to_numeric(nh[col], errors="coerce")

    if nh["Glucose"].median() < 30:
        nh["Glucose"] = nh["Glucose"] * 18.0

    nh = nh.dropna(subset=["BMI", "Age"])
    nh = nh[nh["BMI"] > 0]
    nh = nh[nh["Age"] >= 18]
    nh = nh[nh["Glucose"].notna() & (nh["Glucose"] > 0)]

    for col in PIMA_COLS:
        if nh[col].isnull().any():
            nh[col] = nh[col].fillna(nh[col].median())

    return nh[PIMA_COLS].reset_index(drop=True)


def extract_prob(value):
    if isinstance(value, dict):
        for key in (
            "risk",
            "probability",
            "prob",
            "final_risk",
            "score",
            "prediction",
        ):
            if key in value:
                return float(value[key])
        for v in value.values():
            try:
                return float(v)
            except Exception:
                continue
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value).flatten()
        if arr.size == 0:
            return 0.0
        if arr.size == 2:
            return float(arr[1])
        return float(arr[0])
    try:
        return float(value)
    except Exception:
        return 0.0


def get_probs(agent, df):
    probs = []
    for i in range(len(df)):
        row = df.iloc[[i]]
        pred = agent.predict(row)
        probs.append(extract_prob(pred))
    return np.clip(np.array(probs, dtype=float), 0.0, 1.0)


def ensemble_probs(lab, phys, demo, df):
    return 0.5 * get_probs(lab, df) + 0.3 * get_probs(phys, df) + 0.2 * get_probs(demo, df)


def best_threshold(y_true, probs):
    best_t = 0.5
    best_f1 = -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 2), round(best_f1, 4)


def compute_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision_class_0": precision_score(y_true, preds, pos_label=0, zero_division=0),
        "precision_class_1": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall_class_0": recall_score(y_true, preds, pos_label=0, zero_division=0),
        "recall_class_1": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "f1_class_0": f1_score(y_true, preds, pos_label=0, zero_division=0),
        "f1_class_1": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "brier": brier_score_loss(y_true, probs),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def save_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Diabetic", "Diabetic"])
    ax.set_yticklabels(["Non-Diabetic", "Diabetic"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def save_roc_curve(y_true, probs, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    print("[1/5] Loading datasets...")
    pima = load_pima()
    nhanes = load_nhanes_harmonized()

    x_pima = pima.drop("Outcome", axis=1)
    y_pima = pima["Outcome"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pima, y_pima, test_size=0.2, stratify=y_pima, random_state=42
    )
    train_df = pd.concat([x_train, y_train], axis=1)
    pima_test_df = pd.concat([x_test, y_test], axis=1)
    nhanes_df = nhanes.copy()
    y_nhanes = nhanes_df["Outcome"].astype(int)

    print("[2/5] Training agents on Pima training split only...")
    lab = LabAgentIntelligent().fit(train_df, y_train)
    phys = PhysicalAgentIntelligent().fit(train_df, y_train)
    demo = DemographicAgentIntelligent().fit(train_df, y_train)

    print("[3/5] Generating ensemble probabilities...")
    pima_probs = ensemble_probs(lab, phys, demo, pima_test_df)
    nhanes_probs = ensemble_probs(lab, phys, demo, nhanes_df)

    default_t = 0.5
    optimal_t, optimal_f1 = best_threshold(y_test.values, pima_probs)

    print("[4/5] Computing strict metrics...")
    metrics = []
    for dataset_name, y_true, probs in [
        ("internal_pima", y_test.values, pima_probs),
        ("external_nhanes", y_nhanes.values, nhanes_probs),
    ]:
        for tag, thr in [("default", default_t), ("internal_optimal", optimal_t)]:
            row = compute_metrics(y_true, probs, thr)
            row["dataset"] = dataset_name
            row["threshold_tag"] = tag
            metrics.append(row)

    metrics_df = pd.DataFrame(metrics)
    metrics_csv = os.path.join(RESULTS_DIR, "strict_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    print("[5/5] Saving figures from computed outputs...")
    cm_pima = confusion_matrix(y_test.values, (pima_probs >= default_t).astype(int))
    cm_nhanes = confusion_matrix(y_nhanes.values, (nhanes_probs >= default_t).astype(int))

    save_confusion_matrix(
        cm_pima,
        "Confusion Matrix - Internal (Pima, t=0.50)",
        os.path.join(RESULTS_DIR, "cm_internal_pima.png"),
    )
    save_confusion_matrix(
        cm_nhanes,
        "Confusion Matrix - External (NHANES, t=0.50)",
        os.path.join(RESULTS_DIR, "cm_external_nhanes.png"),
    )
    save_roc_curve(
        y_test.values,
        pima_probs,
        "ROC Curve - Internal (Pima)",
        os.path.join(RESULTS_DIR, "roc_internal_pima.png"),
    )
    save_roc_curve(
        y_nhanes.values,
        nhanes_probs,
        "ROC Curve - External (NHANES)",
        os.path.join(RESULTS_DIR, "roc_external_nhanes.png"),
    )

    print("\nStrict publication evaluation complete.")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved figures: {RESULTS_DIR}")
    print(f"Internal optimal threshold (from Pima holdout): {optimal_t} (F1={optimal_f1})")


if __name__ == "__main__":
    main()
