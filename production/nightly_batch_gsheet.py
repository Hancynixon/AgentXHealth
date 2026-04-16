import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deployment.model_runner import run_model_for_input

from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
    precision_score, recall_score, f1_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from agents.lab_agent_intelligent          import LabAgentIntelligent
from agents.physical_agent_intelligent     import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent  import DemographicAgentIntelligent

import gspread
from google.oauth2.service_account import Credentials

# =========================================================
# GOOGLE SHEETS CONFIG
# =========================================================
GSHEET_CREDENTIALS_JSON = "credentials.json"
GSHEET_SPREADSHEET_NAME = "AgentXHealth_Form"
GSHEET_WORKSHEET_INDEX  = 0

COL_FINAL_RISK     = "Final_Risk"
COL_DOMINANT_AGENT = "Dominant_Agent"
COL_EXPLANATION    = "Explanation"

COLUMN_MAP = {
    "Glucose":       "Glucose",
    "Insulin":       "Insulin",
    "BMI":           "BMI",
    "BloodPressure": "BloodPressure",
    "Age":           "Age",
    "Pregnancies":   "Pregnancies",
    "Email":         "Email",
    "Gender":        "Gender",
}

# =========================================================
# GOOGLE SHEETS → BATCH PROCESSING (PDF GENERATION)
# =========================================================
def run_batch_from_gsheet():
    print("\n[INFO] Connecting to Google Sheets...")

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    creds = Credentials.from_service_account_file(
        GSHEET_CREDENTIALS_JSON, scopes=scope
    )

    client = gspread.authorize(creds)
    sheet = client.open(GSHEET_SPREADSHEET_NAME).get_worksheet(GSHEET_WORKSHEET_INDEX)

    records = sheet.get_all_records()
    headers = [h.strip() for h in sheet.row_values(1)]
    header_to_col = {h: i + 1 for i, h in enumerate(headers)}

    print(f"[INFO] Total records fetched: {len(records)}")

    if len(records) == 0:
        print("[WARNING] No data found in sheet.")
        return

    required_output_headers = [COL_FINAL_RISK, COL_DOMINANT_AGENT, COL_EXPLANATION]
    missing_headers = [h for h in required_output_headers if h not in header_to_col]
    if missing_headers:
        print(f"[ERROR] Missing output columns in sheet header: {missing_headers}")
        return

    processed_count = 0
    skipped_count = 0

    for idx, row in enumerate(records):
        try:
            row_number = idx + 2  # +2 because header + 0-index

            # Skip rows already processed earlier.
            existing_final_risk = str(row.get(COL_FINAL_RISK, "")).strip()
            existing_dominant = str(row.get(COL_DOMINANT_AGENT, "")).strip()
            existing_explanation = str(row.get(COL_EXPLANATION, "")).strip()

            if existing_final_risk or existing_dominant or existing_explanation:
                skipped_count += 1
                print(f"\n[SKIP ROW {idx+1}] Already processed")
                continue

            print(f"\n[PROCESSING ROW {idx+1}]")

            # ---- Build input dict ----
            input_dict = {
                "Glucose": float(row.get("Glucose", 0)),
                "Insulin": float(row.get("Insulin", 0)),
                "BMI": float(row.get("BMI", 0)),
                "BloodPressure": float(row.get("BloodPressure", 0)),
                "Age": float(row.get("Age", 0)),
                "Pregnancies": float(row.get("Pregnancies", 0)),
                "Email": row.get("Email", ""),
                "Gender": row.get("Gender", ""),
            }

            # ---- RUN MODEL + GENERATE PDF ----
            result = run_model_for_input(input_dict)

            final_risk = result.get("final_risk", "")
            dominant = result.get("dominant_agent", "")
            explanation = result.get("explanation", "")

            print(f"  Risk: {final_risk} | Dominant: {dominant}")

            # ---- UPDATE SHEET ----
            sheet.update_cell(row_number, header_to_col[COL_FINAL_RISK], final_risk)
            sheet.update_cell(row_number, header_to_col[COL_DOMINANT_AGENT], dominant)
            sheet.update_cell(row_number, header_to_col[COL_EXPLANATION], explanation)

            print("  Sheet updated + PDF generated ✔")
            processed_count += 1

        except Exception as e:
            print(f"[ERROR] Row {idx+1} failed: {e}")

    print(
        f"\n[INFO] Batch done. New rows processed: {processed_count} | "
        f"Rows skipped (already processed): {skipped_count}"
    )

# =========================================================
# SAFE PROBABILITY EXTRACTOR
# =========================================================
def extract_prob(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("risk", "probability", "prob", "final_risk", "score", "prediction"):
            if key in value:
                return float(value[key])
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
        raise ValueError(f"No numeric value found in dict: {value}")
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            return extract_prob(value[0])
        raise ValueError("Empty list/tuple returned.")
    if isinstance(value, np.ndarray):
        return float(value.flat[0])
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    raise ValueError(f"Cannot extract probability from {type(value)}: {value!r}")

# =========================================================
# ECE (Expected Calibration Error)
# =========================================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if not np.any(mask):
            continue
        avg_conf = np.mean(y_prob[mask])
        avg_acc  = np.mean(y_true[mask])
        ece += (np.sum(mask) / len(y_true)) * abs(avg_conf - avg_acc)
    return ece

# =========================================================
# PREDICT ROW BY ROW
# =========================================================
def predict_all_rows(agent, X_df):
    probas = []
    for i in range(len(X_df)):
        row = X_df.iloc[[i]]
        try:
            prob = extract_prob(agent.predict(row))
        except Exception as e:
            print(f"  Warning row {i}: {e} — defaulting to 0.5")
            prob = 0.5
        probas.append(prob)
    return np.array(probas)

# =========================================================
# MODEL EVALUATION
# =========================================================
def run_evaluation():
    df = pd.read_csv("data/raw/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    Y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    train_df = X_train.copy()
    train_df["Outcome"] = y_train

    print("=" * 57)
    print("  AgentXHealth — Model Evaluation Report")
    print("=" * 57)
    print(f"  Train size : {len(X_train)} samples")
    print(f"  Test size  : {len(X_test)}  samples")
    print(f"  Diabetic % : {Y.mean()*100:.1f}% positive class")
    print("=" * 57)

    AGENTS = [
        ("Lab Agent",         LabAgentIntelligent),
        ("Physical Agent",    PhysicalAgentIntelligent),
        ("Demographic Agent", DemographicAgentIntelligent),
    ]
    all_auc    = {}
    all_probas = {}

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_cm,  axes   = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (name, AgentClass) in enumerate(AGENTS):
        print(f"\n{'─'*57}\n  {name}\n{'─'*57}")
        agent = AgentClass().fit(train_df, y_train)
        probas = predict_all_rows(agent, X_test)
        binary = (probas >= 0.5).astype(int)
        auc    = roc_auc_score(y_test, probas)
        all_auc[name]    = auc
        all_probas[name] = probas

        brier = brier_score_loss(y_test, probas)
        ece   = expected_calibration_error(y_test.values, probas)

        print(f"  Running predictions on {len(X_test)} test rows...")
        print(classification_report(y_test, binary,
              target_names=["Non-Diabetic", "Diabetic"]))
        print(f"  ROC-AUC     : {auc:.4f}")
        print(f"  Brier Score : {brier:.4f}")
        print(f"  ECE         : {ece:.4f}")

        print(f"\n  Threshold Sensitivity (Diabetic class):")
        print(f"  {'Thresh':>8}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}")
        for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            b = (probas >= thresh).astype(int)
            rec  = recall_score(y_test,  b, zero_division=0)
            prec = precision_score(y_test, b, zero_division=0)
            f1   = f1_score(y_test, b, zero_division=0)
            flag = "  ← clinical recommended" if thresh == 0.40 else ""
            print(f"  {thresh:>8.2f}  {rec:>8.4f}  {prec:>10.4f}  {f1:>8.4f}{flag}")

        fpr, tpr, _ = roc_curve(y_test, probas)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        cm = confusion_matrix(y_test, binary)
        ConfusionMatrixDisplay(cm, display_labels=["Non-Diabetic","Diabetic"]).plot(
            ax=axes[idx], colorbar=False)
        axes[idx].set_title(f"{name}\nAUC={auc:.3f}", fontsize=11)

    print(f"\n{'='*57}")
    print("  ENSEMBLE (Lab×0.5 + Physical×0.3 + Demographic×0.2)")
    print(f"{'='*57}")

    ensemble_probas = (
        all_probas["Lab Agent"]          * 0.5 +
        all_probas["Physical Agent"]     * 0.3 +
        all_probas["Demographic Agent"]  * 0.2
    )
    ensemble_binary = (ensemble_probas >= 0.5).astype(int)
    ensemble_auc    = roc_auc_score(y_test, ensemble_probas)
    all_auc["Ensemble"] = ensemble_auc

    ensemble_brier = brier_score_loss(y_test, ensemble_probas)
    ensemble_ece   = expected_calibration_error(y_test.values, ensemble_probas)

    print(classification_report(y_test, ensemble_binary,
          target_names=["Non-Diabetic", "Diabetic"]))
    print(f"  ROC-AUC     : {ensemble_auc:.4f}")
    print(f"  Brier Score : {ensemble_brier:.4f}")
    print(f"  ECE         : {ensemble_ece:.4f}")

    fpr_e, tpr_e, _ = roc_curve(y_test, ensemble_probas)
    ax_roc.plot(fpr_e, tpr_e, "k--", linewidth=2,
                label=f"Ensemble (AUC={ensemble_auc:.3f})")
    ax_roc.plot([0,1],[0,1],"gray",linestyle="dotted",label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — All Agents + Ensemble")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig("evaluation_roc_curves.png", dpi=150)
    fig_cm.suptitle("Confusion Matrices (threshold=0.5)", fontsize=14)
    fig_cm.tight_layout()
    fig_cm.savefig("evaluation_confusion_matrices.png", dpi=150)

    print(f"\n{'='*57}\n  FINE-TUNING VERDICT\n{'='*57}")
    for name, auc in all_auc.items():
        if auc >= 0.82:
            status = "✅ Excellent — no fine-tuning needed"
        elif auc >= 0.78:
            status = "✅ Good — no fine-tuning needed"
        elif auc >= 0.74:
            status = "⚠  Acceptable — threshold calibration suggested"
        else:
            status = "❌ Below threshold — fine-tuning recommended"
        print(f"  {name:<22}: AUC={auc:.4f}  {status}")

    print(f"\n  Saved: evaluation_roc_curves.png")
    print(f"  Saved: evaluation_confusion_matrices.png")
    print(f"{'='*57}")

# (rest of nightly_batch.py: gsheet functions and __main__ unchanged)

if __name__ == "__main__":
    print("[STARTING NIGHTLY BATCH]")
    run_batch_from_gsheet()
