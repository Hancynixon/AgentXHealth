import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import datetime
import smtplib
from email.mime.multipart   import MIMEMultipart
from email.mime.text        import MIMEText
from email.mime.application import MIMEApplication
from io        import BytesIO
from xhtml2pdf import pisa
import math
import numpy as np
import joblib

from agents.lab_agent_intelligent         import LabAgentIntelligent
from agents.physical_agent_intelligent    import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner     import CoordinatorReasoner
from utils.validator                      import validate_inputs


# =========================================================
# CONFIGURATION
# =========================================================
SENDER_EMAIL               = "akp11172000@gmail.com"
APP_PASSWORD               = "xnoefzdynmtpkbbx"   # ← Replace
NORMAL_FASTING_INSULIN_MAX = 25.0
HOMA_IR_THRESHOLD          = 2.5
QUICKI_THRESHOLD           = 0.33
USE_COMBINED_MODELS        = True

LAB_MODEL_PATH  = "models/lab_combined.pkl"
PHYS_MODEL_PATH = "models/phys_combined.pkl"
DEMO_MODEL_PATH = "models/demo_combined.pkl"


# =========================================================
# LAZY AGENT LOADER — trains only once on first call
# =========================================================
_agents_ready = False
lab = physical = demo = coord = None

def _load_agents():
    global lab, physical, demo, coord, _agents_ready
    if _agents_ready:
        return
    import time
    t0 = time.time()
    print("  [AgentXHealth] Loading agents (first call only)...")

    combined_available = all(
        os.path.exists(p) for p in [LAB_MODEL_PATH, PHYS_MODEL_PATH, DEMO_MODEL_PATH]
    )

    if USE_COMBINED_MODELS and combined_available:
        print("  [AgentXHealth] Loading combined pretrained agents...")
        lab = joblib.load(LAB_MODEL_PATH)
        physical = joblib.load(PHYS_MODEL_PATH)
        demo = joblib.load(DEMO_MODEL_PATH)
    else:
        if USE_COMBINED_MODELS and not combined_available:
            print("  [AgentXHealth] Combined models not found. Falling back to Pima training.")
        train_df = pd.read_csv("data/raw/diabetes.csv")
        Y = train_df["Outcome"]
        lab = LabAgentIntelligent().fit(train_df, Y)
        physical = PhysicalAgentIntelligent().fit(train_df, Y)
        demo = DemographicAgentIntelligent().fit(train_df, Y)

    coord    = CoordinatorReasoner()
    _agents_ready = True
    print(f"  [AgentXHealth] Agents ready in {time.time()-t0:.1f}s")


# =========================================================
# SAFE PROBABILITY EXTRACTOR
# =========================================================
def extract_prob(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("risk","probability","prob","final_risk","score","prediction"):
            if key in value:
                return float(value[key])
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
        raise ValueError(f"No numeric value found in dict: {value}")
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            return extract_prob(value[0])
        raise ValueError("Empty list/tuple returned by agent.")
    if isinstance(value, np.ndarray):
        return float(value.flat[0])
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    raise ValueError(f"Cannot extract probability from type {type(value)}: {value!r}")


# =========================================================
# DERIVED CLINICAL INDICES
# =========================================================
def compute_homa_ir(glucose, insulin):
    if glucose <= 0 or insulin <= 0:
        return None
    return round((glucose * insulin) / 405.0, 3)

def compute_quicki(glucose, insulin):
    if glucose <= 0 or insulin <= 0:
        return None
    try:
        return round(1.0 / (math.log(insulin) + math.log(glucose)), 4)
    except (ValueError, ZeroDivisionError):
        return None


# =========================================================
# GUARDRAIL
# =========================================================
def run_sanity_checks(input_data):
    errors = []
    g      = input_data.get("Glucose", 0)
    bp     = input_data.get("BloodPressure", 0)
    bmi    = input_data.get("BMI", 0)
    age    = input_data.get("Age", 0)
    ins    = input_data.get("Insulin", 0)
    p      = input_data.get("Pregnancies", 0)
    gender = input_data.get("Gender", "Female")

    if g < 20 or g > 600:
        errors.append(f"Invalid Glucose ({g}): Outside survivable range (20-600 mg/dL).")
    if bp < 50 or bp > 250:
        errors.append(f"Invalid BloodPressure ({bp}): Life-threatening value.")
    if bmi < 10 or bmi > 100:
        errors.append(f"Invalid BMI ({bmi}): Outside valid bounds (10-100).")
    if age < 0 or age > 120:
        errors.append(f"Invalid Age ({age}): Must be 0-120.")
    if ins < 0 or ins > 2000:
        errors.append(f"Invalid Insulin ({ins}): Outside plausible range (0-2000 uU/mL).")
    if gender.lower() == "male" and p > 0:
        errors.append("Data Error: Male gender with pregnancies > 0.")
    if p > 0 and (age - p) < 12:
        errors.append("Data Error: Age-pregnancy timeline is implausible.")
    return errors


# =========================================================
# RISK BOOST FUNCTIONS
# =========================================================
def glucose_boost(g):
    if g < 90:      return 0.0
    elif g <= 140:  return (g - 90) / 50 * 0.15
    elif g <= 200:  return 0.15 + (g - 140) / 60 * 0.15
    else:           return 0.30

def bmi_boost(b):
    if b < 18.5:    return 0.02
    elif b <= 25:   return 0.0
    elif b <= 35:   return (b - 25) / 10 * 0.10
    else:           return 0.10

def bp_boost(bp):
    if bp < 70:     return 0.05
    elif bp <= 90:  return 0.02
    elif bp <= 120: return 0.0
    elif bp <= 140: return 0.05
    else:           return 0.10

def age_boost(a):
    if a < 30:      return 0.0
    elif a <= 60:   return (a - 30) / 30 * 0.10
    else:           return 0.10

def insulin_boost(i):
    if i < 5:       return 0.05
    elif i <= 25:   return 0.0
    elif i <= 100:  return 0.08
    else:           return 0.15

def pregnancy_boost(p, gender):
    if gender.lower() == "male" or p == 0: return 0.0
    elif p <= 2:    return 0.02
    elif p <= 5:    return 0.05
    else:           return 0.10


# =========================================================
# CLINICAL OVERRIDE ENGINE
# =========================================================
def clinical_override(prob, input_data):
    glucose  = input_data["Glucose"]
    insulin  = input_data["Insulin"]
    bp       = input_data["BloodPressure"]
    homa     = compute_homa_ir(glucose, insulin)
    warnings = []
    floor    = prob

    if glucose < 55:
        floor = max(floor, 0.85)
        warnings.append("CRITICAL: Hypoglycemia (< 55 mg/dL). Immediate clinical intervention required.")
    elif glucose >= 250:
        floor = max(floor, 0.90)
        warnings.append("CRITICAL: Hyperglycemia (>= 250 mg/dL). Emergency medical review required.")
    elif glucose >= 180:
        floor = max(floor, 0.80)
        warnings.append("Severe hyperglycemia (>= 180 mg/dL). Urgent medical evaluation recommended.")
    elif glucose >= 126:
        if insulin <= 5:
            floor = max(floor, 0.70)
            warnings.append(
                "Diabetic glucose threshold met with very low insulin "
                "— possible beta-cell dysfunction."
            )
        else:
            floor = max(floor, 0.65)
            warnings.append("Diabetic-range fasting glucose (>= 126 mg/dL). Confirmatory testing advised.")

    if insulin > NORMAL_FASTING_INSULIN_MAX and glucose < 100:
        floor = max(floor, prob + 0.15)
        msg = (
            f"Elevated fasting insulin ({insulin} uU/mL) with normal glucose — "
            "possible compensatory hyperinsulinemia and early insulin resistance."
        )
        if homa and homa >= HOMA_IR_THRESHOLD:
            msg += f" HOMA-IR = {homa} (threshold: {HOMA_IR_THRESHOLD})."
        warnings.append(msg)

    if bp < 70:
        floor = max(floor, 0.75)
        warnings.append("CRITICAL: Hypotension (BP < 70 mm Hg). Immediate evaluation required.")
    elif bp <= 90:
        floor = max(floor, 0.60)
        warnings.append("Low blood pressure (BP <= 90 mm Hg). Monitor for symptoms.")

    return round(max(prob, floor), 4), warnings


def apply_clinical_logic(base, input_data):
    contributions = {
        "Glucose":       glucose_boost(input_data["Glucose"]),
        "BMI":           bmi_boost(input_data["BMI"]),
        "BloodPressure": bp_boost(input_data["BloodPressure"]),
        "Age":           age_boost(input_data["Age"]),
        "Insulin":       insulin_boost(input_data["Insulin"]),
        "Pregnancies":   pregnancy_boost(
                             input_data.get("Pregnancies", 0),
                             input_data.get("Gender", "Female")
                         ),
    }
    boost  = min(sum(contributions.values()), 0.35)
    final, warnings = clinical_override(base + boost, input_data)
    return max(0.01, min(round(final, 4), 0.99)), contributions, warnings


def get_label(prob):
    if prob < 0.20:   return "LOW"
    elif prob < 0.60: return "MODERATE"
    else:             return "HIGH"

def get_confidence(prob):
    d = abs(prob - 0.5)
    if d > 0.40:   return "Very High"
    elif d > 0.25: return "High"
    elif d > 0.15: return "Moderate"
    else:          return "Low"


# =========================================================
# PDF-SAFE STATUS BADGES
# =========================================================
def status_tag(level):
    styles = {
        "ok":       ("background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7;", "NORMAL"),
        "warn":     ("background:#fff8e1; color:#e65100; border:1px solid #ffcc02;", "BORDERLINE"),
        "high":     ("background:#ffebee; color:#b71c1c; border:1px solid #ef9a9a;", "HIGH RISK"),
        "critical": ("background:#fce4ec; color:#880e4f; border:1px solid #f48fb1;", "CRITICAL"),
        "info":     ("background:#e3f2fd; color:#0d47a1; border:1px solid #90caf9;", "NOTE"),
    }
    style, label = styles.get(level, styles["info"])
    return (
        f"<span style='{style} padding:2px 8px; border-radius:3px; "
        f"font-size:10px; font-weight:bold; font-family:Arial;'>{label}</span>"
    )


# =========================================================
# IMPACT LABEL
# =========================================================
def get_impact_label(metric, value, weight):
    def row(level, explanation):
        return (
            f"{status_tag(level)}"
            f"<br/><span style='font-size:11px; color:#444; line-height:1.6;'>"
            f"{explanation}</span>"
        )
    v = value or 0

    if metric == "Glucose":
        if v < 70:    return row("critical", "Hypoglycemia risk (&lt; 70 mg/dL). Immediate review needed.")
        elif v < 100: return row("ok",       "Normal fasting glucose (70-99 mg/dL).")
        elif v < 126: return row("warn",     "Prediabetes range (100-125 mg/dL). Lifestyle changes recommended.")
        else:         return row("high",     "Diabetic-range fasting glucose (&ge; 126 mg/dL). Confirmatory test advised.")

    elif metric == "BMI":
        if v < 18.5:  return row("warn", "Underweight (BMI &lt; 18.5). Nutritional review advised.")
        elif v < 25:  return row("ok",   "Healthy BMI range (18.5-24.9).")
        elif v < 30:  return row("warn", "Overweight (BMI 25-29.9). Associated with insulin resistance.")
        else:         return row("high", "Obesity (BMI &ge; 30). Significantly elevates T2DM risk.")

    elif metric == "BloodPressure":
        if v == 0:    return row("info", "Not recorded.")
        elif v < 80:  return row("warn", "Possible hypotension. Monitor for dizziness or fatigue.")
        elif v < 120: return row("ok",   "Normal blood pressure (&lt; 120 mm Hg).")
        elif v < 130: return row("warn", "Elevated Stage 0 hypertension (120-129 mm Hg).")
        elif v < 140: return row("warn", "Stage 1 hypertension (130-139 mm Hg). Lifestyle modification advised.")
        else:         return row("high", "Stage 2 hypertension (&ge; 140 mm Hg). Medical evaluation required.")

    elif metric == "Age":
        if v < 30:    return row("ok",   "Young adult (&lt; 30 yrs). Low age-related baseline risk.")
        elif v < 45:  return row("warn", "Early midlife (30-44 yrs). Risk begins to rise gradually.")
        elif v < 60:  return row("warn", "Midlife (45-59 yrs). Increased T2DM susceptibility.")
        else:         return row("high", "Senior (&ge; 60 yrs). Age is a significant independent risk factor.")

    elif metric == "Insulin":
        if v < 5:      return row("critical", "Very low insulin (&lt; 5 uU/mL). Possible beta-cell dysfunction.")
        elif v <= 25:  return row("ok",       "Normal fasting insulin (5-25 uU/mL).")
        elif v <= 100: return row("warn",     "Elevated fasting insulin (&gt; 25 uU/mL). Early compensatory hyperinsulinemia likely.")
        else:          return row("high",     "Severe hyperinsulinemia (&gt; 100 uU/mL). Strong insulin resistance signal.")

    elif metric == "Pregnancies":
        if v == 0:    return row("ok",   "No prior pregnancies. No gestational diabetes history factor.")
        elif v <= 2:  return row("ok",   "1-2 pregnancies. Minimal additional risk.")
        elif v <= 4:  return row("warn", "3-4 pregnancies. Possible gestational diabetes history.")
        else:         return row("high", "5+ pregnancies. Elevated gestational diabetes risk history.")

    return row("info", "Data not available.")


# =========================================================
# PHYSICIAN NARRATIVE
# =========================================================
def glucose_insulin_interpretation(glucose, insulin):
    homa   = compute_homa_ir(glucose, insulin)
    quicki = compute_quicki(glucose, insulin)

    if glucose < 55:
        return (
            "Severe hypoglycemia detected (< 55 mg/dL). The insulin-to-glucose relationship "
            "requires immediate emergency medical review. This finding takes clinical priority "
            "over all other metabolic parameters."
        )
    if glucose >= 126 and insulin <= 5:
        return (
            f"Fasting glucose of {glucose} mg/dL meets the diabetic threshold (&ge; 126 mg/dL), "
            f"combined with critically low insulin (&le; 5 uU/mL). This pattern raises strong concern "
            "for impaired pancreatic beta-cell secretion — a presentation consistent with "
            "early Type 1 diabetes or MODY (Maturity Onset Diabetes of the Young). "
            "Urgent endocrinological assessment including C-peptide and autoantibody testing is recommended."
        )
    if insulin > NORMAL_FASTING_INSULIN_MAX and glucose < 100:
        ir_note = quicki_note = ""
        if homa and homa >= HOMA_IR_THRESHOLD:
            ir_note = (
                f" The computed HOMA-IR of {homa} significantly exceeds the insulin resistance "
                f"threshold of {HOMA_IR_THRESHOLD}, confirming metabolic dysfunction."
            )
        if quicki and quicki <= QUICKI_THRESHOLD:
            quicki_note = (
                f" A QUICKI index of {quicki} (threshold &le; {QUICKI_THRESHOLD}) further validates "
                "reduced insulin sensitivity at the cellular level."
            )
        return (
            f"Fasting glucose of {glucose} mg/dL is within the normal range; however, fasting insulin "
            f"is significantly elevated at {insulin} uU/mL "
            f"(normal upper limit: {int(NORMAL_FASTING_INSULIN_MAX)} uU/mL). "
            "This pattern of normal glucose with high insulin represents compensatory hyperinsulinemia, "
            "where the pancreas overproduces insulin to maintain glucose levels. "
            "This is a well-established early precursor to Type 2 diabetes and metabolic syndrome."
            f"{ir_note}{quicki_note} "
            "Metabolic monitoring, dietary review, and physical activity assessment are strongly advised."
        )
    if glucose >= 126 and insulin > 25:
        return (
            f"Fasting glucose of {glucose} mg/dL (diabetic range) combined with elevated insulin "
            f"of {insulin} uU/mL strongly suggests established insulin resistance — the hallmark "
            "of Type 2 diabetes pathophysiology. The pancreas is producing excess insulin but "
            "peripheral tissues are failing to respond adequately. Prompt medical evaluation is advised."
        )
    if glucose >= 100:
        return (
            f"Fasting glucose of {glucose} mg/dL is in the prediabetes range (100-125 mg/dL). "
            "Without intervention, prediabetes carries a significant risk of progression to "
            "Type 2 diabetes within 5-10 years. Lifestyle modifications including dietary changes "
            "and increased physical activity can substantially reduce this risk."
        )
    return (
        f"Fasting glucose ({glucose} mg/dL) and insulin ({insulin} uU/mL) are both within "
        "normal physiological limits. Current metabolic markers do not indicate active insulin "
        "resistance. Continued routine monitoring is recommended as a preventive measure."
    )


# =========================================================
# HTML REPORT GENERATOR — fully PDF-safe
# =========================================================
def generate_email_report(input_data, decision):
    risk_label = decision["risk_label"]
    risk_prob  = decision["final_risk"]
    xai        = decision["feature_contributions"]
    homa       = compute_homa_ir(input_data["Glucose"], input_data["Insulin"])
    quicki     = compute_quicki(input_data["Glucose"], input_data["Insulin"])
    narrative  = glucose_insulin_interpretation(
                     input_data["Glucose"], input_data["Insulin"])
    today      = datetime.datetime.now().strftime("%B %d, %Y")
    report_id  = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    prob_pct   = int(risk_prob * 100)

    # ── Risk color scheme
    if risk_label == "LOW":
        risk_color = "#1b5e20"; risk_bg = "#e8f5e9"
        risk_border = "#a5d6a7"; bar_color = "#43a047"
    elif risk_label == "MODERATE":
        risk_color = "#e65100"; risk_bg = "#fff8e1"
        risk_border = "#ffb300"; bar_color = "#fb8c00"
    else:
        risk_color = "#b71c1c"; risk_bg = "#ffebee"
        risk_border = "#ef9a9a"; bar_color = "#e53935"

    # ── Action plan
    if risk_label == "LOW":
        action_items = [
            "<b>Maintain Lifestyle:</b> Your metabolic markers are reassuring. Continue your current diet and exercise routine.",
            "<b>Annual Screening:</b> Repeat this assessment in 12-24 months as a preventive measure.",
            "<b>Stay Informed:</b> Be aware of family history and lifestyle factors that can shift risk over time.",
        ]
    elif risk_label == "MODERATE":
        action_items = [
            "<b>Physician Consult:</b> Schedule a follow-up with your primary care physician or endocrinologist within 3-4 weeks.",
            "<b>HbA1c Testing:</b> Ask your doctor about an HbA1c test to measure long-term glucose control over the past 2-3 months.",
            "<b>Fasting Lipid Panel:</b> A full metabolic panel including cholesterol and triglycerides is recommended.",
            "<b>Lifestyle Adjustments:</b> Reduce refined carbohydrate intake and increase daily walking (30+ mins/day).",
            "<b>Weight Management:</b> A 5-7% reduction in body weight can reduce T2DM progression risk by up to 58%.",
        ]
    else:
        action_items = [
            "<b>Urgent Consult:</b> Contact your physician or endocrinologist within 3-5 days. Do not delay.",
            "<b>Comprehensive Metabolic Panel:</b> Full panel including HbA1c, insulin, lipids, and kidney function.",
            "<b>Do Not Self-Medicate:</b> Do not start medications or drastic diets without consulting your doctor.",
            "<b>Monitor Symptoms:</b> Watch for excessive thirst, frequent urination, blurred vision, or fatigue.",
            "<b>Emergency Signs:</b> Confusion, rapid breathing, or loss of consciousness — call emergency services immediately.",
        ]

    action_html = "".join(
        f"<li style='margin-bottom:10px; line-height:1.7; color:#333;'>{item}</li>"
        for item in action_items
    )

    # ── Clinical alerts
    alerts_html = ""
    if decision.get("clinical_warnings"):
        items = "".join(
            f"<li style='margin-bottom:8px; color:#7f0000; line-height:1.6;'>{w}</li>"
            for w in decision["clinical_warnings"]
        )
        alerts_html = f"""
        <div style="background-color:#ffebee; border-left:5px solid #c62828;
                    padding:12px 16px; margin-top:18px; border-radius:4px;">
            <p style="color:#c62828; font-weight:bold; margin:0 0 8px 0;
                      font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">
                URGENT CLINICAL ALERTS
            </p>
            <ul style="margin:0; padding-left:18px;">{items}</ul>
        </div>"""

    # ── Metabolic indices
    # KEY FIX: use <td> styled as header (not <th>) — eliminates xhtml2pdf gap
    indices_html = ""
    if homa is not None or quicki is not None:
        homa_color   = "#b71c1c" if (homa   and homa   >= HOMA_IR_THRESHOLD) else "#1b5e20"
        quicki_color = "#b71c1c" if (quicki and quicki <= QUICKI_THRESHOLD)   else "#1b5e20"
        homa_bg      = "#fff3f3" if (homa   and homa   >= HOMA_IR_THRESHOLD) else "#f1fff1"
        quicki_bg    = "#fff3f3" if (quicki and quicki <= QUICKI_THRESHOLD)   else "#f1fff1"
        homa_interp  = (f"Insulin resistance indicated (&ge; {HOMA_IR_THRESHOLD})"
                        if (homa and homa >= HOMA_IR_THRESHOLD)
                        else f"Normal (&lt; {HOMA_IR_THRESHOLD})")
        quicki_interp= (f"Reduced insulin sensitivity (&le; {QUICKI_THRESHOLD})"
                        if (quicki and quicki <= QUICKI_THRESHOLD)
                        else f"Normal (&gt; {QUICKI_THRESHOLD})")

        indices_html = f"""
        <div style="margin-top:18px;">
            <div style="background-color:#0f4c81; color:white; padding:8px 14px;
                        border-radius:4px 4px 0 0;">
                <span style="font-size:11px; font-weight:bold; letter-spacing:0.5px;">
                    COMPUTED METABOLIC INDICES
                </span>
            </div>
            <p style="font-size:11px; color:#555; margin:5px 0 7px 0; line-height:1.5;">
                HOMA-IR and QUICKI are physiologically validated indices that quantify insulin
                resistance beyond raw glucose and insulin values alone.
            </p>
            <table style="width:100%; border-collapse:collapse; font-size:12px;
                          border:1px solid #c5cae9;">
                <tr>
                    <td style="padding:8px 12px; border:1px solid #c5cae9; width:32%;
                                background-color:#e8eaf6; font-weight:bold; font-size:12px;
                                color:#1a237e; line-height:1.4;">Index</td>
                    <td style="padding:8px 12px; border:1px solid #c5cae9; width:18%;
                                background-color:#e8eaf6; font-weight:bold; font-size:12px;
                                color:#1a237e; text-align:center; line-height:1.4;">Your Value</td>
                    <td style="padding:8px 12px; border:1px solid #c5cae9; width:50%;
                                background-color:#e8eaf6; font-weight:bold; font-size:12px;
                                color:#1a237e; line-height:1.4;">Clinical Interpretation</td>
                </tr>
                <tr>
                    <td style="padding:10px 12px; border:1px solid #ddd;
                                background-color:{homa_bg}; vertical-align:top; line-height:1.5;">
                        <b>HOMA-IR</b><br/>
                        <span style="font-size:10px; color:#666;">(Glucose x Insulin) / 405</span>
                    </td>
                    <td style="padding:10px 12px; border:1px solid #ddd;
                                background-color:{homa_bg}; text-align:center;
                                font-weight:bold; font-size:18px; color:{homa_color};
                                vertical-align:middle; white-space:nowrap;">
                        {homa if homa else 'N/A'}
                    </td>
                    <td style="padding:10px 12px; border:1px solid #ddd;
                                background-color:{homa_bg}; vertical-align:top; line-height:1.5;">
                        <span style="color:{homa_color}; font-weight:bold;">{homa_interp}</span><br/>
                        <span style="font-size:10px; color:#666;">
                            Normal: &lt; 2.5 &nbsp;|&nbsp; Resistance: &ge; 2.5
                        </span>
                    </td>
                </tr>
                <tr>
                    <td style="padding:10px 12px; border:1px solid #ddd;
                                background-color:{quicki_bg}; vertical-align:top; line-height:1.5;">
                        <b>QUICKI</b><br/>
                        <span style="font-size:10px; color:#666;">
                            1 / (log(Insulin) + log(Glucose))
                        </span>
                    </td>
                    <td style="padding:10px 12px; border:1px solid #ddd;
                                background-color:{quicki_bg}; text-align:center;
                                font-weight:bold; font-size:18px; color:{quicki_color};
                                vertical-align:middle; white-space:nowrap;">
                        {quicki if quicki else 'N/A'}
                    </td>
                    <td style="padding:10px 12px; border:1px solid #ddd;
                                background-color:{quicki_bg}; vertical-align:top; line-height:1.5;">
                        <span style="color:{quicki_color}; font-weight:bold;">{quicki_interp}</span><br/>
                        <span style="font-size:10px; color:#666;">
                            Normal: &gt; 0.33 &nbsp;|&nbsp; Reduced: &le; 0.33
                        </span>
                    </td>
                </tr>
            </table>
        </div>"""

    # ── Metrics table row helper
    # KEY FIX: <td> header row in metrics table too — no <th>
    def metric_row(label, value_str, metric_key, raw_value, bg="#ffffff"):
        return f"""
        <tr>
            <td style="padding:10px 12px; border:1px solid #e0e0e0;
                        background-color:{bg}; vertical-align:middle;">
                <b>{label}</b>
            </td>
            <td style="padding:10px 12px; border:1px solid #e0e0e0;
                        background-color:{bg}; text-align:center; font-weight:bold;
                        font-size:13px; vertical-align:middle; white-space:nowrap;">
                {value_str}
            </td>
            <td style="padding:10px 12px; border:1px solid #e0e0e0;
                        background-color:{bg}; vertical-align:top;">
                {get_impact_label(metric_key, raw_value,
                                  decision["feature_contributions"].get(metric_key, 0))}
            </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  body  {{ font-family: Helvetica, Arial, sans-serif; font-size:13px;
           color:#222; margin:0; padding:0; }}
  @page {{ margin: 1.5cm 1.4cm; }}
  table {{ border-collapse:collapse; width:100%; }}
  ul    {{ margin:0; padding-left:20px; }}
</style>
</head>
<body>

<!-- ══════════ HEADER ══════════ -->
<table style="width:100%; background-color:#0f4c81; margin-bottom:16px;">
  <tr>
    <td style="padding:15px 20px; vertical-align:middle;">
      <div style="color:white; font-size:20px; font-weight:bold; letter-spacing:1px;">
        AgentXHealth
      </div>
      <div style="color:#b3d4f5; font-size:10px; margin-top:3px;">
        Comprehensive AI Clinical Assessment &nbsp;|&nbsp; {today}
      </div>
    </td>
    <td style="padding:15px 20px; text-align:right; vertical-align:middle;">
      <div style="color:#b3d4f5; font-size:10px;">
        Confidential &mdash; For Patient Use Only
      </div>
    </td>
  </tr>
</table>

<div style="padding:0 20px 20px 20px;">

  <p style="color:#444; margin:0 0 14px 0; line-height:1.7; font-size:13px;">
    Dear Patient, your metabolic risk assessment has been processed by the
    AgentXHealth multi-agent clinical AI system. This report combines biochemical,
    physical, and demographic data to estimate your Type 2 Diabetes risk profile.
  </p>

  <!-- ══════════ RISK BANNER ══════════ -->
  <div style="background-color:{risk_bg}; border:2px solid {risk_border};
               border-radius:5px; padding:15px 20px; margin-bottom:18px;">
    <table style="width:100%; border:none;">
      <tr>
        <td style="width:55%; vertical-align:middle; border:none;">
          <div style="color:#666; font-size:10px; text-transform:uppercase;
                       letter-spacing:1px; margin-bottom:4px;">
            Overall Assessment Result
          </div>
          <div style="color:{risk_color}; font-size:26px; font-weight:bold;
                       letter-spacing:2px; line-height:1.2;">
            {risk_label} RISK
          </div>
          <div style="color:#888; font-size:11px; margin-top:5px;">
            Ensemble probability score: <b>{prob_pct}%</b>
          </div>
        </td>
        <td style="width:45%; vertical-align:middle; padding-left:16px; border:none;">
          <div style="font-size:10px; color:#666; margin-bottom:5px;">
            Risk Score: <b>{prob_pct}%</b>
          </div>
          <table style="width:100%; border:none; border-collapse:collapse; height:12px;">
            <tr>
              <td style="padding:0; width:{prob_pct}%; background-color:{bar_color};
                          height:12px; border-radius:3px 0 0 3px; border:none;"></td>
              <td style="padding:0; background-color:#e0e0e0; height:12px;
                          border-radius:0 3px 3px 0; border:none;"></td>
            </tr>
          </table>
          <table style="width:100%; border:none; margin-top:3px;">
            <tr>
              <td style="font-size:9px; color:#999; text-align:left;  border:none;">0%</td>
              <td style="font-size:9px; color:#999; text-align:center;border:none;">50%</td>
              <td style="font-size:9px; color:#999; text-align:right; border:none;">100%</td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </div>

  <!-- ══════════ PHYSICIAN SUMMARY ══════════ -->
  <div style="margin-bottom:16px;">
    <div style="background-color:#1a237e; color:white; padding:8px 14px;
                 border-radius:4px 4px 0 0;">
      <span style="font-size:11px; font-weight:bold; letter-spacing:0.5px;">
        PHYSICIAN'S AI SUMMARY
      </span>
    </div>
    <div style="background-color:#f5f5ff; border:1px solid #c5cae9;
                 border-top:none; border-radius:0 0 4px 4px; padding:12px 16px;">
      <p style="color:#333; line-height:1.8; margin:0; font-size:12px;">
        {narrative}
      </p>
    </div>
  </div>

  {alerts_html}
  {indices_html}

  <!-- ══════════ CLINICAL METRICS ══════════ -->
  <div style="margin-top:18px;">
    <div style="background-color:#1a237e; color:white; padding:8px 14px;
                 border-radius:4px 4px 0 0;">
      <span style="font-size:11px; font-weight:bold; letter-spacing:0.5px;">
        YOUR CLINICAL METRICS AND AI RISK IMPACT
      </span>
    </div>
    <p style="font-size:11px; color:#555; margin:5px 0 7px 0; line-height:1.5;">
      Each metric is evaluated against established clinical thresholds.
      AI Risk Impact reflects how each value contributes to your overall risk estimate.
    </p>
    <table style="width:100%; border-collapse:collapse; font-size:12px;
                  border:1px solid #c5cae9;">
      <tr>
        <td style="padding:8px 12px; border:1px solid #c5cae9; width:28%;
                    background-color:#e8eaf6; font-weight:bold; font-size:12px;
                    color:#1a237e;">Clinical Metric</td>
        <td style="padding:8px 12px; border:1px solid #c5cae9; width:18%;
                    background-color:#e8eaf6; font-weight:bold; font-size:12px;
                    color:#1a237e; text-align:center;">Your Result</td>
        <td style="padding:8px 12px; border:1px solid #c5cae9; width:54%;
                    background-color:#e8eaf6; font-weight:bold; font-size:12px;
                    color:#1a237e;">AI Risk Impact</td>
      </tr>
      {metric_row("Fasting Glucose",
                  f"{input_data.get('Glucose','N/A')}&nbsp;mg/dL",
                  "Glucose", input_data.get('Glucose', 0), "#fafafa")}
      {metric_row("Insulin",
                  f"{input_data.get('Insulin','N/A')}&nbsp;uU/mL",
                  "Insulin", input_data.get('Insulin', 0))}
      {metric_row("Body Mass Index (BMI)",
                  str(input_data.get('BMI','N/A')),
                  "BMI", input_data.get('BMI', 0), "#fafafa")}
      {metric_row("Blood Pressure",
                  f"{input_data.get('BloodPressure','N/A')}&nbsp;mm&nbsp;Hg",
                  "BloodPressure", input_data.get('BloodPressure', 0))}
      {metric_row("Age",
                  f"{input_data.get('Age','N/A')}&nbsp;yrs",
                  "Age", input_data.get('Age', 0), "#fafafa")}
      {metric_row("Pregnancies",
                  str(input_data.get('Pregnancies','0')),
                  "Pregnancies", input_data.get('Pregnancies', 0))}
    </table>
  </div>

  <!-- ══════════ NEXT STEPS ══════════ -->
  <div style="margin-top:18px;">
    <div style="background-color:#1a237e; color:white; padding:8px 14px;
                 border-radius:4px 4px 0 0;">
      <span style="font-size:11px; font-weight:bold; letter-spacing:0.5px;">
        RECOMMENDED NEXT STEPS
      </span>
    </div>
    <div style="background-color:#fafafa; border:1px solid #e0e0e0;
                 border-top:none; border-radius:0 0 4px 4px; padding:12px 16px;">
      <ul style="color:#333; padding-left:18px; margin:0; line-height:1.9;">
        {action_html}
      </ul>
    </div>
  </div>

  <!-- ══════════ ABOUT ══════════ -->
  <div style="margin-top:18px; background-color:#f3f4ff; border:1px solid #c5cae9;
               border-radius:4px; padding:12px 16px;">
    <p style="margin:0 0 5px 0; font-size:11px; font-weight:bold; color:#1a237e;">
      About This Report
    </p>
    <p style="margin:0; font-size:10px; color:#555; line-height:1.7;">
      Generated by AgentXHealth — a multi-agent AI framework using three physiologically
      specialized models: <b>Laboratory Agent</b> (biochemical markers, AUC 0.82),
      <b>Physical Agent</b> (anthropometric indicators), and <b>Demographic Agent</b>
      (age and parity). Outputs are combined via weighted probabilistic fusion
      (Lab &times;0.5, Physical &times;0.3, Demographic &times;0.2) and refined
      through clinical rule adjustment for edge-case safety.
    </p>
  </div>

  <!-- ══════════ DISCLAIMER ══════════ -->
  <div style="margin-top:14px; border-top:1px solid #e0e0e0; padding-top:10px;">
    <p style="font-size:9px; color:#aaa; text-align:center; line-height:1.7; margin:0;">
      <b>Disclaimer:</b> AgentXHealth is an investigational AI clinical decision support
      tool. This report does not constitute a formal medical diagnosis and is not a
      substitute for professional medical advice, diagnosis, or treatment. Always consult
      a qualified healthcare provider before making any medical decisions.
      &nbsp;|&nbsp; Report ID: AXH-{report_id}
    </p>
  </div>

</div>
</body>
</html>"""
    return html


# =========================================================
# PDF GENERATOR
# =========================================================
def create_pdf(html_content):
    pdf_buffer  = BytesIO()
    pisa_status = pisa.CreatePDF(
        src=html_content, dest=pdf_buffer, encoding="utf-8"
    )
    pdf_data = pdf_buffer.getvalue()
    if pisa_status.err or not pdf_data:
        print("PDF generation failed or produced empty file.")
        return None
    print(f"PDF generated successfully ({len(pdf_data)} bytes).")
    return pdf_data


# =========================================================
# EMAIL DISPATCHER
# =========================================================
def send_email_with_pdf(recipient_email, pdf_bytes, risk_label):
    print(f"\nAttempting to send email to: {recipient_email}")
    print(f"Sender   : {SENDER_EMAIL}")
    print(f"PDF size : {len(pdf_bytes) if pdf_bytes else 'None'} bytes")

    msg            = MIMEMultipart()
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = recipient_email
    msg["Subject"] = f"AgentXHealth Clinical Report — {risk_label} RISK Assessment"

    body = (
        "Dear Patient,\n\n"
        "Your metabolic risk assessment is complete. "
        "Please find your personalised clinical report attached as a PDF.\n\n"
        "This report includes:\n"
        "  - Overall diabetes risk classification with probability score\n"
        "  - Physician AI narrative summary\n"
        "  - Computed metabolic indices (HOMA-IR, QUICKI)\n"
        "  - Per-metric clinical impact analysis\n"
        "  - Urgent clinical alerts (if applicable)\n"
        "  - Personalised recommended next steps\n\n"
        "IMPORTANT: This is an automated AI assessment tool. "
        "Please consult your physician for formal medical advice.\n\n"
        "AgentXHealth Team"
    )
    msg.attach(MIMEText(body, "plain"))

    if pdf_bytes:
        attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
        attachment.add_header(
            "Content-Disposition", "attachment",
            filename="AgentXHealth_Clinical_Report.pdf"
        )
        msg.attach(attachment)
        print("PDF attached to email.")

    try:
        print("Connecting to smtp.gmail.com:587 ...")
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully to {recipient_email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"Auth failed. Generate App Password at: "
              f"https://myaccount.google.com/apppasswords\n{e}")
        return False
    except Exception as e:
        print(f"Email error: {e}")
        return False


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_model_for_input(input_data):
    _load_agents()

    # Step 1: Guardrail
    sanity_errors = run_sanity_checks(input_data)
    if sanity_errors:
        for err in sanity_errors:
            print(f"Guardrail blocked: {err}")
        return {
            "error_state":    True,
            "details":        sanity_errors,
            "final_risk":     -1.0,
            "dominant_agent": "Guardrail Agent",
            "explanation":    "Data Rejected: " + " | ".join(sanity_errors),
        }

    # Step 2: Validation
    validation = validate_inputs(input_data)
    if validation and validation.get("errors"):
        validation["final_risk"]     = -1.0
        validation["dominant_agent"] = "Input Validator"
        validation["explanation"]    = "Format Error: Invalid input types."
        return validation

    # Step 3: Build feature row
    row = {
        "Pregnancies":              input_data.get("Pregnancies", 0),
        "Glucose":                  input_data["Glucose"],
        "BloodPressure":            input_data["BloodPressure"],
        "SkinThickness":            0,
        "Insulin":                  input_data["Insulin"],
        "BMI":                      input_data["BMI"],
        "DiabetesPedigreeFunction": 0.5,
        "Age":                      input_data["Age"],
        "Outcome":                  0,
    }
    df = pd.DataFrame([row])

    # Step 4: Agent predictions
    raw_outputs = {
        "lab":         lab.predict(df),
        "physical":    physical.predict(df),
        "demographic": demo.predict(df),
    }
    try:
        outputs = {k: extract_prob(v) for k, v in raw_outputs.items()}
    except ValueError as e:
        return {
            "error_state":    True,
            "details":        [str(e)],
            "final_risk":     -1.0,
            "dominant_agent": "Agent Extractor",
            "explanation":    f"Agent output parse error: {e}",
        }

    print(f"Agent outputs -> Lab: {outputs['lab']:.4f} | "
          f"Physical: {outputs['physical']:.4f} | "
          f"Demographic: {outputs['demographic']:.4f}")

    # Step 5: Coordinator fusion
    decision = coord.reason(outputs)
    print(f"Fused base risk : {decision['final_risk']:.4f} | "
          f"Dominant agent  : {decision.get('dominant_agent','N/A')}")

    # Step 6: Clinical logic + boosts
    final_risk, contributions, clinical_warnings = apply_clinical_logic(
        decision["final_risk"], input_data
    )
    print(f"Final risk after clinical logic: {final_risk:.4f}")
    print(f"Feature contributions: "
          f"{ {k: round(v,4) for k,v in contributions.items()} }")
    for w in clinical_warnings:
        print(f"Clinical warning: {w}")

    decision["final_risk"]            = final_risk
    decision["feature_contributions"] = contributions
    decision["clinical_warnings"]     = clinical_warnings
    decision["risk_label"]            = get_label(final_risk)
    decision["confidence"]            = get_confidence(final_risk)
    print(f"Risk label: {decision['risk_label']} | "
          f"Confidence: {decision['confidence']}")

    # Step 7: HTML + PDF
    html_report = generate_email_report(input_data, decision)
    print("HTML report generated.")
    pdf_bytes = create_pdf(html_report)

    # Step 8: Email
    recipient_email = input_data.get("Email")
    if not recipient_email:
        print("No Email in input — skipping email.")
    elif APP_PASSWORD in ("your_16_char_app_password_here", "****", "", None):
        print("APP_PASSWORD not set — skipping email.")
        print("  Generate at: https://myaccount.google.com/apppasswords")
    else:
        send_email_with_pdf(recipient_email, pdf_bytes, decision["risk_label"])

    # Step 9: Local backup
    if pdf_bytes:
        with open("AgentXHealth_Latest_Report.pdf", "wb") as f:
            f.write(pdf_bytes)
        print("Backup PDF saved: 'AgentXHealth_Latest_Report.pdf'")

    # Step 10: Write-back keys for Google Sheets
    decision["email_report"]  = html_report
    decision["explanation"]   = f"Report Generated ({decision['risk_label']})"
    decision.setdefault("dominant_agent", "Coordinator Reasoner")
    return decision
