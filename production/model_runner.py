import pandas as pd

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner
from utils.validator import validate_inputs


# =========================================================
# GLOBAL THRESHOLDS (FREEZE HERE)
# =========================================================

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.60


# =========================================================
# LOAD TRAINING DATA (ONCE)
# =========================================================

TRAIN_DF = pd.read_csv("data/raw/diabetes.csv")
Y = TRAIN_DF["Outcome"]

lab = LabAgentIntelligent()
lab.fit(TRAIN_DF, Y)

physical = PhysicalAgentIntelligent()
physical.fit(TRAIN_DF, Y)

demo = DemographicAgentIntelligent()
demo.fit(TRAIN_DF, Y)

coord = CoordinatorReasoner()


# =========================================================
# RISK LABEL FUNCTION
# =========================================================

def get_risk_label(prob):

    if prob < LOW_THRESHOLD:
        return "LOW"
    elif prob < HIGH_THRESHOLD:
        return "MODERATE"
    else:
        return "HIGH"


# =========================================================
# IMPROVED CONFIDENCE FUNCTION
# Based on distance from decision boundary (0.5)
# =========================================================

def get_confidence(prob):

    distance = abs(prob - 0.5)

    if prob >= 0.85 or prob <= 0.15:
        return "Very High"

    elif distance >= 0.30:
        return "High"

    elif distance >= 0.20:
        return "Moderate"

    else:
        return "Low"


# =========================================================
# CLINICAL BOOST LOGIC (FROZEN)
# =========================================================

def apply_clinical_boost(final_risk, glucose, insulin, bmi, bp, age):

    # Glucose staged boost
    if glucose >= 250:
        final_risk = min(0.97, final_risk + 0.25)
    elif glucose >= 200:
        final_risk = min(0.93, final_risk + 0.20)
    elif glucose >= 180:
        final_risk = min(0.88, final_risk + 0.15)
    elif glucose >= 150:
        final_risk = min(0.82, final_risk + 0.10)

    # Insulin boost
    if insulin >= 300:
        final_risk = min(0.90, final_risk + 0.10)
    elif insulin >= 200:
        final_risk = min(0.85, final_risk + 0.07)

    # Morbid obesity
    if bmi >= 40:
        final_risk = min(0.92, final_risk + 0.10)

    # Hypertensive crisis
    if bp >= 180:
        final_risk = min(0.92, final_risk + 0.10)

    # Elderly + high glucose
    if age >= 65 and glucose >= 180:
        final_risk = min(0.95, final_risk + 0.10)

    return round(final_risk, 4)


# =========================================================
# PATIENT FRIENDLY EXPLANATION GENERATOR
# =========================================================

def generate_patient_explanation(input_data, decision):

    prob = decision["final_risk"]
    label = decision["risk_label"]

    glucose = input_data["Glucose"]
    bmi = input_data["BMI"]
    age = input_data["Age"]

    explanation = f"""
Hello,

Your diabetes risk assessment is ready.

Final Risk Score: {round(prob,3)} ({label} risk)
Confidence Level: {decision["confidence"]}

"""

    # Risk summary
    if label == "HIGH":
        explanation += "A HIGH diabetes risk has been detected.\n"
    elif label == "MODERATE":
        explanation += "A MODERATE diabetes risk has been detected.\n"
    else:
        explanation += "Your diabetes risk is currently LOW.\n"

    
        # Glucose interpretation (Clinically refined wording)
    if glucose < 70:
        explanation += f"- Your blood sugar level ({glucose} mg/dL) is lower than the normal range.\n"

    elif 70 <= glucose < 100:
        explanation += f"- Your blood sugar level ({glucose} mg/dL) is within the normal fasting range.\n"

    elif 100 <= glucose < 126:
        explanation += f"- Your blood sugar level ({glucose} mg/dL) is slightly elevated and may indicate early glycemic imbalance (prediabetes range if fasting).\n"

    elif 126 <= glucose < 200:
        explanation += f"- Your blood sugar level ({glucose} mg/dL) is elevated and requires medical evaluation to rule out diabetes.\n"

    else:  # glucose >= 200
        explanation += f"- Your blood sugar level ({glucose} mg/dL) is in the diabetic range and requires urgent clinical evaluation.\n"


    # BMI interpretation
    if bmi >= 40:
        explanation += f"- Your BMI ({bmi}) indicates severe obesity, increasing metabolic risk.\n"
    elif bmi >= 30:
        explanation += f"- Your BMI ({bmi}) indicates obesity, which increases diabetes risk.\n"
    elif bmi < 18.5:
        explanation += f"- Your BMI ({bmi}) indicates underweight condition.\n"

    # Age interpretation
    if age >= 60:
        explanation += f"- Age ({age} years) increases long-term susceptibility to diabetes.\n"

    explanation += """
It is recommended to consult a healthcare professional for personalized medical advice.
Lifestyle changes including balanced diet, regular exercise, and periodic glucose monitoring can significantly reduce long-term risk.

Regards,
AgentXHealth
"""

    return explanation.strip()


# =========================================================
# MAIN MODEL FUNCTION
# =========================================================

def run_model_for_input(input_data):

    validation = validate_inputs(input_data)

    if validation["errors"]:
        return {
            "error": validation["errors"],
            "alerts": validation["alerts"]
        }

    # Build input row compatible with training structure
    row = {
        "Pregnancies": input_data["Pregnancies"],
        "Glucose": input_data["Glucose"],
        "BloodPressure": input_data["BloodPressure"],
        "SkinThickness": 0,
        "Insulin": input_data["Insulin"],
        "BMI": input_data["BMI"],
        "DiabetesPedigreeFunction": 0.5,
        "Age": input_data["Age"],
        "Outcome": 0
    }

    df_input = pd.DataFrame([row])

    outputs = {
        "lab": lab.predict(df_input),
        "physical": physical.predict(df_input),
        "demographic": demo.predict(df_input)
    }

    decision = coord.reason(outputs)

    # Apply clinical boost
    boosted_risk = apply_clinical_boost(
        decision["final_risk"],
        input_data["Glucose"],
        input_data["Insulin"],
        input_data["BMI"],
        input_data["BloodPressure"],
        input_data["Age"]
    )

    label = get_risk_label(boosted_risk)
    confidence = get_confidence(boosted_risk)

    decision["final_risk"] = boosted_risk
    decision["risk_label"] = label
    decision["confidence"] = confidence
    decision["alerts"] = validation["alerts"]
    decision["explanation"] = generate_patient_explanation(input_data, decision)

    return decision
