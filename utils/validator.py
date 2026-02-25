def validate_inputs(input_data):

    errors = []
    alerts = []

    glucose = float(input_data.get("Glucose", 0))
    insulin = float(input_data.get("Insulin", 0))
    bmi = float(input_data.get("BMI", 0))
    bp = float(input_data.get("BloodPressure", 0))
    age = float(input_data.get("Age", 0))

    # =========================================================
    # HARD VALIDATION RULES (STRICT CLINICAL LIMITS)
    # =========================================================

    # Glucose strict lower bound
    if glucose < 40:
        errors.append("Invalid glucose value (must be ≥ 40 mg/dL)")

    # Insulin
    if insulin < 0:
        errors.append("Invalid insulin value (must be ≥ 0)")

    # BMI
    if bmi < 10 or bmi > 80:
        errors.append("Invalid BMI value (10–80 allowed)")

    # Blood Pressure
    if bp < 40 or bp > 300:
        errors.append("Invalid blood pressure value (40–300 allowed)")

    # Age
    if age <= 0 or age > 120:
        errors.append("Invalid age value")

    # =========================================================
    # SOFT ALERTS (CLINICAL WARNINGS – DO NOT BLOCK)
    # =========================================================

    # Low glucose warning
    if 40 <= glucose < 70:
        alerts.append("Low glucose detected")

    # Severe hyperglycemia
    if glucose >= 250:
        alerts.append("Severe hyperglycemia detected")

    # Morbid obesity
    if bmi >= 40:
        alerts.append("Morbid obesity")

    # Underweight
    if bmi < 18.5:
        alerts.append("Underweight (malnutrition risk)")

    # Hypertensive crisis
    if bp >= 180:
        alerts.append("Hypertensive crisis")

    return {
        "errors": errors,
        "alerts": alerts
    }
