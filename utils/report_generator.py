def generate_explanation(row, result):

    if result["final_risk"] is None:
        return f"Prediction not performed. Errors: {result['error']}"

    risk = result["final_risk"]
    risk_percent = round(risk * 100, 1)

    # Final calibrated threshold
    if risk < 0.35:
        risk_level = "LOW"
    elif risk < 0.60:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    explanation = (
        f"You are at {risk_level} diabetes risk.\n"
        f"Risk Score: {risk_percent}%\n"
        f"Dominant factor: {result['dominant_agent']}.\n"
    )

    if result.get("alerts"):
        explanation += "\n⚠ Clinical alerts:\n"
        for alert in result["alerts"]:
            explanation += f"- {alert}\n"

    explanation += (
        f"\nGlucose = {row['Glucose']} mg/dL\n"
        f"Insulin = {row['Insulin']} µU/mL\n"
        f"BMI = {row['BMI']}\n"
        f"Blood Pressure = {row['BloodPressure']}\n"
        f"Age = {row['Age']}"
    )

    return explanation
