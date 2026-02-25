from production.model_runner import run_model_for_input

case = {
    "Gender": "Female",
    "Email": "demo@patient.com",
    "Glucose": 180,
    "Insulin": 250,
    "BMI": 34,
    "BloodPressure": 150,
    "Age": 48,
    "Pregnancies": 3
}

result = run_model_for_input(case)

print("\n==============================")
print("LIVE DEMO PREDICTION")
print("==============================\n")

if "error" in result:
    print("Invalid input:", result["error"])
else:
    print("Final Risk:", result["final_risk"])
    print("Risk Label:", result["risk_label"])
    print("Dominant Agent:", result["dominant_agent"])
    print("Confidence:", result["confidence"])
    print("\nExplanation:\n")
    print(result["explanation"])
