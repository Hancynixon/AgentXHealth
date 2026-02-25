from production.model_runner import run_model_for_input

print("\n==============================")
print("ADVANCED 50 EDGE CASE STRESS TEST")
print("==============================\n")

cases = [

# ----------------------------
# 1–10: Threshold Edge Flips
# ----------------------------
{"Gender":"Male","Email":"T1","Glucose":125,"Insulin":85,"BMI":24,"BloodPressure":119,"Age":34,"Pregnancies":0},
{"Gender":"Male","Email":"T2","Glucose":126,"Insulin":85,"BMI":24,"BloodPressure":119,"Age":34,"Pregnancies":0},
{"Gender":"Male","Email":"T3","Glucose":149,"Insulin":110,"BMI":29,"BloodPressure":129,"Age":39,"Pregnancies":0},
{"Gender":"Male","Email":"T4","Glucose":150,"Insulin":110,"BMI":29,"BloodPressure":129,"Age":39,"Pregnancies":0},
{"Gender":"Male","Email":"T5","Glucose":179,"Insulin":140,"BMI":32,"BloodPressure":149,"Age":47,"Pregnancies":0},
{"Gender":"Male","Email":"T6","Glucose":180,"Insulin":140,"BMI":32,"BloodPressure":149,"Age":47,"Pregnancies":0},
{"Gender":"Female","Email":"T7","Glucose":199,"Insulin":190,"BMI":34,"BloodPressure":169,"Age":54,"Pregnancies":3},
{"Gender":"Female","Email":"T8","Glucose":200,"Insulin":190,"BMI":34,"BloodPressure":169,"Age":54,"Pregnancies":3},
{"Gender":"Female","Email":"T9","Glucose":249,"Insulin":250,"BMI":36,"BloodPressure":179,"Age":60,"Pregnancies":4},
{"Gender":"Female","Email":"T10","Glucose":250,"Insulin":250,"BMI":36,"BloodPressure":179,"Age":60,"Pregnancies":4},

# ----------------------------
# 11–20: Contradictory Signals
# ----------------------------
{"Gender":"Male","Email":"C1","Glucose":95,"Insulin":350,"BMI":22,"BloodPressure":110,"Age":28,"Pregnancies":0},
{"Gender":"Female","Email":"C2","Glucose":210,"Insulin":20,"BMI":21,"BloodPressure":115,"Age":29,"Pregnancies":1},
{"Gender":"Male","Email":"C3","Glucose":88,"Insulin":400,"BMI":35,"BloodPressure":140,"Age":35,"Pregnancies":0},
{"Gender":"Female","Email":"C4","Glucose":160,"Insulin":15,"BMI":45,"BloodPressure":160,"Age":52,"Pregnancies":5},
{"Gender":"Male","Email":"C5","Glucose":105,"Insulin":10,"BMI":42,"BloodPressure":155,"Age":40,"Pregnancies":0},
{"Gender":"Female","Email":"C6","Glucose":180,"Insulin":500,"BMI":18,"BloodPressure":100,"Age":24,"Pregnancies":0},
{"Gender":"Male","Email":"C7","Glucose":140,"Insulin":18,"BMI":17,"BloodPressure":95,"Age":22,"Pregnancies":0},
{"Gender":"Female","Email":"C8","Glucose":130,"Insulin":300,"BMI":19,"BloodPressure":105,"Age":25,"Pregnancies":1},
{"Gender":"Male","Email":"C9","Glucose":170,"Insulin":50,"BMI":20,"BloodPressure":110,"Age":30,"Pregnancies":0},
{"Gender":"Female","Email":"C10","Glucose":155,"Insulin":200,"BMI":50,"BloodPressure":130,"Age":35,"Pregnancies":2},

# ----------------------------
# 21–30: Elderly Healthy vs Young Severe
# ----------------------------
{"Gender":"Male","Email":"E1","Glucose":90,"Insulin":20,"BMI":23,"BloodPressure":115,"Age":85,"Pregnancies":0},
{"Gender":"Female","Email":"E2","Glucose":95,"Insulin":18,"BMI":22,"BloodPressure":110,"Age":82,"Pregnancies":0},
{"Gender":"Male","Email":"E3","Glucose":200,"Insulin":200,"BMI":24,"BloodPressure":120,"Age":18,"Pregnancies":0},
{"Gender":"Female","Email":"E4","Glucose":220,"Insulin":220,"BMI":25,"BloodPressure":118,"Age":19,"Pregnancies":0},
{"Gender":"Male","Email":"E5","Glucose":250,"Insulin":250,"BMI":26,"BloodPressure":122,"Age":20,"Pregnancies":0},
{"Gender":"Female","Email":"E6","Glucose":100,"Insulin":15,"BMI":19,"BloodPressure":105,"Age":90,"Pregnancies":0},
{"Gender":"Male","Email":"E7","Glucose":300,"Insulin":400,"BMI":30,"BloodPressure":130,"Age":21,"Pregnancies":0},
{"Gender":"Female","Email":"E8","Glucose":85,"Insulin":10,"BMI":18,"BloodPressure":100,"Age":88,"Pregnancies":0},
{"Gender":"Male","Email":"E9","Glucose":160,"Insulin":100,"BMI":28,"BloodPressure":135,"Age":17,"Pregnancies":0},
{"Gender":"Female","Email":"E10","Glucose":140,"Insulin":80,"BMI":27,"BloodPressure":125,"Age":92,"Pregnancies":0},

# ----------------------------
# 31–40: Boundary Valid Inputs
# ----------------------------
{"Gender":"Male","Email":"B1","Glucose":1,"Insulin":0,"BMI":10,"BloodPressure":40,"Age":30,"Pregnancies":0},
{"Gender":"Female","Email":"B2","Glucose":2,"Insulin":1,"BMI":10.1,"BloodPressure":41,"Age":31,"Pregnancies":0},
{"Gender":"Male","Email":"B3","Glucose":300,"Insulin":0,"BMI":79,"BloodPressure":299,"Age":50,"Pregnancies":0},
{"Gender":"Female","Email":"B4","Glucose":299,"Insulin":1,"BMI":78,"BloodPressure":298,"Age":49,"Pregnancies":2},
{"Gender":"Male","Email":"B5","Glucose":50,"Insulin":0,"BMI":15,"BloodPressure":45,"Age":18,"Pregnancies":0},
{"Gender":"Female","Email":"B6","Glucose":70,"Insulin":5,"BMI":18.5,"BloodPressure":60,"Age":20,"Pregnancies":0},
{"Gender":"Male","Email":"B7","Glucose":100,"Insulin":50,"BMI":30,"BloodPressure":80,"Age":40,"Pregnancies":0},
{"Gender":"Female","Email":"B8","Glucose":120,"Insulin":60,"BMI":29.9,"BloodPressure":89,"Age":45,"Pregnancies":1},
{"Gender":"Male","Email":"B9","Glucose":130,"Insulin":70,"BMI":30.1,"BloodPressure":90,"Age":50,"Pregnancies":0},
{"Gender":"Female","Email":"B10","Glucose":140,"Insulin":80,"BMI":25,"BloodPressure":100,"Age":55,"Pregnancies":3},

# ----------------------------
# 41–50: Balanced Ambiguous Cases
# ----------------------------
{"Gender":"Male","Email":"A1","Glucose":135,"Insulin":100,"BMI":28,"BloodPressure":125,"Age":42,"Pregnancies":0},
{"Gender":"Female","Email":"A2","Glucose":145,"Insulin":120,"BMI":30,"BloodPressure":130,"Age":45,"Pregnancies":2},
{"Gender":"Male","Email":"A3","Glucose":155,"Insulin":140,"BMI":32,"BloodPressure":135,"Age":48,"Pregnancies":0},
{"Gender":"Female","Email":"A4","Glucose":165,"Insulin":160,"BMI":34,"BloodPressure":140,"Age":50,"Pregnancies":3},
{"Gender":"Male","Email":"A5","Glucose":175,"Insulin":180,"BMI":36,"BloodPressure":145,"Age":52,"Pregnancies":0},
{"Gender":"Female","Email":"A6","Glucose":185,"Insulin":200,"BMI":38,"BloodPressure":150,"Age":55,"Pregnancies":4},
{"Gender":"Male","Email":"A7","Glucose":195,"Insulin":220,"BMI":40,"BloodPressure":155,"Age":58,"Pregnancies":0},
{"Gender":"Female","Email":"A8","Glucose":205,"Insulin":240,"BMI":42,"BloodPressure":160,"Age":60,"Pregnancies":5},
{"Gender":"Male","Email":"A9","Glucose":215,"Insulin":260,"BMI":44,"BloodPressure":165,"Age":62,"Pregnancies":0},
{"Gender":"Female","Email":"A10","Glucose":225,"Insulin":280,"BMI":46,"BloodPressure":170,"Age":65,"Pregnancies":6},
]

for idx, case in enumerate(cases, start=1):

    print(f"\n--- Case {idx} ---")
    print("Input:", case)

    try:
        result = run_model_for_input(case)

        if "error" in result:
            print("❌ INVALID INPUT")
            print("Errors:", result["error"])
            print("Alerts:", result["alerts"])
            continue

        print("Final Risk:", result["final_risk"])
        print("Risk Label:", result["risk_label"])
        print("Dominant Agent:", result["dominant_agent"])
        print("Confidence:", result["confidence"])

        if result.get("alerts"):
            print("Alerts:", result["alerts"])

    except Exception as e:
        print("❌ ERROR:", str(e))

print("\n==============================")
print("ADVANCED STRESS TEST COMPLETE")
print("==============================")
