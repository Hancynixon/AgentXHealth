class PatientPreprocessor:
    """
    Cleans + validates form inputs before model inference.
    """

    def clean(self, row: dict):

        patient = row.copy()

        gender = str(patient.get("Gender", "")).lower()

        # ---------------------------------
        # Clinical rule
        # ---------------------------------
        if gender == "male":
            patient["Pregnancies"] = 0

        # ---------------------------------
        # Safe numeric casting
        # ---------------------------------
        def to_float(v, default=0):
            try:
                return float(v)
            except:
                return default

        def to_int(v, default=0):
            try:
                return int(float(v))
            except:
                return default

        patient["Glucose"] = to_float(patient.get("Glucose"))
        patient["Insulin"] = to_float(patient.get("Insulin"))
        patient["BMI"] = to_float(patient.get("BMI"))
        patient["BloodPressure"] = to_float(patient.get("BloodPressure"))
        patient["Age"] = to_int(patient.get("Age"))
        patient["Pregnancies"] = to_int(patient.get("Pregnancies"))

        return patient
