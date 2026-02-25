from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from agents.lab_agent_intelligent import LabAgentIntelligent
from agents.physical_agent_intelligent import PhysicalAgentIntelligent
from agents.demographic_agent_intelligent import DemographicAgentIntelligent
from coordinator.coordinator_reasoner import CoordinatorReasoner

app = FastAPI()

df = pd.read_csv("data/raw/diabetes.csv")

lab = LabAgentIntelligent()
lab.fit(df, df["Outcome"])

physical = PhysicalAgentIntelligent()
physical.fit(df, df["Outcome"])

demo = DemographicAgentIntelligent()
demo.fit(df, df["Outcome"])

reasoner = CoordinatorReasoner()


class PatientInput(BaseModel):
    Gender: str
    Email: str
    Glucose: float
    Insulin: float
    BMI: float
    BloodPressure: float
    Age: int
    Pregnancies: int | None = 0


@app.get("/")
def root():
    return {"status": "AgentXHealth API Running"}


@app.post("/predict")
def predict_risk(patient: PatientInput):

    pregnancies = patient.Pregnancies
    if patient.Gender.lower() == "male":
        pregnancies = 0

    sample = pd.DataFrame([{
        "Glucose": patient.Glucose,
        "Insulin": patient.Insulin,
        "BMI": patient.BMI,
        "BloodPressure": patient.BloodPressure,
        "Age": patient.Age,
        "Pregnancies": pregnancies
    }])

    lab_out = lab.predict(sample)
    phys_out = physical.predict(sample)
    demo_out = demo.predict(sample)

    decision = reasoner.reason(
        {
            "lab": lab_out,
            "physical": phys_out,
            "demographic": demo_out
        },
        sample.iloc[0].to_dict()
    )

    return decision
