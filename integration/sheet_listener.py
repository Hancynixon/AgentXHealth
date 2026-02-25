from integration.send_to_make import send_to_make
import gspread
import requests
import time
from oauth2client.service_account import ServiceAccountCredentials
from integration.preprocessing import PatientPreprocessor


print("🔥 AgentXHealth Batch Sheet Processor Started")

# ==================================================
# GOOGLE AUTH
# ==================================================
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    r"C:\Users\srira\AgentXHealth\api\credentials.json",
    scope
)

client = gspread.authorize(creds)
sheet = client.open("AgentXHealth_Form").sheet1

print("✅ Connected to Google Sheet")


API_URL = "http://127.0.0.1:8000/predict"
pre = PatientPreprocessor()


# ==================================================
# PROCESS ONCE (BATCH STYLE)
# ==================================================

def process_batch():

    all_values = sheet.get_all_values()

    if len(all_values) < 2:
        print("No data found.")
        return

    headers = all_values[0]
    rows = all_values[1:]

    risk_col = headers.index("Final_Risk")
    agent_col = headers.index("Dominant_Agent")
    expl_col = headers.index("Explanation")

    total_sent = 0

    for idx, row_values in enumerate(rows):

        row_dict = dict(zip(headers, row_values))

        # ==========================================
        # 🚨 KEY LOGIC (skip already processed rows)
        # ==========================================
        if row_dict.get("Final_Risk"):
            continue

        try:
            patient = pre.clean(row_dict)

            payload = {
                "Gender": patient.get("Gender"),
                "Email": patient.get("Email"),
                "Glucose": patient["Glucose"],
                "Insulin": patient["Insulin"],
                "BMI": patient["BMI"],
                "BloodPressure": patient["BloodPressure"],
                "Age": patient["Age"],
                "Pregnancies": patient["Pregnancies"]
            }

            r = requests.post(API_URL, json=payload)

            if r.status_code != 200:
                print("⚠ API Error:", r.text)
                continue

            result = r.json()

            print("📊 Processing:", patient["Email"])

            row_number = idx + 2

            # ==========================================
            # Update sheet
            # ==========================================
            sheet.update(
                values=[[
                    result["final_risk"],
                    result["dominant_agent"],
                    result["decision_explanation"]
                ]],
                range_name=f"{chr(65+risk_col)}{row_number}:{chr(65+expl_col)}{row_number}"
            )

            # ==========================================
            # Send email
            # ==========================================
            send_to_make(result, patient.get("Email"))

            total_sent += 1

        except Exception as e:
            print("⚠ Row error:", e)

    print(f"\n✅ Batch finished. Emails sent: {total_sent}")


# ==================================================
# RUN ONCE AND EXIT (DAG STYLE)
# ==================================================
if __name__ == "__main__":
    process_batch()
