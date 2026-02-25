import os
import sys
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# ==============================
# Fix import path
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from production.model_runner import run_model_for_input
from utils.validator import validate_inputs
from utils.report_generator import generate_explanation
from utils.email_sender import send_email

# ==============================
# GOOGLE SHEET CONFIG
# ==============================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_NAME = "AgentXHealth_Form"
KEY_PATH = os.path.join(BASE_DIR, "credentials.json")


# ==============================
# CONNECT TO GOOGLE SHEET
# ==============================

def connect_sheet():
    creds = Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1
    return sheet


# ==============================
# MAIN NIGHT BATCH PROCESS
# ==============================

def run_batch():

    print("\nConnecting to Google Sheet...")
    sheet = connect_sheet()

    records = sheet.get_all_records()

    if not records:
        print("No records found.")
        return

    print(f"Total rows found: {len(records)}")

    for index, row in enumerate(records):

        final_risk_value = row.get("Final_Risk")

        # Skip already processed rows
        if final_risk_value not in [None, "", " "]:
            continue

        print(f"\nProcessing row {index+2} | Email: {row.get('Email')}")

        try:

            # ============================
            # Clean + Validate Inputs
            # ============================

            input_data = {
                "Gender": row["Gender"],
                "Email": row["Email"],
                "Glucose": float(row["Glucose"]),
                "Insulin": float(row["Insulin"]),
                "BMI": float(row["BMI"]),
                "BloodPressure": float(row["BloodPressure"]),
                "Age": float(row["Age"]),
                "Pregnancies": float(row["Pregnancies"])
            }

            # Ignore pregnancies if Male
            if input_data["Gender"].lower() == "male":
                input_data["Pregnancies"] = 0

            alerts = validate_inputs(input_data)

            # ============================
            # Run Model
            # ============================

            result = run_model_for_input(input_data)

            explanation = result["explanation"]

            # ============================
            # Update Google Sheet
            # ============================

            sheet.update_cell(index + 2, 10, round(result["final_risk"], 4))
            sheet.update_cell(index + 2, 11, result["dominant_agent"])
            sheet.update_cell(index + 2, 12, explanation)

            print("Sheet updated.")

            # ============================
            # Send Email
            # ============================

            success = send_email(input_data["Email"], explanation)

            if success:
                print("Email sent successfully.")
            else:
                print("Email failed.")


        except Exception as e:
            print(f"Error processing row {index+2}:", e)

    print("\nNight batch completed.")


if __name__ == "__main__":
    run_batch()
