import requests


# -------------------------------------------------
# PASTE YOUR WEBHOOK URL HERE
# -------------------------------------------------
MAKE_WEBHOOK_URL = "https://hook.eu1.make.com/w6wh8m5egjc7fi84it0bo5hd9lz373y6"


def send_to_make(result, email):
    """
    Sends prediction result to Make.com
    """

    payload = {
        "email": email,
        "final_risk": result["final_risk"],
        "dominant_agent": result["dominant_agent"],
        "explanation": result["decision_explanation"]
    }

    try:
        r = requests.post(MAKE_WEBHOOK_URL, json=payload)
        print("📤 Sent to Make:", r.status_code)

    except Exception as e:
        print("❌ Webhook failed:", e)
