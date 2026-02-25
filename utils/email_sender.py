import smtplib
from email.mime.text import MIMEText

SENDER_EMAIL = "akp11172000@gmail.com"
APP_PASSWORD = "xnoefzdynmtpkbbx"



def send_email(receiver_email, message):

    msg = MIMEText(message)
    msg["Subject"] = "AgentXHealth – Diabetes Risk Report"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        server.quit()
        return True

    except Exception as e:
        print("Email sending failed:", e)
        return False
