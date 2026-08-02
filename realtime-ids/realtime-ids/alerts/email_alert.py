"""
Secure email notifications for critical anomalies, sent over SMTP using an
app password (e.g. a Gmail App Password) rather than the account's main
credentials.
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import config


def send_email_alert(subject: str, body: str) -> bool:
    """Send an alert email. Returns True on success, False otherwise.

    No-ops (and returns False) if EMAIL_ALERTS_ENABLED is not set, so the
    project runs out of the box without requiring SMTP credentials.
    """
    if not config.EMAIL_ALERTS_ENABLED:
        print(f"[email_alert] Email alerts disabled — would have sent: {subject}")
        return False

    if not (config.SMTP_SENDER_EMAIL and config.SMTP_APP_PASSWORD and config.ALERT_RECIPIENT_EMAIL):
        print("[email_alert] Missing SMTP configuration — skipping send")
        return False

    message = MIMEMultipart()
    message["From"] = config.SMTP_SENDER_EMAIL
    message["To"] = config.ALERT_RECIPIENT_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(config.SMTP_SERVER, config.SMTP_PORT, context=context) as server:
            server.login(config.SMTP_SENDER_EMAIL, config.SMTP_APP_PASSWORD)
            server.sendmail(config.SMTP_SENDER_EMAIL, config.ALERT_RECIPIENT_EMAIL, message.as_string())
        return True
    except Exception as exc:
        print(f"[email_alert] Failed to send email: {exc}")
        return False
