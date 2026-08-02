"""
Central configuration for the IDS.
All secrets/tunables are read from environment variables (see .env.example)
so nothing sensitive needs to be hard-coded or committed to git.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Flask / JWT ---
    JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "30"))

    # Demo credentials (swap for a real user store / DB in production)
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

    # --- Monitoring thresholds ---
    CPU_THRESHOLD_PERCENT = float(os.getenv("CPU_THRESHOLD_PERCENT", "80"))
    MEMORY_THRESHOLD_PERCENT = float(os.getenv("MEMORY_THRESHOLD_PERCENT", "75"))
    MONITOR_INTERVAL_SECONDS = float(os.getenv("MONITOR_INTERVAL_SECONDS", "5"))

    # --- RSA digital signatures ---
    RSA_PRIVATE_KEY_PATH = os.getenv("RSA_PRIVATE_KEY_PATH", "keys/private_key.pem")
    RSA_PUBLIC_KEY_PATH = os.getenv("RSA_PUBLIC_KEY_PATH", "keys/public_key.pem")

    # --- Email alerts (SMTP with app password) ---
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
    SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL", "")
    SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "")
    ALERT_RECIPIENT_EMAIL = os.getenv("ALERT_RECIPIENT_EMAIL", "")
    EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true"

    # --- Profile database ---
    PROFILE_DB_PATH = os.getenv("PROFILE_DB_PATH", "database/profiles.json")
    ANOMALY_LOG_PATH = os.getenv("ANOMALY_LOG_PATH", "database/anomaly_log.json")


config = Config()
