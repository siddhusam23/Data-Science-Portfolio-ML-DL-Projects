"""
SystemSentinelAgent — continuously monitors system metrics (CPU, memory),
flags anomalies against configured thresholds, digitally signs alerts with
RSA, and dispatches email notifications.

Runs on its own daemon thread so monitoring never blocks the Flask app.
"""

import threading
import time
from datetime import datetime, timezone

import psutil

from alerts.email_alert import send_email_alert
from config import config
from database.profile_db import profile_db
from security.digital_signature import sign_message


class SystemSentinelAgent(threading.Thread):
    def __init__(self, fault_evaluation_agent=None, poll_interval: float = None):
        super().__init__(daemon=True, name="SystemSentinelAgent")
        self.fault_evaluation_agent = fault_evaluation_agent
        self.poll_interval = poll_interval or config.MONITOR_INTERVAL_SECONDS
        self._stop_event = threading.Event()
        self.latest_metrics = {"cpu": 0.0, "memory": 0.0, "anomaly": None}

    def stop(self):
        self._stop_event.set()

    def collect_metrics(self) -> dict:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        return {
            "cpu": cpu,
            "memory": memory,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def detect_anomaly(self, metrics: dict):
        reasons = []
        if metrics["cpu"] > config.CPU_THRESHOLD_PERCENT:
            reasons.append(f"CPU usage {metrics['cpu']}% exceeded threshold "
                            f"{config.CPU_THRESHOLD_PERCENT}%")
        if metrics["memory"] > config.MEMORY_THRESHOLD_PERCENT:
            reasons.append(f"Memory usage {metrics['memory']}% exceeded threshold "
                            f"{config.MEMORY_THRESHOLD_PERCENT}%")
        return reasons or None

    def handle_anomaly(self, metrics: dict, reasons: list) -> dict:
        alert_text = (
            f"Potential threat detected at {metrics['timestamp']}. "
            f"CPU={metrics['cpu']}%, Memory={metrics['memory']}%. "
            f"Reasons: {'; '.join(reasons)}"
        )
        signature = sign_message(alert_text)

        anomaly = {
            "message": alert_text,
            "metrics": metrics,
            "reasons": reasons,
            "signature": signature,
        }
        logged = profile_db.log_anomaly(anomaly)

        send_email_alert(subject="[IDS] Anomaly Detected", body=alert_text)

        if self.fault_evaluation_agent:
            self.fault_evaluation_agent.evaluate(logged)

        return logged

    def run(self):
        while not self._stop_event.is_set():
            metrics = self.collect_metrics()
            profile_db.record_metrics(metrics)

            reasons = self.detect_anomaly(metrics)
            anomaly = self.handle_anomaly(metrics, reasons) if reasons else None

            self.latest_metrics = {
                "cpu": metrics["cpu"],
                "memory": metrics["memory"],
                "timestamp": metrics["timestamp"],
                "anomaly": anomaly["message"] if anomaly else None,
            }

            time.sleep(self.poll_interval)
