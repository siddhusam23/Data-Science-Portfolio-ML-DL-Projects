"""
ProfileDatabase — maintains a rolling profile of "normal" system behavior
and the anomaly log used by the SystemFaultEvaluationAgent for pattern
analysis. Backed by simple JSON files so the project runs with zero external
database dependencies; swap in Postgres/Mongo/etc. for production use.
"""

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone

from config import config

_lock = threading.Lock()


class ProfileDatabase:
    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        self._metric_history = deque(maxlen=history_size)
        self._anomaly_log = []
        self._load()

    # ---- persistence ----
    def _load(self):
        if os.path.exists(config.ANOMALY_LOG_PATH):
            with open(config.ANOMALY_LOG_PATH, "r") as f:
                try:
                    self._anomaly_log = json.load(f)
                except json.JSONDecodeError:
                    self._anomaly_log = []

    def _save_anomaly_log(self):
        os.makedirs(os.path.dirname(config.ANOMALY_LOG_PATH), exist_ok=True)
        with open(config.ANOMALY_LOG_PATH, "w") as f:
            json.dump(self._anomaly_log[-1000:], f, indent=2)

    # ---- metric history (used to build the "normal" baseline) ----
    def record_metrics(self, metrics: dict):
        with _lock:
            self._metric_history.append(metrics)

    def get_recent_metrics(self, n: int = 50):
        with _lock:
            return list(self._metric_history)[-n:]

    def get_average_baseline(self):
        with _lock:
            if not self._metric_history:
                return {"cpu": 0, "memory": 0}
            cpu_avg = sum(m["cpu"] for m in self._metric_history) / len(self._metric_history)
            mem_avg = sum(m["memory"] for m in self._metric_history) / len(self._metric_history)
            return {"cpu": round(cpu_avg, 2), "memory": round(mem_avg, 2)}

    # ---- anomaly log ----
    def log_anomaly(self, anomaly: dict):
        anomaly = {**anomaly, "logged_at": datetime.now(timezone.utc).isoformat()}
        with _lock:
            self._anomaly_log.append(anomaly)
            self._save_anomaly_log()
        return anomaly

    def get_recent_anomalies(self, n: int = 20):
        with _lock:
            return self._anomaly_log[-n:]


# Singleton instance shared by all agents/routes
profile_db = ProfileDatabase()
