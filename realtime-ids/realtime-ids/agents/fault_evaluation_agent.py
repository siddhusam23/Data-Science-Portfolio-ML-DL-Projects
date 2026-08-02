"""
SystemFaultEvaluationAgent — reviews anomalies raised by the
SystemSentinelAgent, looks for recurring patterns (e.g. the same metric
tripping repeatedly within a short window), and escalates to the
SystemReplicationAgent when a fault looks serious enough to warrant a
recovery action.
"""

import threading
from collections import Counter
from datetime import datetime, timedelta, timezone

from database.profile_db import profile_db


class SystemFaultEvaluationAgent:
    def __init__(self, replication_agent=None, recurrence_threshold: int = 3,
                 recurrence_window_minutes: int = 10):
        self.replication_agent = replication_agent
        self.recurrence_threshold = recurrence_threshold
        self.recurrence_window_minutes = recurrence_window_minutes
        self._lock = threading.Lock()

    def _recent_anomalies_within_window(self):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.recurrence_window_minutes)
        recent = []
        for a in profile_db.get_recent_anomalies(n=100):
            logged_at = datetime.fromisoformat(a["logged_at"])
            if logged_at >= cutoff:
                recent.append(a)
        return recent

    def evaluate(self, anomaly: dict) -> dict:
        """Analyze the anomaly in context of recent history and decide whether
        it is a one-off blip or a recurring fault pattern that needs recovery
        action."""
        with self._lock:
            recent = self._recent_anomalies_within_window()
            reason_counts = Counter(
                reason for a in recent for reason in a.get("reasons", [])
            )
            recurring = [reason for reason, count in reason_counts.items()
                         if count >= self.recurrence_threshold]

            evaluation = {
                "anomaly": anomaly["message"],
                "recent_count": len(recent),
                "recurring_patterns": recurring,
                "escalated": bool(recurring),
            }

            if recurring and self.replication_agent:
                self.replication_agent.initiate_recovery(reason=recurring)

            return evaluation
