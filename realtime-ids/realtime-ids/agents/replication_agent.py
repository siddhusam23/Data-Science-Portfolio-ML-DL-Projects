"""
SystemReplicationAgent — responsible for keeping the system stable once a
recurring fault has been flagged by the SystemFaultEvaluationAgent. In this
reference implementation, "recovery" is represented by a logged recovery
action; hook in real remediation (restarting a service, failing over to a
standby node, scaling resources, etc.) for your environment.
"""

import threading
from datetime import datetime, timezone

from database.profile_db import profile_db


class SystemReplicationAgent:
    def __init__(self):
        self._lock = threading.Lock()
        self.recovery_log = []

    def initiate_recovery(self, reason: list) -> dict:
        with self._lock:
            action = {
                "action": "recovery_initiated",
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.recovery_log.append(action)
            profile_db.log_anomaly({
                "message": f"Recovery action triggered due to recurring pattern(s): {reason}",
                "metrics": {},
                "reasons": ["recovery_action"],
                "signature": None,
            })
            print(f"[SystemReplicationAgent] Recovery initiated: {reason}")
            return action

    def get_recovery_log(self, n: int = 20):
        with self._lock:
            return self.recovery_log[-n:]
