"""
Local System Intelligence Agent (LSIA) — coordinates the three worker
agents (SystemSentinelAgent, SystemFaultEvaluationAgent,
SystemReplicationAgent) and exposes a single entry point for starting and
querying the IDS from the Flask app.
"""

from agents.fault_evaluation_agent import SystemFaultEvaluationAgent
from agents.replication_agent import SystemReplicationAgent
from agents.sentinel_agent import SystemSentinelAgent
from database.profile_db import profile_db


class LocalSystemIntelligenceAgent:
    def __init__(self):
        self.replication_agent = SystemReplicationAgent()
        self.fault_evaluation_agent = SystemFaultEvaluationAgent(
            replication_agent=self.replication_agent
        )
        self.sentinel_agent = SystemSentinelAgent(
            fault_evaluation_agent=self.fault_evaluation_agent
        )
        self._started = False

    def start(self):
        if not self._started:
            self.sentinel_agent.start()
            self._started = True

    def stop(self):
        self.sentinel_agent.stop()
        self._started = False

    def get_status(self) -> dict:
        return {
            "latest_metrics": self.sentinel_agent.latest_metrics,
            "baseline": profile_db.get_average_baseline(),
            "recent_anomalies": profile_db.get_recent_anomalies(n=10),
            "recovery_log": self.replication_agent.get_recovery_log(n=10),
        }


# Singleton shared across the Flask app
lsia = LocalSystemIntelligenceAgent()
