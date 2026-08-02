"""
Basic unit tests for anomaly detection logic. Run with:

    pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sentinel_agent import SystemSentinelAgent


def test_detect_anomaly_flags_high_cpu():
    agent = SystemSentinelAgent()
    metrics = {"cpu": 95.0, "memory": 10.0, "timestamp": "2026-01-01T00:00:00+00:00"}
    reasons = agent.detect_anomaly(metrics)
    assert reasons is not None
    assert any("CPU" in r for r in reasons)


def test_detect_anomaly_flags_high_memory():
    agent = SystemSentinelAgent()
    metrics = {"cpu": 5.0, "memory": 90.0, "timestamp": "2026-01-01T00:00:00+00:00"}
    reasons = agent.detect_anomaly(metrics)
    assert reasons is not None
    assert any("Memory" in r for r in reasons)


def test_detect_anomaly_returns_none_when_normal():
    agent = SystemSentinelAgent()
    metrics = {"cpu": 10.0, "memory": 20.0, "timestamp": "2026-01-01T00:00:00+00:00"}
    reasons = agent.detect_anomaly(metrics)
    assert reasons is None
