# Real-Time Anomaly Detection and Secure Intrusion Prevention

A modular, multi-agent Intrusion Detection System (IDS) that monitors system
metrics in real time, flags anomalies against configurable thresholds,
tamper-proofs alerts with RSA digital signatures, and exposes a
JWT-authenticated dashboard and API.

Vinay Nambiar · Siddhesh T S
Department of Computer Science and Engineering, Amrita School of Computing,
Bengaluru, Amrita Vishwa Vidyapeetham, India

## Overview

Traditional monitoring is reactive: by the time CPU spikes or memory
exhaustion trigger an alert, the incident may already be underway. This
project takes a proactive, modular approach — a set of cooperating agents
continuously watch system health, evaluate whether anomalies form a
recurring fault pattern, and can trigger recovery action, all while keeping
alerts cryptographically verifiable and API access locked down.

## Architecture

```
                 ┌─────────────────────┐
                 │        LSIA          │  (coordinates all agents)
                 └──────────┬───────────┘
                             │
      ┌──────────────────────┼───────────────────────┐
      ▼                      ▼                        ▼
┌───────────────┐  ┌──────────────────────┐  ┌───────────────────────┐
│ SystemSentinel │  │ SystemFaultEvaluation │  │ SystemReplicationAgent │
│     Agent      │─▶│        Agent          │─▶│  (recovery actions)    │
│ (psutil metrics,│  │ (pattern/recurrence   │  └───────────────────────┘
│  threshold check,│  │  analysis)            │
│  RSA signing)    │  └──────────────────────┘
└───────┬────────┘
        │
        ▼
┌────────────────┐        ┌─────────────────┐
│  ProfileDatabase │        │  Email Alerts    │
│ (baseline + log)  │       │ (SMTP + app pwd) │
└────────────────┘        └─────────────────┘

Flask app (JWT-protected /metrics, /login) ──▶ real-time dashboard (Chart.js)
```

- **SystemSentinelAgent** — polls CPU/memory via `psutil` on a background
  thread, flags threshold violations, signs the resulting alert with RSA,
  logs it, and fires an email notification.
- **SystemFaultEvaluationAgent** — looks at the recent anomaly history to
  spot recurring patterns (the same failure mode repeating within a time
  window) rather than reacting to every one-off blip.
- **SystemReplicationAgent** — triggered when a fault recurs; represents
  where real recovery actions (restart a service, fail over, scale up)
  would be hooked in.
- **ProfileDatabase** — JSON-backed store of the rolling metric baseline and
  the anomaly log, used by the other agents.
- **LSIA (Local System Intelligence Agent)** — wires the three agents
  together and exposes a single `get_status()` used by the Flask API.

## Security Features

- **RSA digital signatures** — every anomaly alert is signed
  (`security/digital_signature.py`) so a tampered or spoofed alert can be
  detected with the public key.
- **JWT authentication** — `/metrics` requires a valid bearer token issued
  by `/login`; tokens expire after `JWT_EXPIRY_MINUTES` (default 30).
- **SMTP with app password** — email alerts authenticate with an app
  password rather than a primary account password.

## Tech Stack

Python · Flask · psutil · PyJWT · `cryptography` (RSA) · smtplib · Chart.js

## Repository Structure

```
.
├── app.py                       # Flask entry point (routes, dashboard)
├── config.py                    # env-driven settings
├── agents/
│   ├── sentinel_agent.py        # SystemSentinelAgent
│   ├── fault_evaluation_agent.py# SystemFaultEvaluationAgent
│   ├── replication_agent.py     # SystemReplicationAgent
│   └── lsia.py                  # coordinator
├── security/
│   ├── digital_signature.py     # RSA sign/verify
│   └── auth.py                  # JWT issue/verify, @token_required
├── database/
│   └── profile_db.py            # baseline + anomaly log (JSON-backed)
├── alerts/
│   └── email_alert.py           # SMTP email notifications
├── templates/                   # login.html, dashboard.html
├── static/                      # css/js for the dashboard
├── keys/
│   └── generate_keys.py         # one-time RSA key pair generation
├── tests/
│   └── test_sentinel_agent.py
├── docs/
│   └── Project_Report.pdf       # full written report
├── requirements.txt
├── .env.example
└── README.md
```

## Getting Started

```bash
# 1. Clone and enter the repo
git clone https://github.com/siddhusam23/realtime-ids.git
cd realtime-ids

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# edit .env: set JWT_SECRET, ADMIN_USERNAME/PASSWORD, and (optionally)
# SMTP settings if you want real email alerts

# 4. Generate the RSA key pair used for signing alerts
python keys/generate_keys.py

# 5. Run the app
python app.py
```

Then open `http://localhost:5000`, log in with the credentials from your
`.env`, and you'll land on the real-time metrics dashboard.

### Running tests

```bash
pip install pytest
pytest tests/
```

## API Endpoints

| Method | Endpoint    | Auth        | Description                              |
|--------|-------------|-------------|-------------------------------------------|
| POST   | `/login`    | none        | Authenticates a user, returns a JWT       |
| GET    | `/metrics`  | Bearer JWT  | Latest metrics, baseline, recent anomalies, recovery log |
| GET    | `/health`   | none        | Liveness check                            |

## Extending the System

- **New agents** — add a class under `agents/` and register it with the
  `LocalSystemIntelligenceAgent` in `agents/lsia.py`.
- **New metrics** — extend `SystemSentinelAgent.collect_metrics()` (e.g. disk
  I/O, network throughput) and add matching thresholds to `config.py`.
- **Swap storage** — `database/profile_db.py` is a thin JSON-backed store;
  replace it with Postgres/Mongo/etc. behind the same interface.
- **Real recovery actions** — implement actual remediation logic in
  `SystemReplicationAgent.initiate_recovery()`.

## Report

The full write-up — literature survey, methodology, and results — is in
[`docs/Project_Report.pdf`](docs/Project_Report.pdf).

## License

MIT — see [LICENSE](LICENSE).
