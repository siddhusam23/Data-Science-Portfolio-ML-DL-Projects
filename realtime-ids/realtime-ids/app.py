"""
Flask entry point for the Real-Time Anomaly Detection and Secure Intrusion
Prevention system.

Routes:
    GET  /              -> login page
    GET  /dashboard      -> real-time metrics dashboard (requires a token in
                            localStorage, checked client-side)
    POST /login           -> authenticates a user, returns a JWT
    GET  /metrics          -> current + recent metrics (JWT protected)
"""

from flask import Flask, jsonify, render_template, request

from agents.lsia import lsia
from config import config
from security.auth import generate_token, token_required

app = Flask(__name__)


@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or request.form
    username = data.get("username")
    password = data.get("password")

    if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
        token = generate_token(username)
        return jsonify({"token": token, "expires_in_minutes": config.JWT_EXPIRY_MINUTES})

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/metrics", methods=["GET"])
@token_required
def metrics():
    return jsonify(lsia.get_status())


@app.route("/health", methods=["GET"])
def health():
    """Unauthenticated liveness check for load balancers / uptime monitors."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    lsia.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
