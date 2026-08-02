"""
JWT-based authentication for securing the dashboard's API endpoints.

Tokens are issued on successful login (/login) and expire after
config.JWT_EXPIRY_MINUTES. Protected endpoints use the @token_required
decorator to reject requests with a missing, invalid, or expired token.
"""

from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import jsonify, request

from config import config


def generate_token(username: str) -> str:
    payload = {
        "sub": username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=config.JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.split(" ", 1)[1] if auth_header.startswith("Bearer ") else None

        if not token:
            return jsonify({"error": "Missing authentication token"}), 401

        try:
            decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated
