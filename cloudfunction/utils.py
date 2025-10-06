"""
utils.py
---------
General helper utilities for the Serverless Image Recognition project.
"""

import os
import json
from datetime import datetime


def log_event(message: str):
    """Lightweight console + file logger."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp} UTC] {message}"
    print(entry)

    # Optionally store logs locally during tests
    os.makedirs("logs", exist_ok=True)
    with open("logs/events.log", "a") as f:
        f.write(entry + "\n")


def save_json(data: dict, path: str):
    """Save dictionary as formatted JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log_event(f"Saved JSON → {path}")


def read_json(path: str) -> dict:
    """Read JSON safely; return empty dict if missing."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        log_event(f"⚠️ JSON not found → {path}")
        return {}
