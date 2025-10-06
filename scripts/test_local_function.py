"""
scripts/test_local_function.py
--------------------------------
Simulates a GCS upload event to test main.py locally.
"""

import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cloudfunction import main

# Simulated event payload (mimics GCS finalize trigger)
event = {
    "bucket": "local-bucket",
    "name": "sample_images/cat.jpg"
}
context = {"event_id": "1234", "timestamp": "2025-10-06T00:00:00Z"}

if __name__ == "__main__":
    print(f"🚀 Running local simulation at {datetime.utcnow().isoformat()}...\n")
    try:
        result = main.gcs_trigger(event, context)
        print("\n✅ Function returned:\n", json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Local simulation failed: {e}")
        sys.exit(1)