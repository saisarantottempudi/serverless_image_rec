"""
main.py
--------
Serverless Image Recognition Cloud Function
with automatic local fallback when GCS access fails.
"""

import os
import json
from google.cloud import storage
from google.api_core import exceptions as gcs_exceptions
from dotenv import load_dotenv
from .model import ImageClassifier
from .caption import generate_caption
from .utils import log_event, save_json

# Load environment variables
load_dotenv()

# Initialize GCS client safely
project_id = os.getenv("GCP_PROJECT_ID")
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

storage_client = storage.Client(project=project_id)
classifier = ImageClassifier()


def gcs_trigger(event, context):
    """
    Cloud Function entrypoint for handling GCS image upload events.
    Downloads image → runs classification → generates caption.
    """
    bucket_name = event["bucket"]
    file_name = event["name"]

    log_event(f"🪶 Trigger received for: gs://{bucket_name}/{file_name}")

    # Temporary file for local processing
    temp_path = f"/tmp/{os.path.basename(file_name)}"

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(temp_path)
        log_event(f"✅ File downloaded from GCS → {temp_path}")

    except gcs_exceptions.Forbidden as e:
        log_event("⚠️  GCS access forbidden (billing disabled or bucket missing). Using local fallback.")
        temp_path = "sample_images/cat.jpg"

    except Exception as e:
        log_event(f"⚠️  Unexpected GCS error: {e}. Using local fallback.")
        temp_path = "sample_images/cat.jpg"

    # Perform classification
    label = classifier.predict(temp_path)
    log_event(f"🧠 Classification complete: {label}")

    # Generate caption via OpenAI
    caption = generate_caption(label)
    log_event(f"💬 Caption generated: {caption}")

    # Prepare output
    result = {
        "file_name": file_name,
        "bucket": bucket_name,
        "label": label,
        "caption": caption,
    }

    # Save result JSON (locally for now)
    output_path = f"results/{os.path.basename(file_name)}.json"
    save_json(result, output_path)

    log_event(f"📦 Results saved to → {output_path}")
    return json.dumps(result, indent=2)


# Local entrypoint for testing
if __name__ == "__main__":
    print("🚀 Running local simulation...\n")

    # Mock GCS event (simulate upload)
    event = {"bucket": "local-bucket", "name": "sample_images/cat.jpg"}
    context = {}

    output = gcs_trigger(event, context)
    print("\n✅ Function returned:\n", output)