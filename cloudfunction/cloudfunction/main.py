import os
import json
import tempfile
from google.cloud import storage
from model import ImageClassifier
from caption import generate_caption

# Initialize GCS client only once (cold start optimization)
storage_client = storage.Client()

def gcs_trigger(event, context):
    """
    Triggered by a finalized (uploaded) file in Cloud Storage.
    Performs image classification and caption generation.
    """

    bucket_name = event["bucket"]
    file_name = event["name"]
    print(f"🪶 Trigger received for: gs://{bucket_name}/{file_name}")

    # 1️⃣ Download file temporarily
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_name))
    blob.download_to_filename(temp_path)
    print(f"✅ File downloaded to {temp_path}")

    # 2️⃣ Run image classification
    clf = ImageClassifier()
    results = clf.predict_topk(temp_path, k=5)
    top_label = results[0]["label"]
    print(f"🧠 Classification complete: {top_label}")

    # 3️⃣ Generate caption using LLM
    caption = generate_caption(top_label)
    print(f"💬 Generated caption: {caption}")

    # 4️⃣ Prepare response data
    response = {
        "file_name": file_name,
        "bucket": bucket_name,
        "top_predictions": results,
        "caption": caption,
    }

    # 5️⃣ Optionally store result as JSON back to GCS
    output_blob = bucket.blob(f"results/{os.path.splitext(file_name)[0]}.json")
    output_blob.upload_from_string(json.dumps(response, indent=2), content_type="application/json")
    print(f"📦 Results saved to: gs://{bucket_name}/{output_blob.name}")

    return json.dumps(response)