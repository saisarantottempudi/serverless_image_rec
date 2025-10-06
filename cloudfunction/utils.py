import os, json
from google.cloud import storage

client = storage.Client()

def download_blob(bucket_name, src, dst):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    blob.download_to_filename(dst)
    return dst

def upload_text(bucket_name, dst, text):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dst)
    blob.upload_from_string(text)
    return f"gs://{bucket_name}/{dst}"

def upload_json(bucket_name, dst, data):
    upload_text(bucket_name, dst, json.dumps(data, indent=2))
