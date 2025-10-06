import os, tempfile
from utils import download_blob, upload_json
from model import ImageClassifier
from caption import generate_caption

classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = ImageClassifier()
    return classifier

def gcs_trigger(event, context):
    bucket = event["bucket"]
    name = event["name"]
    tmp = tempfile.mktemp()
    download_blob(bucket, name, tmp)
    label = get_classifier().predict(tmp)
    caption = generate_caption(label)
    result = {"file": name, "label": label, "caption": caption}
    upload_json(bucket, f"results/{os.path.basename(name)}.json", result)
    return result
