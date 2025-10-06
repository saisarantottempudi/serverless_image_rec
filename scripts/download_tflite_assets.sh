set -euo pipefail

mkdir -p cloudfunction/models
cd cloudfunction/models

curl -L -o mobilenet_v2_1.0_224.tflite \
  https://storage.googleapis.com/ailab-public/models/tflite/mobilenet_v2_1.0_224.tflite

curl -L -o imagenet_labels.txt \
  https://storage.googleapis.com/ailab-public/models/tflite/imagenet_labels_1001.txt

echo "Downloaded model + labels successfully."
