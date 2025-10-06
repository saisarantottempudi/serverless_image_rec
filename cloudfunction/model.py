"""
model.py
--------
Lightweight TensorFlow Lite image classification module.
Loads a .tflite model and runs inference on uploaded images.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mobilenet_v2_1.0_224.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "models", "labels.txt")

class ImageClassifier:
    def __init__(self):
        # Load TFLite model and allocate tensors
        print(f"🔍 Loading TFLite model from {MODEL_PATH}")
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Detect input data type (float or uint8)
        self.input_type = self.input_details[0]["dtype"]
        print(f"[INFO] Model input type: {self.input_type}")

        # Load labels
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r") as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            self.labels = [f"class_{i}" for i in range(1000)]  # fallback

    def preprocess(self, image_path):
        """Preprocess an image for model input."""
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img = np.array(img)

        if self.input_type == np.float32:
            img = img / 255.0
            img = np.expand_dims(img.astype(np.float32), axis=0)
        else:
            img = np.expand_dims(img.astype(np.uint8), axis=0)

        return img

    def predict_topk(self, image_path, k=5):
        """Return top-k predictions with confidence scores."""
        input_tensor = self.preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # Convert probabilities
        if self.output_details[0]["dtype"] == np.uint8:
            scale, zero_point = self.output_details[0]["quantization"]
            output_data = scale * (output_data - zero_point)

        top_k_indices = np.argsort(output_data)[::-1][:k]
        results = [
            {
                "label": self.labels[i] if i < len(self.labels) else f"class_{i}",
                "confidence": float(output_data[i] * 100),
            }
            for i in top_k_indices
        ]
        return results