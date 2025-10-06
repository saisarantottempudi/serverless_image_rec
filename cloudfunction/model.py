"""
model.py
---------
Handles image classification using a TensorFlow Lite model.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os


class ImageClassifier:
    def __init__(self):
        # Path to the TFLite model
        MODEL_PATH = os.path.join(
            os.path.dirname(__file__), "models", "mobilenet_v2_1.0_224.tflite"
        )

        print(f"🔍 Loading TFLite model from {MODEL_PATH}")
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_dtype = self.input_details[0]["dtype"]
        print(f"[INFO] Model input type: {input_dtype}")

    def preprocess(self, image_path):
        """Load and preprocess image for model inference."""
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Adjust dtype if needed
        if self.input_details[0]["dtype"] == np.uint8:
            img_array = (img_array / 255.0 * 255).astype(np.uint8)
        else:
            img_array = img_array / 255.0

        return img_array

    def predict(self, image_path):
        """Run inference on a given image."""
        x = self.preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Get top-5 predictions
        top_indices = np.argsort(output[0])[::-1][:5]
        top_confidences = output[0][top_indices]

        print("\nTop-5 Predictions:")
        for i, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
            print(f"  class_{idx:<3d} — {conf*100:.2f}%")

        top_label = f"class_{top_indices[0]}"
        return top_label