import os
import numpy as np
from PIL import Image

# ✅ Fallback for macOS TensorFlow vs tflite-runtime
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow.lite as tflite

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mobilenet_v2_1.0_224.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "models", "imagenet_labels.txt")

class ImageClassifier:
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(LABELS_PATH, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.input_dtype = self.input_details[0]['dtype']
        print(f"[INFO] Model input type: {self.input_dtype}")

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.array(img)

        if self.input_dtype == np.uint8:
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32) / 255.0

        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict_topk(self, image_path, k=5):
        x = self.preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Convert to probabilities if quantized
        output = output.astype(np.float32)
        probs = output / np.sum(output)

        top_k = probs.argsort()[-k:][::-1]
        results = [
            {"label": self.labels[i] if i < len(self.labels) else f"class_{i}", "confidence": float(probs[i])}
            for i in top_k
        ]
        return results