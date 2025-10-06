import os, sys
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cloudfunction"))
from model import ImageClassifier
from caption import generate_caption

IMG = "sample_images/cat.jpg"
clf = ImageClassifier()
results = clf.predict_topk(IMG, k=5)

print("\nTop-5 Predictions:")
for r in results:
    print(f"  {r['label']:<25} {r['confidence']*100:.2f}%")

caption = generate_caption(results[0]["label"])
print("\nGenerated Caption:", caption)
