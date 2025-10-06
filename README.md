# 🧠 Serverless Image Recognition API

A fully serverless image recognition and caption generation pipeline built using **Google Cloud Functions**, **TensorFlow Lite**, and **OpenAI GPT models**.

---

## 🚀 Features
- Triggered automatically when an image is uploaded to Google Cloud Storage
- Performs lightweight image classification using TensorFlow Lite
- Generates rich captions using OpenAI GPT (e.g. GPT-4o-mini)
- Stores structured JSON results back into your GCS bucket
- Automated deployment via GitHub Actions + Workload Identity Federation (WIF)

---

## 🧩 Project Structure
serverless-image-rec/
├── cloudfunction/
│   ├── main.py              # Cloud Function entrypoint
│   ├── caption.py           # LLM-based caption generation
│   ├── model.py             # TensorFlow Lite image classifier
│   ├── requirements.txt     # Cloud Function dependencies
│   └── utils.py             # Helper utilities (optional)
├── sample_images/           # Local test images
├── scripts/                 # Local test & automation scripts
├── .dev-requirements.txt    # Local (Mac M1/M2) dependencies
└── .github/workflows/       # CI/CD workflows for deployment

---

## ⚙️ Setup (Local Development on Mac M1/M2)

```bash
# Clone the repo
git clone https://github.com/saisarantottempudi/serverless_image_rec.git
cd serverless_image_rec

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install Apple Silicon-optimized dependencies
pip install -r .dev-requirements.txt

