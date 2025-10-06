"""
caption.py
-----------
Generates short contextual captions for images
using OpenAI's GPT models.
"""

import os
from openai import OpenAI

# Create a single reusable client (uses env variable OPENAI_API_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def generate_caption(label: str) -> str:
    """
    Given a classified image label, generate a short
    descriptive caption using OpenAI.
    """
    if not label:
        return "No label provided."

    prompt = (
        f"Write a short, fun, single-sentence Instagram-style caption "
        f"for a photo classified as '{label}'. Add one emoji if appropriate."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a witty caption generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️  Caption generation failed: {e}")
        return f"Image of {label}."
