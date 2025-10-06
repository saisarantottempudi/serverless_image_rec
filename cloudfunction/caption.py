import os
from openai import OpenAI

def generate_caption(label):
    prompt = f"Write a short human-like caption for a photo containing: {label}."
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=50
    )
    return resp.choices[0].message.content.strip()
