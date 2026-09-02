"""
generate.py
Handles: taking retrieved chunks + user question, building a prompt,
and calling the LLM to generate a grounded answer.

Uses Groq's free-tier API (fast inference, no cost on free tier) with
the open-source Llama 3.1 model. You'll need a free API key from
https://console.groq.com -> stored as an environment variable GROQ_API_KEY.
"""

import os
from groq import Groq
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_prompt(question: str, chunks: List[Dict]) -> str:
    context = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in chunks])
    prompt = f"""You are a helpful assistant answering questions based ONLY on the
provided context. If the answer isn't in the context, say you don't know —
never make up information.

Context:
{context}

Question: {question}

Answer (also mention which chunk id(s) you used as source):"""
    return prompt


def generate_answer(question: str, chunks: List[Dict]) -> str:
    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # free-tier model on Groq
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # lower = more factual, less "creative"
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    test_chunks = [{"id": "chunk_0", "text": "The sky appears blue due to Rayleigh scattering."}]
    print(generate_answer("Why is the sky blue?", test_chunks))
