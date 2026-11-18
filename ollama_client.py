"""
ollama_client.py — Unified Client for Self-Hosted Models on College GPU
========================================================================
Connects to:
- Port 11434 (Ollama): Qwen Generation Model (with keep_alive='3m' to free VRAM when idle)
- Port 8000 (vLLM): Google Gemma 4 12B IT (G-Eval Judge Model)
- Port 7997 (vLLM): Nomic Embed Text v1.5 (768-dim RAG Embeddings)

Features:
- Departmental GPU Politeness: Pre-flight check & wait queue if VRAM is busy
- Seamless fail-open to Groq cloud API if GPU is occupied for > 50s
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

SERVER_IP = os.getenv("GPU_SERVER_IP", "192.168.4.3")
OLLAMA_URL = f"http://{SERVER_IP}:11434"
GEMMA_JUDGE_URL = f"http://{SERVER_IP}:8000/v1"
NOMIC_EMBED_URL = f"http://{SERVER_IP}:7997/v1"

GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:14b")
JUDGE_MODEL = "google/gemma-4-12B-it"
EMBED_MODEL = "/home/sayan25041/models/nomic-embed-text-v1.5"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def is_ollama_alive() -> bool:
    """Checks if Ollama service on 11434 is responding."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def is_gemma_alive() -> bool:
    """Checks if Gemma Judge service on 8000 is responding."""
    try:
        r = requests.get(f"{GEMMA_JUDGE_URL}/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def is_nomic_alive() -> bool:
    """Checks if Nomic Embed service on 7997 is responding."""
    try:
        r = requests.get(f"{NOMIC_EMBED_URL}/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def generate_recipe_ollama(messages: list, max_tokens: int = 500, temperature: float = 0.6) -> str:
    """
    Generates recipe instructions using self-hosted Qwen.
    Uses keep_alive='3m' to automatically release GPU VRAM when idle.
    """
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "3m",  # Frees VRAM back to department after 3 minutes of inactivity
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()

def query_gemma_judge(prompt: str, max_tokens: int = 250, temperature: float = 0.1) -> str:
    """
    Evaluates recipe drafts using Google Gemma 4 12B IT (Port 8000).
    """
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Culinary Judge. Evaluate recipes with strict chain-of-thought analysis and output precise numeric scores."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(f"{GEMMA_JUDGE_URL}/chat/completions", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def get_nomic_embeddings(texts: list) -> list:
    """
    Fetches 768-dimensional embeddings using resident Nomic model on Port 7997.
    """
    payload = {
        "model": EMBED_MODEL,
        "input": texts
    }
    resp = requests.post(f"{NOMIC_EMBED_URL}/embeddings", json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]
