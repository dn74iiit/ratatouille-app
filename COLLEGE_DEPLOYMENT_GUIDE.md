# Ratatouille College Server Deployment Guide

This guide provides instructions for deploying the Ratatouille application onto the CoSyLab research server environment with self-hosted GPU inference and G-Eval judging.

---

## 1. System Overview

- **Backend**: FastAPI (`api.py`), running on port `8000` (or `8085` if port 8000 is reserved for vLLM).
- **Frontend**: React + Vite SPA (`frontend/`), built to static files (`dist/`) and served on port `3000` (or reverse-proxied via Nginx on port `80`/`443`).
- **AI Inference Architecture (Blackwell RTX Pro 5000 GPU @ 192.168.4.3)**:
  - **Recipe Generator**: Self-hosted `qwen2.5:14b` running via user-space Ollama on port `11434` with `keep_alive="3m"` to automatically yield VRAM when idle.
  - **Culinary Judge**: Self-hosted `google/gemma-4-12B-it` running via vLLM on port `8000` executing the G-Eval 4-dimension audit framework.
  - **RAG Embeddings**: Self-hosted `nomic-embed-text-v1.5` (768-dim) running via vLLM on port `7997`.
  - **Cloud Fail-Open Fallback**: Automatic seamless fallback to Groq API (`qwen/qwen3.8-27b`) if cluster GPU is busy.
- **Database**:
  - **MongoDB Atlas**: Persistent storage for recipes, community kitchen recipes, and user profile saves.

---

## 2. Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18+ and `npm`
- **Process Manager** (Recommended): `pm2` (`npm install -g pm2`) or `systemd`

---

## 3. GPU Cluster Setup (Blackwell RTX Pro 5000)

If hosting generation on the departmental GPU cluster (`192.168.4.3`):

1. **Start User-Space Ollama**:
   ```bash
   export OLLAMA_HOST=0.0.0.0:11434
   export OLLAMA_MODELS=/home/nindra25074/ollama/models
   nohup /home/nindra25074/ollama/bin/ollama serve > /home/nindra25074/ollama/ollama.log 2>&1 &
   ```

2. **Verify Model Availability**:
   ```bash
   /home/nindra25074/ollama/bin/ollama list
   # Output should include qwen2.5:14b
   ```

3. **Verify vLLM Services**:
   - Gemma 4 12B IT (Judge): `curl http://192.168.4.3:8000/v1/models`
   - Nomic Embed (RAG): `curl http://192.168.4.3:7997/v1/models`

---

## 4. Backend Deployment (FastAPI)

From the repository root:

1. **Set up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Create or update `.env` in the repository root:
   ```bash
   GPU_SERVER_IP="192.168.4.3"
   OLLAMA_GEN_MODEL="qwen2.5:14b"
   GROQ_API_KEY="your_groq_api_key_here"  # For cloud fail-open fallback
   MONGO_URI="your_mongodb_connection_uri_here"
   ```

3. **Start the Backend Server**:
   Using PM2:
   ```bash
   pm2 start "uvicorn api:app --host 0.0.0.0 --port 8000" --name ratatouille-api
   ```
   Or run directly:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

4. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   # Response includes: {"status":"healthy","model_display":"Qwen 2.5","judge_model":"Google Gemma 4 12B IT"}
   ```

---

## 5. Frontend Deployment (React / Vite)

From the repository root:

1. **Navigate and Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Build for Production**:
   ```bash
   npm run build
   ```

3. **Serve the Production Build**:
   Using PM2 Serve:
   ```bash
   pm2 serve dist 3000 --spa --name ratatouille-web
   ```

4. **Network & Hostname Auto-Detection**:
   - The frontend automatically detects the server's current IP/hostname (`window.location.hostname:8000`) to communicate with the backend.
   - If deploying behind an Nginx reverse proxy with a custom route, set `VITE_BACKEND_URL` in `frontend/.env` before running `npm run build`.

---

## 6. Summary of Architecture & Capabilities

1. **Self-Hosted LLM Generation & Judging**:
   - ⚡ **Qwen 2.5 with Guardrails**: Self-hosted generator on college GPU with Mandi median constraints and live CVS validation.
   - ⚖️ **G-Eval 4-Dimension Judge**: Google Gemma 4 12B IT evaluates Consistency, Constraints, Coherence, and Fluency (0.00 - 1.00 scale) with Chain-of-Thought reasoning.
2. **Two-Stage High-Dimensional RAG**:
   - FAISS vector candidate search powered by 768-dim embeddings, re-ranked by ingredient overlap and archetype matching.
3. **Mandi Market Robustness**:
   - Mandi rates computed via row-level and state-level median pricing across min, max, and modal rates.
   - Dynamic condiment bounds scaling with total bulk dish mass.
   - Hot peppers and green chillies correctly calibrated as intense condiments (0.5g - 4.0g).
4. **🌱 Agentic Vegan Engine**:
   - Chemical profile alignment and dairy/meat substitution checking.
