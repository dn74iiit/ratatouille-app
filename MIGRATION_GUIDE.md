# Ratatouille System Migration & Deployment Guide

This document outlines the architecture, dependencies, and steps required to migrate and deploy the Ratatouille application onto the CoSyLab (Complex Systems Lab) servers.

## 1. Repository Structure Overview

When migrating, you only need to push the essential application files. Large datasets and training notebooks should generally be excluded from the production deployment repository unless specifically required for retraining on the server.

### What to Push (Essential Files)
- **Backend Core**: `api.py`, `vegan_engine.py` (FastAPI backend and core generation logic)
- **Frontend**: The `frontend/` directory (React/Vite SPA)
- **Dependencies**: `requirements.txt` (Python packages), `frontend/package.json` (Node packages)
- **Data Configuration**: 
  - `chemical_features.json`
  - `deconstruction_map.json`
  - `vegan_alternatives_db.json`
  - `vegan_alternatives_db.csv`
- **Environment Template**: `.env.example`
- **Documentation**: All `.md` files (like this migration guide and architectural documentation).

### What NOT to Push (Add to `.gitignore`)
- `.env` (contains sensitive API keys and database URIs)
- `__pycache__/` and `.pytest_cache/`
- Large CSV files: `RecipeDB_*.csv`, `final_clean_50k_recipes_grams.csv`, `recipedb_*.csv`
- Large Jupyter Notebooks (`*.ipynb`) (unless training is moving to the server too)
- `node_modules/` (inside the frontend folder)
- `headroom_memory.db`

---

## 2. Dependencies & Prerequisites

### Server Requirements
- **Python**: Version 3.9 or higher.
- **Node.js**: Version 18+ (for building and serving the frontend).
- **Environment**: A Linux/Unix-based server is recommended, but Windows Server works as well.

### Required API Keys and External Services
You will need to set up the `.env` file on the new server with the following credentials (refer to `.env.example`):
- `HF_TOKEN`: HuggingFace token for inference API access.
- `GROQ_API_KEY`: Groq API key for LLM operations.
- `MONGO_URI`: MongoDB connection string.
- `GITHUB_PAT`: GitHub Personal Access Token (if required by your scripts).
- `HF_SPACE_URL` & `HF_SPACE_URL_V9`: Endpoints for Hugging Face Spaces.

---

## 3. Deployment Walkthrough

### Step A: Clone the Repository
Clone the new repository onto the target server:
```bash
git clone https://github.com/<organization>/ratatouille-cost.git
cd ratatouille-cost
```

### Step B: Setup the Backend
1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:
   Copy the example `.env` file and populate it with actual keys.
   ```bash
   cp .env.example .env
   nano .env  # Add your keys here
   ```
4. **Run the FastAPI Server**:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
   *(For production, consider running this via a process manager like `pm2`, `systemd`, or `gunicorn` with `uvicorn` workers).*

### Step C: Setup the Frontend
1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```
2. **Install Node Dependencies**:
   ```bash
   npm install
   ```
3. **Configure Frontend Environment (If Applicable)**:
   Ensure any frontend environment variables (like `VITE_API_URL`) are set so the React app points to the backend server (e.g., `http://<server-ip>:8000`). You can create a `.env` in the `frontend/` folder.
4. **Build for Production**:
   ```bash
   npm run build
   ```
5. **Serve the Frontend**:
   The `dist/` folder created by the build command contains static files. You can serve this using a tool like `serve`, Nginx, or by configuring the FastAPI backend to mount and serve the static files.

---

## 4. Notes for the Administrator (PhD Student)
- **CORS Configuration**: Depending on how the frontend is served (same port vs different port), ensure `api.py` has the correct origins configured in `CORSMiddleware`.
- **Database**: Ensure the CoSyLab server's firewall allows outbound connections to the MongoDB cluster URI provided in the `.env` file.
- **Port Forwarding**: Expose port `8000` (or whichever port you choose to run the backend on) and the frontend port (e.g., `80` or `3000`) so they can be accessed externally.
