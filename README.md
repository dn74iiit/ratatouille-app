# Ratatouille 🍲🤖

Welcome to **Ratatouille**, an intelligent, AI-powered recipe generation and ingredient substitution engine! 

This project explores computational gastronomy by combining Large Language Models (via Groq), Hugging Face spaces, and deterministic computational pipelines to generate context-aware, vegan-friendly, and cost-constrained recipes.

Whether you're a Computer Science student interested in AI integrations, full-stack web development, or complex systems, this repository offers a great sandbox to explore and learn.

## 🚀 Features

- **Vegan Recipe Engine (`vegan_engine.py`)**: Automatically converts traditional recipes into vegan alternatives using chemical feature mapping and a customized substitution database.
- **Cost Constraints Integration**: Uses advanced prompting and data structures to formulate recipes that fit within a specified budget.
- **Modern Tech Stack**:
  - **Backend**: FastAPI (Python) for blazing-fast asynchronous endpoints.
  - **Frontend**: React + Vite for a snappy, modern User Interface.
  - **AI Integration**: Groq API for rapid LLM inference, paired with custom models hosted on Hugging Face Spaces.
  - **Database**: MongoDB (motor_asyncio) for non-blocking database queries.
- **Deconstruction Mapping**: Maps complex ingredients into atomic chemical features to find the most accurate vegan substitutes.

## 📁 Repository Structure

```text
├── frontend/                     # React/Vite Single Page Application
├── api.py                        # FastAPI backend server
├── vegan_engine.py               # Core logic for recipe processing and conversion
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for required environment variables
├── MIGRATION_GUIDE.md            # Detailed instructions for server deployments
├── *.json / *.csv                # Datasets (chemical features, vegan alternatives)
└── docs/                         # Additional architectural documentation
```

*(Note: Large datasets and Jupyter notebooks used for training the underlying models are managed separately or documented within `MIGRATION_GUIDE.md`.)*

## 🛠️ Getting Started (Local Development)

If you're a student looking to run this locally, hack on the features, or understand how the components tie together, follow these steps:

### 1. Prerequisites
- **Python 3.9+**
- **Node.js 18+**
- **API Keys**: You will need a Groq API Key and a MongoDB URI to fully test the backend.

### 2. Backend Setup
Clone the repository and set up your Python environment:
```bash
git clone https://github.com/dn74iiit/ratatouille-app.git
cd ratatouille-app

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

Set up your environment variables by copying the example file:
```bash
cp .env.example .env
```
Open `.env` and fill in your keys (e.g., `GROQ_API_KEY`, `MONGO_URI`, etc.).

Start the FastAPI server:
```bash
uvicorn api:app --reload
```
*The API will be available at `http://localhost:8000`. You can view the interactive Swagger documentation at `http://localhost:8000/docs`.*

### 3. Frontend Setup
Open a new terminal window, and navigate to the frontend directory:
```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```
*The React app will typically be available at `http://localhost:5173`.*

## 🧠 Areas to Explore for CS Students

If you want to contribute or just study the code, here are some interesting areas:
1. **The AI Pipeline**: Check out `api.py` to see how we stream responses from Groq and handle Gradio client requests to Hugging Face.
2. **Computational Gastronomy**: Look into `vegan_engine.py` and `deconstruction_map.json` to understand how ingredients are mathematically and chemically substituted.
3. **Asynchronous Python**: The backend heavily relies on `async/await` patterns with FastAPI and Motor (MongoDB).
4. **Server Deployment**: Read through the `MIGRATION_GUIDE.md` to understand how this system is deployed to academic research servers (e.g., CoSyLab).

## 🤝 Contributing
Feel free to fork this repository, submit Pull Requests, or open Issues if you find bugs or have feature ideas. 

Happy coding! 👨‍🍳💻
