# Ratatouille: AI-Powered Recipe Generation and Substitution Engine

Welcome to Ratatouille, an intelligent application designed to bridge computational gastronomy with Large Language Models. 

This project leverages advanced natural language processing via the Groq API and customized Hugging Face spaces, alongside deterministic computational pipelines, to generate context-aware, vegan-friendly, and cost-constrained recipes.

## Core Features and System Workflow

The Ratatouille system is built on a modular architecture that separates recipe generation, ingredient deconstruction, and constraint optimization. Below is a detailed walkthrough of the internal workflows and the intricacies of each subsystem.

### 1. The Generation Pipeline (api.py)
The generation pipeline acts as the primary orchestrator for user requests, ensuring low-latency responses by streaming data directly to the client.
- **Asynchronous Processing**: Built on FastAPI, the backend handles concurrent requests without blocking.
- **LLM Integration**: The system interfaces with the Groq API for rapid natural language processing. It constructs dense, context-heavy prompts based on user dietary restrictions and preferences.
- **Hugging Face Delegation**: For specific domain tasks (like specialized vegan recipe modeling), the API seamlessly hands off requests to custom inference endpoints hosted on Hugging Face Spaces via the Gradio client.
- **Data Persistence**: Asynchronous MongoDB operations (via Motor) log generated recipes and user interactions, allowing the system to maintain a conversational context without slowing down the primary generation loop.

### 2. The Vegan Conversion Engine (vegan_engine.py)
The Vegan Engine is a deterministic subsystem that applies computational gastronomy principles to convert traditional recipes into fully vegan alternatives while maintaining flavor profiles and structural integrity.
- **Ingredient Deconstruction**: When a non-vegan recipe is detected, the engine parses the ingredient list and maps complex animal-based products to their fundamental components using `deconstruction_map.json`.
- **Chemical Feature Mapping**: Each deconstructed component is analyzed for its chemical properties (e.g., binding capabilities, fat content, moisture) using `chemical_features.json`.
- **Intelligent Substitution**: The engine queries the `vegan_alternatives_db` to find plant-based ingredients that match the exact chemical and functional profile of the original ingredient. For example, it understands whether an egg is being used for binding, leavening, or moisture, and selects the optimal substitute accordingly.

### 3. Cost-Constraint Optimization
A defining feature of Ratatouille is its ability to formulate recipes that strictly adhere to a user-defined budget.
- **Pricing Data Structures**: The system cross-references generated ingredients against a localized pricing matrix.
- **Iterative Refinement**: If a generated recipe exceeds the budget, the system utilizes linear programming and iterative LLM prompting to swap out high-cost ingredients for cheaper, nutritionally equivalent alternatives without compromising the dish's core identity.

## Repository Structure

```text
├── frontend/                     # React/Vite Single Page Application
├── api.py                        # FastAPI backend server and orchestrator
├── vegan_engine.py               # Core logic for recipe processing and conversion
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for required environment variables
├── MIGRATION_GUIDE.md            # Detailed instructions for server deployments
├── *.json / *.csv                # Datasets (chemical features, vegan alternatives)
└── docs/                         # Additional architectural documentation
```

*(Note: Large datasets and Jupyter notebooks used for training the underlying models are managed separately or documented within MIGRATION_GUIDE.md.)*

## Deployment and Administration
For instructions regarding deploying this system to academic research servers or production environments, please refer to the MIGRATION_GUIDE.md included in this repository.

## Contributing
Contributions and architectural reviews are welcome. Please submit Pull Requests or open Issues for any bugs or feature proposals.
