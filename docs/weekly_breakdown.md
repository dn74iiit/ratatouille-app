# 📅 Ratatouille Capstone Project: Detailed Weekly Work Breakdown
**Project Name:** Ratatouille – Cost-Constraint Extension  
**Student:** Nindra Dhanush (MT25074)  
**Supervisor:** Dr. Ganesh Bagler  
**Timeline:** January 2026 – May 2026 (Semester 2)  
**Tech Stack (FARM):** FastAPI, React (Vite), MongoDB Atlas, Hugging Face (Llama 3 3B, Unsloth, Gradio Client)

---

## 🏗️ Phase 1: Generative Baseline & LLM Fine-Tuning (Weeks 1 – 4)
*Goal: Establish a high-quality, open-source recipe text generator using fine-tuned LLMs while addressing local compute constraints.*

### Week 1: Environment Setup & Baseline Evaluation
* Reviewed the previous *Ratatouille* generative pipeline codebase and architecture.
* Evaluated prospective open-source models; selected Meta's **Llama 3 (3B)** as the core generative engine.
* Configured local environments and Google Colab runtimes to run large-scale model pipelines.

### Week 2: Data Cleaning, Formatting & Prompt Engineering
* Cleaned and structured the 50,000-recipe dataset (`final_clean_50k_recipes_grams.csv`).
* Designed custom prompt structures to force sequential inference: parsing raw ingredients first, establishing the recipe title next, and finally writing step-by-step cooking directions.
* Managed tokenizer configurations (vocabulary mappings, padding tokens, special formatting tokens).

### Week 3: Supervised Fine-Tuning (SFT) using Unsloth
* Integrated the **Unsloth** library to load Llama 3 in 4-bit precision, bypassing Google Colab's strict GPU RAM limitations.
* Coded a custom checkpoint-saving loop that pushes model weights directly to the Hugging Face Hub, protecting training progress from runtime timeouts.
* Executed training epochs on the preprocessed 50k recipe dataset.

### Week 4: Model Evaluation & Serialization (V8)
* Sequestered a 100-recipe test slice for objective validation (`Sligth data slicing for eval.ipynb`).
* Built evaluation scripts utilizing standard translation metrics (**BLEU, ROUGE-1/2/L, METEOR**) to measure structural correctness.
* Merged LoRA/PEFT adapters with the base model at 16-bit precision and saved the consolidated weights (`nd1490/ratatouille-llama3-3b-v8-MERGED`) to Hugging Face.

---

## 📈 Phase 2: Budget Constraints & Data Scaffolding (Weeks 5 – 8)
*Goal: Address the primary project mandate—imposing regional pricing and strict financial budgets—by separating mathematical calculations from language modeling.*

### Week 5: Constraint Feasibility & Architectural Design
* Investigated LLM reasoning limits; confirmed that generative models hallucinate algebraic constraints (e.g., failing to keep a recipe mathematically under Rs. 200).
* Designed a hybrid microservices architecture: **SciPy Optimization Engine** (deterministic math) + **Llama 3** (generative text instructions).

### Week 6: Market Price Ingestion Pipeline
* Extracted and formatted real-world Indian agricultural market databases (`RecipeDB_general.csv` and `RecipeDB_instructions.csv`).
* Configured a dynamic CSV parser in Python to ingest regional Mandi prices (e.g., Delhi state) from remote storage.

### Week 7: Ingredient Mapping & Culinary Ratio Rules
* Developed an ingredient categorization engine using token grouping (classifying ingredients into Proteins, Bases, Carbs, Sweets, etc.).
* Translated cooking principles into inequality guidelines (e.g., in curries, ensuring weight ratios satisfy: $\text{Weight of Protein} \ge \text{Weight of Base Sauce}$).

### Week 8: Linear Programming Optimization Formulation
* Defined the objective function: **Maximize total cooked portion size (gram weight)** for a specified family size.
* Formulated budget parameters: $\sum (\text{Ingredient Weight} \times \text{Regional Price}) \le \text{User Budget}$.
* Modeled nutritional boundary floors (minimum carbs, fats, and proteins) as mathematical constraints.

---

## 🧮 Phase 3: Math Engine Development & API Orchestration (Weeks 9 – 12)
*Goal: Develop the Python-based solver engine and construct the central web orchestrator using FastAPI.*

### Week 9: SciPy Optimization Solver Integration
* Coded the mathematical engine in the Python backend using `scipy.optimize.linprog`.
* Translated dynamic user inputs (list of ingredients, budget constraint, family size) into the solver's matrix coefficients ($A_{ub}, b_{ub}, A_{eq}, b_{eq}$).

### Week 10: Solver Robustness & Fallback Mechanisms
* Implemented handling for "Solver Infeasibility" (e.g., when a user budget is too low to buy enough food for a family size).
* Built a tiered relaxation algorithm (**Super Fallback** and **Nuclear Fallback**) that programmatically scales down minimum portions and relaxes constraints until a mathematically valid solution is guaranteed.

### Week 11: Backend Setup (FastAPI)
* Scaffolded the backend orchestrator application (`api.py`).
* Set up CORS middlewares to allow secure requests from browser frontends.
* Implemented the `/health` endpoint to monitor connected microservices.

### Week 12: Backend-to-AI Handshake & Sanitization
* Integrated the `huggingface_hub.InferenceClient` and `gradio_client` to communicate with Hugging Face's serverless GPUs.
* Wrote parsing utilities to intercept Llama 3 output, stripping chat padding (e.g., "Here is your recipe!") and extracting clean markdown formatting.

---

## 🌐 Phase 4: Full-Stack Integration, V9 Evolution & Cloud Launch (Weeks 13 – 16)
*Goal: Build the React frontend, persist user histories via MongoDB, refine the model with V9, and deploy the entire system to production.*

### Week 13: Persistence Layer (MongoDB Atlas)
* Configured a cloud MongoDB Atlas cluster.
* Integrated the asynchronous `motor` driver into the FastAPI backend to ensure non-blocking database queries.
* Implemented "Username Tagging" to save and load user history cards (`POST /save-recipe`, `GET /my-recipes/{username}`) without complex signup friction.

### Week 14: Frontend Development (React + Vite)
* Developed a responsive Single Page Application (SPA) using React 18 and Vite.
* Designed a premium glassmorphic UI (`backdrop-filter: blur(12px)`) featuring user input forms, loading indicators (reflecting HF cold starts), a markdown recipe viewer, and an expandable history sidebar.

### Week 15: V9 Fine-Tuning & Multi-Model Architecture
* Executed model retraining for refined cost-constraint alignments (`RAT_V9_CONTINUE_TRAIN_ON_RECIPEDB.ipynb`).
* Deployed a new Hugging Face inference space (`nd1490/ratatouille-inference-v9`).
* Updated `api.py` to support **Dual-Client switching**, letting users toggle between the V8 (Default) and V9 (Optimized) models on the fly.

### Week 16: Cloud Deployment & Verification (Vercel & Render)
* Deployed the FastAPI orchestrator to **Render.com**, keeping secrets secure in `.env` config environments.
* Configured a unified `vercel.json` build specification to automatically build and compile the React SPA from the `/frontend` sub-directory onto **Vercel**.
* Completed end-to-end system validation: verified the mathematical solver outputs, checked the MongoDB collection writes, and ensured model generation works across desktop and mobile browsers.
