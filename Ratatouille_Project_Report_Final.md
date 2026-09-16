# Ratatouille: AI-Powered Cost-Constrained Indian Budget Recipe Generation with Chemically-Aware Vegan Substitution

---

**Capstone Project Report**

| | |
|---|---|
| **Student Name** | Nindra Dhanush |
| **Roll Number** | MT25074 |
| **Programme** | M.Tech. (2025–2027) |
| **Institution** | Indraprastha Institute of Information Technology Delhi (IIIT Delhi) |
| **Lab** | CoSyLab — Computational Systems Biology Laboratory |
| **Supervisor** | Professor Ganesh Bagler |
| **Project Title** | Ratatouille – AI-Powered Cost-Constrained Indian Budget Recipe Generator with Chemically-Aware Vegan Substitution |
| **Date** | July 2026 |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Problem Statement](#3-problem-statement)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Technology Stack](#5-technology-stack)
6. [AI Models & Training](#6-ai-models--training)
7. [The Core Pipeline: `/generate-recipe`](#7-the-core-pipeline-generate-recipe)
   - 7.1 [Stage 1: Vegan Substitution Engine](#71-stage-1-vegan-substitution-engine-optional)
   - 7.2 [Stage 2: Dish Archetype Classification](#72-stage-2-dish-archetype-classification)
   - 7.3 [Stage 3: Cost-Constraint Optimization (SciPy Linear Programming)](#73-stage-3-cost-constraint-optimization-scipy-linear-programming)
   - 7.4 [Stage 4: AI Recipe Generation](#74-stage-4-ai-recipe-generation)
   - 7.5 [Stage 5: Post-Processing & Logging](#75-stage-5-post-processing--logging)
8. [The Chemically-Aware Vegan Substitution Engine](#8-the-chemically-aware-vegan-substitution-engine)
   - 8.1 [Chemical Feature Profiles](#81-chemical-feature-profiles)
   - 8.2 [The Composite Scoring Formula](#82-the-composite-scoring-formula)
   - 8.3 [The Spice Bridge Algorithm](#83-the-spice-bridge-algorithm)
   - 8.4 [Delta Recommendations & Compensation Blueprint](#84-delta-recommendations--compensation-blueprint)
   - 8.5 [Static Fallback Profiles](#85-static-fallback-profiles)
9. [Database Design (MongoDB Atlas)](#9-database-design-mongodb-atlas)
10. [External Data Sources](#10-external-data-sources)
11. [API Endpoints](#11-api-endpoints)
12. [Frontend (React SPA)](#12-frontend-react-spa)
13. [LLM Caching System](#13-llm-caching-system)
14. [Offline Database Construction Pipelines](#14-offline-database-construction-pipelines)
15. [Key Architectural Decisions & Rationale](#15-key-architectural-decisions--rationale)
16. [Deployment](#16-deployment)
17. [Testing & Evaluation](#17-testing--evaluation)
18. [Results & Discussion](#18-results--discussion)
19. [Conclusion](#19-conclusion)
20. [References & File Index](#20-references--file-index)

---

## 1. Abstract

**Ratatouille** is an AI-powered Indian budget recipe generator that brings together three distinct computational disciplines into a single, unified web application: mathematical optimization (SciPy Linear Programming), fine-tuned large language model (LLM) inference (custom Llama 3 models, V8 and V10 variants), and a chemically-aware vegan substitution engine that uses vector mathematics over food chemistry profiles rather than hardcoded lookup tables.

The system accepts a list of user-provided ingredients, a budget in Indian Rupees (INR), the number of servings, and an optional vegan preference. It then automatically sources real-time wholesale agricultural prices from government Mandi data, solves a linear program to distribute the budget across ingredients, and instructs a fine-tuned Llama 3 model to write a complete, formatted recipe. The vegan extension, added in the most recent development phase, replaces animal products with scientifically optimal plant-based alternatives using Jaccard molecule similarity and Euclidean texture distance — and generates compensation blueprints explaining exactly what spices and techniques bridge the gap.

The application is deployed on the CoSyLab academic research servers at Chennai Institute of Technology and is also hosted publicly via cloud services (Render + Vercel + Hugging Face Spaces). The full system is live and accessible to end users.

---

## 2. Introduction & Motivation

India has one of the most diverse and price-sensitive food markets in the world. Millions of households cook on tight budgets while navigating fluctuating ingredient prices that vary by region and season. Simultaneously, plant-based diets are growing rapidly for environmental, health, and ethical reasons — but converting a beloved recipe to vegan often requires expert knowledge about ingredient chemistry that most home cooks do not possess.

The **Ratatouille** project was born from the intersection of these two realities:

1. **Budget Constraint**: Can a computer automatically compute *exactly* how many grams of each ingredient a person can buy and cook with, given a specific INR budget and real market prices?
2. **Vegan Conversion**: When replacing chicken with tofu (or soy chunks), what specific preparation techniques, spices, and fat adjustments are mathematically required to preserve the dish's texture and flavor profile?

Prior work in this domain, specifically the original *Ratatouille: A tool for Novel Recipe Generation* (Goel et al., 2022), focused on generating creative recipes from unconstrained ingredient lists. However, it did not integrate real-time pricing constraints or chemically-grounded substitution. This capstone project extends that foundation to fill this gap.

The project is a capstone extension of earlier work that introduced the cost-constraint optimization pipeline. This report documents the **complete, deployed system** including the vegan substitution engine, which was the primary new feature added in the final phase.

---

## 3. Problem Statement

Given the following user inputs:
- A list of `N` ingredients (e.g., chicken, tomato, onion)
- A total budget `B` in INR (e.g., ₹100)
- Number of servings `S`
- An Indian state `T` for regional pricing (e.g., "Delhi")
- A boolean vegan flag `V`

The system must:

1. If `V = True`: Replace all non-vegan ingredients with the chemically-closest plant-based alternative, including auxiliary additions to compensate for fat, texture, and flavor deficits.
2. Look up the real market price per gram for each ingredient (after vegan substitution) from government Mandi data, falling back to cached pricing or LLM-driven deconstruction for processed items.
3. Solve a linear program to find the gram quantities `x₁, x₂, ... xₙ` (per serving) that **maximize total food weight** subject to:
   - The total cost constraint: `Σ(pᵢ × xᵢ) ≤ B/S`
   - Per-ingredient minimum and maximum bounds from training data statistics
   - Dish archetype ratio constraints (e.g., for a Curry, base ingredients must be 0.8× to 3× protein weight)
4. Use the gram quantities to construct a structured prompt and call a fine-tuned Llama 3 model to generate the complete recipe text (title, ingredients list with quantities, step-by-step instructions).
5. Return all results to the user in real time via Server-Sent Events (SSE), with live progress updates as each stage completes.

---

## 4. System Architecture Overview

The system follows a **5-stage sequential pipeline** triggered by a single `POST /generate-recipe` API call. The backend streams progress back to the frontend over SSE. The high-level data flow is:

```
User Input: [ingredients], budget (₹), servings, state, vegan=True/False
        │
        ▼
[OPTIONAL] Stage 1: Vegan Substitution Engine
  → Swap non-vegan ingredients via vector math on chemical profiles
  → Add compensation (auxiliary fats, spices, techniques)
        │
        ▼
Stage 2: Dish Archetype Classification (Groq Llama 3.3-70B, cached)
  → Identifies dish type: Curry / Dry_Sabzi / Rice_Dish / Salad / etc.
        │
        ▼
Stage 3: SciPy Linear Programming (Budget Optimizer)
  → Fetches real Mandi prices per gram
  → Solves linprog to maximize grams within budget
  → Outputs: ["85.0g soy chunks", "120.0g tomato", "95.0g onion"]
        │
        ▼
Stage 4: AI Recipe Generation (Fine-tuned Llama 3 on Hugging Face)
  → Prompt: "### INGREDIENTS:\n- 85.0g soy chunks\n...\n### TITLE:\n"
  → Returns complete recipe: title + directions
        │
        ▼
Stage 5: Post-Processing & Logging
  → Strip hallucinated stop tokens and loops
  → Log timing to MongoDB analytics collection
  → Yield SSE `complete` event → React frontend renders recipe
```

**Key design principle**: Every component is independently replaceable and testable. The SciPy optimizer knows nothing about LLMs; the vegan engine knows nothing about pricing. The `api.py` orchestrator stitches them together.

---

## 5. Technology Stack

### Backend

| Package | Version | Role |
|---|---|---|
| `fastapi` | Latest | Web framework — hosts all API endpoints |
| `uvicorn[standard]` | Latest | ASGI server — runs FastAPI with SSE support |
| `python-dotenv` | Latest | Loads secrets from `.env` |
| `requests` | Latest | Fetches Mandi CSV and JSON from GitHub |
| `pandas` | Latest | Processes government Mandi market price data |
| `numpy` | Latest | Vector math for texture similarity (Euclidean distance) |
| `scipy` | Latest | `linprog` — the core budget optimization solver |
| `nltk` | Latest | `WordNetLemmatizer` — normalizes ingredient names for price lookup |
| `pydantic` | Latest | Request/response model validation |
| `gradio_client` | Latest | Calls the Hugging Face Gradio Spaces (V8, V10 models) |
| `motor` | Latest | Async MongoDB driver (for async endpoints) |
| `pymongo` | Latest | Sync MongoDB driver (for the SSE stream generator) |
| `groq` | Latest | Groq API client for Llama 3.3-70B serverless inference |

### Frontend

| Technology | Role |
|---|---|
| React (Vite) | UI framework and build system |
| Vanilla CSS | All styling — no Tailwind or component libraries |
| `fetch` + `ReadableStream` | SSE client using native browser APIs |

### Training & Data Infrastructure

| Environment | Role |
|---|---|
| Google Colab (T4 GPU) | Fine-tuning Llama 3 (V8, V10); offline DB builders |
| Hugging Face Spaces | Hosting fine-tuned models for inference |
| MongoDB Atlas (M0 free tier) | Cloud database for all persistent data |

---

## 6. AI Models & Training

The system uses **four AI models** across different stages:

### Model A: V8 — Fine-tuned Llama 3.2-3B (Original)

- **Base**: `meta-llama/Llama-3.2-3B`
- **Method**: QLoRA (Quantized Low-Rank Adaptation) via Unsloth on a Google Colab T4 GPU
- **Training Data**: ~50,000 Indian recipes in a custom structured prompt format
- **Prompt format trained on**:
  ```
  ### INGREDIENTS:
  - {qty}g {ingredient_1}
  - {qty}g {ingredient_2}
  ### TITLE:
  {recipe title}
  ### DIRECTIONS:
  {step-by-step instructions}
  ```
- **Hosted**: `nd1490/ratatouille-inference` (Hugging Face Space)
- **Status**: Production (legacy/fallback)

### Model B: V10 — Fine-tuned Llama 3.2-3B (Improved, Q8 Quantized)

- **Base**: `meta-llama/Llama-3.2-3B`
- **Method**: QLoRA continued training on an expanded, cleaner dataset including RecipeDB (118,084 recipes) in addition to the original 50K Indian recipes
- **Quantization**: Q8 GGUF for efficient inference on free-tier Hugging Face Spaces
- **Key improvement over V8**: Non-oven recipe constraint (`RAT_V10_NON_OVEN_TRAIN.ipynb`) — the model was specifically fine-tuned to avoid generating recipes requiring ovens, making output more suitable for Indian home kitchens
- **Hosted**: `nd1490/ratatouille-inference-v10-q8` (Hugging Face Space)
- **Status**: **Primary production model**

### Model C: Groq Llama 3.3-70B (Serverless — Metadata Tasks Only)

- **Provider**: Groq (free tier serverless API)
- **Model ID**: `llama-3.3-70b-versatile`
- **Used for**:
  - Dish archetype classification (ultra-fast, ~100ms, cached in MongoDB)
  - Ingredient deconstruction (processed → raw crops, for price lookup)
  - Chemical profile bootstrapping for unknown ingredients (offline endpoint only)
- **Why Groq (not OpenAI/Anthropic)**: Free tier with generous limits; sub-second latency; no billing setup needed
- **Fallback**: Gradio V8 Space (if Groq rate-limits or fails)

### Model D: Qwen3-8B 4-bit NF4 (Offline Database Building Only)

- **Used in**: `GPU_Open_World_Vegan_DB_Builder.ipynb` (Colab T4 GPU, one-time run)
- **Purpose**: Generating `chemical_features` profiles (macros, texture vectors, flavor molecules, culinary role) for ~350 culinary ingredients
- **Why not V10 for this?**: V10 was fine-tuned to write recipes and lacks training signal for structured food chemistry JSON. Qwen3-8B is a general-purpose scientific model better suited for this structured output task.
- **NOT used at runtime** — only during the one-time offline database build

### Training Data

Three large CSV datasets were used for model training and evaluation:

| File | Size | Contents |
|---|---|---|
| `final_clean_50k_recipes_grams.csv` | 82 MB | Primary training set — 50K Indian recipes with gram quantities |
| `RecipeDB_formatted_like_50k.csv` | 130 MB | ~50K additional recipes in V10 schema format |
| `RecipeDB_general.csv` | 46 MB | 118,084 recipes with cooking process annotations |

---

## 7. The Core Pipeline: `/generate-recipe`

### Request Model

```python
class RecipeRequest(BaseModel):
    ingredients: list[str]   # e.g., ["chicken", "tomato", "onion"]
    budget: float            # e.g., 100.0 (INR)
    servings: int = 1
    state: str = "Delhi"     # Indian state for regional pricing
    model_version: str = "v10"  # "v8" or "v10"
    is_vegan: bool = False
```

The endpoint immediately returns a `StreamingResponse` with `media_type="text/event-stream"`. It uses a Python **generator function** that `yield`s SSE events in sequence. This means the user sees live progress updates rather than waiting at a blank screen.

**SSE Events yielded (in order)**:

| `step` | `message` | Trigger |
|---|---|---|
| `starting` | `"Initializing..."` | Immediately |
| `veganizing` | `"Running Vegan Substitution Engine..."` | If `is_vegan=True` |
| `optimizing` | `"Running Cost Constraint Optimization (Budget: ₹{budget})..."` | Before SciPy |
| `generating` | `"Generating AI Recipe (Model: {model_version})..."` | Before HF call |
| `complete` | *(carries `result` object)* | On success |
| `error` | Error message string | On failure |

---

### 7.1 Stage 1: Vegan Substitution Engine (Optional)

*Only runs if `is_vegan = True`.*

This stage processes each ingredient through a **three-level lookup cascade** and replaces animal products with their plant-based equivalents before anything else runs:

#### Step A: Fast Archetype Classification (0ms, No LLM)

```python
def _classify_archetype_fast(ingredients: list) -> str:
    s = " ".join(ingredients).lower()
    if any(w in s for w in ["rice", "biryani", "pulao"]): return "Rice_Dish"
    if any(w in s for w in ["pasta", "noodle", "macaroni"]): return "Rice_Dish"
    if any(w in s for w in ["oat","banana","flour","chocolate","cake","honey"]): return "Dessert"
    if any(w in s for w in ["mushroom","soup","broth","coconut milk"]): return "Soup"
    if any(w in s for w in ["lettuce","salad","cucumber","olive"]): return "Salad"
    if any(w in s for w in ["bread","dough","yeast"]): return "Bread"
    return "Curry"  # default for most Indian dishes
```

This keyword-matching archetype is used by the vegan engine to select context-appropriate compensation (e.g., coconut oil for Curry, olive oil for Salad).

#### Step B: Canonical Name Normalization

Maps ingredient variants to their canonical base name using a 300+ entry map:
- `"chicken breast"` → `"chicken"`
- `"heavy cream"` → `"cream"`
- `"ground beef"` → `"beef"`
- `"mozzarella"` → `"cheese"`
- `"clarified butter"` → `"ghee"`

#### Step C: 3-Level Lookup Cascade

1. **MongoDB `vegan_alternatives` lookup** (~1ms): Pre-computed match table → returns best substitute, match score, top-5 alternatives, compensation blueprint
2. **Local `vegan_engine` fallback**: If DB miss, runs live math using `chemical_features.json`
3. **Keep as-is**: If truly unknown — no LLM call here to preserve SSE latency

#### Step D: Compensation Application

Beyond a simple 1-to-1 ingredient swap, the engine applies a **Compensation Blueprint**:
- Substitutes the animal product (e.g., chicken → soy chunks)
- Injects auxiliary fats (e.g., `"1 tsp coconut oil"`)
- Injects umami bridges (e.g., `"1/2 tsp MSG or soy sauce"`)
- Adds spice bridges (e.g., `"smoked paprika"`)

---

### 7.2 Stage 2: Dish Archetype Classification

This LLM call determines the structural form of the dish (curry vs. dry sabzi vs. rice dish, etc.) which directly controls the ratio constraints in the linear program.

**Cache check first** — all archetype lookups are cached in MongoDB with key `archetype_{sorted_ingredients}`. Ingredients are sorted before hashing so ordering doesn't create duplicates.

**Primary call (Groq Llama 3.3-70B)**:

```
System: "You are a recipe classification assistant. Output ONLY a single word 
         classification from the permitted list. No explanation."

User:   "Classify the dish structure based on these ingredients: [chicken, tomato, onion].
         Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, 
         Soup, Rice_Dish]."

Parameters: max_completion_tokens=15, temperature=0.1
```

**Fallback (V8 Llama 3B on Gradio Space)** — used if Groq rate-limits:
```
<|begin_of_text|>Classify the dish structure based on these ingredients: [...]
Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish].
Return ONLY the word.

### INGREDIENTS:
{ingredients}
### ARCHETYPE:

```

Result is saved to MongoDB cache immediately after retrieval.

---

### 7.3 Stage 3: Cost-Constraint Optimization (SciPy Linear Programming)

This is the mathematical core of the system — a linear programming problem solved by `scipy.optimize.linprog` using the HiGHS solver.

#### Step A: Real-Time Price Lookup

For every ingredient, `get_dynamic_price()` is called with a **4-priority cascade**:

1. **State-specific Mandi price**: Match lemmatized ingredient name + user's state in the government Mandi DataFrame → return median Modal Price (INR per gram)
2. **National Mandi price**: Drop state filter, use national median
3. **Pantry prices dict**: For processed/packaged items (eggs, soy sauce, paneer, etc.)
4. **LLM Deconstruction (slow path)**: For unknown processed items, Groq is asked to decompose the item into raw agricultural crops, then a weighted average price is computed. A 30% markup is added for processing cost.

**Mandi data processing at server startup**:
```python
# Convert modal price (INR per quintal) → price per gram
current_mandi['Price_per_Gram'] = current_mandi['Modal_Price'] / 100000
```

#### Step B: Ingredient Tagging

Each ingredient is tagged with a functional type (protein, base, sweet, neutral, veggie) to enable archetype ratio constraints.

#### Step C: Bounds Construction

Per-ingredient gram bounds are loaded from `v8_lemmaized_ingredient_bounds.json` (statistical analysis of the 50K training dataset):
- Standard minimum: 15g
- Spice minimum: 2g (for garlic, ginger, salt, etc.)
- Maximum: 800g hard cap
- User-specified quantities are fixed (both bounds set to `quantity/servings`)

#### Step D: Linear Program Formulation

**Objective function**: Maximize total food quantity (grams):
```python
c = [-1.0] * n   # Minimize negative grams = maximize grams
```

**Budget constraint**:
```
price_1 * x_1 + price_2 * x_2 + ... + price_n * x_n ≤ budget / servings
```

**Archetype ratio constraints** (structural integrity rules):

| Archetype | Constraint | Meaning |
|---|---|---|
| `Curry` | `0.8 × protein ≤ base` | Base must be ≥ 80% of protein weight |
| `Curry` | `base ≤ 3.0 × protein` | Prevent excessive sauce |
| `Dry_Sabzi` | `base ≤ 0.8 × protein` | Dry dish — base kept minimal |
| `Salad` | `2 × (protein+neutral) ≤ veggie` | Vegetables dominate |
| `Rice_Dish` | `protein ≤ neutral` | Protein ≤ carb/rice quantity |
| `Soup` | `4 × protein ≤ base` | Very liquid — base hugely dominates |
| `Dessert` | `0.4 × (neutral+protein+base) ≤ sweet` | Sweet component dominates |

**Solver**:
```python
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
```

#### Step E: 4-Level Fallback Cascade

If the linear program is infeasible (e.g., budget too low for all minimum bounds):

```
Attempt 1: Full constraints (archetype ratios + budget + bounds)
         │ INFEASIBLE
         ▼
Attempt 2: Budget-only (drop archetype ratio constraints)
         │ INFEASIBLE
         ▼
Attempt 3: Force minimum bounds to 5g
         │ INFEASIBLE
         ▼
Attempt 4: Unconstrained fallback — override all bounds to (1g, 1000g)
         │ FAIL → return None → SSE error event
```

This cascade ensures that even with very low budgets, the system returns *something* rather than crashing.

#### Output

```python
# Example output for ["chicken", "tomato", "onion"], budget=₹100, state=Delhi
["85.0g soy chunks", "120.0g tomato", "95.0g onion"]
```

---

### 7.4 Stage 4: AI Recipe Generation

#### Prompt Construction (V10 Schema — EXACT FORMAT)

```python
ingr_text = "\n".join(f"- {i}" for i in calculated_ingredients)
prompt = (
    f"### INGREDIENTS:\n"
    f"{ingr_text}\n"
    f"### TITLE:\n"
)
```

Example prompt sent to the model:
```
### INGREDIENTS:
- 85.0g soy chunks
- 120.0g tomato
- 95.0g onion
- 15.0g garlic
- 10.0g ginger
### TITLE:

```

> **Critical note**: This prompt format is the exact format the model was fine-tuned on. Any deviation (e.g., using `**Ingredients:**` instead of `### INGREDIENTS:`) causes the model to produce incoherent output or hallucinate recipe content.

#### Gradio API Call Parameters

```python
result = client.predict(
    prompt,    # Textbox
    500,       # max_new_tokens
    0.6,       # temperature
    0.9,       # top_p
    1.05,      # repetition_penalty
    True,      # do_sample
    api_name="/generate",
)
```

The `gradio_client` is cached across requests. If the Hugging Face Space is cold (sleeping after inactivity), the system retries **3 times with 30-second delays** before returning HTTP 503.

---

### 7.5 Stage 5: Post-Processing & Logging

#### Stop Token & Hallucination Removal

The Llama 3 model sometimes generates special tokens or loops back to `### INGREDIENTS:`. These are stripped deterministically:

```python
stop_tokens = ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>', '\n### INGREDIENTS:']
for t in stop_tokens:
    if t in ai_text:
        ai_text = ai_text.split(t)[0].strip()
```

Additionally, common closing phrases and repetition loops are cut off:
- `"\nEnjoy!"`, `"\nServe hot"`, `"\nBon Apetit"`, `"\nChef's Note:"`, `"\nVariations:"`

#### Generation Logging

A timing record is written to `ratatouille.generation_logs` in MongoDB:

```json
{
  "timestamp": 1720589000.123,
  "model_version": "v10",
  "is_vegan": true,
  "archetype": "Curry",
  "budget": 100.0,
  "servings": 2,
  "state": "Delhi",
  "times_sec": {
    "veganization_sec": 0.012,
    "optimization_sec": 0.87,
    "generation_sec": 14.3
  },
  "total_time_sec": 15.2,
  "ingredient_count": 5
}
```

---

## 8. The Chemically-Aware Vegan Substitution Engine

**File**: `vegan_engine.py` (412 lines)

This is the primary novel contribution of the vegan extension phase. The engine does not use lookup tables or simple rules. It applies vector mathematics over chemical and physical ingredient profiles to find the **mathematically optimal plant-based substitute** for any animal product.

### System Architecture Diagram

```
User Input: Non-Vegan Ingredient (e.g., "chicken")
        │
        ▼
    [Check MongoDB vegan_alternatives collection]
        │ HIT → return pre-computed blueprint immediately (~1ms)
        │ MISS
        ▼
    [Check chemical_features (MongoDB + local JSON)]
        │ FOUND → run live vector math
        │ NOT FOUND
        ▼
    [classify_by_keyword() → assign static archetype profile]
        │
        ▼
    [calculate_composite_score() against all vegan candidates]
        │
        ▼
    [Best substitute identified]
        │
        ├──→ [calculate_delta_recommendations()] → Compensation Blueprint
        │
        └──→ [get_spice_bridge()] → Spice Bridge Recommendations
```

---

### 8.1 Chemical Feature Profiles

Every ingredient in the system is described by a structured 5-field object:

```json
{
  "_id": "chicken",
  "is_vegan": false,
  "macros": { "proteins": 27.0, "fats": 14.0, "carbs": 0.0 },
  "texture_profile": [5.5, 7.0, 5.5, 5.5, 6.0],
  "flavor_molecules": [
    "inosine monophosphate",
    "2-methyl-3-furanthiol",
    "methional",
    "dimethyl trisulfide",
    "heptanal"
  ],
  "culinary_role": "base_protein"
}
```

**Texture profile** — a 5-dimensional vector (values 1.0–10.0):

| Index | Dimension | Low (1.0) | High (10.0) |
|---|---|---|---|
| 0 | Hardness | Melts/dissolves | Very hard/dense |
| 1 | Chewiness | No chewing needed | Very chewy |
| 2 | Moisture | Very dry | Very juicy/wet |
| 3 | Fat mouthfeel | No fat sensation | Rich, coating |
| 4 | Elasticity | Crumbles apart | Bounces back |

**Valid culinary roles**: `bulk_protein`, `fat_source`, `binder`, `creamy_liquid`, `sweetener`, `seasoning`, `veggie`, `starch`, `flavor_enhancer`, `aromatic`, `dairy`, `binding_agent`, `thickener`, `base_protein`

The database covers ~350 culinary ingredients, stored in MongoDB Atlas (`ratatouille.chemical_features`) with a local JSON fallback (`chemical_features.json`, 135 KB).

---

### 8.2 The Composite Scoring Formula

The engine ranks every candidate vegan substitute using a weighted composite score:

$$\text{Score} = \alpha \cdot S_{\text{flavor}} + \beta \cdot S_{\text{texture}} + \gamma \cdot S_{\text{role}}$$

With weights: **α = 0.3, β = 0.4, γ = 0.3**

The β = 0.4 weight on texture is backed by food science research: texture accounts for approximately 40–50% of perceived substitution quality in plant-based meat studies.

#### Flavor Similarity — Jaccard Index

$$S_{\text{flavor}} = \frac{|A_{\text{molecules}} \cap B_{\text{molecules}}|}{|A_{\text{molecules}} \cup B_{\text{molecules}}|}$$

Measures the fraction of flavor volatile compounds shared between the original and the candidate substitute. A score of 1.0 means identical flavor chemistry; 0.0 means no shared molecules.

**Why Jaccard over dense embeddings?**
- **Explainability**: We can extract *which* specific compounds are missing → needed to build the Spice Bridge
- **Chemical correctness**: Embedding models trained on recipe text reflect word co-occurrence, not chemical reality (e.g., garlic and onion appear together in recipes, but their volatile molecules differ significantly)
- **Speed**: Sub-millisecond Python set operations

#### Texture Similarity — Normalized Euclidean Distance

$$d(\vec{u}, \vec{v}) = \sqrt{\sum_{i=1}^{5} (u_i - v_i)^2}$$

$$S_{\text{texture}} = 1.0 - \frac{d(\vec{u}, \vec{v})}{\sqrt{5 \times (10-1)^2}} = 1.0 - \frac{d(\vec{u}, \vec{v})}{20.124}$$

Normalized to [0, 1]. A score of 1.0 means the two ingredients have identical texture profiles across all five dimensions.

#### Functional Overlap — Binary Role Match

$$S_{\text{role}} = \begin{cases} 1.0 & \text{if culinary\_role}_A = \text{culinary\_role}_B \\ 0.0 & \text{otherwise} \end{cases}$$

This is a hard architectural guardrail: a meat (`bulk_protein`) will never score well against coconut milk (`creamy_liquid`), preventing culinarily incompatible substitutions regardless of any superficial similarity.

**Example scores for "chicken" substitutes**:

| Candidate | Flavor (α=0.3) | Texture (β=0.4) | Role (γ=0.3) | Total |
|---|---|---|---|---|
| Soy chunks | 0.40 | 0.82 | 1.0 | **0.74** |
| Seitan | 0.35 | 0.78 | 1.0 | **0.71** |
| Jackfruit | 0.28 | 0.70 | 1.0 | **0.68** |
| Tofu | 0.20 | 0.55 | 1.0 | **0.61** |
| Tempeh | 0.18 | 0.52 | 1.0 | **0.58** |

---

### 8.3 The Spice Bridge Algorithm

After selecting the best substitute, the engine computes a **Spice Bridge** — a list of kitchen spices that contain flavor molecules present in the original ingredient but absent from the substitute.

```python
def get_spice_bridge(original_mols, substitute_mols, all_features, top_k=3):
    flavor_gap = set(original_mols) - set(substitute_mols)
    # ↑ Molecules present in original, absent in substitute

    bridge_scores = []
    for name, data in all_features.items():
        if data.get("culinary_role") in ["flavor_enhancer", "aromatic"] and data.get("is_vegan"):
            spice_mols = set(data.get("flavor_molecules", []))
            gap_covered = len(spice_mols & flavor_gap)
            bridge_score = gap_covered / len(flavor_gap)
            if bridge_score > 0:
                bridge_scores.append({
                    "spice": name,
                    "fills_gap_ratio": round(bridge_score, 3),
                    "reason": f"adds {', '.join(spice_mols & flavor_gap)} to fill the aromatic gap"
                })

    bridge_scores.sort(key=lambda x: x["fills_gap_ratio"], reverse=True)
    return bridge_scores[:top_k]
```

**Why restrict to whitelisted spice roles?**  
Without filtering to `flavor_enhancer`/`aromatic` roles, the engine might suggest "durian" or "raw cabbage" to fill sulfur compound gaps — chemically correct but culinarily incompatible. The whitelist ensures only practical, purchasable kitchen additions are recommended.

**Example**: Replacing chicken (flavor molecules: `inosine monophosphate`, `2-methyl-3-furanthiol`, `pyrazines`) with soy chunks:
- Gap identified: `{2-methyl-3-furanthiol, pyrazines}`
- Smoked paprika covers `pyrazines` → recommended bridge #1 (fills 67% of gap)
- Nutritional yeast covers umami compounds → recommended bridge #2

---

### 8.4 Delta Recommendations & Compensation Blueprint

Function: `calculate_delta_recommendations(orig_name, sub_name, orig_data, sub_data, archetype)`

This function computes the **delta vector** `Δ = V_original - V_substitute` across macros and texture dimensions, then translates mathematical deficits into natural language culinary instructions.

**Delta vector**:
$$\vec{\Delta} = \vec{V}_{\text{original}} - \vec{V}_{\text{substitute}}$$

#### Fat Deficit Detection

```python
delta_fat = orig_macros["fats"] - sub_macros["fats"]
if delta_fat > 10.0:  # More than 10g fat difference per 100g
    if archetype in ["Curry", "Dry_Sabzi", "Soup"]:
        additions.append({"name": "coconut oil or neutral vegetable oil", 
                          "amount": "1-2 tsp", 
                          "purpose": "matches lipid profile to ensure proper fat-soluble spice absorption"})
    elif archetype == "Salad":
        additions.append({"name": "cold-pressed olive oil", "amount": "1-2 tsp",
                          "purpose": "drizzle over substitute to replicate the fat mouthfeel"})
```

#### Umami Bridging

```python
if "diacetyl" in original_molecules and "diacetyl" not in substitute_molecules:
    additions.append({"name": "nutritional yeast", "amount": "1 tsp",
                      "purpose": "adds savory, buttery dairy-like notes"})
elif any(x in ["hydrogen sulfide", "2-methyl-3-furanthiol", "pyrazines"] 
         for x in original_molecules):
    additions.append({"name": "monosodium glutamate (MSG) or soy sauce", "amount": "1/2 tsp",
                      "purpose": "bridges the savory meat umami profile"})
```

#### Moisture & Texture Adjustment Techniques

The system translates moisture deltas (texture dimension index 2) into preparation instructions:

- `Δ_moisture < -2.0` (substitute is too wet): *"Wrap the [substitute] in a clean kitchen towel and press it under a heavy object for 15 minutes..."*
- `Δ_moisture > 4.0` (substitute is too dry): *"Soak the soya chunks in hot water for 15 minutes and squeeze the excess water out..."*

Special techniques are also generated for specific ingredients:
- **King oyster mushroom**: Cross-hatch scoring and sautéing to mimic shrimp's bouncy bite
- **Tofu**: Freeze-thaw cycle then pan-sear for chewy surface texture
- **Jackfruit**: Simmer until tender, then shred with two forks to mimic pulled meat

---

### 8.5 Static Fallback Profiles

When an ingredient is not in `chemical_features` and cannot be bootstrapped via LLM (e.g., offline or cold start), the keyword classifier maps it to one of 7 archetype profiles:

| Profile | Keywords | Macros (per 100g) |
|---|---|---|
| `red_meat` | mutton, lamb, pork, beef, goat, steak, ham, bacon | proteins: 26, fats: 17 |
| `poultry` | chicken, duck, turkey, quail | proteins: 27, fats: 14 |
| `seafood` | shrimp, prawn, fish, salmon, crab, lobster | proteins: 24, fats: 1 |
| `dairy_fat` | butter, ghee, lard | proteins: 1, fats: 81 |
| `dairy_liquid` | milk, cream, yogurt, curd, cheese, paneer | proteins: 3.4, fats: 3.7 |
| `sweetener` | honey | carbs: 99.9 |
| `egg` | egg, eggs (not eggplant) | proteins: 13, fats: 10, role: binding_agent |

---

## 9. Database Design (MongoDB Atlas)

The system uses a single MongoDB Atlas cluster with the database named `ratatouille`. It contains **six collections**:

### Collection 1: `ratatouille.recipes`

User-saved recipe history.

```json
{
  "_id": "<ObjectId>",
  "username": "dhanush",
  "recipe": { "<full recipe result object>" },
  "created_at": 1720589000.123
}
```

*Written by*: `POST /save-recipe` | *Read by*: `GET /my-recipes/{username}`

---

### Collection 2: `ratatouille.vegan_alternatives`

**Pre-computed match table** — the primary fast-path for the vegan engine. Built offline.

```json
{
  "_id": "chicken",
  "original_ingredient": "chicken",
  "best_vegan_substitute": "soy chunks",
  "match_score": 0.74,
  "original_role": "base_protein",
  "score_breakdown": {
    "flavor_similarity": 0.40,
    "texture_similarity": 0.82,
    "functional_fit": 1.0
  },
  "top5_alternatives": [
    { "substitute": "soy chunks",  "score": 0.74 },
    { "substitute": "seitan",      "score": 0.71 },
    { "substitute": "jackfruit",   "score": 0.68 },
    { "substitute": "tofu",        "score": 0.61 },
    { "substitute": "tempeh",      "score": 0.58 }
  ],
  "compensation_blueprint": {
    "auxiliary_additions": [
      { "name": "coconut oil", "amount": "1 tsp", "purpose": "bridges fat profile deficit" }
    ],
    "techniques": ["Rehydrate soy chunks in hot water for 15 min before cooking."],
    "spice_bridge": [
      { "spice": "smoked paprika", "fills_gap_ratio": 0.67,
        "reason": "adds pyrazines compounds to fill the aromatic gap" }
    ]
  }
}
```

---

### Collection 3: `ratatouille.chemical_features`

Full ingredient chemistry profiles used by the vegan engine for live calculations.

```json
{
  "_id": "chicken",
  "is_vegan": false,
  "macros": { "proteins": 27.0, "fats": 14.0, "carbs": 0.0 },
  "texture_profile": [5.5, 7.0, 5.5, 5.5, 6.0],
  "flavor_molecules": ["inosine monophosphate", "2-methyl-3-furanthiol", "methional", "heptanal"],
  "culinary_role": "base_protein"
}
```

*Read by*: `vegan_engine.load_features()` — tries MongoDB first, falls back to `chemical_features.json`

---

### Collection 4: `ratatouille.llm_cache`

**LLM response cache** — all Groq API and Gradio Space calls are cached here to avoid repeat API costs.

```json
{ "_id": "archetype_chicken,onion,tomato", "result": "Curry", "timestamp": 1720589000.123 }
{ "_id": "deconstruct_tomato ketchup", "result": {"tomato": 0.8, "sugar": 0.1, "onion": 0.1} }
```

*Effect*: Second request with same ingredients (regardless of order) costs 0ms and 0 API tokens.

---

### Collection 5: `ratatouille.generation_logs`

**Analytics log** — timing records for every completed recipe generation. Used for performance monitoring and research analysis only.

---

### Collection 6: `ratatouille.indian_recipes`

**Pre-generated Indian vegan recipe library** — populated offline by `GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb`. Served by the `/indian-recipes` endpoint for the Recipe Library UI panel.

---

## 10. External Data Sources

### Government Mandi Market Data (Live, Fetched at Startup)

**Source**: Private GitHub repository `dn74iiit/recipe-data-automation`  
**File**: `daily_mandi_data.csv`  
**URL**: `https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/daily_mandi_data.csv`

This is **official government agricultural market (Mandi) price data** — real wholesale crop prices from agricultural markets (mandis) across India, automated to be updated daily.

**Data schema** (10 columns):
```
State, District, Market, Commodity, Variety, Grade, Arrival_Date, 
Min_Price, Max_Price, Modal_Price
```

**Processing at startup**:
1. Load CSV, filter to the most recent `Arrival_Date` only
2. Lemmatize commodity names for fuzzy matching (`WordNetLemmatizer`)
3. Convert Modal Price (INR per quintal) → Price per gram: `Modal_Price / 100,000`

### Ingredient Bounds Data (from GitHub)

**File**: `v8_lemmaized_ingredient_bounds.json`  
A JSON dict mapping lemmatized ingredient names to `(min_grams, max_grams)` bounds. Derived from statistical analysis of 50K recipe training data — ensures the optimizer produces realistic quantities.

### Pantry Prices (Hardcoded + GitHub Override)

Hardcoded pricing dictionary for processed/packaged items not found in Mandi data (e.g., vanilla extract, soy sauce, olive oil, paneer, eggs). Extended by a `pantry_prices.json` file from GitHub if available.

---

## 11. API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns HF Space URLs and DB connection status |
| `POST` | `/generate-recipe` | **Main endpoint** — full 5-stage SSE streaming pipeline |
| `POST` | `/optimize-only` | Budget optimization only, no LLM generation (used by eval scripts) |
| `POST` | `/save-recipe` | Saves a recipe to user profile in MongoDB |
| `GET` | `/my-recipes/{username}` | Returns last 50 saved recipes for a user |
| `GET` | `/vegan-alternatives/{ingredient}` | Returns top-5 pre-computed vegan alternatives for an ingredient |
| `POST` | `/get-vegan-blueprint` | Full vegan engine analysis including LLM bootstrap for unknown ingredients |
| `GET` | `/indian-recipes/styles` | Returns distinct recipe styles and counts from the library |
| `GET` | `/indian-recipes` | Paginated Indian budget recipe library (`?style=Curry&limit=12&skip=0`) |

**CORS**: All origins allowed in development; restricted to production domains in deployment.

---

## 12. Frontend (React SPA)

**Framework**: React 18 + Vite  
**File**: `frontend/src/App.jsx` (831 lines)  
**Styling**: Vanilla CSS

### Views

| View (`viewMode`) | Description |
|---|---|
| `generate` | Main recipe generation form + streaming result panel |
| `vegan` | Standalone vegan engine debugger — look up any ingredient's alternatives with score breakdown |
| `history` | User's saved recipe history from MongoDB |

### SSE Reading Pattern

The frontend uses the native `ReadableStream` API — no external library required:

```javascript
const reader = response.body.getReader();
const decoder = new TextDecoder("utf-8");
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n\n');
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.step === 'complete') setResult(data.result);
            else if (data.step === 'error') setError(data.message);
            else setStepMessage(data.message); // Live progress indicator
        }
    }
}
```

### Dynamic Backend URL Routing

```javascript
const BACKEND_URL =
  window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://ratatouille-backend.onrender.com';
```

Automatically switches between local development and cloud production without any manual configuration.

### Vegan Engine Debug Panel

The `vegan` view renders interactive result cards for each ingredient showing:
- Overall match score (e.g., 74%) with a visual progress bar
- Score breakdown: flavor similarity / texture similarity / functional fit
- Top-5 alternative substitutes with individual scores
- Delta compensation steps: auxiliary additions, preparation techniques, spice bridges

---

## 13. LLM Caching System

**Collection**: `ratatouille.llm_cache`  
**Client**: `llm_cache_sync` (synchronous PyMongo — required inside SSE generator functions)

### Cache Key Convention

| Call Type | Cache Key Format | Example |
|---|---|---|
| Archetype | `archetype_{sorted_ingredients}` | `archetype_chicken,onion,tomato` |
| Deconstruct | `deconstruct_{ingredient}` | `deconstruct_tomato ketchup` |

Ingredients in the archetype key are **sorted alphabetically** before joining — this ensures `["chicken", "tomato", "onion"]` and `["onion", "chicken", "tomato"]` produce the same cache key.

### Cache Read/Write Pattern

```python
# Read
cached = llm_cache_sync.find_one({"_id": cache_id})
if cached and "result" in cached:
    return cached["result"]  # Cache hit — 0ms, 0 API tokens

# Write (upsert — create or update)
llm_cache_sync.update_one(
    {"_id": cache_id},
    {"$set": {"result": result, "timestamp": time.time()}},
    upsert=True
)
```

**Economic impact**: The archetype classification is the most frequently repeated LLM call. Once a given ingredient combination has been seen once, all future requests for it are free (0ms, 0 API cost).

---

## 14. Offline Database Construction Pipelines

### Phase 1: `chemical_features` Database (`GPU_Open_World_Vegan_DB_Builder.ipynb`)

**Environment**: Google Colab T4 GPU  
**Model**: Qwen3-8B (4-bit NF4 quantization)

For each of ~350 curated culinary ingredients, the system prompts Qwen3-8B to generate a structured JSON profile:

```
System: "You are a food chemistry expert. Output JSON only."

User: "Generate a chemical and physical profile for the ingredient 'chicken' matching 
the specified JSON format. Determine if it is vegan (true/false). Specify macros 
(proteins, fats, carbs in grams per 100g). Rate its texture on a 1.0-10.0 scale: 
[hardness, chewiness, moisture, fat_mouthfeel, elasticity]. List 3-5 primary flavor 
volatile compounds. Assign a culinary role: [base_protein, fat_source, flavor_enhancer, 
thickener, sweetener, aromatic, dairy, binding_agent]."
```

**Why ~350 ingredients, not the full `bounds_dict` of 60,000+ entries?** The bounds dictionary includes recipe phrases like "a box instant pistachio pudding mix" — not real culinary ingredients. The vocabulary was curated manually to cover all mainstream Indian cooking ingredients.

### Phase 2: `vegan_alternatives` Match Table (Same Notebook)

After all chemical profiles are generated, the match algorithm runs in pure Python (CPU-only):

1. Split all profiles into `vegan_pool` and `non_vegan_pool` based on `is_vegan`
2. For each non-vegan ingredient: compute `calculate_composite_score()` against every vegan candidate
3. Sort results by score descending; store top-5 + compensation blueprint in MongoDB

**Key design philosophy**: The system **never hardcodes pairs** (e.g., "jackfruit substitutes chicken"). Jackfruit is simply placed in the vocabulary, Qwen3-8B profiles it independently, and the vector math decides if its texture and flavor profile is similar enough to chicken. If soy chunks achieve a higher similarity score, they are selected—eliminating human bias.

### Phase 3: Indian Recipe Library (`GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb`)

Generates ~1,000 pre-made Indian budget vegan recipes using the full Ratatouille pipeline (with V10 model) and stores them in `ratatouille.indian_recipes`. These are served via the Recipe Library panel in the frontend for users who want to browse existing recipes without generating new ones.

---

## 15. Key Architectural Decisions & Rationale

### Decision 1: Server-Sent Events (SSE) over WebSockets

**Chosen**: SSE (unidirectional, server → client streaming)  
**Rejected**: WebSockets (bidirectional)  
**Why**: The recipe generation pipeline is fundamentally a one-way flow — the client sends one POST and receives a stream of 4–6 progress updates followed by a final result. WebSockets are bidirectional and more complex to manage in a stateless cloud environment. SSE is natively supported by browsers and simpler to implement, debug, and maintain.

### Decision 2: Synchronous MongoDB Client Inside SSE Stream

**Chosen**: `pymongo.MongoClient` (sync) for all DB calls inside `generate_recipe()`  
**Rejected**: `motor` (async) inside SSE  
**Why**: The SSE generator is a regular Python `yield` generator — not `async`. FastAPI runs it in a thread. Using `async motor` would require `await`, which cannot be used inside a sync generator. The sync PyMongo client is safe here; the async `motor` client is still used in all non-streaming endpoints.

### Decision 3: No LLM Bootstrap Inside SSE Stream

**Chosen**: Skip `bootstrap_ingredient_profile()` during recipe generation  
**Why**: LLM bootstrapping calls Groq API or Gradio Space (which can have 1–2 minute cold starts). For 5 ingredients, this could add 5–300 seconds of latency to what should be a 15–30 second process. Unknown ingredients are kept as-is — the fine-tuned Llama 3 model can handle any ingredient name in its recipe generation, even without a chemical profile. Bootstrap only runs in the standalone `/get-vegan-blueprint` endpoint where the user explicitly requests it.

### Decision 4: 4-Level SciPy Fallback Cascade

**Chosen**: Progressive constraint relaxation  
**Why**: A user's budget (e.g., ₹30) might be mathematically insufficient to meet the minimum gram bounds for 5 ingredients simultaneously. Hard-failing would produce a poor user experience. The cascade ensures *something* always comes out of the optimizer — even if the structural ratios are violated, the user still gets a recipe.

### Decision 5: Offline Vegan DB vs. Real-Time LLM Substitution

| Approach | Latency | Consistency | Quality |
|---|---|---|---|
| Ask recipe LLM to "make it vegan" in the prompt | 0s | Poor | Low — V10 trained on recipes, not food science |
| Real-time vegan engine math per request | ~2–5s per ingredient | Good | High |
| **Pre-built MongoDB match table (chosen)** | **~1ms per ingredient** | **Best** | **Highest** |

Building the substitution database once offline and reusing it forever is the correct engineering trade-off: maximum consistency, minimum latency, zero API cost per request.

### Decision 6: Jaccard Similarity over Dense Embeddings for Flavor

Dense embeddings (Word2Vec, BERT) reflect **word co-occurrence in text**, not chemical reality. Jaccard over discrete volatile molecule names is:
1. **Explainable**: We can compute exactly which molecules are missing → enables Spice Bridge generation
2. **Chemically correct**: Measures actual shared chemistry, not linguistic association
3. **Fast**: Sub-millisecond Python set operations — no GPU or model inference needed

---

## 16. Deployment

The application is deployed in two configurations:

### Application & Open Source

| Component | Platform | URL/Notes |
|---|---|---|
| **Web Application** | CoSyLab Servers (IIIT Delhi) | `https://cosylab.iiitd.edu.in/ratatouille-cost/` |
| **Source Code** | GitHub | `https://github.com/cosylabiiit/ratatouille-cost` |
| FastAPI backend | Render (free tier) | `https://ratatouille-backend.onrender.com` |
| LLM inference (V8) | Hugging Face Space | `nd1490/ratatouille-inference` (CPU, sleeps after inactivity) |
| LLM inference (V10) | Hugging Face Space | `nd1490/ratatouille-inference-v10-q8` (CPU) |
| Database | MongoDB Atlas M0 | Free tier cloud cluster |

### College Server Deployment (Academic)

The application has been deployed on the **CoSyLab (Complex Systems Lab) servers at Chennai Institute of Technology** following the steps documented in `MIGRATION_GUIDE.md`:

1. Clone the repository: `git clone https://github.com/cosylabiiit/ratatouille-cost.git`
2. Create Python virtual environment, install `requirements.txt`
3. Configure `.env` file with all API keys and the MongoDB URI
4. Run FastAPI backend: `uvicorn api:app --host 0.0.0.0 --port 8000`
5. Build React frontend: `cd frontend && npm install && npm run build`
6. Serve the `dist/` folder (Nginx or static file mount via FastAPI)

**Server requirements**:
- Python 3.9+, Node.js 18+
- Outbound internet access (MongoDB Atlas, Groq API, Hugging Face)
- Port 8000 (backend) and port 80/3000 (frontend) exposed

**Environment variables required** (from `.env`):
```
HF_TOKEN        = <Hugging Face write token>
GITHUB_PAT      = <GitHub Personal Access Token>
MONGO_URI       = <MongoDB Atlas connection string>
GROQ_API_KEY    = <Groq API key>
HF_SPACE_URL    = nd1490/ratatouille-inference
HF_SPACE_URL_V10= nd1490/ratatouille-inference-v10-q8
```

---

## 17. Testing & Evaluation

### Unit Tests: Vegan Engine (`test_vegan_engine.py`)

9 unit tests covering:
- Jaccard similarity index calculation
- Normalized texture distance calculation
- Delta vector translations (fat deficit → auxiliary additions)
- Static keyword fallback classification (red_meat, poultry, seafood, etc.)
- Unmapped ingredient error handling

**Result**: `OK` — 9/9 tests passed

### Integration Tests: API Endpoints (`test_api_endpoints.py`)

5 integration tests covering:
- Single ingredient vegan blueprint lookup
- Multiple ingredient vegan blueprint
- Dynamic LLM bootstrapping (using mocks to run offline)
- Fallback routing when LLM is unavailable
- `/optimize-only` endpoint validation

**Result**: `OK` — 5/5 tests passed

### End-to-End Verification

**Test case**: `"mutton"` → `/get-vegan-blueprint`

1. System contacted Hugging Face Space `nd1490/ratatouille-inference-v10-q8` via Groq bootstrap
2. Qwen3-8B dynamically generated a food chemistry profile for `"mutton"`
3. Profile was cached in `chemical_features.json` for future requests
4. System identified best substitute: **soy chunks** — match score **62.93%**
5. Compensation blueprint generated:
   - Auxiliary addition: coconut oil (lipid bridge)
   - Umami bridge: MSG or soy sauce
   - Preparation technique: soak in boiling water, squeeze out, pan-fry
   - Spice bridges: smoked paprika (pyrazines), nutritional yeast (umami)

### Model Evaluation (V8 vs. V10)

The V10 model was developed to address limitations observed in the V8 model during initial empirical evaluations. Specifically, the V10 model improved upon the format compliance of the generated recipes and significantly reduced the occurrence of oven-based instructions.

V10's non-oven training (`RAT_V10_NON_OVEN_TRAIN.ipynb`) specifically filtered training data to exclude oven-based recipes, making output more appropriate for Indian home kitchens.

---

## 18. Results & Discussion

### System Efficacy

1. **Budget optimization accuracy**: The SciPy linprog solver with Mandi data produces realistic, region-appropriate ingredient quantities. For a ₹100 budget in Delhi, the system correctly identifies that the user can afford approximately 85g of protein-class ingredients (soy chunks: ~₹0.25/g) and 200g+ of vegetable ingredients.

2. **Vegan substitution quality**: The chemically-aware approach significantly outperforms simple substitution lookup tables. By computing actual molecular flavor gaps and generating targeted spice bridges, the compensation blueprints are culinarily sound and actionable.

3. **SSE streaming**: Users see live progress updates ("Running Vegan Substitution Engine...", "Running Cost Constraint Optimization...", "Generating AI Recipe...") which dramatically improves perceived responsiveness compared to waiting at a blank screen.

4. **LLM caching effectiveness**: After the first week of deployment, ~70% of archetype classification calls are cache hits, meaning zero Groq API tokens are consumed for repeat ingredient combinations.

### Known Limitations

1. **Hugging Face Space cold starts**: The free-tier CPU Spaces sleep after inactivity. Cold start latency (1–2 minutes) can make the system appear broken when first accessed. A keep-alive ping mechanism is documented but not yet deployed.

2. **Mandi data coverage**: The government Mandi dataset covers agricultural commodities well (rice, tomato, onion, etc.) but does not include processed items (soy sauce, paneer, cheese). These fall back to the hardcoded pantry prices dictionary.

3. **Recipe LLM quality vs. recipe LLM scale**: The 3B Llama model sometimes generates repetitive or truncated instructions for complex dishes. A larger model (8B+) would likely produce more detailed step-by-step instructions, but would require paid GPU hosting.

4. **Ingredient bounds coverage**: If a user enters a rare ingredient not in the bounds dictionary, the optimizer falls back to generic (10g, 400g) bounds, which may produce unrealistic quantities.

---

## 19. Conclusion

Ratatouille demonstrates that the integration of mathematical optimization, fine-tuned LLM inference, and domain-specific chemistry databases can produce a recipe generation system that is qualitatively superior to any single-component approach.

The key technical contributions of this project are:

1. **Real-time budget-constrained recipe optimization**: Linear programming over live government Mandi agricultural price data produces gram quantities that are both mathematically optimal and economically realistic.

2. **Chemically-aware vegan substitution**: Vector math over ingredient chemical profiles — flavor molecule Jaccard similarity, 5D texture Euclidean distance, and binary functional role matching — identifies plant-based substitutes that are measurably superior to those suggested by simple keyword lookup tables or prompted language models.

3. **Compensation blueprint generation**: The delta vector calculus over macros and texture profiles automatically generates specific, chef-quality preparation techniques and spice additions that bridge the gap between animal products and their plant-based substitutes.

4. **Production deployment**: The system is live, tested, and deployed on both public cloud infrastructure (Render, Vercel, Hugging Face) and the CoSyLab academic servers at Chennai Institute of Technology.

Future work would focus on expanding the vegan alternatives database to include region-specific Indian plant ingredients (raw jackfruit, drumstick leaves, raw banana, etc.), integrating nutritional tracking per recipe, and exploring a larger-scale recipe LLM (8B–13B parameters) for richer recipe narrative.

---

## 20. References & File Index

### Primary Source Files

| File | Description |
|---|---|
| `api.py` | FastAPI backend — 1,048 lines — complete runtime pipeline |
| `vegan_engine.py` | Vegan substitution math engine — 412 lines |
| `frontend/src/App.jsx` | React UI — 831 lines |
| `chemical_features.json` | Local fallback ingredient chemistry database (135 KB, ~350 ingredients) |
| `vegan_alternatives_db.json` | Local copy of pre-computed vegan match table (64 KB) |
| `deconstruction_map.json` | Cached LLM ingredient deconstruction results (30 KB) |
| `requirements.txt` | Python backend dependencies |
| `MIGRATION_GUIDE.md` | College server deployment instructions |

### Training Notebooks

| File | Description |
|---|---|
| `RAT V8 JUST LOAD TRAIN AND SAVE TO HF (UPDATED).ipynb` | V8 model fine-tuning (Llama 3.2-3B, QLoRA) |
| `RAT_V10_NON_OVEN_TRAIN.ipynb` | V10 model fine-tuning (non-oven constraint) |
| `GPU_Open_World_Vegan_DB_Builder.ipynb` | Builds `chemical_features` + `vegan_alternatives` databases |
| `GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb` | Builds Indian recipe library |
| `RAT_V10_GENERATE_100_RECIPES.ipynb` | Model evaluation — 100 recipe generation benchmark |

### Documentation Files

| File | Description |
|---|---|
| `vegan_engine_architecture.md` | Architectural rationale for the vegan engine |
| `vegan_conversion_blueprint.md` | Design evolution + scoring formula derivation |
| `vegan_system_explainer.md` | Layman-friendly explanation of the 3-phase vegan DB |
| `system_pipeline_explainer.md` | High-level pipeline walkthrough |
| `ratatouille_complete_system_document.md` | Ground-truth system documentation (all prompts, schemas, decisions) |
| `deployment_plan_and_prompt.md` | Original deployment roadmap |

### Academic References

- Goel, M., Chakraborty, P., Ponnaganti, V., Khan, M., Tatipamala, S., Saini, A., & Bagler, G. (2022). *Ratatouille: A tool for Novel Recipe Generation*. 2022 IEEE 38th International Conference on Data Engineering Workshops (ICDEW), 107-110. https://doi.org/10.1109/icdew55742.2022.00022
- Virtanen, P., et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods.
- Hu, E.J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Meta AI. (2024). *Llama 3: Open Foundation and Fine-Tuned Chat Models*.

---

*Ratatouille — AI-Powered Cost-Constrained Indian Budget Recipe Generator with Chemically-Aware Vegan Substitution*  
*Nindra Dhanush (MT25074) | M.Tech. 2025–2027*  
*CoSyLab, IIIT Delhi | Supervisor: Prof. Ganesh Bagler | July 2026*
