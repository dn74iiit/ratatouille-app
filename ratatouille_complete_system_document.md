# 🍲 Ratatouille — Complete System Documentation
### Every Prompt, Database, API Call, Data Pipeline, and Design Decision

> This document is a ground-truth, exhaustive reference for the Ratatouille AI recipe generation system. It is derived directly from reading every source file in the repository. Every prompt shown is the exact string used in production code.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack & Dependencies](#2-technology-stack--dependencies)
3. [Infrastructure & Environment](#3-infrastructure--environment)
4. [Databases (MongoDB Atlas)](#4-databases-mongodb-atlas)
5. [External Data Sources](#5-external-data-sources)
6. [AI Models Used](#6-ai-models-used)
7. [The Complete Request Pipeline (`/generate-recipe`)](#7-the-complete-request-pipeline-generate-recipe)
   - 7.1 [Request Initialization & SSE Streaming](#71-request-initialization--sse-streaming)
   - 7.2 [Stage 1: Vegan Substitution Engine](#72-stage-1-vegan-substitution-engine)
   - 7.3 [Stage 2: Archetype Classification (Live LLM)](#73-stage-2-archetype-classification-live-llm)
   - 7.4 [Stage 3: Cost Constraint Optimization (SciPy)](#74-stage-3-cost-constraint-optimization-scipy)
   - 7.5 [Stage 4: AI Recipe Generation (HF Gradio)](#75-stage-4-ai-recipe-generation-hf-gradio)
   - 7.6 [Stage 5: Post-Processing & Logging](#76-stage-5-post-processing--logging)
8. [The Vegan Engine Deep Dive](#8-the-vegan-engine-deep-dive)
   - 8.1 [Chemical Feature Profiles](#81-chemical-feature-profiles)
   - 8.2 [The Composite Scoring Formula](#82-the-composite-scoring-formula)
   - 8.3 [The Spice Bridge Algorithm](#83-the-spice-bridge-algorithm)
   - 8.4 [Delta Recommendations (Compensation Blueprint)](#84-delta-recommendations-compensation-blueprint)
   - 8.5 [Static Fallback Profiles](#85-static-fallback-profiles)
9. [How We Built the Offline Databases](#9-how-we-built-the-offline-databases)
   - 9.1 [chemical_features Database](#91-chemical_features-database)
   - 9.2 [vegan_alternatives Match Table](#92-vegan_alternatives-match-table)
10. [All LLM Prompts (Production Code)](#10-all-llm-prompts-production-code)
11. [All API Endpoints](#11-all-api-endpoints)
12. [The Frontend (React SSE Client)](#12-the-frontend-react-sse-client)
13. [The LLM Cache System](#13-the-llm-cache-system)
14. [Key Architectural Decisions & Why](#14-key-architectural-decisions--why)
15. [File Reference Map](#15-file-reference-map)

---

## 1. System Overview

Ratatouille is an **AI-powered Indian budget recipe generator**. It is a capstone research project that combines:

- **Mathematical optimization** (SciPy Linear Programming) to allocate real-time Indian market prices across ingredients within a strict INR budget
- **Fine-tuned LLM inference** (custom Llama 3 model, V8 and V10 variants) to write the actual recipe
- **A chemically-aware vegan substitution engine** that uses vector math over food chemistry profiles — not hardcoded lookup tables — to find the best plant-based alternative for any animal product
- **Real-time streaming** (Server-Sent Events) so the user sees live progress updates instead of waiting at a blank screen

### The Core User Journey

```
User enters: ["chicken", "tomato", "onion"], Budget: ₹100, Servings: 2, State: Delhi, Vegan: ON
         │
         ▼
1. Swap chicken → soy chunks (vector math over chemical profiles)
         │
         ▼
2. Classify dish as "Curry" (Groq Llama 3.3 70B call, cached in MongoDB)
         │
         ▼
3. Look up real Delhi market prices (Mandi data from GitHub)
   Run SciPy linprog to maximize grams within ₹100 budget
   → 85g soy chunks, 120g tomato, 95g onion
         │
         ▼
4. Send to fine-tuned Llama 3 on Hugging Face Gradio:
   "### INGREDIENTS:\n- 85.0g soy chunks\n- 120.0g tomato\n- 95.0g onion\n### TITLE:\n"
         │
         ▼
5. Return SSE event: complete → recipe text rendered in React UI
```

---

## 2. Technology Stack & Dependencies

### Backend (`requirements.txt`)

| Package | Role |
|---|---|
| `fastapi` | Web framework — hosts all API endpoints |
| `uvicorn[standard]` | ASGI server — runs FastAPI with WebSocket/SSE support |
| `python-dotenv` | Loads secrets from `.env` file |
| `requests` | Fetches Mandi CSV and JSON files from GitHub |
| `pandas` | Processes the Mandi market data CSV into a searchable DataFrame |
| `numpy` | Vector math for texture similarity (Euclidean distance) |
| `scipy` | `linprog` — the core budget optimizer |
| `nltk` | `WordNetLemmatizer` — normalizes ingredient names for price lookup |
| `pydantic` | Request/response model validation |
| `gradio_client` | Calls the Hugging Face Gradio Spaces (V8, V10 models) |
| `motor` | Async MongoDB driver (for async endpoints) |
| `pymongo` | Sync MongoDB driver (for the SSE stream which is sync) |
| `groq` | Groq API client for Llama 3.3 70B serverless inference |

### Frontend

| Technology | Role |
|---|---|
| React (Vite) | UI framework |
| Vanilla CSS | All styling |
| `fetch` + `ReadableStream` | SSE client (no library needed — native browser API) |

### Training / Data Notebooks

| Environment | Role |
|---|---|
| Google Colab (T4 GPU) | Fine-tuning Llama 3 (V8, V10); running offline DB builders |
| Hugging Face Spaces | Hosting the fine-tuned models for inference |
| MongoDB Atlas | Cloud database for all persistent data |

---

## 3. Infrastructure & Environment

### Environment Variables (`.env`)

```
HF_TOKEN        = <Hugging Face token — needed to call Gradio Spaces>
GITHUB_PAT      = <GitHub Personal Access Token — fetches private recipe-data-automation repo>
MONGO_URI       = <MongoDB Atlas connection string>
GROQ_API_KEY    = <Groq API key — Llama 3.3 70B serverless>
HF_SPACE_URL    = nd1490/ratatouille-inference        (V8 model Space)
HF_SPACE_URL_V10= nd1490/ratatouille-inference-v10-q8 (V10 model Space)
```

### Deployment

| Component | Platform |
|---|---|
| FastAPI backend | Render (free tier) |
| React frontend | Vercel |
| LLM inference | Hugging Face Spaces (free T4 GPU, sleeps after inactivity) |
| Database | MongoDB Atlas |

### Backend URL Routing (Frontend)

```javascript
// App.jsx line 4
const BACKEND_URL = 
  window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://ratatouille-backend.onrender.com';
```

---

## 4. Databases (MongoDB Atlas)

The system uses a single MongoDB Atlas cluster with the database named `ratatouille`. It has **five collections**:

### Collection 1: `ratatouille.recipes`

Stores recipes saved by users to their profile.

```json
{
  "_id": "<ObjectId>",
  "username": "dhanush",
  "recipe": { "<full recipe result object>" },
  "created_at": 1720589000.123
}
```

Written by: `POST /save-recipe`  
Read by: `GET /my-recipes/{username}`

---

### Collection 2: `ratatouille.vegan_alternatives`

**Pre-computed match table.** For every non-vegan ingredient we've ever processed, this stores the best vegan substitute and the full compensation blueprint. This is the **primary fast-path** for the vegan substitution engine during recipe generation. Built offline by the Colab notebooks.

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
    { "substitute": "soy chunks",   "score": 0.74 },
    { "substitute": "seitan",       "score": 0.71 },
    { "substitute": "jackfruit",    "score": 0.68 },
    { "substitute": "tofu",         "score": 0.61 },
    { "substitute": "tempeh",       "score": 0.58 }
  ],
  "compensation_blueprint": {
    "auxiliary_additions": [
      { "name": "coconut oil", "amount": "1 tsp", "purpose": "bridges fat profile deficit" }
    ],
    "techniques": ["Rehydrate soy chunks in hot water for 15 min before cooking."],
    "spice_bridge": [
      { "spice": "smoked paprika", "fills_gap_ratio": 0.67, "reason": "adds pyrazines compounds to fill the aromatic gap" }
    ]
  }
}
```

Read by: `vegan_alternatives_sync.find_one({"_id": canonical_name})` (sync path inside SSE)  
Read by: `GET /vegan-alternatives/{ingredient}` (async path for frontend debug panel)

---

### Collection 3: `ratatouille.chemical_features`

**Full ingredient chemistry profiles.** This is the raw science database — every ingredient described by its macros, texture vector, flavor molecules, and culinary role. This powers the vegan engine's live math when a direct match isn't in `vegan_alternatives`.

```json
{
  "_id": "chicken",
  "is_vegan": false,
  "macros": { "proteins": 27.0, "fats": 14.0, "carbs": 0.0 },
  "texture_profile": [5.5, 7.0, 5.5, 5.5, 6.0],
  "flavor_molecules": ["inosine monophosphate", "2-methyl-3-furanthiol", "methional", "dimethyl trisulfide", "heptanal"],
  "culinary_role": "base_protein"
}
```

Read by: `vegan_engine.load_features()` — tries MongoDB first, falls back to `chemical_features.json`  
Written by: Colab notebooks (offline) + `vegan_engine.save_new_feature()` (live bootstrap)

**Local fallback**: `chemical_features.json` (135 KB) — kept in sync with the cloud collection.

---

### Collection 4: `ratatouille.llm_cache`

**LLM response cache.** Every time we call the Groq API or Gradio Space for structured classification or deconstruction, the result is saved here so we never pay for the same call twice.

```json
{
  "_id": "archetype_chicken,onion,tomato",
  "result": "Curry",
  "timestamp": 1720589000.123,
  "original_ingredients": ["chicken", "onion", "tomato"]
}
```

```json
{
  "_id": "deconstruct_tomato ketchup",
  "result": { "tomato": 0.8, "sugar": 0.1, "onion": 0.1 },
  "timestamp": 1720589000.123
}
```

Read/Written by: `get_recipe_archetype()` and `deconstruct_ingredient()` in `api.py`

---

### Collection 5: `ratatouille.generation_logs`

**Analytics/performance log.** Every completed recipe generation writes a timing record here.

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

Written by: end of `generate_recipe()` SSE handler  
Never read by the app — used for monitoring and performance analysis only.

---

### Collection 6: `ratatouille.indian_recipes`

**Pre-generated Indian vegan recipe library.** Populated offline by the `GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb` notebook. Served by the `/indian-recipes` endpoint for the "Recipe Library" UI panel.

```json
{
  "_id": "<ObjectId>",
  "style": "Curry",
  "title": "...",
  "ingredients": [...],
  "instructions": "...",
  "created_at": 1720589000.123
}
```

---

## 5. External Data Sources

### Mandi Market Data (Live, fetched at server startup)

**Source**: Private GitHub repo `dn74iiit/recipe-data-automation`  
**File**: `daily_mandi_data.csv`  
**URL**: `https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/daily_mandi_data.csv`

This is government agricultural market (Mandi) price data — real wholesale crop prices from markets across India, updated daily.

**Schema** (10 columns):
```
State, District, Market, Commodity, Variety, Grade, Arrival_Date, Min_Price, Max_Price, Modal_Price
```

**How it's processed at startup:**

```python
df_mandi = pd.read_csv(io.StringIO(mandi_response.text), header=None,
    names=['State','District','Market','Commodity','Variety','Grade',
           'Arrival_Date','Min_Price','Max_Price','Modal_Price'])

# Filter to ONLY the most recent date
latest_date = df_mandi['Arrival_Date'].max()
current_mandi = df_mandi[df_mandi['Arrival_Date'] == latest_date].copy()

# Lemmatize commodity names for fuzzy matching
current_mandi['Processed_Ingredient'] = current_mandi['Commodity'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word.lower()) for word in str(x).split()])
)

# Convert modal price (INR per quintal) → price per gram
current_mandi['Price_per_Gram'] = current_mandi['Modal_Price'] / 100000
```

**Price lookup logic in `get_dynamic_price()`:**
1. Exact match on `state` + lemmatized ingredient name → `median(Modal_Price)`
2. Fallback to national median (ignore state filter) if state has no data
3. Fallback to `pantry_prices` dict
4. Fallback to LLM deconstruction

---

### Ingredient Bounds Data (from GitHub)

**File**: `v8_lemmaized_ingredient_bounds.json`  
**URL**: `https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/v8_lemmaized_ingredient_bounds.json`

A JSON dict mapping lemmatized ingredient names to `(min_grams, max_grams)` bounds used in the linear program. Derived from statistical analysis of our 50K recipe training dataset.

```json
{
  "chicken": [50, 350],
  "tomato": [30, 300],
  "onion": [20, 250],
  ...
}
```

---

### Pantry Prices (Hardcoded + GitHub override)

Hardcoded dictionary for processed/packaged items not in Mandi data:

```python
pantry_prices = {
    "vanilla extract": 4.50, "dark chocolate": 1.20, "soy sauce": 0.40,
    "olive oil": 0.80, "macaroni": 0.30, "vegan cashew mozzarella": 2.50,
    "sugar": 0.05, "salt": 0.02, "egg": 0.12, "chicken": 0.25, "milk": 0.06,
    "mutton": 0.80, "fish": 0.30, "paneer": 0.40, "pork": 0.30, "beef": 0.30,
    "cheese": 0.50, "butter": 0.60, "yogurt": 0.10, "curd": 0.10
}
```

This is then overridden/extended by `pantry_prices.json` from GitHub if available.

---

### RecipeDB (Training Data — Offline Only)

Three large CSV files used to train and evaluate our models. Not used at runtime.

| File | Size | Contents |
|---|---|---|
| `RecipeDB_formatted_like_50k.csv` | 130 MB | ~50K recipes formatted in our V10 training schema |
| `RecipeDB_general - RecipeDB_general.csv` | 46 MB | 118,084 rows with `Processes` column (pipe-separated cooking verbs) |
| `RecipeDB_instructions.csv` | 96 MB | Full recipe instructions text |
| `final_clean_50k_recipes_grams.csv` | 82 MB | Final cleaned training set with gram quantities |

---

## 6. AI Models Used

### Model A: V8 — Fine-tuned Llama 3.2 3B (Original)

- **Base model**: `meta-llama/Llama-3.2-3B`
- **Fine-tuning**: QLoRA on 50K Indian recipes in our custom prompt format
- **Hosted at**: `nd1490/ratatouille-inference` (HF Space)
- **Called via**: `gradio_client` (Gradio Spaces API)

### Model B: V10 — Fine-tuned Llama 3.2 3B (Improved, Q8 Quantized)

- **Base model**: `meta-llama/Llama-3.2-3B`
- **Fine-tuning**: Continued training on a larger, cleaner dataset including RecipeDB
- **Quantization**: Q8 GGUF for efficient inference
- **Hosted at**: `nd1490/ratatouille-inference-v10-q8` (HF Space)
- **Called via**: `gradio_client`

### Model C: Groq Llama 3.3 70B (Serverless — Metadata Tasks)

- **Provider**: Groq (free tier serverless API)
- **Model ID**: `llama-3.3-70b-versatile`
- **Used for**:
  - Dish archetype classification
  - Ingredient deconstruction (processed → raw crops)
  - Dynamic ingredient profile bootstrapping
- **Fallback**: Gradio V8 Space (if Groq API fails/rate-limits)

### Model D: Qwen3-8B 4-bit NF4 (Offline — Database Building Only)

- **Used in**: `GPU_Open_World_Vegan_DB_Builder.ipynb` (Colab T4 GPU)
- **Purpose**: Generating `chemical_features` profiles for ~350 ingredients
- **Why not our V10 model?** V10 was fine-tuned to write recipes. It has no training signal for structured food chemistry JSON. Qwen3-8B is a general-purpose scientific model — correct tool for this job.
- **NOT used at runtime** — only during the one-time offline database build.

---

## 7. The Complete Request Pipeline (`/generate-recipe`)

### 7.1 Request Initialization & SSE Streaming

**Endpoint**: `POST /generate-recipe`  
**Request model**:

```python
class RecipeRequest(BaseModel):
    ingredients: list[str]   # e.g., ["chicken", "tomato", "onion"]
    budget: float            # e.g., 100.0 (INR)
    servings: int = 1
    state: str = "Delhi"     # Indian state for regional pricing
    model_version: str = "v10"  # "v8" or "v10"
    is_vegan: bool = False
```

The endpoint immediately returns a `StreamingResponse` with `media_type="text/event-stream"`. It uses a Python **generator function** (`event_stream()`) that `yield`s SSE events:

```python
return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**SSE Events yielded (in order)**:

| `step` value | `message` | When |
|---|---|---|
| `starting` | `"Initializing..."` | Immediately |
| `veganizing` | `"Running Vegan Substitution Engine..."` | If `is_vegan=True` |
| `optimizing` | `"Running Cost Constraint Optimization (Budget: ₹{budget})..."` | Before SciPy |
| `generating` | `"Generating AI Recipe (Model: {model_version})..."` | Before HF call |
| `complete` | *(none — carries `result` object)* | On success |
| `error` | Error message string | On failure |

**React SSE client** (in `App.jsx`):

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
      else setStepMessage(data.message); // Updates progress indicator
    }
  }
}
```

---

### 7.2 Stage 1: Vegan Substitution Engine

*Only runs if `is_vegan = True`.*

#### Step A: Ingredient Cleaning

Comma-separated input strings are split:
```python
clean_ingredients = []
for item in request.ingredients:
    clean_ingredients.extend([i.strip() for i in item.split(',') if i.strip()])
```

#### Step B: Fast Archetype Classification (0ms, No LLM)

```python
archetype_fast = _classify_archetype_fast([parse_ingredient_input(i)[1] for i in clean_ingredients])
```

`_classify_archetype_fast()` is a pure keyword-matching function — instant, no LLM:

```python
def _classify_archetype_fast(ingredients: list) -> str:
    s = " ".join(ingredients).lower()
    if any(w in s for w in ["rice", "biryani", "pulao", "fried rice"]): return "Rice_Dish"
    if any(w in s for w in ["pasta", "noodle", "macaroni", "spaghetti", "basil"]): return "Rice_Dish"
    if any(w in s for w in ["oat","banana","sugar","flour","chocolate","cake","honey","cream","milk","vanilla"]): return "Dessert"
    if any(w in s for w in ["mushroom","soup","broth","carrot","ginger","coconut milk"]): return "Soup"
    if any(w in s for w in ["lettuce","salad","cucumber","olive"]): return "Salad"
    if any(w in s for w in ["bread","dough","yeast"]): return "Bread"
    return "Curry"  # default for most Indian dishes
```

#### Step C: Per-Ingredient Processing Loop

For each ingredient, the system runs a **3-step lookup cascade**:

##### Step C-1: Canonical Normalization

Maps variant ingredient names to their canonical base:

```python
canonical_name = canonicalize_ingredient(name_lower)
# Examples from CANONICAL_MAP (300+ entries):
# "chicken breast" → "chicken"
# "egg yolk" → "egg"
# "heavy cream" → "cream"
# "ground beef" → "beef"
# "lamb chop" → "mutton"
# "mozzarella" → "cheese"
# "sour cream" → "yogurt"
# "clarified butter" → "ghee"
```

##### Step C-2: MongoDB `vegan_alternatives` Lookup (~1ms)

```python
db_doc = vegan_alternatives_sync.find_one({"_id": canonical_name})
if db_doc and db_doc.get("best_vegan_substitute"):
    blueprint = { "status": "success", ... }
```

If found, we have the full pre-computed result: substitute name, score, compensation blueprint.

##### Step C-3: Local vegan_engine Fallback (if DB miss)

```python
features = vegan_engine.load_features()
if canonical_name in features or vegan_engine.classify_by_keyword(canonical_name) is not None:
    blueprint = vegan_engine.generate_vegan_blueprint(canonical_name, archetype=archetype_fast)
else:
    # Truly unknown — keep as-is. No LLM call here.
    blueprint = {"status": "unknown"}
```

> **Important design note**: The `bootstrap_ingredient_profile()` LLM call is **deliberately NOT called** inside the SSE stream. It costs 1–60s per ingredient (due to HF Space cold starts). Unknown ingredients are simply kept as-is during recipe generation. The bootstrap path is only used in the standalone `/get-vegan-blueprint` endpoint.

#### Step D: Apply Substitution + Compensation

```python
if blueprint.get("status") == "success" and original != best_substitute:
    # Replace the animal product
    veganized_ingredients.append(f"{qty}g {substitute}")
    # Add auxiliary additions (e.g., "1 tsp coconut oil")
    for add in blueprint["compensation_blueprint"]["auxiliary_additions"]:
        veganized_ingredients.append(f"{add['amount']} {add['name']}")
    # Add spice bridges (e.g., "smoked paprika")
    for spice in blueprint["compensation_blueprint"]["spice_bridge"]:
        veganized_ingredients.append(spice["spice"])
else:
    veganized_ingredients.append(raw_ing)  # Keep as-is

# Override the working ingredient list
clean_ingredients = veganized_ingredients
```

---

### 7.3 Stage 2: Archetype Classification (Live LLM)

Called via `optimize_recipe_v2()` which calls `get_recipe_archetype()`.

#### Cache Key Construction

```python
ingr_str = ", ".join(ingredients_list)
sorted_ingrs = ",".join(sorted([i.strip().lower() for i in ingredients_list]))
cache_id = f"archetype_{sorted_ingrs}"
```

Ingredients are **sorted before hashing** so `["onion", "chicken", "tomato"]` and `["chicken", "tomato", "onion"]` produce the same cache key.

#### Cache Check

```python
cached = llm_cache_sync.find_one({"_id": cache_id})
if cached and "result" in cached:
    return cached["result"]  # ← e.g., "Curry"
```

#### Live Groq Call (Primary)

**Model**: `llama-3.3-70b-versatile`  
**Parameters**: `max_completion_tokens=15`, `temperature=0.1`

**System prompt** (exact string from `api.py` line 387):
```
You are a recipe classification assistant. Output ONLY a single word classification from the permitted list. No explanation.
```

**User prompt** (exact string from `api.py` line 388):
```
Classify the dish structure based on these ingredients: [{ingr_str}].
Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish].
```

**Valid outputs**: `Curry`, `Dry_Sabzi`, `Salad`, `Dessert`, `Bread`, `Soup`, `Rice_Dish`  
**Default if parsing fails**: `"Curry"`

#### Gradio Fallback (if Groq fails)

**Prompt sent to V8 Llama 3B Space**:
```
<|begin_of_text|>Classify the dish structure based on these ingredients: [{ingr_str}].
Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish].
Return ONLY the word.

### INGREDIENTS:
{ingr_str}
### ARCHETYPE:

```

#### Cache Save

```python
llm_cache_sync.update_one(
    {"_id": cache_id},
    {"$set": {"result": result, "timestamp": time.time(), "original_ingredients": ingredients_list}},
    upsert=True
)
```

---

### 7.4 Stage 3: Cost Constraint Optimization (SciPy)

Function: `optimize_recipe_v2(raw_user_ingredients, total_budget, servings, user_state, archetype)`

#### Step A: Price Lookup for Every Ingredient

Function: `get_dynamic_price(clean_ingredient, user_state, skip_llm=False)`

**Priority 1 — State-specific Mandi price:**
```python
lemmatized_ing = " ".join([lemmatizer.lemmatize(w) for w in clean_ingredient.split()])
state_data = current_mandi[
    (current_mandi['Processed_Ingredient'] == lemmatized_ing) &
    (current_mandi['State'].str.lower() == user_state.lower())
]
if not state_data.empty:
    return state_data['Price_per_Gram'].median()
```

**Priority 2 — National Mandi price (ignore state):**
```python
nat_data = current_mandi[current_mandi['Processed_Ingredient'] == lemmatized_ing]
if not nat_data.empty:
    return nat_data['Price_per_Gram'].median()
```

**Priority 3 — Pantry prices dict:**
```python
if clean_ingredient in pantry_prices:
    return pantry_prices[clean_ingredient]
```

**Priority 4 — LLM Deconstruction (slow path, only if `skip_llm=False`):**

Cache key: `f"deconstruct_{ingredient.strip().lower()}"`

**Groq System prompt** (exact, `api.py` line 341):
```
You are a food chemistry and agricultural database. Output only valid JSON. Do not write any explanations or conversational text outside the JSON.
```

**Groq User prompt** (exact, `api.py` line 342):
```
Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops with approximate weight percentages (total summing to 1.0).
Example: for 'tomato ketchup', return exactly: {"tomato": 0.8, "sugar": 0.1, "onion": 0.1}
```

**Parameters**: `max_tokens=100`, `temperature=0.1`

**Gradio fallback prompt**:
```
<|begin_of_text|>Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops.
Assign approximate weight percentages. Return ONLY valid JSON.
Example for 'tomato ketchup': {"tomato": 0.8, "sugar": 0.1, "onion": 0.1}

### INGREDIENT:
{ingredient}
### JSON:

```

**Price calculation from deconstruction**:
```python
avg_price = 0
for sub_ing, weight in deconstructed.items():
    sub_lem = lemmatizer.lemmatize(sub_ing.lower())
    sub_price = current_mandi[current_mandi['Processed_Ingredient'] == sub_lem]['Price_per_Gram'].median()
    if pd.isna(sub_price): sub_price = pantry_prices.get(sub_ing, 0.5)
    avg_price += sub_price * (weight / total_weight)
return avg_price * 1.3  # 30% markup for processing cost
```

**Default fallback price**: `0.5` INR/gram (if all else fails, or `skip_llm=True`)

#### Step B: Ingredient Tagging

```python
def tag_ingredient(ing_name):
    proteins = ['paneer','chicken','soya','tofu','dal','lentil','egg','meat','fish']
    bases    = ['onion','tomato','garlic','ginger','puree']
    sweets   = ['sugar','jaggery','chocolate','vanilla','syrup']
    carbs    = ['rice','flour','wheat','bread','noodle','pasta','potato']
    if any(p in ing_name for p in proteins): return 'protein'
    if any(b in ing_name for b in bases):    return 'base'
    if any(s in ing_name for s in sweets):   return 'sweet'
    if any(c in ing_name for c in carbs):    return 'neutral'
    return 'veggie'
```

#### Step C: Bounds Construction

```python
lower, upper = bounds_dict.get(lemmatizer.lemmatize(clean_name.split()[-1]), (10, 400))
# Minimum lower bound: 15g for most, 2g for spices
if clean_name not in ['garlic','ginger','chili','salt','pepper']:
    lower = max(lower, 15.0)
else:
    lower = max(lower, 2.0)
upper = min(upper, 800.0)  # Never exceed 800g
if lower > upper: lower = upper
```

If the user specified a quantity (e.g., `"200g chicken"`), that quantity is **fixed** (both bounds set to `quantity/servings`).

#### Step D: Linear Program Construction

**Objective**: Maximize total food weight (minimize `-grams` for each ingredient)
```python
c = [-1] * n  # Minimize negative grams = maximize grams
```

**Budget constraint**:
```python
A_ub = [prices]          # [price_A, price_B, ...]
b_ub = [total_budget / servings]
```

**Archetype ratio constraints** (structural integrity rules):

| Archetype | Constraint | Meaning |
|---|---|---|
| `Curry` | `0.8*protein - 1.0*base <= 0` | Base must be ≥ 80% of protein weight |
| `Curry` | `-3.0*protein + 1.0*base <= 0` | Base must be ≤ 3× protein (not too saucy) |
| `Dry_Sabzi` | `-0.8*protein + 1.0*base <= 0` | Base ≤ protein (dry, not saucy) |
| `Salad` | `2.0*(protein+neutral) - 1.0*veggie <= 0` | Veggies dominate |
| `Rice_Dish` | `1.0*protein - 1.0*neutral <= 0` | Protein ≤ rice/carbs |
| `Soup` | `4.0*protein - 1.0*base <= 0` | Very liquid — base hugely dominates |
| `Dessert` | `-0.4*(neutral+protein+base) + 1.0*sweet <= 0` | Sweet component dominates |

**Solver**:
```python
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
```

#### Step E: 4-Level Fallback Cascade

```
Attempt 1: Full constraints (archetype ratios + budget + bounds)
         │ FAIL (infeasible)
         ▼
Attempt 2: Budget-only (drop archetype ratio constraints)
         │ FAIL (minimum bounds too high for budget)
         ▼
Attempt 3: Force lower bounds to 5g (emergency floor)
         │ FAIL (theoretically impossible matrix)
         ▼
Attempt 4: Nuclear — override all bounds to (1g, 1000g)
         │ FAIL → return None (very rare)
         ▼
         Error SSE event sent to frontend
```

#### Step F: Result Assembly

```python
estimated_grams_per_serving = np.round(result.x, 1)
final_quantities = estimated_grams_per_serving * servings
return [f"{qty}g {parse_ingredient_input(r)[1]}" for qty, r in zip(final_quantities, raw_ingredients)]
# → ["85.0g soy chunks", "120.0g tomato", "95.0g onion"]
```

---

### 7.5 Stage 4: AI Recipe Generation (HF Gradio)

#### Prompt Construction (EXACT FORMAT — V10 Schema)

```python
ingr_text = "\n".join(f"- {i}" for i in calculated_ingredients)
prompt = (
    f"### INGREDIENTS:\n"
    f"{ingr_text}\n"
    f"### TITLE:\n"
)
```

**Example prompt sent to the model**:
```
### INGREDIENTS:
- 85.0g soy chunks
- 120.0g tomato
- 95.0g onion
- 15.0g garlic
- 10.0g ginger
### TITLE:

```

This prompt format is critical — the model was **fine-tuned on exactly this structure**, so any deviation (e.g., using `**Ingredients:**` instead of `### INGREDIENTS:`) produces garbage output.

#### Gradio API Call

```python
result = client.predict(
    prompt,           # Textbox: prompt
    500,              # Slider: max_new_tokens
    0.6,              # Slider: temperature
    0.9,              # Slider: top_p
    1.05,             # Slider: repetition_penalty
    True,             # Checkbox: do_sample
    api_name="/generate",
)
```

The client is retrieved via `_get_client(version)` which:
- Returns the cached `gradio_client_v8` or `gradio_client_v10` if already connected
- If not connected, retries **3 times with 30-second delays** (handles HF Space cold starts)
- Raises `HTTP 503` if all 3 attempts fail (Space won't wake up)

---

### 7.6 Stage 5: Post-Processing & Logging

#### Stop Token Removal

The V10 Llama model sometimes generates Llama 3 special tokens or loops back to `### INGREDIENTS:`. These are stripped:

```python
stop_tokens = ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>', '\n### INGREDIENTS:']
for t in stop_tokens:
    if t in ai_text:
        ai_text = ai_text.split(t)[0].strip()
```

#### Section Clean-up

```python
# Remove double TITLE headers (model sometimes hallucinates a second one)
if "### TITLE:\n" in ai_text:
    ai_text = ai_text.split("### TITLE:\n")[1].strip()

# Cut off closing phrases / repetition loops
cut_phrases = ["\nEnjoy!", "\nServe hot", "\nBon Apetit", "\nChef's Note:",
               "\nVariations:", "\nServing suggestion:", "\nNote:"]
for phrase in cut_phrases:
    if phrase in ai_text:
        ai_text = ai_text.split(phrase)[0].strip()

# Cut off any second ### section (only ### DIRECTIONS: is valid in output)
if "### DIRECTIONS:\n" in ai_text:
    parts = ai_text.split("### DIRECTIONS:\n")
    directions_part = parts[1]
    if "\n### " in directions_part:
        directions_part = directions_part.split("\n### ")[0]
    ai_text = f"{parts[0]}### DIRECTIONS:\n{directions_part}".strip()
```

#### Final SSE Payload

```python
final_result = {
    "status": "success",
    "archetype": archetype,
    "calculated_ingredients": calculated_ingredients,  # ["85.0g soy chunks", ...]
    "recipe": ai_text  # Full recipe markdown string
}
yield f"data: {json.dumps({'step': 'complete', 'result': final_result})}\n\n"
```

---

## 8. The Vegan Engine Deep Dive

File: [`vegan_engine.py`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_engine.py)

### 8.1 Chemical Feature Profiles

Every ingredient is stored as a structured object with 5 fields:

```json
{
  "_id": "chicken",
  "is_vegan": false,
  "macros": { "proteins": 27.0, "fats": 14.0, "carbs": 0.0 },
  "texture_profile": [5.5, 7.0, 5.5, 5.5, 6.0],
  "flavor_molecules": ["inosine monophosphate", "2-methyl-3-furanthiol", "methional", "dimethyl trisulfide", "heptanal"],
  "culinary_role": "base_protein"
}
```

**Texture profile dimensions** (5D vector, values 1.0–10.0):

| Index | Dimension | Low (1) | High (10) |
|---|---|---|---|
| 0 | Hardness | Melts/dissolves | Very hard/dense |
| 1 | Chewiness | No chewing | Very chewy |
| 2 | Moisture | Very dry | Very juicy/wet |
| 3 | Fat mouthfeel | No fat sensation | Rich, coating |
| 4 | Elasticity | Crumbles | Bounces back |

**Culinary roles** (valid values):
`bulk_protein`, `fat_source`, `binder`, `creamy_liquid`, `sweetener`, `seasoning`, `veggie`, `starch`, `flavor_enhancer`, `aromatic`, `dairy`, `binding_agent`, `thickener`, `base_protein`

---

### 8.2 The Composite Scoring Formula

$$\text{Score} = \alpha \cdot S_{\text{flavor}} + \beta \cdot S_{\text{texture}} + \gamma \cdot S_{\text{role}}$$

With weights: **α = 0.3, β = 0.4, γ = 0.3**

The β=0.4 weight on texture is backed by food science literature: texture accounts for ~40-50% of perceived substitution quality in plant-based meat research.

#### Flavor Similarity — Jaccard Index

```python
def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
```

Takes the **set** of flavor molecule names. Measures what fraction of all molecules are shared.

**Why Jaccard over dense embeddings?**
1. **Explainability**: We can extract *which* compounds are missing to build the Spice Bridge
2. **Correctness**: Embedding models reflect word co-occurrence, not chemical overlap — garlic and onion appear together but their molecules differ
3. **Speed**: Sub-millisecond Python set operations

#### Texture Similarity — Normalized Euclidean Distance

```python
def texture_similarity(vec_a, vec_b):
    arr_a = np.array(vec_a, dtype=float)
    arr_b = np.array(vec_b, dtype=float)
    dist = np.linalg.norm(arr_a - arr_b)
    max_dist = 20.124  # sqrt(5 * (10-1)^2) for 5D vectors with range 1-10
    return float(1.0 - (dist / max_dist))
```

Normalized to [0, 1]. 1.0 = identical texture profile.

#### Functional Overlap — Binary Role Match

```python
def functional_overlap(role_a, role_b):
    return 1.0 if role_a == role_b else 0.0
```

A meat (`bulk_protein`) will never score well against coconut milk (`creamy_liquid`). This prevents culinarily absurd substitutions.

---

### 8.3 The Spice Bridge Algorithm

After selecting the best substitute, the engine computes which **kitchen spices** can fill the aromatic gap left by removing the animal product.

```python
def get_spice_bridge(original_mols, substitute_mols, all_features, top_k=3):
    orig_set = set(original_mols)
    sub_set  = set(substitute_mols)
    flavor_gap = orig_set - sub_set   # Molecules present in original, absent in substitute

    if not flavor_gap:
        return []  # Perfect aromatic match — no bridge needed

    bridge_scores = []
    for name, data in all_features.items():
        # Only consider items that are vegan AND labeled as flavor_enhancer or aromatic
        if data.get("culinary_role") in ["flavor_enhancer", "aromatic"] and data.get("is_vegan"):
            spice_mols = set(data.get("flavor_molecules", []))
            gap_covered = len(spice_mols & flavor_gap)
            bridge_score = gap_covered / len(flavor_gap)
            if bridge_score > 0:
                matching = list(spice_mols & flavor_gap)
                bridge_scores.append({
                    "spice": name,
                    "fills_gap_ratio": round(bridge_score, 3),
                    "reason": f"adds {', '.join(matching)} compounds to fill the aromatic gap"
                })

    bridge_scores.sort(key=lambda x: x["fills_gap_ratio"], reverse=True)
    return bridge_scores[:top_k]
```

**Why whitelisted spices, not any ingredient?**  
Without filtering to `flavor_enhancer`/`aromatic` roles, the engine might suggest "durian" or "raw cabbage" to fill sulfur compound gaps — which would be culinarily disastrous. The whitelist ensures only practical kitchen additions are recommended.

---

### 8.4 Delta Recommendations (Compensation Blueprint)

Function: `calculate_delta_recommendations(orig_name, sub_name, orig_data, sub_data, archetype)`

**Fat deficit detection**:
```python
delta_fat = orig_macros.get("fats", 0) - sub_macros.get("fats", 0)
if delta_fat > 10.0:  # More than 10g fat difference per 100g
    if archetype in ["Curry", "Dry_Sabzi", "Soup"]:
        additions.append({"name": "coconut oil or neutral vegetable oil", "amount": "1-2 tsp",
                          "purpose": "matches lipid profile to ensure proper fat-soluble spice absorption"})
    elif archetype == "Salad":
        additions.append({"name": "cold-pressed olive oil", "amount": "1-2 tsp",
                          "purpose": "drizzle over substitute to replicate the fat mouthfeel"})
```

**Umami bridging**:
```python
if "diacetyl" in orig_data.get("flavor_molecules", []) and "diacetyl" not in sub_data.get("flavor_molecules", []):
    additions.append({"name": "nutritional yeast", "amount": "1 tsp",
                      "purpose": "adds savory, buttery dairy-like notes missing from the plant substitute"})
elif any(x in ["hydrogen sulfide", "2-methyl-3-furanthiol", "pyrazines"] for x in orig_data.get("flavor_molecules", [])):
    additions.append({"name": "monosodium glutamate (MSG) or soy sauce", "amount": "1/2 tsp",
                      "purpose": "bridges the savory meat umami profile"})
```

**Moisture adjustments**:
```python
delta_moisture = orig_text[2] - sub_text[2]  # texture_profile index 2 = moisture
if delta_moisture < -2.0:
    # Substitute is too wet — press it
    techniques.append(f"Wrap the {sub_name} in a clean kitchen towel and press it under a heavy object for 15 minutes...")
elif delta_moisture > 4.0:
    if sub_name == "soya chunks":
        techniques.append("Soak the soya chunks in hot water for 15 minutes and squeeze the excess water out...")
    else:
        techniques.append(f"Hydrate the dry {sub_name} by soaking in water or simmering before adding...")
```

**Texture-specific techniques**:
```python
if sub_name == "king oyster mushroom":
    techniques.append("Slice the king oyster mushroom stalks into round coins and score them in a cross-hatch pattern, then sauté in oil to mimic the firm, bouncy bite of shrimp.")
elif delta_chewiness > 3.0:
    if sub_name == "soya chunks":
        techniques.append("Soak in boiling water for 15 minutes, squeeze out completely, and pan-fry before adding to sauce to mimic poultry fibers.")
    elif sub_name == "tofu":
        techniques.append("Freeze and thaw the tofu beforehand to open up its pores, then pan-sear until golden-brown to create a chewy surface texture.")
    elif sub_name == "jackfruit":
        techniques.append("Simmer the jackfruit pieces until tender, then shred with two forks to mimic pulled meat fibers.")
```

---

### 8.5 Static Fallback Profiles

When an ingredient is not in `chemical_features` and can't be bootstrapped, keyword classification maps it to one of 7 archetype profiles:

| Profile | Keywords | Representative |
|---|---|---|
| `red_meat` | mutton, lamb, pork, beef, goat, steak, ham, bacon | `proteins:26, fats:17`, molecules: heptanal, hexanal, 2-methyl-3-furanthiol |
| `poultry` | chicken, duck, turkey, quail | `proteins:27, fats:14`, molecules: inosine monophosphate, methional |
| `seafood` | shrimp, prawn, fish, salmon, crab, lobster | `proteins:24, fats:1`, molecules: trimethylamine, dimethyl sulfide |
| `dairy_fat` | butter, ghee, lard | `proteins:1, fats:81`, molecules: diacetyl, butyric acid, delta-decalactone |
| `dairy_liquid` | milk, cream, yogurt, curd, cheese, paneer | `proteins:3.4, fats:3.7`, molecules: lactose, casein, diacetyl |
| `sweetener` | honey | `carbs:99.9`, molecules: sucrose, glucose, fructose |
| `egg` | egg, eggs (not eggplant) | `proteins:13, fats:10`, role: binding_agent |

---

## 9. How We Built the Offline Databases

### 9.1 `chemical_features` Database

**Notebook**: `GPU_Open_World_Vegan_DB_Builder.ipynb`  
**Environment**: Google Colab T4 GPU  
**Model**: `Qwen/Qwen3-8B` (4-bit NF4 quantization)

**The offline generation prompt** (exact):

```
System: "You are a food chemistry expert. Output JSON only."

User: "You are a food chemistry and culinary database. Output only valid JSON. No explanations.
Generate a chemical and physical profile for the ingredient '{ingredient}' matching the specified JSON format.
Determine if it is vegan (true/false).
Specify macros (fat, protein, carb, water as ratios summing to 1.0).
Rate its texture on a 1-5 scale: [hardness, chewiness, fibrousness, moisture, elasticity, granularity].
List 3-5 primary flavor volatile compounds.
Assign a culinary role: [bulk_protein, fat_source, binder, creamy_liquid, sweetener, seasoning, veggie, starch]."
```

**Ingredient vocabulary**: ~350 real cooking ingredients, curated manually after rejecting two failed approaches:

| Attempt | Problem | Rejected because |
|---|---|---|
| Use `bounds_dict` keys from GitHub | 60,069 entries including recipe phrases like "a box instant pistachio pudding mix" | Garbage data |
| Ask Qwen3-8B to generate vocabulary | Model looped: "almond butter, almond milk, almond oil..." for 2048 tokens | LLM repetition degeneration |
| **Final: curated list of ~350 real items** | Clean, covers all mainstream Indian cooking | ✅ Used in production |

**Storage**: Each profile stored as a MongoDB document in `ratatouille.chemical_features` with `_id = ingredient_name`.

**Local backup**: `chemical_features.json` (135 KB) — write-through cache.

---

### 9.2 `vegan_alternatives` Match Table

**Same notebook** (`GPU_Open_World_Vegan_DB_Builder.ipynb`), Phase 2:

After all profiles are generated, the notebook runs the matching algorithm in Python (CPU-only math):

1. **Self-classify** all profiles into `vegan_pool` and `non_vegan_pool` based on `is_vegan` field
2. **For each non-vegan ingredient**, compute `calculate_composite_score()` against every vegan candidate
3. **Sort** candidates by score descending
4. **Store** the top-5 results + compensation blueprint in `ratatouille.vegan_alternatives`

**Why no preconceived pairs?** We never tell the system "jackfruit could substitute chicken." We put jackfruit in the vocabulary, let Qwen3-8B profile it independently, and let the math decide if it's similar to chicken. If soy chunks score higher, soy chunks win.

---

## 10. All LLM Prompts (Production Code)

### Prompt 1: Archetype Classification (Groq — Primary)

```python
messages = [
    {"role": "system", "content": "You are a recipe classification assistant. Output ONLY a single word classification from the permitted list. No explanation."},
    {"role": "user",   "content": f"Classify the dish structure based on these ingredients: [{ingr_str}].\nChoose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish]."}
]
# max_tokens=15, temperature=0.1
```

### Prompt 2: Archetype Classification (Gradio — Fallback)

```
<|begin_of_text|>Classify the dish structure based on these ingredients: [{ingr_str}].
Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish].
Return ONLY the word.

### INGREDIENTS:
{ingr_str}
### ARCHETYPE:

```

### Prompt 3: Ingredient Deconstruction (Groq — Primary)

```python
messages = [
    {"role": "system", "content": "You are a food chemistry and agricultural database. Output only valid JSON. Do not write any explanations or conversational text outside the JSON."},
    {"role": "user",   "content": f"Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops with approximate weight percentages (total summing to 1.0).\nExample: for 'tomato ketchup', return exactly: {{\"tomato\": 0.8, \"sugar\": 0.1, \"onion\": 0.1}}"}
]
# max_tokens=100, temperature=0.1
```

### Prompt 4: Ingredient Deconstruction (Gradio — Fallback)

```
<|begin_of_text|>Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops.
Assign approximate weight percentages. Return ONLY valid JSON.
Example for 'tomato ketchup': {"tomato": 0.8, "sugar": 0.1, "onion": 0.1}

### INGREDIENT:
{ingredient}
### JSON:

```

### Prompt 5: Chemical Profile Bootstrap (Groq — Runtime, `/get-vegan-blueprint` only)

```python
messages = [
    {"role": "system", "content": "You are a food chemistry and culinary database. Output only valid JSON. No explanations."},
    {"role": "user",   "content": (
        f"Generate a chemical and physical profile for the ingredient '{ingredient_name}' matching the specified JSON format.\n"
        f"Determine if it is vegan (true/false).\n"
        f"Specify macros (proteins, fats, carbs in grams per 100g, summing up to at most 100).\n"
        f"Rate its texture on a 1.0-10.0 scale: [hardness, chewiness, moisture, fat_mouthfeel, elasticity].\n"
        f"List 3-5 primary flavor volatile compounds (e.g. aldehydes, pyrazines, esters, terpenes, or specific molecules).\n"
        f"Assign a culinary role: [base_protein, fat_source, flavor_enhancer, thickener, sweetener, aromatic, dairy, binding_agent].\n\n"
        f"Return ONLY valid JSON matching this exact structure:\n"
        f"{{\n"
        f"  \"is_vegan\": false,\n"
        f"  \"macros\": {{\"proteins\": 22.0, \"fats\": 20.0, \"carbs\": 0.0}},\n"
        f"  \"texture_profile\": [4.5, 4.0, 4.0, 6.0, 2.0],\n"
        f"  \"flavor_molecules\": [\"methanethiol\", \"dimethyl sulfide\", \"pyrazines\"],\n"
        f"  \"culinary_role\": \"base_protein\"\n"
        f"}}\n"
    )}
]
# max_tokens=250, temperature=0.1
```

### Prompt 6: Recipe Generation (Fine-tuned Llama 3, V10 Schema)

```
### INGREDIENTS:
- {qty}g {ingredient_1}
- {qty}g {ingredient_2}
...
### TITLE:

```

Parameters: `max_new_tokens=500`, `temperature=0.6`, `top_p=0.9`, `repetition_penalty=1.05`, `do_sample=True`

### Prompt 7: Offline DB Builder (Qwen3-8B, Colab — NOT runtime)

```
System: "You are a food chemistry expert. Output JSON only."

User: "You are a food chemistry and culinary database. Output only valid JSON. No explanations.
Generate a chemical and physical profile for the ingredient '{ingredient}' matching the specified JSON format.
Determine if it is vegan (true/false).
Specify macros (fat, protein, carb, water as ratios summing to 1.0).
Rate its texture on a 1-5 scale: [hardness, chewiness, fibrousness, moisture, elasticity, granularity].
List 3-5 primary flavor volatile compounds.
Assign a culinary role: [bulk_protein, fat_source, binder, creamy_liquid, sweetener, seasoning, veggie, starch]."
```

---

## 11. All API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Health check — returns Space URLs and DB connection status |
| `POST` | `/generate-recipe` | None | **Main endpoint** — full SSE streaming pipeline |
| `POST` | `/optimize-only` | None | Budget optimization only, no LLM recipe generation. Used by eval scripts. |
| `POST` | `/save-recipe` | None | Saves a recipe to a user's profile in MongoDB |
| `GET` | `/my-recipes/{username}` | None | Returns the last 50 saved recipes for a user |
| `GET` | `/vegan-alternatives/{ingredient}` | None | Returns top-5 pre-computed vegan alternatives for an ingredient |
| `POST` | `/get-vegan-blueprint` | None | Full vegan engine analysis including LLM bootstrap for unknown ingredients |
| `GET` | `/indian-recipes/styles` | None | Returns distinct recipe styles and counts from the library |
| `GET` | `/indian-recipes` | None | Paginated Indian budget recipe library (`?style=Curry&limit=12&skip=0`) |

---

## 12. The Frontend (React SSE Client)

**File**: `frontend/src/App.jsx` (831 lines)

### Views

| `viewMode` | Panel |
|---|---|
| `generate` | Main recipe generation form + result panel |
| `vegan` | Standalone vegan engine debugger (lookup any ingredient's alternatives) |
| `history` | User's saved recipe history |

### State Variables

```javascript
// Form
const [ingredients, setIngredients] = useState('rice, egg, potato, tomato, onion');
const [budget, setBudget]           = useState(150);
const [servings, setServings]       = useState(1);
const [stateName, setStateName]     = useState('Delhi');
const [modelVersion, setModelVersion] = useState('v10');
const [isVegan, setIsVegan]         = useState(false);

// Response
const [loading, setLoading]         = useState(false);
const [stepMessage, setStepMessage] = useState('');  // Live SSE step label
const [result, setResult]           = useState(null);
const [error, setError]             = useState('');

// Timer (shows elapsed seconds while loading)
const [elapsedSeconds, setElapsedSeconds] = useState(0);
```

### SSE Reading Pattern

The frontend uses the native `ReadableStream` API — no external library:

```javascript
const reader = response.body.getReader();
const decoder = new TextDecoder("utf-8");
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    // Parse "data: {...}\n\n" format
    const lines = chunk.split('\n\n');
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.step === 'complete') setResult(data.result);
            else if (data.step === 'error') setError(data.message);
            else setStepMessage(data.message);
        }
    }
}
```

---

## 13. The LLM Cache System

**Collection**: `ratatouille.llm_cache`  
**Client**: `llm_cache_sync` (synchronous PyMongo — needed inside the SSE generator)

### Cache Key Convention

| Call type | Cache key format | Example |
|---|---|---|
| Archetype | `archetype_{sorted_ingredients}` | `archetype_chicken,onion,tomato` |
| Deconstruct | `deconstruct_{ingredient}` | `deconstruct_tomato ketchup` |

### Cache Read Pattern (used in both functions)

```python
cached = llm_cache_sync.find_one({"_id": cache_id})
if cached and "result" in cached:
    print(f"[CACHE HIT] ...")
    return cached["result"]
```

### Cache Write Pattern

```python
llm_cache_sync.update_one(
    {"_id": cache_id},
    {"$set": {"result": result, "timestamp": time.time()}},
    upsert=True  # Creates if not exists, updates if exists
)
```

**Effect**: The second time anyone generates a recipe with the same ingredients (regardless of order), the archetype call is **free** (0ms, no Groq API usage). The same applies to any previously seen processed ingredient.

---

## 14. Key Architectural Decisions & Why

### Decision 1: SSE over WebSockets

**Chosen**: Server-Sent Events (SSE)  
**Why**: SSE is unidirectional (server → client), which is exactly what we need — the client sends one POST and receives a stream of progress updates. WebSockets are bidirectional and more complex to manage. SSE is also natively supported by browsers with `EventSource` or plain `ReadableStream`.

### Decision 2: Sync MongoDB Client Inside SSE Stream

**Chosen**: `pymongo.MongoClient` (sync) for all DB calls inside `generate_recipe()`  
**Why**: The SSE generator function is a regular Python `yield` generator — it is **not** `async`. FastAPI runs it in a thread. Using `async motor` would require `await`, which is not possible inside a sync generator. The sync client is safe here.

### Decision 3: No LLM Bootstrap Inside SSE Stream

**Chosen**: Skip `bootstrap_ingredient_profile()` during recipe generation  
**Why**: The bootstrap calls the Groq API (or Gradio Space with up to 2-min cold start). For a list of 5 ingredients, this could add 5–300 seconds of latency. Unknown ingredients are simply kept as-is — the recipe LLM can handle them. Bootstrap only runs in the standalone `/get-vegan-blueprint` endpoint where the user explicitly requests it.

### Decision 4: 4-Level SciPy Fallback Cascade

**Chosen**: Progressive constraint relaxation  
**Why**: A user's budget (e.g., ₹30) might be mathematically insufficient even for 15g of each ingredient. Hard-failing would be a bad UX. The cascade ensures **something** always comes out of the optimizer, even if it violates the ideal structural ratios.

### Decision 5: Offline Vegan DB vs. Real-Time LLM Substitution

| Approach | Latency | Consistency |
|---|---|---|
| Ask recipe LLM to "make it vegan" | 0s extra | Poor — V10 trained on recipes, not food science |
| Real-time vegan_engine per request | 2–5s per ingredient | Good |
| **Pre-built MongoDB match table** | **~1ms per ingredient** | **Best** |

The database is built once offline and reused forever. This is the correct engineering trade-off.

### Decision 6: Jaccard Similarity over Dense Embeddings for Flavor

Dense embeddings (Word2Vec, BERT) reflect **word co-occurrence** in text, not chemical overlap. Jaccard over discrete volatile molecule names is:
1. **Explainable**: We can compute exactly which molecules are missing (the flavor gap)
2. **Correct for chemistry**: It measures actual shared chemistry, not linguistic association
3. **Fast**: Sub-millisecond Python set operations

---

## 15. File Reference Map

| File | Purpose |
|---|---|
| [`api.py`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/api.py) | FastAPI backend — 1048 lines — entire runtime pipeline |
| [`vegan_engine.py`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_engine.py) | Vegan substitution math engine — 412 lines |
| [`chemical_features.json`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/chemical_features.json) | Local fallback ingredient chemistry DB (135 KB) |
| [`vegan_alternatives_db.json`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_alternatives_db.json) | Local copy of pre-computed match table (64 KB) |
| [`deconstruction_map.json`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/deconstruction_map.json) | Cached LLM deconstruction results (30 KB) |
| [`frontend/src/App.jsx`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/frontend/src/App.jsx) | React UI — 831 lines |
| `GPU_Open_World_Vegan_DB_Builder.ipynb` | Colab notebook — builds `chemical_features` + `vegan_alternatives` |
| `GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb` | Colab notebook — builds `indian_recipes` library |
| `RAT V8 JUST LOAD TRAIN AND SAVE TO HF (UPDATED).ipynb` | V8 model fine-tuning notebook |
| `RAT_V10_NON_OVEN_TRAIN.ipynb` | V10 model fine-tuning notebook |
| [`requirements.txt`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/requirements.txt) | Python backend dependencies |
| [`vegan_engine_architecture.md`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_engine_architecture.md) | Architectural rationale for vegan engine |
| [`vegan_conversion_blueprint.md`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_conversion_blueprint.md) | Design evolution + scoring formula derivation |
| [`vegan_system_explainer.md`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_system_explainer.md) | Layman explanation of 3-phase vegan DB architecture |
| [`system_pipeline_explainer.md`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/system_pipeline_explainer.md) | High-level pipeline walkthrough |

---

*Document generated from ground-truth source reading of all production files. Every prompt, formula, and decision shown above is derived directly from the codebase.*
