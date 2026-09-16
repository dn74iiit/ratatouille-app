# 🌱 Ratatouille — Vegan Substitution System: Full Explainer

## What is Ratatouille?

Ratatouille is an **AI-powered Indian budget recipe generator**. A user enters:
- A list of ingredients they have at home
- Their budget (in INR)
- Their dietary preference (including a **"Make it Vegan"** toggle)

The app then generates a complete, budget-optimised Indian recipe using a fine-tuned LLM (Llama 3.2 3B, our V10 model).

---

## The Problem: "Make it Vegan"

When a user toggles **Make it Vegan**, the app receives ingredients like:

```
["200g chicken", "paneer", "2 eggs", "ghee", "mutton"]
```

It needs to convert these into plant-based equivalents *before* the recipe is generated — because the LLM is a recipe writer, not a food scientist. If you hand it `chicken` and say "make it vegan", it will either ignore the request or hallucinate a bad substitution.

> **Core challenge**: How do you replace `chicken` with the most chemically and texturally similar plant-based ingredient — without just hardcoding `chicken → tofu`?

---

## Why Not Just Hardcode Substitutions?

The obvious approach would be a lookup table:

```python
SUBSTITUTIONS = {
    "chicken": "tofu",
    "paneer":  "tofu",
    "ghee":    "coconut oil",
    "egg":     "flax egg",
}
```

**This is wrong for several reasons:**

| Problem | Why it matters |
|---|---|
| `paneer → tofu` and `chicken → tofu` are the same | They have completely different textures and roles in a recipe |
| No chemical compensation | Ghee has fat that coconut oil partially replicates, but nothing bridges the aroma gap |
| No cooking guidance | Tofu needs pressing; jackfruit needs shredding — the recipe won't reflect this |
| Can't discover better matches | What if `soy chunks` are a better match for chicken than tofu in a specific context? |

---

## Our Solution: Profile-Based Open-World Matching

Instead of hardcoding pairs, we built a **food chemistry matching engine**.

### The Core Idea

Every food ingredient has a chemical-physical profile that can be described by:

```json
{
  "is_vegan": false,
  "macros": { "fat": 0.14, "protein": 0.27, "carb": 0.0, "water": 0.59 },
  "texture": [3, 4, 3, 3, 2, 2],
  "flavor_molecules": ["methanethiol", "dimethyl_sulfide", "pyrazines", "aldehydes"],
  "role": "bulk_protein"
}
```

- `texture` = a 6-dimensional vector `[hardness, chewiness, fibrousness, moisture, elasticity, granularity]` rated 1–5
- `flavor_molecules` = the primary volatile aroma compounds
- `role` = the culinary function (bulk protein, fat source, binder, etc.)

If we profile **every ingredient** and represent it as a point in this multi-dimensional space, then finding the best vegan substitute for `chicken` is simply a **nearest-neighbour search** in that space.

### The Matching Math

```
composite_score = 0.30 × flavor_jaccard
               + 0.40 × texture_euclidean
               + 0.30 × role_match

flavor_jaccard      = |shared flavor molecules| / |all flavor molecules|
texture_euclidean   = 1 - (Euclidean distance in 6D space / max possible distance)
role_match          = 1.0 if same culinary role, 0.0 otherwise
```

The ingredient in the vegan pool with the **highest composite score** is selected as the best substitute. No human decision involved.

---

## The Three-Phase Architecture

### Phase 1 — Profile Generation (Notebook, GPU)

We run `GPU_Open_World_Vegan_DB_Builder.ipynb` once on Google Colab T4 GPU.

**Model used**: `Qwen/Qwen3-8B` in 4-bit NF4 quantisation  
**Why not our V10 Llama model?** Our V10 was fine-tuned to *write recipes*. It has no training signal for structured food chemistry data. Qwen3-8B is a general-purpose model with deep scientific knowledge — it's the right tool for this job.

The notebook generates a chemical-physical profile for every ingredient in our vocabulary (~350 real cooking ingredients) and stores them in:

```
MongoDB Atlas → ratatouille.chemical_features
```

Each document looks like:
```json
{
  "_id": "chicken",
  "is_vegan": false,
  "macros": { "fat": 0.14, "protein": 0.27, "carb": 0.0, "water": 0.59 },
  "texture": [3, 4, 3, 3, 2, 2],
  "flavor_molecules": ["methanethiol", "dimethyl_sulfide", "pyrazines"],
  "role": "bulk_protein"
}
```

### Phase 2 — Self-Classification + Open-World Matching (Notebook, CPU)

After profiling, the notebook reads all profiles from MongoDB and:

1. **Self-classifies** into two pools based on the model's own `is_vegan` output:
   - `non_vegan_pool` → the items that need substitutes (chicken, paneer, ghee…)
   - `vegan_pool` → the entire candidate search space (tofu, jackfruit, soy chunks, coconut milk…)

2. **For each non-vegan ingredient**, scores it against **every** item in the vegan pool using the composite score formula above

3. Picks the winner (highest score) and also stores the top-5 alternatives

> **Why no preconceived pairs?**  
> We never tell the system "jackfruit could substitute chicken." We just put `jackfruit` in the ingredient vocabulary, let the model profile it independently, and let the math discover whether it's actually chemically similar to `chicken`. If it is, it wins. If `soy chunks` scores higher, that wins instead.

### Phase 3 — Match Table Storage (MongoDB)

The results are stored in:

```
MongoDB Atlas → ratatouille.vegan_alternatives
```

Each document:
```json
{
  "_id": "chicken",
  "original_ingredient": "chicken",
  "best_vegan_substitute": "soy chunks",
  "match_score": 0.74,
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
      { "spice": "smoked paprika", "fills_gap_ratio": 0.67, "supplies": ["pyrazines"] }
    ]
  }
}
```

The `compensation_blueprint` answers: *even if soy chunks is the best match, what do you need to add to make it taste more like chicken?*

---

## How This Connects to the API (`api.py`)

When a user hits `/generate-recipe` with `is_vegan: true`, the API runs this lookup for each ingredient:

```
                    ┌─────────────────────────────────────┐
                    │  User sends: "chicken", is_vegan=True│
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  STEP 1: MongoDB lookup (~1ms)       │
                    │  db.vegan_alternatives.find("chicken")│
                    └──────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐   ┌─────────────────────────────┐
              │   DB HIT ✅              │   │   DB MISS ⚠️                │
              │   "soy chunks" returned │   │   Live vegan_engine fallback │
              │   instantly             │   │   (slower, ~2-5s per item)   │
              └────────────┬────────────┘   └──────────────┬──────────────┘
                           │                               │
                    ┌──────▼───────────────────────────────▼──────┐
                    │  STEP 3: Apply substitution + compensation    │
                    │  → "soy chunks" replaces "chicken"            │
                    │  → "1 tsp coconut oil" added                  │
                    │  → "smoked paprika" added (spice bridge)      │
                    └─────────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │  Modified ingredient list passed to          │
                    │  Cost Constraint Optimizer → V10 LLM        │
                    │  → Recipe generated with vegan ingredients   │
                    └─────────────────────────────────────────────┘
```

---

## Why This Approach Is Better Than Real-Time LLM Substitution

| Approach | Latency | Quality | Consistency |
|---|---|---|---|
| Ask the recipe LLM to "make it vegan" inline | 0s extra | Poor — V10 was trained on recipes, not food science | Inconsistent |
| Real-time vegan_engine calculation per request | 2–5s per ingredient | Good | Consistent |
| **Pre-built MongoDB match table (our approach)** | **~1ms per ingredient** | **Best — Qwen3-8B + math** | **Consistent** |

The database is built once offline and reused forever. The recipe LLM's job stays clean: it receives a fully-substituted ingredient list and just writes a recipe — which is exactly what it was trained to do.

---

## Vocabulary Design Decision

The matching is only as good as the candidate pool. We went through several approaches:

| Attempt | What happened | Why rejected |
|---|---|---|
| `bounds_dict` keys from GitHub | 60,069 entries — full recipe phrases like *"a box instant pistachio pudding mix buy a box"* | Garbage data, not ingredient names |
| Ask Qwen3-8B to generate vegan vocabulary | Model looped on "almond butter, almond milk, almond oil…" for 2048 tokens | LLM repetition degeneration on long lists |
| **Curated list of ~350 real cooking ingredients** | Clean, correct, covers all mainstream Indian cooking | ✅ **This is what we use** |

> **Important clarification on "preconceived":**  
> The curated list defines *what food items exist in the world* — this is unavoidable in any finite system (even the USDA food database must define its boundary).  
> What we do NOT preconceive is *which vegan item maps to which non-vegan item* — that is purely determined by the math.  
> `chicken → tofu` is a preconceived **pair**. `chicken scored against [tofu, soy chunks, jackfruit, seitan, ...]` is an open-world **search**.

---

## Files Involved

| File | Role |
|---|---|
| [`GPU_Open_World_Vegan_DB_Builder.ipynb`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/GPU_Open_World_Vegan_DB_Builder.ipynb) | Colab notebook — profiles all ingredients, runs matching, uploads to MongoDB |
| [`build_notebooks.py`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/build_notebooks.py) | Source that generates the `.ipynb` file |
| [`vegan_engine.py`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_engine.py) | Live fallback engine (used when ingredient not in MongoDB) |
| [`api.py`](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/api.py) | FastAPI backend — DB-first lookup → fallback → apply substitution |
| `MongoDB: ratatouille.chemical_features` | All ingredient profiles |
| `MongoDB: ratatouille.vegan_alternatives` | Pre-computed match table (non-vegan → best vegan substitute) |
