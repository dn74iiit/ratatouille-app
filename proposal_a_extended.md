# Proposal A (Revised): FlavorDB-Grounded GraphRAG for Ratatouille

## What Changed in This Revision

> [!IMPORTANT]
> **Removed:** The general-purpose `/smart-substitute` endpoint, allergy/unavailability substitution, and the hybrid query router. These don't fit Ratatouille's core design — your user provides the ingredients they want, and the system generates a recipe with *those* ingredients within budget. Adding "replace peanuts because you're allergic" is a different product.
>
> **Added:** Recipe Grounding RAG (Phase 3) — using your existing 50K recipe corpus to validate V10's generated output. This directly improves the existing pipeline's output quality instead of adding a new out-of-scope feature.

### Where GraphRAG now touches your pipeline (revised)

```
User Input: [chicken, tomato, onion], ₹100, vegan=True
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 1: Vegan Substitution  ← ★ PHASE 2 + PHASE 4  │
│   Graph traversal with REAL FlavorDB2 molecules      │
│   Agentic loop validates substitute quality          │
│   Spice bridge from real molecule gap analysis       │
│   Compensation from real FooDB macros                │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 2: Dish Archetype Classification               │
│   (unchanged — Groq LLM call, cached)               │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 3: SciPy Optimizer                             │
│   Now uses real FooDB macros for delta calculations  │
│   (data improvement from Phase 1, no code change)    │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 4: V10 Recipe Generation                       │
│   (unchanged — fine-tuned Llama 3 on HF Spaces)     │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 5: Post-Processing  ← ★ PHASE 3               │
│   NEW: Recipe Grounding RAG                          │
│   Retrieve 3 similar REAL recipes from 50K corpus    │
│   LLM validates cooking techniques & proportions     │
│   Confidence badge + "Grounded By" references        │
└──────────────────────────────────────────────────────┘
```

**Every piece of GraphRAG work now maps to an existing pipeline stage.** No new user-facing features that change the product's scope.

---

## Time Budget

| Parameter | Value |
|---|---|
| Duration | 3 months (~12 weeks) |
| Weekly commitment | 3 hours |
| **Total available hours** | **~36 hours** |

---

## The 4 Phases

---

### Phase 1: Real Data Foundation (Weeks 1–3, 9 hours)

> **Goal:** Replace LLM-generated `chemical_features.json` with scientifically curated data from FlavorDB2 + FooDB.

*Unchanged from previous version — this is the critical foundation.*

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 1 | 3 hrs | **Obtain FlavorDB2 data.** Download pre-scraped CSV from `wannasleepforlong/flavordb` on GitHub (contains `flavordb.csv` + `molecules.csv`). If coverage is insufficient for your 350 ingredients, supplement by scraping FlavorDB2's JSON endpoint (`entities_json?id=x`) for missing items. Extract `{ingredient_name, category, molecules[], is_vegan}`. | `flavordb_molecules.json` |
| 2 | 3 hrs | **Download and process FooDB.** Download CSV bulk dump from [foodb.ca/downloads](https://foodb.ca/downloads). Extract `foods.csv` + `nutrients.csv`. Pandas script to produce `{food_name, proteins_per_100g, fats_per_100g, carbs_per_100g}` for your 350 ingredients. | `foodb_macros.json` |
| 3 | 3 hrs | **Merge into `chemical_features_real.json`.** Fuzzy-match ingredient names (using `rapidfuzz`) across FlavorDB2, FooDB, and your current file. Merge: **molecules** from FlavorDB2 (replacing LLM-hallucinated ones), **macros** from FooDB (replacing LLM-estimated ones), keep **culinary_role** and **texture_profile** from current data. Document match rates. | `chemical_features_real.json` + match report |

> [!TIP]
> **Why keep texture_profile from LLM data?** Texture is a 5D ordinal vector (hardness, chewiness, etc.) that no public database provides in this exact format. Your LLM values are reasonable for relative ordering. The critical data to replace is **flavor molecules** (where LLM hallucination directly corrupts Jaccard similarity) and **macros** (where precise numbers matter for delta/compensation calculations).

---

### Phase 2: Knowledge Graph + Vegan Engine Upgrade (Weeks 4–6, 9 hours)

> **Goal:** Build a knowledge graph from the real data and upgrade the vegan engine to use graph traversal instead of flat vector math.

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 4 | 3 hrs | **Build the knowledge graph.** Load `chemical_features_real.json` into a `networkx` graph. | `food_knowledge_graph.graphml` |

**Graph structure:**

```
Node Types:
  - Ingredient (350 nodes) — attrs: is_vegan, culinary_role, macros, texture_profile
  - Molecule (~500-1000 nodes from FlavorDB2)
  - Role (14 nodes: bulk_protein, fat_source, binder, etc.)

Edge Types:
  - has_molecule (Ingredient → Molecule)
  - has_role (Ingredient → Role)
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 5 | 3 hrs | **Build graph-powered substitution functions.** These replace the flat Jaccard/Euclidean calculations in `vegan_engine.py` with graph traversal equivalents: | `graph_rag.py` module |

```python
def find_vegan_substitutes_graph(ingredient: str, top_k: int = 5) -> list:
    """
    Traverse shared-molecule edges to find vegan ingredients 
    with highest molecular overlap (replaces flat Jaccard).
    """
    orig_molecules = {n for n in G.neighbors(ingredient) 
                      if G.nodes[n]["type"] == "molecule"}
    
    candidates = []
    for node in G.nodes:
        if (G.nodes[node].get("type") == "ingredient" 
            and G.nodes[node].get("is_vegan") 
            and node != ingredient):
            
            cand_molecules = {n for n in G.neighbors(node) 
                              if G.nodes[n]["type"] == "molecule"}
            
            # Graph-based Jaccard: shared edges / total edges
            jaccard = len(orig_molecules & cand_molecules) / len(orig_molecules | cand_molecules)
            
            # Still use texture distance (from node attributes)
            texture_sim = 1.0 - euclidean_distance(
                G.nodes[ingredient]["texture"], G.nodes[node]["texture"]
            ) / 20.124
            
            # Still use role match (from graph edges)
            role_match = 1.0 if (set(G.neighbors(ingredient)) & role_nodes) == \
                                (set(G.neighbors(node)) & role_nodes) else 0.0
            
            score = 0.3 * jaccard + 0.4 * texture_sim + 0.3 * role_match
            candidates.append((node, score, jaccard))
    
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]


def trace_molecule_gap(original: str, substitute: str) -> dict:
    """
    Return shared molecules, missing molecules, and spice bridges.
    This is the graph-native version of get_spice_bridge().
    """
    orig_mols = {n for n in G.neighbors(original) if G.nodes[n]["type"] == "molecule"}
    sub_mols = {n for n in G.neighbors(substitute) if G.nodes[n]["type"] == "molecule"}
    
    gap = orig_mols - sub_mols
    shared = orig_mols & sub_mols
    
    # Find spices/aromatics connected to gap molecules (graph traversal)
    bridges = []
    for mol in gap:
        for neighbor in G.neighbors(mol):
            if (G.nodes[neighbor].get("type") == "ingredient" 
                and G.nodes[neighbor].get("culinary_role") in ["flavor_enhancer", "aromatic"]
                and G.nodes[neighbor].get("is_vegan")):
                bridges.append({
                    "spice": neighbor,
                    "covers_molecule": mol,
                    "reason": f"contains {mol}, filling the aromatic gap from {original}"
                })
    
    return {"shared": list(shared), "gap": list(gap), "bridges": bridges}
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 6 | 3 hrs | **Integrate into the vegan pipeline.** Modify `vegan_engine.py` to use the graph functions instead of the old flat-math approach. The `find_best_substitute()` function now calls `find_vegan_substitutes_graph()`. The `get_spice_bridge()` function now calls `trace_molecule_gap()`. Compensation blueprint still uses `calculate_delta_recommendations()` but now with **real FooDB macros** instead of LLM-estimated ones. | Updated `vegan_engine.py` |

**What the user sees (unchanged UX, better data):**
The user still checks "vegan", clicks generate, and gets a recipe. But internally:
- Substitutes are now based on **real FlavorDB2 molecule overlaps**, not LLM-hallucinated ones
- Spice bridges are computed by **tracing actual molecular paths** in the graph
- Fat/protein compensations use **real FooDB nutritional values**

---

### Phase 3: Recipe Grounding RAG (Weeks 7–9, 9 hours)

> **Goal:** After V10 generates a recipe, retrieve the 3 most similar REAL recipes from your 50K training corpus and use them to validate the AI output. This adds a quality assurance layer to Stage 5 of your pipeline.

**Why this fits your project perfectly:**
- Your V10 model occasionally generates weird cooking steps or unusual ingredient proportions (hallucinations)
- You already have the ground-truth data (50K recipes the model was trained on)
- This directly improves the quality of the existing `/generate-recipe` endpoint — no new product scope

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 7 | 3 hrs | **Embed the recipe corpus.** Write a one-time Colab script: load `final_clean_50k_recipes_grams.csv`, for each recipe concatenate its ingredient list into a single text string, generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`. Store in a FAISS index. Save as a downloadable `.faiss` file (~50–100 MB). | `recipe_vectors.faiss` + `recipe_metadata.json` |

```python
# One-time embedding script (Colab)
from sentence_transformers import SentenceTransformer
import faiss, pandas as pd, json

df = pd.read_csv("final_clean_50k_recipes_grams.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
metadata = []
for _, row in df.iterrows():
    ingredient_text = row["ingredients"]  # or however the column is named
    texts.append(ingredient_text)
    metadata.append({
        "title": row.get("title", ""),
        "ingredients": ingredient_text,
        "directions": row.get("directions", "")
    })

embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)

index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (cosine after normalization)
faiss.normalize_L2(embeddings)
index.add(embeddings)
faiss.write_index(index, "recipe_vectors.faiss")

with open("recipe_metadata.json", "w") as f:
    json.dump(metadata, f)
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 8 | 3 hrs | **Build the grounding check.** After V10 generates a recipe, embed the generated ingredient list, retrieve Top-3 most similar real recipes from FAISS, and send them to Groq for validation: | `recipe_grounding.py` module |

```python
def ground_recipe(generated_ingredients: list, generated_directions: str) -> dict:
    """
    Retrieve similar real recipes and validate the AI output against them.
    """
    # Step 1: Embed the generated recipe's ingredients
    query_text = ", ".join(generated_ingredients)
    query_vec = embedding_model.encode([query_text])
    faiss.normalize_L2(query_vec)
    
    # Step 2: Retrieve Top-3 similar real recipes
    distances, indices = faiss_index.search(query_vec, k=3)
    references = [recipe_metadata[i] for i in indices[0]]
    
    # Step 3: Ask Groq to validate
    prompt = f"""You are a culinary expert reviewing an AI-generated recipe.

AI-GENERATED RECIPE:
Ingredients: {', '.join(generated_ingredients)}
Directions: {generated_directions}

REFERENCE RECIPES FROM REAL COOKBOOKS (similar ingredients):
{chr(10).join(f"Recipe {i+1}: {ref['title']}{chr(10)}Ingredients: {ref['ingredients']}{chr(10)}Directions: {ref['directions'][:200]}..." for i, ref in enumerate(references))}

Compare the AI recipe against the reference recipes. In 2-3 sentences:
1. Are the cooking techniques consistent with how these ingredients are traditionally used?
2. Are the ingredient proportions reasonable compared to the real recipes?
3. Flag any steps that seem unsupported by the reference recipes."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0.2
    )
    
    assessment = response.choices[0].message.content
    avg_similarity = float(distances[0].mean())
    
    return {
        "confidence": "high" if avg_similarity > 0.75 else "medium" if avg_similarity > 0.5 else "low",
        "similarity_score": round(avg_similarity, 3),
        "reference_recipes": [{"title": r["title"], "ingredients": r["ingredients"]} for r in references],
        "assessment": assessment
    }
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 9 | 3 hrs | **Integrate into the SSE pipeline + frontend.** Add a new SSE event `grounding` after `complete`. Call `ground_recipe()` with the V10 output. In the React frontend, add a collapsible "Recipe Validation" section below the generated recipe showing: confidence badge (green/yellow/red), the 3 reference recipe titles, and the LLM's assessment. | Updated `api.py` + React component |

**New SSE event flow:**
```
starting → [veganizing] → optimizing → generating → complete → grounding
                                                                   │
                                                          "Validating recipe
                                                           against 50K real
                                                           recipes..."
```

**What the user sees:**
After the recipe is generated, a brief "Validating..." step runs, then a panel appears:

```
┌─────────────────────────────────────────────────┐
│  ✅ Recipe Confidence: HIGH (0.82)              │
│                                                 │
│  Grounded by 3 similar real recipes:            │
│  1. "Soy Chunks Masala Curry"                   │
│  2. "Textured Protein Tomato Gravy"             │
│  3. "Mock Chicken Onion Stew"                   │
│                                                 │
│  Assessment: "The cooking technique of          │
│  rehydrating soy chunks before sautéing with    │
│  tomato-onion base is consistent with all 3     │
│  reference recipes. The 85g:120g protein-to-    │
│  base ratio aligns with Reference 1."           │
└─────────────────────────────────────────────────┘
```

---

### Phase 4: Agentic Vegan Loop + Evaluation (Weeks 10–12, 9 hours)

> **Goal:** Add an iterative validation loop to the vegan substitution (agentic pattern), then run a comprehensive evaluation comparing old engine vs. new GraphRAG engine.

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 10 | 3 hrs | **Build the agentic vegan loop.** Instead of returning the first graph-traversal result, iterate up to 3 candidates until one meets a quality threshold: | Updated `vegan_engine.py` |

```python
def agentic_vegan_substitution(ingredient: str, archetype: str) -> dict:
    """
    Agentic pattern: Retrieve → Evaluate → Accept/Retry.
    Scoped ONLY to vegan substitution within the existing pipeline.
    """
    candidates = find_vegan_substitutes_graph(ingredient, top_k=5)
    
    for iteration, (candidate_name, score, jaccard) in enumerate(candidates[:3]):
        if score >= 0.60:
            # ACCEPT: generate compensation + explanation
            gap_analysis = trace_molecule_gap(ingredient, candidate_name)
            compensation = calculate_delta_recommendations(
                ingredient, candidate_name,
                chemical_features[ingredient], chemical_features[candidate_name],
                archetype
            )
            
            # Graph-grounded explanation via Groq
            explanation = groq_explain(
                ingredient, candidate_name, gap_analysis, score
            )
            
            return {
                "substitute": candidate_name,
                "score": round(score, 3),
                "iterations": iteration + 1,
                "graph_evidence": gap_analysis,
                "compensation": compensation,
                "explanation": explanation,
                "confidence": "high" if score >= 0.70 else "medium"
            }
    
    # Fallback: best available
    best = candidates[0]
    return {
        "substitute": best[0], "score": round(best[1], 3),
        "iterations": 3, "confidence": "low",
        "note": "Best available substitute below quality threshold"
    }
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 11 | 3 hrs | **Evaluation: Old vs. New.** Run both systems on 30 test ingredients. Compare: | Evaluation report |

**What to measure:**

| Metric | How |
|---|---|
| **Substitute accuracy** | Did the top substitute change when using real FlavorDB2 data? |
| **Score shift** | Did composite scores go up or down with real data? |
| **Molecule coverage** | How many of the original's molecules does the substitute actually share? (verifiable against FlavorDB2) |
| **Agentic iterations** | How often did the loop need >1 iteration? |
| **Grounding confidence** | What's the average recipe grounding score from Phase 3? |

**Example evaluation table:**

| Ingredient | Old Substitute (LLM data) | New Substitute (FlavorDB2) | Old Score | New Score | Iterations | Molecules Verified? |
|---|---|---|---|---|---|---|
| chicken | soy chunks | soy chunks | 0.74 | 0.71 | 1 | ✅ Real overlap |
| butter | coconut oil | coconut cream | 0.68 | 0.73 | 1 | ✅ Real overlap |
| egg | flax meal | chickpea flour | 0.55 | 0.62 | 2 | ✅ Real overlap |
| cream | coconut cream | cashew cream | 0.72 | 0.76 | 1 | ✅ Real overlap |
| ghee | coconut oil | coconut oil | 0.70 | 0.72 | 1 | ✅ Real overlap |

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 12 | 3 hrs | **Frontend polish + documentation + demo.** | Final deliverables |

Frontend additions:
- Vegan results now show a "Why This Substitute?" expandable panel with molecule gap analysis and spice bridge (data from Phase 2)
- All recipes now show a "Recipe Validation" panel with grounding confidence (data from Phase 3)

Documentation:
- Write up the research approach citing LightRAG, Agentic RAG patterns
- Include the evaluation comparison table
- Prepare a 3-minute demo flow

---

## Summary: What You Deliver

| # | Deliverable | Pipeline Stage | Research Pattern |
|---|---|---|---|
| 1 | `chemical_features_real.json` — real FlavorDB2 molecules + FooDB macros | Foundation for all stages | Data quality upgrade |
| 2 | `food_knowledge_graph.graphml` — networkx graph with ingredient-molecule-role edges | Powers Stage 1 (vegan) | Knowledge Graph construction |
| 3 | Graph-powered vegan substitution with molecule gap analysis and spice bridges | Stage 1: Vegan Substitution | GraphRAG traversal |
| 4 | Agentic vegan loop (retrieve → evaluate → retry) | Stage 1: Vegan Substitution | Agentic RAG (2025/2026) |
| 5 | Recipe Grounding RAG — FAISS retrieval over 50K real recipes + LLM validation | Stage 5: Post-Processing | Vector RAG + grounding |
| 6 | Quantitative evaluation: old engine vs. new GraphRAG on 30 ingredients | Evaluation | Benchmarking |

> [!TIP]
> **The key narrative for your report:** "We identified that our vegan substitution engine was operating on LLM-hallucinated chemical data. We replaced this with peer-reviewed FlavorDB2 and FooDB data, built a food chemistry knowledge graph, and implemented two state-of-the-art RAG patterns: (1) graph-based agentic substitution for the vegan engine, and (2) vector-based recipe grounding to validate AI-generated output against 50,000 real recipes."

---

## Tech Stack Additions (All Free)

| Component | Tool | Purpose |
|---|---|---|
| Real molecule data | FlavorDB2 (GitHub CSV / scrape) | Replace LLM flavor molecules |
| Real nutrition data | FooDB (CSV download) | Replace LLM macros |
| Graph library | `networkx` (Python) | Knowledge graph |
| Fuzzy matching | `rapidfuzz` | Name matching across databases |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Recipe corpus embedding |
| Vector store | `faiss-cpu` | Recipe similarity search |
| RAG LLM | Groq Llama 3.3-70B (existing) | Explanations + validation |

---

## Research References

| Technique | Paper/Source | Year |
|---|---|---|
| Knowledge Graph for food | Ahn et al., "Flavor Network and Principles of Food Pairing" (Nature Sci. Reports) | 2011 |
| FlavorDB | Garg et al., "FlavorDB: a database of flavor molecules" (Nucleic Acids Research, CoSyLab) | 2018 |
| Agentic Retrieve-Evaluate-Retry | Agentic RAG pattern (LangGraph docs, NODES 2025) | 2025 |
| RAG for hallucination detection | Industry best practice, GraphRAG Benchmark (2025/2026) | 2025/2026 |
| Dual-level retrieval concept | LightRAG (ICLR 2025) — we use the low-level entity query pattern | 2025 |
