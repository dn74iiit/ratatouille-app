# Proposal B: KG-RAG — Knowledge Graph-Augmented Recipe Generation for Ratatouille
## (RecipeDB-Only Edition)

---

## Executive Summary

> [!IMPORTANT]
> **Core Research Question:**
> *"Can conditioning a fine-tuned Small Language Model on linearized subgraphs retrieved from an empirically-mined Culinary Knowledge Graph (CKG) reduce hallucination and improve procedural coherence in recipe generation — beyond what zero-shot prompting or fine-tuning alone achieves?"*
>
> **Scope (this version):** The CKG is built **entirely from RecipeDB** (Garg et al., 2022). No external ontologies are required; all structural knowledge — co-occurrence, technique association, ordering — is **mined statistically from the 50K recipe corpus** already in the pipeline.
>
> **What stays untouched:** V10's fine-tuned Llama 3 (HF Spaces), SciPy budget optimizer, Groq API, React frontend, and the 50K corpus.
> **What changes:** The generation prompt is augmented with a structured KG subgraph context block, and a new Culinary Validity Score (CVS) evaluates output quality beyond BLEU/BERTScore.

---

## Motivating Gap

Standard metrics like **BLEU** and **BERTScore** measure surface-level lexical overlap — they cannot detect whether:
- A recipe tells the user to *"add garlic"* but garlic was never listed as an ingredient (**hallucination**)
- A recipe says *"serve, then sauté"* — a physically impossible step order (**procedural incoherence**)
- A recipe simply ignores one of the user-provided ingredients (**coverage failure**)

Fine-tuning alone reduces hallucination somewhat, but at inference time the model has no access to *which* techniques empirically co-occur with the given ingredient set. **KG-RAG** addresses this by injecting relevant culinary context at the prompt level, conditioning the model's generation on statistically-grounded knowledge without requiring any additional training.

---

## Revised End-to-End Architecture

```
User Input: [chicken, tomato, onion], budget: ₹100
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 0: Input Normalization                             │
│   Fuzzy-match user ingredient strings → CKG node IDs    │
│   (handles: "chilli" → "chili", typos, plurals)         │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: CKG Subgraph Retrieval  ← ★ PHASE 1 + 2        │
│   N-hop expansion from seed ingredient nodes:            │
│   - co_occurs_with edges (top-K weighted neighbors)      │
│   - has_technique edges (empirical technique assoc.)     │
│   - precedes edges (step ordering constraints)           │
│   - belongs_to_archetype edges (dish type)               │
│   Output: filtered subgraph → linearized context string  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Dish Archetype Classification                   │
│   (unchanged — Groq LLM call, result cached per session) │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: SciPy Budget Optimizer                          │
│   (unchanged — macro-nutrient + cost optimization)       │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: KG-RAG Recipe Generation  ← ★ PHASE 3          │
│   Prompt = [SYSTEM] + [KG CONTEXT] + [TASK]              │
│   KG context provides: pairings, techniques, step order  │
│   Model = V10 (fine-tuned Llama 3 8B) via HF Spaces      │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 5: CVS Evaluation  ← ★ PHASE 4                    │
│   Culinary Validity Score (3 components):                │
│   (1) Ingredient Coverage  (2) Step Order Fidelity       │
│   (3) Technique Plausibility                             │
│   + LLM-as-Judge (Groq Llama-3.3-70B, rubric scoring)   │
│   Frontend: confidence badge + CVS breakdown panel       │
└──────────────────────────────────────────────────────────┘
```

> **Every stage maps to an existing pipeline component.** The academic contribution lies in (a) the KG construction methodology, (b) the retrieval and linearization design, and (c) the CVS metric.

---

## Time Budget

| Parameter | Value |
|---|---|
| Total duration | 12 weeks (3 months) |
| Weekly commitment | ~3 hours |
| **Total available hours** | **~36 hours** |
| Buffer allocation | ~2 hrs across weeks (absorbed within phases) |

---

## The 4 Phases

---

### Phase 1: RecipeDB Culinary Knowledge Graph — Construction (Weeks 1–3, 9 hours)

> **Goal:** Mine the RecipeDB dataset and your existing 50K corpus to build a statistically grounded Culinary Knowledge Graph (CKG). No external ontologies. All relationships are derived empirically from real recipe data.

#### Data Strategy & Held-Out Split

Before any graph construction, define the corpus split to prevent data leakage:

```
50K Recipe Corpus
├── Training set: 80%  (40K recipes) — used to BUILD the CKG
├── Validation set: 10% (5K recipes) — used to tune retrieval hyperparams
└── Test set: 10%       (5K recipes) — used ONLY for final experiment evaluation
                                        (500 test ingredient sets drawn from here)
```

> [!WARNING]
> **Data leakage risk:** The CKG must be built **only from the training split**. If test recipes contribute to co-occurrence statistics, KG-RAG scores will be inflated. Enforce this split in `build_ckg.py` before any processing.

---

#### Week-by-Week Breakdown

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 1 | 3 hrs | **Download & parse RecipeDB + align with 50K corpus.** From [cosylab.iiitd.edu.in/recipedb](https://cosylab.iiitd.edu.in/recipedb), extract `{recipe_id, title, ingredient_list, cooking_steps, cuisine_region}`. Align with your 50K training corpus by matching recipe titles and ingredient overlap using `rapidfuzz`. Retain only the **40K training split** for graph construction. | `recipedb_aligned_train.json` |
| 2 | 3 hrs | **Extract co-occurrence, technique association, and step ordering.** See detailed NLP pipeline below. | `co_occurrence_matrix.csv` + `technique_ingredient_map.json` + `technique_order_stats.json` |
| 3 | 3 hrs | **Build the CKG with `networkx`.** Load all extracted entities and relationships. Prune low-frequency edges (threshold: co-occurrence < 5 across corpus). Export artifacts. | `culinary_kg.graphml` + `culinary_kg.pkl` + `kg_stats.json` |

---

#### NLP Pipeline for Feature Extraction (Week 2 Detail)

**Step A — Ingredient co-occurrence:**
```python
from collections import defaultdict
from itertools import combinations

co_occurrence = defaultdict(int)

for recipe in train_recipes:
    ingredients = recipe["ingredient_list"]  # Already normalized list
    for pair in combinations(sorted(ingredients), 2):
        co_occurrence[pair] += 1

# Normalize: PMI-weighted frequency
# pmi(a,b) = log[P(a,b) / (P(a) * P(b))]
# Store both raw count and PMI score on each edge
```

**Step B — Technique extraction from steps (rule-based + spaCy):**
```python
# Canonical technique vocabulary (60 terms, curated from RecipeDB)
TECHNIQUE_VOCAB = {
    "sauté", "fry", "stir-fry", "deep-fry", "boil", "simmer",
    "braise", "roast", "bake", "steam", "blanch", "grill",
    "marinate", "season", "blend", "fold", "temper", "reduce",
    "deglaze", "caramelize", "dice", "mince", "chop", "crush",
    # ... full list in culinary_techniques.txt
}

def extract_techniques_from_steps(steps: list[str]) -> list[str]:
    """
    Uses token lemmatization (spaCy) to identify cooking verbs.
    Returns ordered list of techniques as they appear in the recipe.
    """
    doc = nlp(" ".join(steps))
    return [token.lemma_ for token in doc
            if token.lemma_ in TECHNIQUE_VOCAB and token.pos_ == "VERB"]
```

**Step C — Empirical step ordering:**
```python
# Count how often technique A appears before technique B in the same recipe
# Build a technique transition matrix
# Normalize to get P(A precedes B | both A and B present)

from collections import Counter

precedes_counter = Counter()
for recipe in train_recipes:
    techniques = extract_techniques_from_steps(recipe["cooking_steps"])
    for i, t1 in enumerate(techniques):
        for t2 in techniques[i+1:]:
            precedes_counter[(t1, t2)] += 1

# Threshold: include precedes edge if P(A before B) > 0.70 in co-occurring recipes
# This ensures only high-confidence ordering constraints enter the KG
```

---

#### CKG Schema

```
═══════════════════════════════════════════════════════
 NODE TYPES
═══════════════════════════════════════════════════════

 Ingredient  (~500–800 nodes, from RecipeDB vocabulary)
   attrs:
     - name            : str (canonical)
     - aliases         : list[str] (from fuzzy matching)
     - avg_quantity_g  : float (mean across recipes)
     - cuisine_freq    : dict[str, float]  # e.g. {"Indian": 0.73}
     - top_techniques  : list[str]  # top-3 by freq

 Technique   (~60 nodes)
   attrs:
     - name            : str
     - frequency       : float (across all train recipes)
     - typical_duration_min : float (optional, if available)

 DishArchetype  (8 nodes)
   attrs:
     - name            : str
     - canonical_step_template : list[str]  # most common step order

═══════════════════════════════════════════════════════
 EDGE TYPES
═══════════════════════════════════════════════════════

 co_occurs_with  (Ingredient ↔ Ingredient, undirected)
   weight_count : int    (raw co-occurrence frequency)
   weight_pmi   : float  (pointwise mutual information)

 has_technique   (Ingredient → Technique, directed)
   weight        : float  (fraction of recipes using this technique for this ingredient)

 belongs_to_archetype  (Ingredient → DishArchetype, directed)
   weight        : float  (fraction of recipes containing this ingredient that belong to archetype)

 precedes        (Technique → Technique, directed)
   probability   : float  (P(A before B | both present), only if ≥ 0.70)
   support       : int    (number of recipes supporting this constraint)
```

> [!NOTE]
> **Why PMI instead of raw co-occurrence?** Raw counts favor common ingredients like `oil` and `salt` that appear in nearly every recipe — PMI corrects for marginal frequency, surfacing more informative pairings (e.g., `cumin + coriander`, `ginger + garlic`). Both are stored on the edge so the retrieval function can switch weighting strategy.

---

#### Risk Mitigation — Phase 1

| Risk | Mitigation |
|---|---|
| RecipeDB field format changes | Parse defensively; write a validator script that checks all required fields before processing |
| Technique extraction noise (verbs like "love", "enjoy") | POS filter (only `VERB` tokens) + allow-list against `TECHNIQUE_VOCAB` |
| Low-overlap between RecipeDB and 50K corpus | Use title + ingredient Jaccard similarity ≥ 0.5 for alignment; fall back to direct ingredient list processing if title match fails |
| Sparse precedes constraints for rare technique pairs | Enforce `support ≥ 10` before including a `precedes` edge |

---

### Phase 2: Subgraph Retrieval + Linearization Engine (Weeks 4–6, 9 hours)

> **Goal:** Given a user's ingredient list, extract a compact, relevant subgraph from the CKG and linearize it into a structured string for prompt injection. The whole retrieval + linearization pipeline must run in under 300ms.

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 4 | 3 hrs | **Build `kg_retriever.py`.** N-hop expansion with weighted neighbor pruning. Handle graceful fallback when seed ingredients are not in graph. | `kg_retriever.py` |
| 5 | 3 hrs | **Build `kg_linearizer.py`.** Three serialization formats to A/B test. Add prompt token budget guard (truncate at 400 tokens to protect model context). | `kg_linearizer.py` |
| 6 | 3 hrs | **Integrate into `api.py` + benchmark latency.** New `kg_pipeline.py` orchestrator. Latency target: KG retrieval + linearization < 300ms (measured with `time.perf_counter`). | `kg_pipeline.py` + updated `api.py` |

---

#### `kg_retriever.py` — Full Implementation

```python
import networkx as nx
from rapidfuzz import process
from typing import Optional

# Global: loaded once at server startup
CKG: nx.DiGraph = None

def load_ckg(path: str = "culinary_kg.pkl") -> None:
    import pickle
    global CKG
    with open(path, "rb") as f:
        CKG = pickle.load(f)

def retrieve_subgraph(
    ingredients: list[str],
    top_k_neighbors: int = 8,
    hops: int = 1,
    weight_key: str = "weight_pmi"  # or "weight_count"
) -> Optional[nx.DiGraph]:
    """
    Retrieve a relevant subgraph from the CKG for a given ingredient set.
    Returns None if no seed nodes are matched.
    """
    if CKG is None:
        raise RuntimeError("CKG not loaded. Call load_ckg() first.")

    # ── Step 1: Fuzzy-match ingredients to graph nodes ────────────────
    all_nodes = list(CKG.nodes())
    seed_nodes = []
    for ing in ingredients:
        result = process.extractOne(ing.lower(), all_nodes, score_cutoff=78)
        if result:
            seed_nodes.append(result[0])

    if not seed_nodes:
        return None  # Caller should handle: fall back to flat prompt

    # ── Step 2: N-hop expansion with weight-ranked pruning ─────────────
    subgraph_nodes = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            # Ingredient neighbors (co_occurs_with)
            co_neighbors = sorted(
                [n for n in CKG.neighbors(node)
                 if CKG.nodes[n].get("type") == "ingredient"],
                key=lambda n: CKG[node][n].get(weight_key, 0),
                reverse=True
            )[:top_k_neighbors]

            # Technique neighbors (has_technique)
            tech_neighbors = sorted(
                [n for n in CKG.neighbors(node)
                 if CKG.nodes[n].get("type") == "technique"],
                key=lambda n: CKG[node][n].get("weight", 0),
                reverse=True
            )[:4]  # Top-4 techniques per ingredient

            next_frontier.update(co_neighbors + tech_neighbors)

        subgraph_nodes.update(next_frontier)
        frontier = next_frontier

    # ── Step 3: Add technique ordering edges (precedes) ─────────────
    technique_nodes = {n for n in subgraph_nodes
                       if CKG.nodes[n].get("type") == "technique"}
    for t in technique_nodes:
        for successor in CKG.successors(t):
            edge = CKG.edges[t, successor]
            if edge.get("type") == "precedes" and edge.get("probability", 0) >= 0.70:
                subgraph_nodes.add(successor)

    # ── Step 4: Add archetype nodes for seed ingredients ─────────────
    for node in seed_nodes:
        for neighbor in CKG.neighbors(node):
            if CKG.nodes[neighbor].get("type") == "archetype":
                subgraph_nodes.add(neighbor)

    return CKG.subgraph(subgraph_nodes).copy()
```

---

#### `kg_linearizer.py` — Three Formats for Ablation

```python
import json

def linearize(subgraph: nx.DiGraph, fmt: str = "B", max_tokens: int = 400) -> str:
    """
    Convert a subgraph to a structured string for prompt injection.
    fmt: "A" (triplets) | "B" (structured prose) | "C" (JSON-LD)
    Truncates to max_tokens (word-count proxy) to protect context window.
    """
    if fmt == "A":
        return _format_triplets(subgraph, max_tokens)
    elif fmt == "B":
        return _format_prose(subgraph, max_tokens)
    elif fmt == "C":
        return _format_json(subgraph, max_tokens)
    raise ValueError(f"Unknown format: {fmt}")
```

**Format A — Relation Triplets** *(most compact, token-efficient)*
```
chicken  co_occurs_with  tomato        [pmi=1.42, count=3,842]
chicken  co_occurs_with  onion         [pmi=1.31, count=4,120]
chicken  has_technique   sauté         [freq=0.71]
chicken  has_technique   braise        [freq=0.58]
tomato   has_technique   roast         [freq=0.52]
tomato   has_technique   purée         [freq=0.47]
sauté    precedes        simmer        [P=0.83, n=1,200]
simmer   precedes        serve         [P=0.92, n=2,100]
chicken  belongs_to      curry         [freq=0.61]
```

**Format B — Structured Prose** *(most human-readable, best for instruction-tuned models)*
```
CULINARY KNOWLEDGE CONTEXT:

INGREDIENTS:
• chicken  — most often sautéed or braised. Top co-ingredients: tomato (pmi=1.42),
             onion (pmi=1.31), garlic (pmi=1.28). Common dish type: curry (61%).
• tomato   — most often roasted or puréed. Top co-ingredients: chicken, onion, cumin.
• onion    — most often sautéed or caramelized. Foundational aromatic.

RECOMMENDED STEP ORDER (from recipe corpus):
  1. sauté → 2. add aromatics → 3. simmer → 4. season → 5. serve
  (Constraints: sauté before simmer [P=0.83]; simmer before serve [P=0.92])
```

**Format C — JSON-LD (compact)** *(structured, machine-parseable)*
```json
{
  "ingredients": {
    "chicken": {"techniques": ["sauté","braise"], "top_pairs": ["tomato","onion"], "archetype": "curry"},
    "tomato":  {"techniques": ["roast","purée"],  "top_pairs": ["chicken","onion"]},
    "onion":   {"techniques": ["sauté","caramelize"], "top_pairs": ["chicken","tomato"]}
  },
  "step_order": ["sauté","simmer","season","serve"],
  "order_constraints": [["sauté","simmer"],["simmer","serve"]]
}
```

---

#### KG-Conditioned Prompt (replaces current flat prompt in `api.py`)

```python
def build_kg_conditioned_prompt(
    ingredients: list[str],
    archetype: str,
    budget_g: dict,
    linearized_subgraph: str
) -> str:
    return f"""You are a professional recipe generator.

[CULINARY KNOWLEDGE GRAPH CONTEXT — sourced from RecipeDB corpus]
{linearized_subgraph}

[TASK]
Generate a complete recipe using ONLY these ingredients: {', '.join(ingredients)}.
Dish type: {archetype}. Budget-optimized quantities (grams): {budget_g}.

Requirements (strictly enforce):
1. Every ingredient listed above MUST appear in the cooking steps.
2. Follow the STEP ORDER constraints from the KNOWLEDGE block above exactly.
3. Use ONLY cooking techniques mentioned in the KNOWLEDGE block.
4. Do NOT introduce any ingredient not in the user's list.
5. Do NOT skip or reorder mandatory steps.

Output format:
Title: <dish name>
Ingredients:
  - <ingredient>: <quantity>g
Steps:
  1. <step>
  2. <step>
  ...
"""
```

---

#### Risk Mitigation — Phase 2

| Risk | Mitigation |
|---|---|
| Subgraph too large → exceeds prompt token limit | Hard cap: max 20 subgraph nodes, max 400 word tokens via `linearize(..., max_tokens=400)` |
| No CKG match for unusual ingredient | Graceful degradation: fall back to Format B prose with only the matched nodes; if 0 matches, use flat prompt (log as `kg_miss`) |
| Latency > 300ms | Pre-build adjacency index on server startup; profile with `cProfile` if needed |
| Format B produces inconsistent phrasing | Use a template string, not free-form LLM; all text is deterministically generated from graph data |

---

### Phase 3: Baseline Comparison Experiments (Weeks 7–9, 9 hours)

> **Goal:** Run controlled experiments comparing three generation conditions over **1,500 generated recipes** to measure the effect of KG-RAG conditioning on hallucination, ingredient coverage, and procedural alignment.

#### Experimental Design

| Condition | Generation Method | Model |
|---|---|---|
| **C1 — Zero-Shot** | Flat prompt: *"Generate a recipe using {ingredients}."* | Llama 3.1-8B via Groq (no fine-tuning) |
| **C2 — Fine-Tuned** | V10 prompt format, no KG context block | V10 fine-tuned Llama 3 8B (HF Spaces) |
| **C3 — KG-RAG** | KG-conditioned prompt (best linearization format from Phase 2 ablation) | V10 fine-tuned Llama 3 8B (HF Spaces) |

> 500 held-out test ingredient sets × 3 conditions = **1,500 total generated recipes**

The 500 test sets are sampled from the **test split only** (10% hold-out). They are stratified by cuisine region (Indian, Mexican, Italian, Chinese, Mediterranean) to ensure diversity.

---

#### Generation Script (Google Colab)

```python
import json, time
from kg_retriever import load_ckg, retrieve_subgraph
from kg_linearizer import linearize
from kg_pipeline import build_kg_conditioned_prompt, build_flat_prompt, call_model

load_ckg("culinary_kg.pkl")
test_sets = json.load(open("test_ingredient_sets.json"))  # 500 items, test split only

BEST_FORMAT = "B"  # Determined by ablation in Week 9

for condition in ["zero_shot", "finetuned", "kgrag"]:
    results = []
    for item in test_sets:
        ingredient_set = item["ingredients"]
        archetype      = item["archetype"]
        budget_g       = item["budget_g"]
        cuisine        = item["cuisine_region"]

        if condition == "kgrag":
            subgraph = retrieve_subgraph(ingredient_set)
            if subgraph:
                linearized = linearize(subgraph, fmt=BEST_FORMAT)
                prompt = build_kg_conditioned_prompt(ingredient_set, archetype, budget_g, linearized)
                kg_used = True
            else:
                prompt = build_flat_prompt(ingredient_set, archetype, budget_g)
                kg_used = False  # Log as kg_miss
        else:
            prompt = build_flat_prompt(ingredient_set, archetype, budget_g)
            kg_used = False

        t0 = time.perf_counter()
        recipe = call_model(prompt, condition)
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "id": item["id"],
            "cuisine": cuisine,
            "input_ingredients": ingredient_set,
            "archetype": archetype,
            "generated_recipe": recipe,
            "kg_used": kg_used,
            "latency_ms": latency_ms
        })

    with open(f"generated_recipes_{condition}.json", "w") as f:
        json.dump(results, f, indent=2)

print("Done. 1,500 recipes generated.")
```

---

#### Metrics — Week 8

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 7 | 3 hrs | **Generate 1,500 recipes** across 3 conditions as above. Log `kg_miss` rate for KG-RAG condition. | 3× JSON files |
| 8 | 3 hrs | **Standard metrics: BLEU + BERTScore.** Run `sacrebleu` (corpus BLEU) and `bert-score` (F1, `microsoft/deberta-xlarge-mnli`) against reference recipes in test split. Compute mean ± std per condition. Explicitly document metric limitations. | `standard_metrics_results.csv` |
| 9 | 3 hrs | **Linearization ablation.** Re-run 500 KG-RAG recipes under all 3 formats. Evaluate with BERTScore F1 as proxy. Select best-performing format as canonical for CVS evaluation. | `linearization_ablation.csv` + format decision |

**Expected standard-metrics limitations (document in Week 8):**

> BLEU penalizes valid paraphrases ("heat oil" vs. "warm the oil"). BERTScore cannot detect that "boil" before "marinate" is physically impossible. Neither metric checks whether the user's three input ingredients all appear in the steps. These limitations directly motivate CVS (Phase 4).

---

#### Risk Mitigation — Phase 3

| Risk | Mitigation |
|---|---|
| HF Spaces V10 rate-limiting for 1,500 calls | Batch with exponential backoff; run across 2 Colab sessions if needed; log each recipe as it completes |
| Groq API cold-start latency | Cache archetype classification output per ingredient set across all three conditions |
| Test set overlap with CKG training data | Enforced by strict train/test split in Phase 1 |
| Result reproducibility | Set `temperature=0.7, seed=42` for all model calls; log model version and timestamp |

---

### Phase 4: CVS — Culinary Validity Score + LLM-as-Judge (Weeks 10–12, 9 hours)

> **Goal:** Implement a 3-component Culinary Validity Score that acts as a semantic compiler for recipe output, then deploy LLM-as-Judge for reference-free evaluation. Compile the full evaluation report and update the frontend.

#### CVS — Three Components (upgraded from 2)

```python
def culinary_validity_score(
    generated_recipe: dict,
    input_ingredients: list[str],
    culinary_kg: nx.DiGraph
) -> dict:
    """
    Three-component semantic validity metric for recipe evaluation.
    All components scored in [0, 1]. Final CVS = weighted average.
    """

    # ── Component 1: Ingredient Coverage ────────────────────────────────
    # Does every user-provided ingredient appear in the generated steps?

    mentioned = extract_ingredient_mentions(generated_recipe["steps"])
    missing   = [i for i in input_ingredients
                 if not any(fuzzy_match(i, m, threshold=82) for m in mentioned)]
    ingredient_coverage = 1.0 - (len(missing) / max(len(input_ingredients), 1))

    # ── Component 2: Step Order Fidelity ────────────────────────────────
    # Does the generated step sequence respect empirical ordering constraints from the CKG?

    generated_techniques = extract_techniques(generated_recipe["steps"])
    kg_constraints       = get_ordering_constraints(input_ingredients, culinary_kg)
    # Returns: [("sauté","simmer"), ("simmer","serve")] from CKG precedes edges

    violations = sum(
        1 for (a, b) in kg_constraints
        if a in generated_techniques and b in generated_techniques
        and generated_techniques.index(a) > generated_techniques.index(b)
    )
    step_order_score = 1.0 - (violations / max(len(kg_constraints), 1))

    # ── Component 3: Technique Plausibility ─────────────────────────────
    # Does the recipe introduce techniques that are NOT associated with any
    # of the input ingredients in the CKG? Penalty for implausible techniques.

    valid_techniques = set()
    for ing in input_ingredients:
        matched = process.extractOne(ing, list(culinary_kg.nodes()), score_cutoff=78)
        if matched:
            valid_techniques.update(
                n for n in culinary_kg.neighbors(matched[0])
                if culinary_kg.nodes[n].get("type") == "technique"
            )

    if generated_techniques and valid_techniques:
        plausible = [t for t in generated_techniques if t in valid_techniques]
        technique_plausibility = len(plausible) / len(generated_techniques)
    else:
        technique_plausibility = 1.0  # No constraints to violate

    # ── Weighted Final CVS ───────────────────────────────────────────────
    # Ingredient coverage weighted highest (most critical failure mode)
    cvs = (0.45 * ingredient_coverage +
           0.35 * step_order_score    +
           0.20 * technique_plausibility)

    return {
        "ingredient_coverage"  : round(ingredient_coverage, 3),
        "step_order_score"     : round(step_order_score, 3),
        "technique_plausibility": round(technique_plausibility, 3),
        "missing_ingredients"  : missing,
        "order_violations"     : violations,
        "cvs"                  : round(cvs, 3)
    }
```

> [!NOTE]
> **Why three components?** The original 2-component CVS penalizes missing ingredients and wrong step order, but not the case where a model invents techniques (e.g., "deep-fry" when only "boil" and "steam" are empirically associated with the given ingredients). Component 3 closes this gap. The weights (0.45 / 0.35 / 0.20) can be treated as a hyperparameter and reported in the ablation section of the paper.

---

#### Cuisine-Stratified Subgroup Analysis (new in Phase 4)

Because the 500 test sets span 5 cuisine regions, report CVS and LLM-Judge scores **per cuisine** in addition to aggregate scores:

| Condition | Cuisine | CVS | Ingr. Coverage | Step Order | Tech. Plausibility | LLM Judge |
|---|---|---|---|---|---|---|
| KG-RAG | Indian | — | — | — | — | — |
| KG-RAG | Mexican | — | — | — | — | — |
| Fine-Tuned | Indian | — | — | — | — | — |
| … | … | … | … | … | … | … |

> This analysis tests whether KG-RAG benefits are consistent across cuisines or concentrated in cuisine regions that are over-represented in RecipeDB.

---

#### LLM-as-Judge Protocol — Week 11

Random sample: **150 recipes** (50 per condition), stratified by cuisine.

```python
JUDGE_PROMPT = """You are a professional culinary evaluator with expertise in food science.
Rate the following AI-generated recipe on four criteria. Score each from 1–5 (5 = excellent).

RECIPE:
Title: {title}
Ingredients listed: {ingredients}
Steps: {steps}

SCORING CRITERIA:
1. Ingredient–Instruction Alignment
   Does every listed ingredient appear correctly in the cooking steps?
   (1 = many missing; 3 = most used; 5 = all used correctly and meaningfully)

2. Procedural Logic
   Are cooking steps in a physically executable order? (e.g., can't serve before cooking)
   (1 = impossible/illogical; 3 = mostly sensible; 5 = perfectly ordered)

3. Hallucination Control
   Does the recipe introduce ingredients or equipment NOT given by the user?
   (1 = many unlisted additions; 3 = minor extras; 5 = no hallucinations)

4. Overall Culinary Quality
   Would a home cook successfully produce a good dish from these instructions?
   (1 = incoherent; 3 = adequate; 5 = excellent and detailed)

Respond ONLY as valid JSON (no prose before/after):
{{"alignment": X, "procedural_logic": X, "hallucination": X, "overall_quality": X, "comments": "..."}}"""
```

> **Upgrading from 3 to 4 criteria:** The addition of `overall_quality` gives an aggregate human-interpretable score that can serve as the headline result in your report, while the three targeted criteria provide diagnostic detail.

---

#### Final Deliverables — Week 12

**Aggregate results table:**

| Condition | BLEU ↑ | BERTScore ↑ | CVS ↑ | Ingr. Coverage ↑ | Step Order ↑ | Tech. Plausibility ↑ | LLM Judge (avg/5) ↑ |
|---|---|---|---|---|---|---|---|
| Zero-Shot (C1) | — | — | — | — | — | — | — |
| Fine-Tuned (C2) | — | — | — | — | — | — | — |
| **KG-RAG (C3)** | — | — | — | — | — | — | — |

**Frontend CVS panel (Stage 5 update):**

```
┌─────────────────────────────────────────────────────────────┐
│  🧪 Culinary Validity Score: 0.91  ✅ HIGH CONFIDENCE       │
│                                                             │
│  📋 Ingredient Coverage:     0.95   (1 ingredient missing)  │
│  📐 Step Order Fidelity:     0.88   (1 ordering violation)  │
│  🔧 Technique Plausibility:  0.93   (all techniques valid)  │
│                                                             │
│  KG Source: RecipeDB (50K training corpus)                  │
│  Subgraph: 14 nodes, 21 edges retrieved                     │
│  Linearization: Format B (Structured Prose)                 │
│  Retrieval latency: 142ms                                   │
└─────────────────────────────────────────────────────────────┘
```

---

#### Risk Mitigation — Phase 4

| Risk | Mitigation |
|---|---|
| CVS order constraints empty (no `precedes` edges for rare ingredient sets) | Fall back to step_order_score = 1.0 with a flag `no_kg_constraints`; report this separately |
| LLM-Judge scoring variance across API calls | Use temperature=0.0 for the judge model; run each recipe twice and average if budget allows |
| CVS component 3 too permissive if ingredient is not in CKG | If valid_techniques is empty, skip component 3 and re-weight to (0.55 / 0.45) |
| Confirmation bias in judge prompt | Include a negative-example calibration prefix in the few-shot judge system message |

---

## Summary: Full Deliverable Registry

| # | Deliverable | File | Phase | Research Contribution |
|---|---|---|---|---|
| 1 | Aligned RecipeDB corpus (train split) | `recipedb_aligned_train.json` | 1 | Data sourcing |
| 2 | Co-occurrence matrix (PMI + raw count) | `co_occurrence_matrix.csv` | 1 | Statistical relationship mining |
| 3 | Technique sequence statistics | `technique_sequences.json` | 1 | Empirical step ordering |
| 4 | Culinary Knowledge Graph | `culinary_kg.graphml` + `.pkl` | 1 | **Core KG construction** |
| 5 | Subgraph retrieval module | `kg_retriever.py` | 2 | Graph-based RAG retrieval |
| 6 | Linearization module (3 formats) | `kg_linearizer.py` | 2 | **Linearization ablation** |
| 7 | KG-conditioned prompt + API integration | `kg_pipeline.py` + `api.py` | 2 | **SLM symbolic grounding** |
| 8 | 1,500 generated recipes (3 conditions) | `generated_recipes_*.json` | 3 | Controlled experiment |
| 9 | BLEU + BERTScore baselines | `standard_metrics_results.csv` | 3 | Baseline benchmarking |
| 10 | Linearization ablation results | `linearization_ablation.csv` | 3 | Format selection evidence |
| 11 | 3-component CVS evaluator | `cvs_evaluator.py` | 4 | **Novel domain metric** |
| 12 | LLM-as-Judge results (4-criteria) | `llm_judge_results.csv` | 4 | Reference-free evaluation |
| 13 | Full evaluation report (stratified) | `evaluation_report.csv` | 4 | Final results |
| 14 | Frontend CVS panel | React component update | 4 | UX contribution |

> [!TIP]
> **Report narrative:** *"Standard metrics such as BLEU and BERTScore cannot detect physical or procedural inconsistency in recipe generation. We mine a Culinary Knowledge Graph entirely from RecipeDB, condition a fine-tuned SLM on linearized subgraphs at inference time, and evaluate outputs using the Culinary Validity Score — a 3-component semantic-compiler metric verifying ingredient usage, step ordering fidelity, and technique plausibility. Across 1,500 generated recipes and five cuisine regions, KG-RAG reduces hallucination and improves procedural coherence over zero-shot and fine-tuned baselines, as confirmed by both automated CVS and LLM-as-Judge evaluation."*

---

## Tech Stack (All Free / Already Available)

| Component | Tool | Notes |
|---|---|---|
| Recipe data source | RecipeDB (cosylab.iiitd.edu.in) | Free academic download |
| NLP — technique extraction | `spaCy` (`en_core_web_sm`) | POS tagging for verb extraction |
| Graph library | `networkx` | KG construction + subgraph ops |
| Fuzzy matching | `rapidfuzz` | Ingredient canonicalization |
| Metrics | `sacrebleu`, `bert-score` | Baseline metrics |
| Generation model | V10 (HF Spaces, existing) | Fine-tuned Llama 3 8B — unchanged |
| Archetype classification | Groq Llama-3.1-8B (existing) | Cached, unchanged |
| LLM-as-Judge | Groq Llama-3.3-70B (existing) | Reference-free rubric scoring |
| Experimentation | Google Colab (free tier) | GPU batch generation |

---

## Future Work (Scope for Extension if Time Allows)

> [!NOTE]
> These are explicitly **out of scope** for the 12-week plan but demonstrate awareness of limitations and natural extensions — important for a capstone research narrative.

1. **Hybrid KG (Phase 1+):** Add FoodOn OWL ontology to augment the empirical CKG with symbolic culinary role annotations (e.g., PoultryMeat, AromaVegetable), providing categorical constraints beyond statistical co-occurrence.
2. **Personalization:** Weight the co-occurrence subgraph by a user's cuisine preference or dietary profile.
3. **CVS-guided re-ranking:** Generate multiple candidate recipes and select the one with the highest CVS, rather than taking the first output.
4. **Graph Neural Network retrieval:** Replace heuristic N-hop retrieval with a GNN-based relevance scorer trained on held-out recipe quality labels.

---

## Research References

| Technique | Paper / Source | Year |
|---|---|---|
| KG-RAG for generation | Edge et al., *"From Local to Global: A Graph RAG Approach"* (Microsoft Research) | 2024 |
| RecipeDB dataset | Garg et al., *"RecipeDB: A resource for exploring recipes"* (iScience, CosyLab IIITD) | 2022 |
| KG linearization | Agarwal et al., *"Knowledge Graph Based Synthetic Corpus Generation for Question Answering"* (NAACL) | 2021 |
| LLM-as-Judge | Zheng et al., *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"* (NeurIPS) | 2023 |
| SLM prompt grounding | Ding et al., *"Knowledge Graph-Augmented Language Model Prompting"* | 2023 |
| PMI for co-occurrence | Church & Hanks, *"Word association norms, mutual information"* (Computational Linguistics) | 1990 |
| CVS concept | Derived from semantic compiler / type-checker analogy for structured output validation | — |
