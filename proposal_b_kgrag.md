# Proposal B: KG-RAG — Knowledge Graph-Augmented Recipe Generation for Ratatouille

## What This Proposal Does

> [!IMPORTANT]
> **This proposal replaces the vegan substitution focus** of Proposal A with a fundamentally different research question:
> *"Can conditioning a Small Language Model on linearized subgraphs from a Hybrid Culinary Knowledge Graph reduce hallucination and improve procedural reliability in recipe generation — more than zero-shot prompting or fine-tuning alone?"*
>
> **Kept from the existing system:** V10's fine-tuned Llama 3 model, the SciPy budget optimizer, Groq API integration, the React frontend, and the 50K recipe corpus.
> **Replaced:** The vegan substitution engine and FlavorDB2 graph → replaced by a **Hybrid Culinary KG** (RecipeDB + FoodOn) and **subgraph-conditioned generation**.

### Where KG-RAG touches the pipeline (revised architecture)

```
User Input: [chicken, tomato, onion], ₹100
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 1: KG Subgraph Retrieval  ← ★ PHASE 1 + 2     │
│   Query the Hybrid Culinary KG for:                  │
│   - Ingredient nodes (from RecipeDB)                 │
│   - Ontology class + culinary role (from FoodOn)     │
│   - Co-occurrence edges (which ingredients cook      │
│     together most in 50K recipes)                    │
│   Linearize top subgraph into a structured string    │
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
│ Stage 3: SciPy Budget Optimizer                      │
│   (unchanged — uses existing macro estimates)        │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 4: KG-RAG Recipe Generation  ← ★ PHASE 3      │
│   Prompt = linearized subgraph + archetype + budget  │
│   Model = V10 (fine-tuned Llama 3) via HF Spaces     │
│   Subgraph provides: allowed techniques, typical     │
│   pairings, ordering constraints from ontology       │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Stage 5: CVS Evaluation  ← ★ PHASE 4                │
│   Culinary Validity Score:                           │
│   - Ingredient Coverage Check                        │
│   - Instruction Order Verification                   │
│   - LLM-as-Judge reference-free scoring              │
│   Confidence badge + CVS breakdown panel             │
└──────────────────────────────────────────────────────┘
```

**Every phase maps to an existing pipeline stage.** The research contribution is in *how* generation is conditioned and *how* output is evaluated.

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

### Phase 1: Hybrid Culinary Knowledge Graph — Data Foundation (Weeks 1–3, 9 hours)

> **Goal:** Build a two-source Hybrid Culinary KG from RecipeDB and the FoodOn ontology — the symbolic backbone that conditions all downstream generation.

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 1 | 3 hrs | **Obtain RecipeDB data.** Download the publicly available RecipeDB dataset (Garg et al., 2022) from [cosylab.iiitd.edu.in/recipedb](https://cosylab.iiitd.edu.in/recipedb). Extract `{recipe_id, title, ingredient_list, cooking_steps, cuisine_region, flavor_profile}` for the subset that overlaps your 50K training corpus. Parse co-occurrence frequency: how often do pairs of ingredients appear together across recipes? | `recipedb_ingredients.json` + `co_occurrence_matrix.csv` |
| 2 | 3 hrs | **Obtain FoodOn ontology data.** Download the FoodOn OWL file from [github.com/FoodOntology/foodon](https://github.com/FoodOntology/foodon). Use `owlready2` to extract: ingredient class hierarchy (e.g. `ChickenBreast → PoultryMeat → AnimalDerivedFood`), culinary role annotations (protein source, fat source, binder, aromatic), and permitted cooking processes per class (e.g. `PoultryMeat → [roasting, grilling, braising]`). | `foodon_classes.json` + `cooking_processes.json` |
| 3 | 3 hrs | **Merge into the Hybrid Culinary KG.** Load both sources into a `networkx` graph. Fuzzy-match ingredient names across RecipeDB and FoodOn using `rapidfuzz`. Build the final graph structure documented below. Export as `.graphml` for inspection and `.pkl` for fast runtime loading. | `culinary_kg.graphml` + `culinary_kg.pkl` |

**Graph structure:**

```
Node Types:
  - Ingredient (~500 nodes)
      attrs: culinary_role, ontology_class, is_vegan,
             permitted_techniques[], avg_quantity_g
  - Technique (~60 nodes: sauté, braise, fold, temper…)
      attrs: heat_level, timing_range_min, applies_to[]
  - DishArchetype (8 nodes: curry, stir-fry, salad, soup…)
      attrs: typical_steps_order[]

Edge Types:
  - co_occurs_with (Ingredient ↔ Ingredient)
      weight: co-occurrence frequency from RecipeDB
  - has_technique (Ingredient → Technique)
      weight: frequency in RecipeDB for that ingredient
  - belongs_to_archetype (Ingredient → DishArchetype)
  - ontology_subclass (Ingredient → Ingredient)
      source: FoodOn class hierarchy
  - requires_before (Technique → Technique)
      meaning: step A must precede step B (from FoodOn process constraints)
```

> [!TIP]
> **Why two sources?** RecipeDB gives you *empirical* co-occurrence (what humans actually cook together). FoodOn gives you *symbolic* constraints (what is ontologically permitted). The hybrid captures both statistical and structural culinary knowledge — this is the core academic contribution of the KG construction.

---

### Phase 2: Subgraph Retrieval + Linearization Engine (Weeks 4–6, 9 hours)

> **Goal:** Given a user's ingredient list, query the Hybrid Culinary KG to extract a relevant subgraph, then linearize it into a structured string that can be injected into the SLM's prompt.

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 4 | 3 hrs | **Build the subgraph retrieval function.** Given a query ingredient set, extract the N-hop neighbourhood relevant to those ingredients from the KG. Score candidate nodes/edges by centrality and co-occurrence weight. Return the top-K subgraph as a filtered NetworkX subgraph. | `kg_retriever.py` module |

```python
def retrieve_subgraph(ingredients: list[str], top_k_neighbors: int = 10) -> nx.Graph:
    """
    Extract a relevant subgraph from the Hybrid Culinary KG
    for a given ingredient set.
    """
    seed_nodes = []
    for ing in ingredients:
        # Fuzzy match to graph node names
        match = process.extractOne(ing, list(G.nodes()), score_cutoff=80)
        if match:
            seed_nodes.append(match[0])

    # 1-hop expansion: get direct neighbors (co-occurring ingredients + techniques)
    subgraph_nodes = set(seed_nodes)
    for node in seed_nodes:
        neighbors = sorted(
            G.neighbors(node),
            key=lambda n: G[node][n].get("weight", 0),
            reverse=True
        )[:top_k_neighbors]
        subgraph_nodes.update(neighbors)

    # Add technique ordering edges (requires_before)
    technique_nodes = {n for n in subgraph_nodes
                       if G.nodes[n].get("type") == "technique"}
    for t in technique_nodes:
        for successor in G.successors(t):
            if G.edges[t, successor].get("type") == "requires_before":
                subgraph_nodes.add(successor)

    return G.subgraph(subgraph_nodes).copy()
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 5 | 3 hrs | **Build the linearization function.** Convert the subgraph into a human-readable structured string for prompt injection. Three serialization formats will be tested (see below) — the best-performing one is used in Phase 3. | `kg_linearizer.py` module |

**Three linearization formats to A/B test:**

```
Format A — Relation Triplets:
  "chicken co_occurs_with tomato (weight: 0.87)
   chicken has_technique sauté (freq: 0.72)
   chicken has_technique braise (freq: 0.61)
   tomato has_technique roast (freq: 0.55)
   sauté requires_before simmer
   chicken belongs_to_archetype curry"

Format B — Structured Prose:
  "INGREDIENT CONTEXT:
   chicken: protein source (PoultryMeat), typically sautéed or braised.
     Common pairings: tomato (0.87), onion (0.84), garlic (0.79).
   tomato: acid source (FruitVegetable), typically roasted or puréed.
     Common pairings: chicken (0.87), onion (0.91), cumin (0.68).
   TECHNIQUE ORDER: sauté chicken → add aromatics → simmer with tomato."

Format C — JSON-LD (compact):
  {"ingredients":{"chicken":{"role":"protein","techniques":["sauté","braise"],
   "top_pairs":["tomato","onion"]},"tomato":{"role":"acid","techniques":["roast"],
   "top_pairs":["chicken","onion"]}},"step_order":["sauté","simmer","finish"]}"
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 6 | 3 hrs | **Integrate into the API pipeline.** Replace the current flat prompt in `api.py` with the KG-conditioned prompt. The linearized subgraph is injected as a `[KNOWLEDGE]` block between the system message and the user instruction. Benchmark latency: subgraph retrieval + linearization should add <300ms. | Updated `api.py` + `kg_pipeline.py` |

**New prompt structure (replacing current flat prompt):**

```python
def build_kg_conditioned_prompt(
    ingredients: list,
    archetype: str,
    budget_g: dict,
    linearized_subgraph: str
) -> str:
    return f"""You are a professional recipe generator.

[CULINARY KNOWLEDGE GRAPH CONTEXT]
{linearized_subgraph}

[TASK]
Generate a complete recipe using these ingredients: {', '.join(ingredients)}.
Dish type: {archetype}. Budget quantities (grams): {budget_g}.

Requirements:
- Only use cooking techniques listed in the KNOWLEDGE block above.
- Follow the step order constraints from the KNOWLEDGE block.
- Ensure every listed ingredient appears in the instructions.
- Do not introduce ingredients not listed in the user input.

Output a structured recipe with: Title, Ingredients (with quantities), Steps."""
```

---

### Phase 3: Baseline Comparison Experiments (Weeks 7–9, 9 hours)

> **Goal:** Run controlled experiments comparing three generation conditions across 1,500 generated recipes to measure the impact of KG-RAG conditioning on hallucination and procedural alignment.

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 7 | 3 hrs | **Set up the three baselines and generate recipes.** Use Colab to run all three conditions on the same 500 test ingredient sets (drawn from your 50K corpus, held-out split). Each condition generates 1 recipe per input = 1,500 total outputs. | `generated_recipes_zeroshot.json`, `generated_recipes_finetuned.json`, `generated_recipes_kgrag.json` |

**Three experimental conditions:**

| Condition | Description | Model |
|---|---|---|
| **Baseline 1 — Zero-Shot** | No subgraph. Flat prompt: "Generate a recipe using {ingredients}." | Llama 3.1-8B (Groq, no fine-tuning) |
| **Baseline 2 — Fine-Tuned** | No subgraph. Same V10 fine-tuned prompt format, no KG. | V10 (your existing fine-tuned Llama 3 on HF Spaces) |
| **KG-RAG** | Subgraph-conditioned prompt (best linearization format from Phase 2). | V10 fine-tuned Llama 3 |

```python
# Colab generation script (pseudocode)
test_sets = load_held_out_test_sets(n=500)  # From 50K corpus, held-out split

for condition in ["zero_shot", "finetuned", "kgrag"]:
    results = []
    for ingredient_set in test_sets:
        if condition == "kgrag":
            subgraph = retrieve_subgraph(ingredient_set)
            linearized = linearize_subgraph(subgraph, format="B")  # Best format
            prompt = build_kg_conditioned_prompt(ingredient_set, ..., linearized)
        else:
            prompt = build_flat_prompt(ingredient_set, ...)

        recipe = call_model(prompt, condition)
        results.append({"input": ingredient_set, "output": recipe})

    save_json(results, f"generated_recipes_{condition}.json")
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 8 | 3 hrs | **Implement standard metrics (BLEU + BERTScore).** Run `sacrebleu` and `bert-score` on all 1,500 outputs against reference recipes from the held-out split. Compute mean ± std per condition. Document why these metrics are *insufficient* for culinary validity — this motivates CVS in Phase 4. | `standard_metrics_results.csv` |
| 9 | 3 hrs | **A/B test linearization formats.** Run the 500 KG-RAG recipes under all 3 formats (triplets, prose, JSON-LD). Evaluate with a fast proxy (BERTScore against reference). Pick the best-performing format. Lock it in for all further experiments. | `linearization_ablation.csv` + format decision |

---

### Phase 4: CVS — Culinary Validity Score + LLM-as-Judge (Weeks 10–12, 9 hours)

> **Goal:** Implement the Culinary Validity Score (CVS) — a semantic-compiler-style metric that verifies (1) ingredient usage coverage and (2) cooking step execution order — then run LLM-as-Judge for reference-free assessment. Compare all three conditions on CVS.

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 10 | 3 hrs | **Implement CVS.** CVS has two sub-scores, averaged into a final score in [0, 1]. | `cvs_evaluator.py` module |

**CVS — Two Components:**

```python
def culinary_validity_score(
    generated_recipe: dict,
    input_ingredients: list,
    culinary_kg: nx.Graph
) -> dict:
    """
    Semantic-compiler-style metric for recipe validity.
    Returns ingredient_coverage, step_order_score, and cvs.
    """

    # ── Component 1: Ingredient Coverage Score ──────────────────────────
    # Every user-provided ingredient must appear in the generated instructions.
    # Penalty for each missing ingredient.

    mentioned_in_steps = extract_ingredient_mentions(generated_recipe["steps"])
    missing = [ing for ing in input_ingredients
               if not any(fuzzy_match(ing, m) for m in mentioned_in_steps)]
    ingredient_coverage = 1.0 - (len(missing) / len(input_ingredients))

    # ── Component 2: Step Order Score ───────────────────────────────────
    # Extract technique sequence from generated steps.
    # Compare against KG's requires_before ordering constraints for this ingredient set.

    generated_techniques = extract_techniques(generated_recipe["steps"])  # e.g. ["sauté","simmer","serve"]
    kg_order_constraints = get_ordering_constraints(input_ingredients, culinary_kg)
    # kg_order_constraints = [("sauté","simmer"), ("simmer","serve")]  ← from requires_before edges

    violations = 0
    for (before, after) in kg_order_constraints:
        if before in generated_techniques and after in generated_techniques:
            idx_before = generated_techniques.index(before)
            idx_after = generated_techniques.index(after)
            if idx_before > idx_after:  # Violation: after appears before before
                violations += 1

    step_order_score = 1.0 - (violations / max(len(kg_order_constraints), 1))

    # ── Final CVS ────────────────────────────────────────────────────────
    cvs = 0.5 * ingredient_coverage + 0.5 * step_order_score

    return {
        "ingredient_coverage": round(ingredient_coverage, 3),
        "step_order_score": round(step_order_score, 3),
        "missing_ingredients": missing,
        "order_violations": violations,
        "cvs": round(cvs, 3)
    }
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 11 | 3 hrs | **Implement LLM-as-Judge protocol.** For a random sample of 150 recipes (50 per condition), send each to Groq Llama-3.3-70B with a structured rubric for reference-free assessment. Aggregate scores. | `llm_judge_results.csv` |

**LLM-as-Judge prompt (reference-free):**

```python
JUDGE_PROMPT = """You are a professional culinary evaluator. Rate this AI-generated recipe on three criteria.
Score each from 1–5 (5 = excellent).

RECIPE:
Title: {title}
Ingredients: {ingredients}
Steps: {steps}

CRITERIA:
1. Ingredient–Instruction Alignment: Does every listed ingredient appear in the cooking steps?
   (1=many missing, 5=all used correctly)

2. Procedural Logic: Are the cooking steps in a sensible order?
   (1=steps are impossible/illogical, 5=perfectly ordered and executable)

3. Hallucination: Does the recipe introduce ingredients or techniques not in the input?
   (1=many hallucinations, 5=no hallucinations)

Respond ONLY as JSON: {{"alignment": X, "procedural_logic": X, "hallucination": X, "comments": "..."}}"""
```

| Week | Hours | Task | Deliverable |
|---|---|---|---|
| 12 | 3 hrs | **Compile full evaluation report + frontend CVS panel + documentation.** | Final deliverables |

**Full results table (to be populated):**

| Condition | BLEU ↑ | BERTScore ↑ | CVS ↑ | Ingr. Coverage ↑ | Step Order ↑ | LLM Judge (avg/5) ↑ |
|---|---|---|---|---|---|---|
| Zero-Shot | — | — | — | — | — | — |
| Fine-Tuned (V10) | — | — | — | — | — | — |
| **KG-RAG (ours)** | — | — | — | — | — | — |

**Frontend CVS panel (new Stage 5 output):**

```
┌─────────────────────────────────────────────────────────┐
│  🧪 Culinary Validity Score: 0.91  ✅ HIGH              │
│                                                         │
│  📋 Ingredient Coverage:  0.95   (1/20 ingredients      │
│                                   not found in steps)  │
│  📐 Step Order Score:     0.88   (1 ordering violation) │
│                                                         │
│  KG Source: RecipeDB + FoodOn                           │
│  Subgraph: 12 nodes, 18 edges retrieved                 │
│  Linearization: Structured Prose (Format B)             │
└─────────────────────────────────────────────────────────┘
```

---

## Summary: What You Deliver

| # | Deliverable | Pipeline Stage | Research Contribution |
|---|---|---|---|
| 1 | `culinary_kg.graphml` — Hybrid KG from RecipeDB + FoodOn (co-occurrence + ontology edges) | Foundation for all stages | Knowledge Graph construction (dual-source) |
| 2 | `kg_retriever.py` — N-hop subgraph retrieval for any ingredient set | Stage 1: KG Retrieval | Graph-based RAG retrieval |
| 3 | `kg_linearizer.py` — Three linearization formats A/B tested | Stage 1: KG Retrieval | Linearization ablation |
| 4 | KG-conditioned prompt injection into V10 generation | Stage 4: Generation | Symbolic grounding of SLM |
| 5 | `cvs_evaluator.py` — Culinary Validity Score (ingredient coverage + step order) | Stage 5: Evaluation | Novel domain metric (semantic-compiler style) |
| 6 | LLM-as-Judge protocol — reference-free rubric scoring on 150 recipes | Stage 5: Evaluation | Reference-free evaluation |
| 7 | Quantitative comparison across 1,500 generated recipes (3 conditions) | Evaluation | Benchmarking against zero-shot + fine-tuned |

> [!TIP]
> **The key narrative for your report:** "Standard metrics such as BLEU and BERTScore cannot detect physical or procedural inconsistency in recipe generation. We construct a Hybrid Culinary Knowledge Graph from RecipeDB and the FoodOn ontology, condition a fine-tuned SLM on linearized subgraphs at inference time, and evaluate output using the Culinary Validity Score — a semantic-compiler-style metric verifying ingredient usage and execution order. Across 1,500 generated recipes, KG-RAG substantially reduces hallucination and improves ingredient–instruction alignment over zero-shot and fine-tuned baselines."

---

## Tech Stack Additions (All Free)

| Component | Tool | Purpose |
|---|---|---|
| Recipe knowledge | RecipeDB (cosylab.iiitd.edu.in) | Co-occurrence graph source |
| Food ontology | FoodOn OWL (GitHub) | Culinary role + process constraints |
| Ontology parsing | `owlready2` (Python) | Load and query FoodOn OWL file |
| Graph library | `networkx` (Python) | Hybrid KG construction + subgraph retrieval |
| Fuzzy matching | `rapidfuzz` | Ingredient name matching across sources |
| Metrics | `sacrebleu`, `bert-score` | Standard baseline metrics |
| KG-RAG generation | V10 (HF Spaces, existing) | Fine-tuned Llama 3 — unchanged |
| LLM-as-Judge | Groq Llama-3.3-70B (existing) | Reference-free evaluation |
| Experimentation | Google Colab | Batch generation of 1,500 recipes |

---

## Research References

| Technique | Paper/Source | Year |
|---|---|---|
| KG-RAG for generation | Edge et al., "From Local to Global: A Graph RAG Approach" (Microsoft Research) | 2024 |
| RecipeDB | Garg et al., "RecipeDB: A resource for exploring recipes" (iScience, CosyLab IIITD) | 2022 |
| FoodOn ontology | Dooley et al., "FoodOn: a harmonized food ontology to increase global food traceability" (npj Science of Food) | 2018 |
| Linearization of KGs | Agarwal et al., "Knowledge Graph Based Synthetic Corpus Generation" (NAACL) | 2021 |
| LLM-as-Judge | Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS) | 2023 |
| SLM grounding | Ding et al., "Knowledge Graph-Augmented Language Model Prompting" | 2023 |
| CVS concept (inspiration) | Derived from semantic compiler / type-checker analogy for structured output validation | — |
