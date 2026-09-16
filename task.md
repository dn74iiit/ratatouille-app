# Task Tracker: FlavorDB-Grounded GraphRAG for Ratatouille

## Phase 1: Real Data Foundation (Weeks 1–3)

- [/] **Week 1: Scrape/obtain FlavorDB2 data**
  - [/] Search GitHub/Kaggle for pre-scraped FlavorDB CSVs
  - [ ] If not available, write a Python scraper for FlavorDB2 ingredient pages
  - [ ] Extract `{ingredient_name, category, molecules[], is_vegan}` for ~350 ingredients
  - [ ] Save as `flavordb_molecules.json`

- [ ] **Week 2: Download and process FooDB**
  - [ ] Download CSV bulk dump from foodb.ca/downloads
  - [ ] Extract `foods.csv` + `nutrients.csv`
  - [ ] Write pandas script to produce `{food_name, proteins_per_100g, fats_per_100g, carbs_per_100g}`
  - [ ] Save as `foodb_macros.json`

- [ ] **Week 3: Merge into `chemical_features_real.json`**
  - [ ] Fuzzy-match 350 ingredient names against FlavorDB2 and FooDB entries
  - [ ] Merge: molecules from FlavorDB2, macros from FooDB, keep culinary_role + texture_profile
  - [ ] Document match rates
  - [ ] Save as `chemical_features_real.json`

## Phase 2: Dual-Level Knowledge Graph (Weeks 4–6)

- [ ] **Week 4: Build low-level networkx graph**
  - [ ] Node types: Ingredient, Molecule, Role
  - [ ] Edge types: has_molecule, has_role
  - [ ] Attributes: is_vegan, macros
  - [ ] Save as `food_knowledge_graph.graphml`

- [ ] **Week 5: High-level layer (community detection)**
  - [ ] Run Louvain community detection
  - [ ] Generate cuisine cluster summaries via Groq
  - [ ] Save community mappings

- [ ] **Week 6: Build graph query functions**
  - [ ] `find_substitutes_graph()` — low-level molecule traversal
  - [ ] `trace_molecule_path()` — gap analysis
  - [ ] `get_cuisine_context()` — high-level community queries
  - [ ] Create `graph_rag.py` module

## Phase 3: Hybrid Retrieval + Query Router (Weeks 7–9)

- [ ] **Week 7: Build FAISS vector index**
  - [ ] Embed 350 ingredient profiles
  - [ ] Store in FAISS index
  - [ ] Save as `ingredient_vectors.faiss`

- [ ] **Week 8: Build Query Router**
  - [ ] Groq-based intent classifier (SPECIFIC/EXPLORE/CONTEXT)
  - [ ] Route to graph or vector path
  - [ ] Create `query_router.py`

- [ ] **Week 9: Integrate into pipeline**
  - [ ] Modify substitution pipeline to use router
  - [ ] Add `/smart-substitute` API endpoint
  - [ ] Connect to existing vegan flow

## Phase 4: Agentic Loop + Evaluation (Weeks 10–12)

- [ ] **Week 10: Build agentic substitution loop**
  - [ ] Retrieve → Evaluate → Accept/Retry pattern
  - [ ] Max 3 iterations with score threshold (0.60)
  - [ ] Graph-based explanation generation via Groq
  - [ ] Create `agentic_substitute.py`

- [ ] **Week 11: Evaluation**
  - [ ] Run old engine vs. new GraphRAG on 30 test ingredients
  - [ ] Compare: substitutes chosen, scores, iterations needed
  - [ ] Build comparison table
  - [ ] Write evaluation report

- [ ] **Week 12: Frontend + Documentation + Demo**
  - [ ] Add "Why This Substitute Works" panel to React frontend
  - [ ] Molecule trace visualization
  - [ ] Confidence badge
  - [ ] Write report section citing LightRAG, Agentic RAG, Hybrid Retrieval
  - [ ] Prepare 3-minute demo
