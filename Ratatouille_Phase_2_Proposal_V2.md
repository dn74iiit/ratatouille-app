# Capstone Project Proposal (V2)
## Phase II: Advanced Knowledge Graph-Augmented Recipe Generation for Ratatouille

**Student Name:** Nindra Dhanush
**Roll Number:** MT25074
**Programme:** M.Tech. (2025–2027)
**Institution:** Indraprastha Institute of Information Technology Delhi (IIIT-D)
**Laboratory:** CoSyLab (Computational Systems Biology Laboratory)
**Supervisor:** Prof. Ganesh Bagler

---

### Executive Summary
This project is a continuation of the Ratatouille Capstone Project, which successfully delivered an AI-powered, cost-constrained Indian budget recipe generator with chemically-aware vegan substitution. While the first phase established a robust pipeline for mathematical budget optimization and plant-based substitutions, this new phase focuses on significantly advancing the core recipe generation engine.

I will explore how to enhance Small Language Models (SLMs) using a state-of-the-art Knowledge Graph-Augmented Generation (KG-RAG) pipeline. Building upon baseline KG-RAG concepts, this proposal introduces advanced methodologies including **Hybrid Graph Construction**, **Context-Aware Subgraph Retrieval**, **Advanced Hybrid Graph-RAG**, and **Iterative Self-Correction**. The goal is to construct a system that not only retrieves culinary logic but leverages semantic vector similarity and self-heals its own hallucinations.

### Problem Statement
In the previous phase, the fine-tuned Llama 3 model demonstrated strong creative capabilities. However, SLMs acting as isolated reasoners frequently suffer from the "Semantic Gap"—producing recipes that are linguistically fluent but culinarily flawed (e.g., hallucinations, procedural errors, ignoring allocated ingredients).

Standard KG-RAG attempts to solve this via "Prompt Injection" (feeding graph data into the context window). However, SLMs can still ignore prompt instructions (Attention Disconnect). Furthermore, single-pass generation relies on getting it right the first time, which is brittle. This phase solves these architectural shortcomings by moving from passive retrieval to active, constraint-enforced agentic generation.

### System Architecture Integration
The upgraded recipe generation pipeline will integrate into the existing architecture as follows:

1. **Vegan Substitution & Archetype Classification:** Upgraded from Phase I to include an optional **Deterministic Rule-Based Engine**. This will utilize hard-coded mappings of ingredient groups (e.g., meat, vegetable) and their combinations to reliably infer the dish archetype (e.g., rice_dish, curry) without needing an LLM call.
2. **Budget Optimization:** Unchanged from Phase I.
3. **Hybrid Culinary Knowledge Graph (New):** Combining statistical co-occurrence (PMI) for functional relationships (e.g., `PAIRS_WITH`) with a lightweight taxonomy for hierarchical constraints (e.g., `IS_A`). *Implementation Note: The graph will be hosted In-Memory via NetworkX for sub-millisecond MVP traversal, with a planned migration to Neo4j as the graph scales beyond RAM constraints.*
4. **Context-Aware Subgraph Retrieval (New):** Expanding outward from the user's ingredients, but weighting the graph traversal based on the **Archetype Classification** from Phase I (e.g., penalizing "baking" nodes if the archetype is "South Indian").
5. **Advanced Hybrid Graph-RAG (New):** Expanding the KG-RAG pipeline by pairing the logical rules of the NetworkX Graph with a Vector Database. The system will perform semantic similarity searches against the historical recipe database, guided and filtered by the graph constraints, to inject highly relevant few-shot examples into the SLM's context window.
6. **Iterative Self-Correction (New):** Utilizing the Culinary Validity Score (CVS) and an LLM-as-a-judge inside the generation loop. If the draft recipe fails validation (e.g., serving before cooking), the error is fed back to the generator for autonomous correction before user delivery.

### Implementation Plan
The project will run over 12 weeks, requiring an estimated 3 hours of work per week.

#### Phase 1: Hybrid Culinary Knowledge Graph & Deterministic Classification
**Goal:** Build a robust, hybrid CKG alongside a reliable rule-based archetype engine.
*   **Develop a Deterministic Archetype Classifier:** Construct hard-coded datasets classifying raw ingredients (meat, veg, spice) and mapping specific input combinations directly to archetypes.
*   Extract simple, lightweight taxonomic mappings (e.g., Ingredient categories) from RecipeDB metadata or simpler external datasets.
*   Calculate functional ingredient co-occurrence using Pointwise Mutual Information (PMI).
*   Map cooking techniques to specific ingredients based on the dataset.
*   Merge taxonomic (`IS_A`) and statistical (`PAIRS_WITH`, `PREPARED_BY`) edges into a unified hybrid graph (saved as a flat JSON artifact for NetworkX ingestion, with Neo4j Cypher compatibility planned for Phase 4).

#### Phase 2: Context-Aware Retrieval and Agentic Pipeline
**Goal:** Extract highly relevant, culturally appropriate subgraphs and set up the self-healing generation loop.
*   Implement archetype-weighted graph search algorithms (e.g., personalized PageRank) to ensure retrieved culinary rules match the requested cuisine.
*   Develop the **Iterative Self-Correction** loop: hook up the LLM-as-a-judge so that it evaluates draft recipes in real-time and passes critique back to the generator if CVS thresholds are not met.

#### Phase 3: Advanced Hybrid Graph-RAG & System Analysis
**Goal:** Enhance contextual accuracy via Vector Search and comprehensively analyze system performance.
*   Implement a Vector Database (e.g., Pinecone, FAISS, or MongoDB Vector Search) to encode the 12,000+ recipes from the RecipeDB dataset.
*   Develop a Hybrid Search algorithm: Use the retrieved subgraph (from Phase 2) to filter the vector search space, ensuring retrieved example recipes logically match the graph's culinary constraints.
*   Inject the top matching historical recipes into the prompt as "Few-Shot" examples to dynamically guide the SLM's formatting.
*   **System Analysis & Logging:** Generate a large batch of example recipes to actively track and analyze generation times, Culinary Validity Scores (CVS), and the average number of self-correction attempts required to pass the Judge.
*   **Reference Database:** Compile these successfully generated and validated recipes into a searchable, user-friendly database catalog for easy academic and public reference.

#### Phase 4: Vegan Engine Upgrade & Full System Evaluation
**Goal:** Modernize the core substitution engine and evaluate the final end-to-end architecture.
*   **Vegan Engine Tech Update:** Upgrade the legacy Phase 1 Vegan Substitution Engine to a more advanced, state-of-the-art methodology (e.g., using dense vector embeddings for ingredient matching or automated graph traversal for finding vegan analogs).
*   Test setups: Zero-Shot SLM, Standard KG-RAG (Prompt Injection only), and Advanced Hybrid Graph-RAG.
*   Run subgroup analysis to ensure the system performs well across different regional cuisines.
*   Document results, finalize the CVS metric, and integrate the final pipeline into the React frontend.

### Expected Deliverables
This project will deliver a fully working, advanced KG-RAG pipeline seamlessly integrated into the Ratatouille system. I aim to show that combining structured knowledge grounding with constrained decoding and agentic self-correction completely bridges the Semantic Gap in SLMs for culinary applications. The final outputs will include the Hybrid Culinary Knowledge Graph (NetworkX MVP / Neo4j Production), the integrated generation codebase, the CVS evaluation framework, and a comprehensive thesis report detailing the findings.
