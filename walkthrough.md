# Walkthrough: Chemically-Aware Vegan Substitution Engine

This walkthrough documents the completed implementation of the **Chemically-Aware Vegan Substitution Engine** for *Ratatouille*, which calculates the physical-chemical properties of non-vegan ingredients, maps them to optimal vegan substitutes, and compiles natural language culinary corrections for texture, fat absorption, and flavor.

---

## 🛠️ Changes Implemented

### 1. Database Layer
*   **[chemical_features.json](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/chemical_features.json):** Created a curated chemical, textural, and flavor volatile database covering 25+ essential culinary ingredients and whitelisted spices (e.g. paneer, tofu, chicken, soya chunks, smoked paprika, nutritional yeast).
*   **Dynamic Cache & Offline Generalization:** Added static fallbacks for unrecognized ingredients (red meat, poultry, seafood, dairy fat, dairy liquid, sweeteners) so that they automatically map to standard profiles when offline, and are dynamically cached to this JSON database when successfully bootstrapped.

### 2. Math & Language Engine Layer
*   **[vegan_engine.py](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/vegan_engine.py):** Built a standalone Python engine that:
    *   Computes **Flavor Jaccard Similarity** (molecular overlap).
    *   Computes **Textural Euclidean Distance** over a 6D space (hardness, chewiness, fibrousness, moisture, elasticity, granularity).
    *   Computes **Functional Overlap** based on culinary role tags (e.g. bulk protein, binder).
    *   Calculates the mathematical **Delta Vector** ($\vec{\Delta} = \vec{V}_{\text{orig}} - \vec{V}_{\text{sub}}$) to extract fat deficits, moisture excess/deficit, and chewiness deficits.
    *   Translates mathematical deficits into highly natural, professional chef directions using a **Language Guardrail System** with context-aware templates (e.g., suggesting *pressing tofu for 15 minutes* to handle excess moisture, or *soaking and squeezing soya chunks* for moisture deficits).
    *   Calculates **Spice Bridges** using FlavorDB volatile compound overlap, whitelisting with culinary spices to recommend aroma-matching enhancers (e.g., adding smoked paprika to soya chunks to bridge pyrazines).
    *   Includes **Keyword Fallback Rules** (`classify_by_keyword`) to classify unrecognized ingredients into major category structures (red_meat, poultry, seafood, etc.).

### 3. FastAPI Endpoint Layer
*   **[api.py](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/api.py):** Exposed a new `/get-vegan-blueprint` endpoint.
    *   Accepts a single ingredient or list of ingredients.
    *   Resolves duplicates, computes the vegan blueprint for each, and returns a detailed JSON payload of substitutions and culinary technique blueprints.
    *   **LLM Bootstrapping Layer:** Implemented `bootstrap_ingredient_profile` which prompts the Llama-3-8B-Instruct space client to dynamically generate chemical and physical JSON profiles for unrecognized ingredients at runtime, writing them directly into the database cache.
    *   **Offline Fallback Routing:** In case the LLM space is cold, offline, or times out, the endpoint automatically falls back to keyword-based static classifications to prevent page blanking or API crashes.

### 4. React Frontend UI Layer
*   **[App.jsx](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/frontend/src/App.jsx):** Integrated the feature into the React single-page application:
    *   Added a new **"Vegan Converter 🌿"** view tab in the top navigation bar.
    *   Added form fields for inputs and archetype selectors.
    *   Rendered interactive cards presenting:
        *   The overall match score (e.g., $85\%$) with visual progress bars.
        *   Textural, flavor, and role breakdown meters.
        *   The **Delta Compensation Steps** panel showing additions, physical preparation techniques, and spice bridges.
    *   Configured a dynamic `BACKEND_URL` environment toggle that automatically swaps to `localhost:8000` when running locally, and points to the Render cloud deployment in production.
    *   Added UI safety guards to display clean error/unmapped notices and prevent screen blanking when unknown ingredients are entered.

---

## 🧪 Verification Results

### 1. Mathematical Unit Tests
*   **[test_vegan_engine.py](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/test_vegan_engine.py):** Evaluates Jaccard indices, normalized texture distance, delta translations, static keyword fallbacks, and unmatched error handling.
    *   *Result:* `OK` (9/9 tests passed).

### 2. HTTP Endpoint Integration Tests
*   **[test_api_endpoints.py](file:///c:/Users/dhanu/OneDrive/Desktop/Capstone%20Proj%20CB/Rat-Model2V/RAT%20V3/V8/repo/test_api_endpoints.py):** Validates single/multiple ingredient lookups, dynamic LLM bootstrapping (using mocks to preserve offline execution), and fallback routing.
    *   *Result:* `OK` (5/5 tests passed).

### 3. Dynamic Local E2E Verification
*   We executed a simulated API call for `"mutton"` to `/get-vegan-blueprint`. The system:
    1. Successfully contacted the Hugging Face Llama-3 Space `nd1490/ratatouille-inference-v10-q8`.
    2. Dynamically generated a food chemistry profile for `"mutton"`.
    3. Cached the profile inside `chemical_features.json`.
    4. Generated a complete vegan substitute blueprint recommending `"soya chunks"` matched at $62.93\%$, accompanied by lipid bridging (coconut oil), umami bridging (MSG/soy sauce), poultry-mimicking texture prep, and spice bridges (smoked paprika and nutritional yeast).
