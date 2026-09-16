# Ratatouille API: Recipe Generation Pipeline Architecture

This document breaks down the exact journey of a request flowing through the `/generate-recipe` endpoint in `api.py`. It details the sequence of function calls, caching mechanisms, AI interactions, and mathematical optimizations that occur to generate a cost-constrained, culturally appropriate recipe.

---

## 1. Request Initialization
The pipeline begins when the React frontend sends a `POST /generate-recipe` request containing:
- `ingredients`: List of user-provided ingredients.
- `budget`: The maximum allowed cost in INR (₹).
- `servings`: Number of people to serve.
- `state`: The Indian state for regional pricing (e.g., "Delhi").
- `model_version`: The selected Hugging Face model (`v8` or `v10`).
- `is_vegan`: Boolean toggle.

The endpoint immediately begins yielding **Server-Sent Events (SSE)** to the frontend to update the live progress indicator.

---

## 2. Vegan Substitution Engine (Optional)
*Triggered only if `is_vegan = True`.*

Before any math is done, the system must swap out animal products for plant-based alternatives.
1. **Fast Archetype Classification:** Runs `_classify_archetype_fast()`, a hardcoded keyword matcher (0ms) to figure out the dish type (e.g., Curry, Soup) so the vegan engine knows how to compensate.
2. **Canonicalization:** Standardizes ingredient names using `canonicalize_ingredient()` (e.g., `"chicken breast"` → `"chicken"`).
3. **MongoDB Cache Lookup:** Queries the `vegan_alternatives` MongoDB collection for a pre-calculated substitution blueprint.
4. **Local Engine Fallback:** If not found in the DB, it queries the local `vegan_engine` which uses static features to generate a match (e.g., swapping "chicken" for "soy curls").
5. **Compensation Application:** The engine doesn't just swap 1-to-1. It applies a **Compensation Blueprint**, which injects auxiliary ingredients (e.g., `msg`, `soy sauce`, `nutritional yeast`) to bridge the missing umami and fat gaps left by the animal product.

---

## 3. Dish Archetype Classification
*Triggered here if normal pipeline, or in Step 2 if Vegan.*

The system must know *what* it is building to apply the correct mathematical ratio constraints later.
1. **Cache Check:** Queries the new `llm_cache` MongoDB collection for `archetype_<ingredients>`. 
2. **Serverless LLM Call:** If no cache exists, it calls `get_recipe_archetype()`, which hits **Groq's Serverless Llama 3.3 70B** API. 
   - **System Prompt:** *"You are a recipe classification assistant. Output ONLY a single word classification from the permitted list. No explanation."*
   - **User Prompt:** *"Classify the dish structure based on these ingredients: [{ingredients}]. Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish]."*
3. **Gradio Fallback:** If Groq rate-limits or fails, it falls back to your custom Hugging Face Space using this prompt:
   - *`<|begin_of_text|>Classify the dish structure based on these ingredients: [{ingredients}]. Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish]. Return ONLY the word.\n\n### INGREDIENTS:\n{ingredients}\n### ARCHETYPE:\n`*
4. **Cache Save:** The resulting archetype (e.g., `Curry`, `Dry_Sabzi`, `Rice_Dish`) is instantly saved to MongoDB for future use.

---

## 4. Cost Constraint Optimization
This is the mathematical core of the engine, executed via `optimize_recipe_v2()`. It uses SciPy's Linear Programming solver to maximize food quantity without breaking the user's budget.

### 4a. Dynamic Pricing Lookup
For *every single ingredient* in the list, the system runs `get_dynamic_price()`:
1. **Mandi Database:** Searches the local `current_mandi` DataFrame for the exact crop price in the user's specific `state` (or a national average fallback).
2. **Pantry Fallback:** If it's a processed item (like "egg" or "soy sauce"), it checks the hardcoded/GitHub `pantry_prices` dictionary.
3. **LLM Deconstruction (The Slow Path):** If the ingredient is completely unknown (e.g., "vegan cashew mozzarella"), it queries `llm_cache`. If missing, it calls `deconstruct_ingredient()` to ask the LLM to break the ingredient down into raw base crops. It calculates a weighted average price, then saves it to `llm_cache`.
   - **Serverless System Prompt:** *"You are a food chemistry and agricultural database. Output only valid JSON. Do not write any explanations or conversational text outside the JSON."*
   - **Serverless User Prompt:** *"Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops with approximate weight percentages (total summing to 1.0). Example: for 'tomato ketchup', return exactly: {{\"tomato\": 0.8, \"sugar\": 0.1, \"onion\": 0.1}}"*
   - **Gradio Fallback Prompt:** *`<|begin_of_text|>Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops.\nAssign approximate weight percentages. Return ONLY valid JSON.\nExample for 'tomato ketchup': {{"tomato": 0.8, "sugar": 0.1, "onion": 0.1}}\n\n### INGREDIENT:\n{ingredient}\n### JSON:\n`*

### 4b. Linear Programming Matrix (SciPy)
The system constructs a complex constraint matrix:
- **Cost Constraint:** `(Price_A * Grams_A) + (Price_B * Grams_B) <= Total_Budget`
- **Minimum/Maximum Bounds:** Enforces limits (e.g., you can't have 500g of salt, or less than 15g of tomato).
- **Archetype Ratios:** Enforces structural integrity. For a `Curry`, the ratio of Base (tomato/onion) to Protein must be mathematically sound so you don't get a dry curry.

### 4c. Solvers and Fallbacks
The SciPy `linprog(method='highs')` engine runs. 
- If the budget is too strict for the matrix, the system automatically falls back:
  1. Relaxes structural archetype constraints.
  2. Forces the minimum allowed gram bounds down to a tiny `5g`.
  3. "Nuclear Fallback" removes all bounds to guarantee a mathematical solution.

---

## 5. AI Recipe Generation
With the ingredients perfectly weighed and costed, the system generates the actual cooking instructions.
1. **Prompt Construction:** The ingredients are formatted into the strict V10 prompt schema. The final string sent to the model looks exactly like this:
   ```text
   ### INGREDIENTS:
   - 150.0g paneer
   - 20.0g tomato
   - 10.0g onion
   ### TITLE:
   ```
2. **Inference:** Calls the `query_hf_model` function, sending the prompt to your Hugging Face API Gradio client (V8 or V10).
3. **Post-Processing:** Llama 3 often hallucinates signatures or repetitive loops (e.g., `"Enjoy!"`, `"Chef's Note:"`, or randomly restarting the recipe). A custom Python loop slices the string to forcefully cut off these hallucinations.

---

## 6. Logging & Completion
1. **Metrics:** The total time taken for Veganization, Optimization, and AI Generation is calculated.
2. **Database Write:** A log entry containing the model version, budget, archetype, and distinct execution times is stored in the `generation_logs` MongoDB collection.
3. **Stream End:** The final, parsed recipe and optimized ingredient lists are yielded via SSE (`step: complete`) to the frontend, which renders the markdown UI.
