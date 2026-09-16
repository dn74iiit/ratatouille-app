# Remaining Integration Plan: MISKG + FooDB

You've successfully implemented **Priority 1 & 2** (Canonical Normalization and the Egg fallback). Those two changes alone dramatically improve the engine's coverage of variant names.

Here is the step-by-step plan for implementing the remaining integration tasks when you have time later.

---

## Task 1: FooDB Integration (Priority 3)

**Goal:** Use FooDB to grab chemical flavor molecules and macros for ingredients that aren't in your `chemical_features.json` so you don't have to rely entirely on the LLM bootstrapper.

### 1. Download the Data
1. Go to [FooDB Downloads](https://foodb.ca/downloads)
2. Download the **CSV archive** (it will be large).
3. Extract `Foods.csv` (contains macros) and `Compounds.csv` / `Flavor.csv` (contains flavor molecules mapping).

### 2. Build a Pre-processor Script
Don't load massive CSVs into your live server memory. Write a script `build_foodb_cache.py` that:
- Reads the CSVs.
- Extracts ONLY the name, macros, and flavor molecules.
- Saves it to a lightweight JSON file (e.g., `foodb_cache.json`) in exactly the format your engine expects.

### 3. Integrate into `api.py`
In `api.py`, add a new step before the LLM bootstrapper:
```python
if canonical_name not in features:
    # NEW STEP: Check FooDB cache
    if canonical_name in foodb_cache:
        # Load profile from FooDB, then run engine math!
        pass
    else:
        # LLM Bootstrap as last resort
        pass
```

---

## Task 2: MISKG & USDA Fallback (Priority 4)

**Goal:** Provide a fallback substitution name for truly obscure items, and optionally grab basic macros from the USDA API if FooDB misses.

### 1. Extract MISKG Vegan Pairs
1. Create a Kaggle account and download the **MISKG dataset**.
2. Run a python script over `substitution_pairs.json` to filter out non-vegan swaps (like chicken → turkey).
3. Save the result as `miskg_vegan_pairs.json`.

### 2. Integrate MISKG in `api.py`
If a non-veg ingredient fails the MongoDB check AND fails the `vegan_engine` keyword check, look it up in `miskg_vegan_pairs.json`.
- **If found:** You now have a vegan substitute name (e.g., "gelatin" → "agar agar").
- Now, feed "agar agar" back through your engine pipeline to get its chemical profile (from MongoDB, FooDB, or LLM) so you can do the delta math!

### 3. USDA FoodData Central (Optional)
If you just want fast macros and don't want to use FooDB:
1. Get a free API key at `fdc.nal.usda.gov`.
2. Write a helper function in `api.py` that calls `https://api.nal.usda.gov/fdc/v1/foods/search?query=INGREDIENT&api_key=YOUR_KEY`.
3. Extract Protein, Fat, Carbs to calculate the `chemical_delta` if other databases fail.

---

### Why leave the Groq LLM running?
Keep the LLM bootstrapper (your current setup) active at the very end of the chain. No matter how many databases you add, someone will eventually ask for an ingredient that isn't in any of them. The LLM acts as the ultimate safety net.
