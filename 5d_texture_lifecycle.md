# The 5D Texture Data Lifecycle
### From LLM Prompt → Validation → Composite Score → Prep Techniques → Spice Bridge

> This document traces one complete path through the live Ratatouille production system:
> **how the 5 texture numbers are generated, validated, and then used to drive every piece of substitution advice the user sees.**

---

## Step 1 — The Two Production Prompts That Create 5D Texture Data

There are **exactly two prompts** in the live codebase (`api.py`) that ask an LLM to produce a 5-dimensional texture vector. Both live inside `bootstrap_ingredient_profile()` — the function that creates a chemical profile for an ingredient not already in the MongoDB database.

---

### Primary Prompt — Groq Llama 3.3 70B
**Source**: `api.py` lines 755–773

```
SYSTEM:
You are a food chemistry and culinary database. Output only valid JSON. No explanations.

USER:
Generate a chemical and physical profile for the ingredient '{ingredient_name}' matching the specified JSON format.
Determine if it is vegan (true/false).
Specify macros (proteins, fats, carbs in grams per 100g, summing up to at most 100).
Rate its texture on a 1.0-10.0 scale: [hardness, chewiness, moisture, fat_mouthfeel, elasticity].
List 3-5 primary flavor volatile compounds (e.g. aldehydes, pyrazines, esters, terpenes, or specific molecules).
Assign a culinary role: [base_protein, fat_source, flavor_enhancer, thickener, sweetener, aromatic, dairy, binding_agent].

Return ONLY valid JSON matching this exact structure:
{
  "is_vegan": false,
  "macros": {"proteins": 22.0, "fats": 20.0, "carbs": 0.0},
  "texture_profile": [4.5, 4.0, 4.0, 6.0, 2.0],
  "flavor_molecules": ["methanethiol", "dimethyl sulfide", "pyrazines"],
  "culinary_role": "base_protein"
}
```

The critical line:
```
Rate its texture on a 1.0-10.0 scale: [hardness, chewiness, moisture, fat_mouthfeel, elasticity].
```

This single instruction defines the schema. The JSON example reinforces the format with a 5-element array.
**Parameters**: `max_tokens=250`, `temperature=0.1`

---

### Fallback Prompt — Gradio V10 Llama 3B Space
**Source**: `api.py` lines 779–796 (fires only when Groq fails/rate-limits)

```
<|begin_of_text|>You are a food chemistry and culinary database. Generate a chemical
and physical profile for the ingredient '{ingredient_name}' matching the specified JSON format.
Determine if it is vegan (true/false).
Specify macros (proteins, fats, carbs in grams per 100g, summing up to at most 100).
Rate its texture on a 1.0-10.0 scale: [hardness, chewiness, moisture, fat_mouthfeel, elasticity].
List 3-5 primary flavor volatile compounds.
Assign a culinary role: [base_protein, fat_source, flavor_enhancer, thickener, sweetener, aromatic, dairy, binding_agent].

Return ONLY valid JSON matching this example:
{
  "is_vegan": false,
  "macros": {"proteins": 22.0, "fats": 20.0, "carbs": 0.0},
  "texture_profile": [4.5, 4.0, 4.0, 6.0, 2.0],
  "culinary_role": "base_protein"
}

### INGREDIENT: {ingredient_name}
### JSON:
```

**Parameters**: `max_new_tokens=250`, `temperature=0.0`, `do_sample=False`

---

## Step 2 — Hard Validation: Exactly 5 Values Required

After the LLM responds, `api.py` line 807 runs a **hard validation** before the profile is accepted:

```python
# api.py lines 804–808 — exact production code
if profile and isinstance(profile, dict) and "is_vegan" in profile and \
   "macros" in profile and "texture_profile" in profile:
    if len(profile["texture_profile"]) == 5 and \
       all(isinstance(v, (int, float)) for v in profile["texture_profile"]):
        return profile   # accepted only if BOTH checks pass
```

**What happens on failure**: If the LLM returns 6 values, a string, or any non-numeric value, the function returns `None`. The calling code then falls back to `classify_by_keyword()` — 7 hand-crafted archetype profiles in `vegan_engine.py` that already use the correct 5D schema. A bad LLM response never corrupts the database.

---

## Step 3 — The 5 Dimensions and What Each One Controls

Once accepted, the `texture_profile` array is read by index in `vegan_engine.py`:

```python
# vegan_engine.py lines 113–120
orig_text = orig_data.get("texture_profile", [0]*5)
sub_text  = sub_data.get("texture_profile",  [0]*5)

# 5D texture indices: 0=hardness, 1=chewiness, 2=moisture, 3=fat_mouthfeel, 4=elasticity
delta_chewiness = orig_text[1] - sub_text[1]
delta_moisture  = orig_text[2] - sub_text[2]
```

| Index | Dimension | Range | Role in the system |
|---|---|---|---|
| `[0]` | hardness | 1.0 – 10.0 | Scoring only — part of Euclidean distance calculation |
| `[1]` | chewiness | 1.0 – 10.0 | **Scoring + prep techniques** — delta > 3.0 triggers freeze/press/shred instructions |
| `[2]` | moisture | 1.0 – 10.0 | **Scoring + prep techniques** — delta < -2.0 → press; delta > 4.0 → soak |
| `[3]` | fat_mouthfeel | 1.0 – 10.0 | **Scoring + fat additions** — high differences reinforce the fat deficit check from macros |
| `[4]` | elasticity | 1.0 – 10.0 | Scoring only — part of Euclidean distance calculation |

---

## Step 4 — How the 5D Vector Drives the Composite Score

The texture similarity function (`vegan_engine.py` lines 48–57):

```python
def texture_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.5   # neutral score if no data
    # Vectors have length 5, values between 1.0 and 10.0
    # Max Euclidean distance = sqrt(5 * (10-1)^2) = sqrt(405) = 20.124
    arr_a    = np.array(vec_a, dtype=float)
    arr_b    = np.array(vec_b, dtype=float)
    dist     = np.linalg.norm(arr_a - arr_b)
    max_dist = 20.124
    return float(1.0 - (dist / max_dist))
```

This feeds into the composite score with **beta = 0.4** (the highest weight):

```
Final Score = 0.3 * flavor_similarity
            + 0.4 * texture_similarity    <-- 5D vector drives this
            + 0.3 * role_match
```

Every vegan candidate in the database is ranked against the original by this formula. The 5D Euclidean distance is what decides **which substitute wins**.

**How max_dist is derived**:
```
Worst case per dimension = 10.0 - 1.0 = 9.0
max_dist = sqrt(5 * 9.0^2) = sqrt(405) = 20.124

So scores range:
  identical profiles  -> distance = 0.0  -> similarity = 1.0
  maximum opposites   -> distance = 20.124 -> similarity = 0.0
```

---

## Step 5 — How Moisture and Chewiness Drive Prep Techniques

After the best substitute is selected, `calculate_delta_recommendations()` reads dimensions `[1]` and `[2]` to generate cooking instructions. These are triggered **mathematically**, not hardcoded:

```python
# --- MOISTURE (dimension index 2) ---
delta_moisture = orig_text[2] - sub_text[2]

if delta_moisture < -2.0:
    # Substitute is wetter than original by more than 2 units on the 1-10 scale
    -> "Wrap the {sub_name} in a clean kitchen towel and press it under
        a heavy object for 15 minutes to extract excess moisture and prevent sogginess."

elif delta_moisture > 4.0:
    # Substitute is much drier — needs rehydration
    if sub_name == "soya chunks":
        -> "Soak the soya chunks in hot water for 15 minutes and squeeze
            the excess water out completely before cooking."
    else:
        -> "Hydrate the dry {sub_name} by soaking in water or
            simmering before adding to the dish."

# --- CHEWINESS (dimension index 1) ---
delta_chewiness = orig_text[1] - sub_text[1]

elif delta_chewiness > 3.0:
    # Original is much chewier — substitute needs texture building
    if sub_name == "soya chunks":
        -> "Soak in boiling water for 15 min, squeeze dry, pan-fry
            before adding to the sauce to mimic poultry fibers."
    elif sub_name == "tofu":
        -> "Freeze and thaw beforehand to open up its pores,
            then pan-sear until golden-brown."
    elif sub_name == "jackfruit":
        -> "Simmer until tender, then shred with two forks
            to mimic pulled meat fibers."
    else:
        -> "Saute the {sub_name} in a pan with a splash of oil to firm up its texture."
```

**The rule**: if the delta is below the threshold, zero technique is generated. Only a meaningful gap produces an instruction. This is why two similar-texture ingredients get no prep advice, while swapping chicken (chewiness 7.0) for silken tofu (chewiness 1.5) triggers a freeze-thaw instruction.

---

## Step 6 — How Flavor Molecules Drive the Spice Bridge

The `flavor_molecules` field — also generated by the same bootstrap prompt — is used in a completely separate calculation. This is **set math**, not vector math:

```python
# vegan_engine.py lines 83–108
def get_spice_bridge(original_mols, substitute_mols, all_features, top_k=3):

    # A — Compute what aromas are lost by the substitution
    flavor_gap = set(original_mols) - set(substitute_mols)
    # example: chicken {inosine monophosphate, 2-methyl-3-furanthiol, methional, heptanal}
    #          soy chunks {pyrazines, hexanal}
    # gap = {inosine monophosphate, 2-methyl-3-furanthiol, methional, heptanal}

    if not flavor_gap:
        return []   # substitute already covers all aromas, no bridge needed

    # B — Scan every vegan ingredient labeled flavor_enhancer or aromatic
    for name, data in all_features.items():
        if data.get("culinary_role") in ["flavor_enhancer", "aromatic"] \
           and data.get("is_vegan"):
            spice_mols   = set(data.get("flavor_molecules", []))
            gap_covered  = len(spice_mols & flavor_gap)
            bridge_score = gap_covered / len(flavor_gap)  # fraction of gap this spice fills

            if bridge_score > 0:
                matching = list(spice_mols & flavor_gap)
                bridge_scores.append({
                    "spice": name,
                    "fills_gap_ratio": round(bridge_score, 3),
                    "reason": f"adds {', '.join(matching)} compounds to fill the aromatic gap"
                })

    # C — Return top 3 spices ranked by coverage
    bridge_scores.sort(key=lambda x: x["fills_gap_ratio"], reverse=True)
    return bridge_scores[:3]
```

The `flavor_molecules` field came from this line of the **same bootstrap prompt**:
```
List 3-5 primary flavor volatile compounds (e.g. aldehydes, pyrazines, esters, terpenes, or specific molecules).
```

So the spice bridge is entirely driven by what the LLM described as the ingredient's chemical aroma signature.

---

## Step 7 — Full Concrete Example: "duck" (unknown ingredient)

**User input**: `["duck", "tomato", "onion"]`, `is_vegan=True`, state=Kerala

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: bootstrap_ingredient_profile("duck")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"duck" is not in MongoDB or chemical_features.json.
-> Groq Llama 3.3 70B called with the production prompt.

LLM response:
{
  "is_vegan": false,
  "macros": {"proteins": 27.0, "fats": 28.0, "carbs": 0.0},
  "texture_profile": [6.0, 8.0, 5.5, 8.5, 7.0],
                      ^h   ^ch  ^mo  ^fm   ^el
  "flavor_molecules": ["heptanal", "2-methyl-3-furanthiol", "dimethyl trisulfide", "nonanal"],
  "culinary_role": "base_protein"
}

Validation (api.py line 807):
  len([6.0, 8.0, 5.5, 8.5, 7.0]) == 5  -> PASS
  all values are float               -> PASS
  -> Profile accepted and saved to MongoDB ratatouille.chemical_features


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: Score all vegan candidates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scoring duck vs. soy chunks (texture_profile: [5.0, 6.5, 6.0, 3.0, 5.5]):

  flavor_sim = jaccard({heptanal, 2-methyl-3-furanthiol, ...}
                     & {pyrazines, hexanal})
             = 0 / 6 = 0.00

  text_sim   = 1.0 - dist([6.0, 8.0, 5.5, 8.5, 7.0],
                           [5.0, 6.5, 6.0, 3.0, 5.5]) / 20.124
  distances:  sqrt((6-5)^2 + (8-6.5)^2 + (5.5-6)^2 + (8.5-3)^2 + (7-5.5)^2)
            = sqrt(  1.00  +   2.25   +   0.25   +   30.25   +   2.25  )
            = sqrt(36.0) = 6.0
  text_sim   = 1.0 - 6.0/20.124 = 0.70

  role_match = 1.0  (both are base_protein)

  SCORE = 0.3*0.00 + 0.4*0.70 + 0.3*1.00 = 0.58
  -> soy chunks ranked #1


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: calculate_delta_recommendations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

delta_fat       = 28.0 - 2.5 = 25.5  EXCEEDS threshold 10.0
  -> additions: [{name: "coconut oil or neutral vegetable oil",
                  amount: "1-2 tsp",
                  purpose: "matches lipid profile..."}]

delta_chewiness = 8.0 - 6.5 = 1.5   below threshold 3.0  -> no technique
delta_moisture  = 5.5 - 6.0 = -0.5  above threshold -2.0 -> no technique
  -> techniques: []


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4: get_spice_bridge
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

flavor_gap = {heptanal, 2-methyl-3-furanthiol, dimethyl trisulfide, nonanal}

Scanning flavor_enhancer/aromatic candidates:
  smoked paprika: flavor_molecules = {pyrazines, 2-methyl-3-furanthiol, capsaicin}
    covered = {2-methyl-3-furanthiol} -> ratio = 1/4 = 0.25  ✓ included
  cumin:          flavor_molecules = {cuminaldehyde, pyrazines}
    covered = {}                    -> ratio = 0/4 = 0.00    skipped

  -> spice_bridge: [{spice: "smoked paprika", fills_gap_ratio: 0.25,
                     reason: "adds 2-methyl-3-furanthiol compounds to fill the aromatic gap"}]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL OUTPUT (sent to React UI via SSE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "best_vegan_substitute": "soy chunks",
  "match_score": 0.58,
  "compensation_blueprint": {
    "auxiliary_additions": [
      { "name": "coconut oil or neutral vegetable oil", "amount": "1-2 tsp" }
    ],
    "techniques": [],
    "spice_bridge": [
      { "spice": "smoked paprika", "fills_gap_ratio": 0.25,
        "reason": "adds 2-methyl-3-furanthiol compounds to fill the aromatic gap" }
    ]
  }
}
```

---

## Summary: Complete 5D Data Flow

```
Bootstrap Prompt (Groq Llama 3.3 70B, api.py line 761):
  "Rate its texture on a 1.0-10.0 scale:
   [hardness, chewiness, moisture, fat_mouthfeel, elasticity]"
                  |
                  v
LLM returns: "texture_profile": [h, ch, mo, fm, el]  (5 floats)
                  |
                  | api.py line 807 — hard validation
                  | len == 5  AND  all(isinstance(v, (int, float)))
                  | FAIL -> return None -> keyword fallback (static 5D profiles)
                  | PASS -> save to MongoDB + chemical_features.json
                  |
                  v
            texture_profile stored in DB
                  |
        .---------+-----------.-----------.
        |                     |           |
        v                     v           v
texture_similarity()    [index 1]     [index 2]      flavor_molecules
Euclidean distance      chewiness     moisture       (also from prompt)
across all 5 dims       delta         delta               |
/ 20.124                   |              |               v
        |              if > 3.0       if < -2.0    get_spice_bridge()
        |              -> freeze/     -> press      set subtraction:
        |                 shred/      technique     orig_mols - sub_mols
        v                 pan-fry                       |
0.40 * text_sim           |          if > 4.0           v
into composite            |          -> soak        fills_gap_ratio
score -> picks            |          technique      ranked spice list
BEST substitute           |
                      technique
                      output
```
