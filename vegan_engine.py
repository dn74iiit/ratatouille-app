import os
import json
import numpy as np

# Load chemical features database
DB_PATH = os.path.join(os.path.dirname(__file__), "chemical_features.json")

def load_features():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")
    with open(DB_PATH, "r") as f:
        return json.load(f)

def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def texture_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.5
    # Vectors have length 6, values between 1 and 5
    # Max Euclidean distance is sqrt(6 * (5-1)^2) = sqrt(96) ~= 9.798
    arr_a = np.array(vec_a, dtype=float)
    arr_b = np.array(vec_b, dtype=float)
    dist = np.linalg.norm(arr_a - arr_b)
    max_dist = 9.798
    return float(1.0 - (dist / max_dist))

def functional_overlap(role_a, role_b):
    # Perfect match = 1.0, otherwise 0.0
    return 1.0 if role_a == role_b else 0.0

def calculate_composite_score(orig_data, cand_data, alpha=0.3, beta=0.4, gamma=0.3):
    flavor_sim = jaccard_similarity(
        set(orig_data.get("flavor_molecules", [])),
        set(cand_data.get("flavor_molecules", []))
    )
    text_sim = texture_similarity(
        orig_data.get("texture", []),
        cand_data.get("texture", [])
    )
    func_sim = functional_overlap(
        orig_data.get("role", ""),
        cand_data.get("role", "")
    )
    total_score = alpha * flavor_sim + beta * text_sim + gamma * func_sim
    return round(total_score, 4), {
        "flavor_similarity": round(flavor_sim, 4),
        "texture_similarity": round(text_sim, 4),
        "functional_fit": round(func_sim, 4)
    }

def get_spice_bridge(original_mols, substitute_mols, all_features, top_k=3):
    orig_set = set(original_mols)
    sub_set = set(substitute_mols)
    flavor_gap = orig_set - sub_set
    
    if not flavor_gap:
        return []
        
    bridge_scores = []
    for name, data in all_features.items():
        if data.get("is_spice") and data.get("is_vegan"):
            spice_mols = set(data.get("flavor_molecules", []))
            gap_covered = len(spice_mols & flavor_gap)
            bridge_score = gap_covered / len(flavor_gap) if len(flavor_gap) > 0 else 0.0
            if bridge_score > 0:
                # Add natural language reason based on compound overlap
                matching_compounds = list(spice_mols & flavor_gap)
                reason = f"adds {', '.join(matching_compounds)} compounds to fill the aromatic gap"
                bridge_scores.append({
                    "spice": name,
                    "fills_gap_ratio": round(bridge_score, 3),
                    "reason": reason
                })
                
    bridge_scores.sort(key=lambda x: x["fills_gap_ratio"], reverse=True)
    return bridge_scores[:top_k]

def calculate_delta_recommendations(orig_name, sub_name, orig_data, sub_data, archetype="Curry"):
    orig_macros = orig_data.get("macros", {})
    sub_macros = sub_data.get("macros", {})
    orig_text = orig_data.get("texture", [0]*6)
    sub_text = sub_data.get("texture", [0]*6)
    
    delta_fat = orig_macros.get("fat", 0) - sub_macros.get("fat", 0)
    delta_water = orig_macros.get("water", 0) - sub_macros.get("water", 0)
    
    # 6D texture indices: 0=hardness, 1=chewiness, 2=fibrousness, 3=moisture, 4=elasticity, 5=granularity
    delta_chewiness = orig_text[1] - sub_text[1]
    delta_fibrous = orig_text[2] - sub_text[2]
    
    additions = []
    techniques = []
    
    # 1. Lipid Bridging (Fat Deficit)
    if delta_fat > 0.1:
        if archetype in ["Curry", "Dry_Sabzi", "Soup"]:
            additions.append({
                "name": "coconut oil or neutral vegetable oil",
                "amount": "1-2 tsp",
                "purpose": "matches lipid profile to ensure proper fat-soluble spice (like curcumin) absorption"
            })
        elif archetype == "Salad":
            additions.append({
                "name": "cold-pressed olive oil",
                "amount": "1-2 tsp",
                "purpose": "drizzle over substitute to replicate the fat mouthfeel of the dairy original"
            })
        else:
            additions.append({
                "name": "neutral oil",
                "amount": "1 tsp",
                "purpose": "bridges fat profile deficit"
            })
            
    # Umami bridging (Dairy or Meat original flavor compounds)
    if not sub_data.get("is_vegan", False) == orig_data.get("is_vegan", False):
        if "diacetyl" in orig_data.get("flavor_molecules", []) and "diacetyl" not in sub_data.get("flavor_molecules", []):
            additions.append({
                "name": "nutritional yeast",
                "amount": "1 tsp",
                "purpose": "adds savory, buttery dairy-like notes missing from the plant substitute"
            })
        elif any(x in ["hydrogen_sulfide", "2-methyl-3-furanthiol", "pyrazines"] for x in orig_data.get("flavor_molecules", [])):
            additions.append({
                "name": "monosodium glutamate (MSG) or soy sauce",
                "amount": "1/2 tsp",
                "purpose": "bridges the savory meat umami profile"
            })
            
    # 2. Moisture Adjustments
    if delta_water < -0.1:
        techniques.append(
            f"Wrap the {sub_name} in a clean kitchen towel and press it under a heavy object for 15 minutes to extract excess moisture and prevent sogginess."
        )
    elif delta_water > 0.2:
        if sub_name == "soya chunks":
            techniques.append(
                "Soak the soya chunks in hot water for 15 minutes and squeeze the excess water out completely before cooking to allow it to absorb the recipe flavors."
            )
        else:
            techniques.append(
                f"Hydrate the dry {sub_name} by soaking in water or simmering before adding to the dish."
            )
        
    # 3. Textural Adjustment (Chewiness or fibrousness deficit)
    if sub_name == "king oyster mushroom":
        techniques.append(
            "Slice the king oyster mushroom stalks into round coins and score them in a cross-hatch pattern, then sauté in oil to mimic the firm, bouncy bite of shrimp."
        )
    elif delta_fibrous > 2 or delta_chewiness > 1:
        if sub_name == "soya chunks":
            techniques.append(
                "Soak the soya chunks in boiling water for 15 minutes, squeeze the water out completely, and pan-fry before adding to the sauce to mimic poultry fibers."
            )
        elif sub_name == "tofu":
            techniques.append(
                "Freeze and thaw the tofu beforehand to open up its pores, then pan-sear until golden-brown to create a chewy surface texture."
            )
        elif sub_name == "jackfruit":
            techniques.append(
                "Simmer the jackfruit pieces until tender, then shred with two forks to mimic pulled meat fibers."
            )
        else:
            techniques.append(
                f"Sauté the {sub_name} in a pan with a splash of oil to firm up its texture."
            )
            
    return {
        "additions": additions,
        "techniques": techniques
    }

def generate_vegan_blueprint(ingredient_name, archetype="Curry"):
    features = load_features()
    clean_name = ingredient_name.lower().strip()
    
    if clean_name not in features:
        fallback_data = classify_by_keyword(clean_name)
        if fallback_data:
            orig_data = fallback_data
        else:
            return {
                "status": "error",
                "message": f"Ingredient '{ingredient_name}' not found in the database.",
                "original_ingredient": clean_name
            }
    else:
        orig_data = features[clean_name]
    if orig_data.get("is_vegan"):
        return {
            "status": "already_vegan",
            "message": f"'{ingredient_name}' is already vegan, no substitution needed.",
            "original_ingredient": clean_name
        }
        
    # Find all vegan candidates
    candidates = []
    for name, data in features.items():
        if data.get("is_vegan") and not data.get("is_spice"):
            candidates.append((name, data))
            
    if not candidates:
        return {
            "status": "error",
            "message": "No vegan substitution candidates found in the database.",
            "original_ingredient": clean_name
        }
        
    scored_candidates = []
    for cand_name, cand_data in candidates:
        score, breakdown = calculate_composite_score(orig_data, cand_data)
        scored_candidates.append({
            "substitute": cand_name,
            "score": score,
            "breakdown": breakdown,
            "data": cand_data
        })
        
    # Sort candidates by score descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    best_candidate = scored_candidates[0]
    
    sub_name = best_candidate["substitute"]
    sub_data = best_candidate["data"]
    
    # Calculate delta recommendations and spice bridge
    recommendations = calculate_delta_recommendations(
        clean_name, sub_name, orig_data, sub_data, archetype=archetype
    )
    spice_bridge = get_spice_bridge(
        orig_data.get("flavor_molecules", []),
        sub_data.get("flavor_molecules", []),
        features
    )
    
    return {
        "status": "success",
        "original_ingredient": clean_name,
        "best_vegan_substitute": sub_name,
        "match_score": best_candidate["score"],
        "score_breakdown": best_candidate["breakdown"],
        "chemical_delta": {
            "lipid_deficit": round(float(orig_data["macros"]["fat"] - sub_data["macros"]["fat"]), 3),
            "moisture_excess": round(float(sub_data["macros"]["water"] - orig_data["macros"]["water"]), 3)
        },
        "compensation_blueprint": {
            "auxiliary_additions": recommendations["additions"],
            "techniques": recommendations["techniques"],
            "spice_bridge": spice_bridge
        }
    }

# ============================================================
# DYNAMICGeneralization & Caching Logic
# ============================================================

STATIC_FALLBACKS = {
    "red_meat": {
        "is_vegan": False,
        "macros": { "fat": 0.20, "protein": 0.22, "carb": 0.0, "water": 0.57 },
        "texture": [4, 4, 4, 2, 2, 2],
        "flavor_molecules": ["methanethiol", "dimethyl_sulfide", "2-methyl-3-furanthiol", "pyrazines"],
        "role": "bulk_protein"
    },
    "poultry": {
        "is_vegan": False,
        "macros": { "fat": 0.14, "protein": 0.25, "carb": 0.0, "water": 0.60 },
        "texture": [3, 4, 4, 3, 2, 1],
        "flavor_molecules": ["hydrogen_sulfide", "methyl_furan", "dimethyl_disulfide", "hexanal", "nonanal", "pyrazines"],
        "role": "bulk_protein"
    },
    "seafood": {
        "is_vegan": False,
        "macros": { "fat": 0.01, "protein": 0.24, "carb": 0.0, "water": 0.75 },
        "texture": [3, 4, 1, 4, 3, 2],
        "flavor_molecules": ["dimethyl_sulfide", "trimethylamine", "pyrrole", "hexanal", "sweet_esters"],
        "role": "bulk_protein"
    },
    "dairy_fat": {
        "is_vegan": False,
        "macros": { "fat": 0.81, "protein": 0.01, "carb": 0.01, "water": 0.16 },
        "texture": [2, 1, 0, 2, 1, 1],
        "flavor_molecules": ["diacetyl", "butyric_acid", "delta-decalactone", "lactones"],
        "role": "fat_source"
    },
    "dairy_liquid": {
        "is_vegan": False,
        "macros": { "fat": 0.03, "protein": 0.03, "carb": 0.05, "water": 0.88 },
        "texture": [1, 1, 0, 5, 1, 1],
        "flavor_molecules": ["dimethyl_sulfide", "diacetyl", "lactone"],
        "role": "creamy_liquid"
    },
    "sweetener": {
        "is_vegan": False,
        "macros": { "fat": 0.0, "protein": 0.0, "carb": 0.82, "water": 0.18 },
        "texture": [2, 1, 0, 2, 2, 1],
        "flavor_molecules": ["phenylacetaldehyde", "floral_esters"],
        "role": "sweetener"
    }
}

def classify_by_keyword(name):
    clean_name = name.lower().strip()
    
    red_meat_kws = ["mutton", "lamb", "pork", "beef", "goat", "meat", "steak", "veal", "ham", "bacon", "venison"]
    poultry_kws = ["chicken", "duck", "poultry", "turkey", "quail", "goose", "fowl", "pheasant"]
    seafood_kws = ["shrimp", "prawn", "fish", "salmon", "tuna", "crab", "lobster", "seafood", "cod", "sardine", "mackerel", "clam", "scallop", "halibut", "shrimps", "prawns"]
    dairy_fat_kws = ["butter", "ghee", "lard"]
    dairy_liquid_kws = ["milk", "cream", "yogurt", "curd", "cheese", "paneer", "whey"]
    sweetener_kws = ["honey"]
    
    if any(kw in clean_name for kw in red_meat_kws):
        return STATIC_FALLBACKS["red_meat"]
    if any(kw in clean_name for kw in poultry_kws):
        return STATIC_FALLBACKS["poultry"]
    if any(kw in clean_name for kw in seafood_kws):
        return STATIC_FALLBACKS["seafood"]
    if any(kw in clean_name for kw in dairy_fat_kws):
        return STATIC_FALLBACKS["dairy_fat"]
    if any(kw in clean_name for kw in dairy_liquid_kws):
        return STATIC_FALLBACKS["dairy_liquid"]
    if any(kw in clean_name for kw in sweetener_kws):
        return STATIC_FALLBACKS["sweetener"]
        
    return None

def save_new_feature(name, data):
    try:
        features = load_features()
        clean_name = name.lower().strip()
        features[clean_name] = data
        with open(DB_PATH, "w") as f:
            json.dump(features, f, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save new feature to DB: {e}")
        return False

