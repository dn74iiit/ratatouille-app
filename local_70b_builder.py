import os, json, time, re, pymongo
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")

client = pymongo.MongoClient(MONGO_URI)
db = client.ratatouille
features_col = db.chemical_features
alternatives_col = db.vegan_alternatives

# The full vocabulary
with open("vocab_list.txt", "r", encoding="utf-8") as f:
    full_vocab = [line.strip() for line in f if line.strip()]

hf_client = InferenceClient(model="meta-llama/Llama-3.3-70B-Instruct", token=HF_TOKEN)

PROMPT_TEMPLATE = """You are a food chemistry and culinary database. Output only valid JSON. No explanations.
Generate a chemical and physical profile for the ingredient '{ingredient}' matching the specified JSON format.
Determine if it is vegan (true/false).
Specify macros (fat, protein, carb, water as ratios summing to 1.0).
Rate its texture on a 1-5 scale: [hardness, chewiness, fibrousness, moisture, elasticity, granularity].
List 3-5 primary flavor volatile compounds.
Assign a culinary role: [bulk_protein, fat_source, binder, creamy_liquid, sweetener, seasoning, veggie, starch].

Return ONLY valid JSON matching this exact structure:
{{
  "is_vegan": false,
  "macros": {{"fat": 0.20, "protein": 0.22, "carb": 0.0, "water": 0.58}},
  "texture": [4, 4, 4, 2, 2, 2],
  "flavor_molecules": ["methanethiol", "dimethyl_sulfide", "pyrazines"],
  "role": "bulk_protein"
}}"""

profiles = {}

def fetch_profile(ing):
    messages = [
        {"role": "system", "content": "You are a food chemistry expert. Output JSON only."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(ingredient=ing)}
    ]
    for attempt in range(3):
        try:
            res = hf_client.chat_completion(messages=messages, max_tokens=250, temperature=0.1)
            text = res.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                profile = json.loads(match.group(0))
                profile["_id"] = ing
                return ing, profile
        except Exception as e:
            time.sleep(1)
    return ing, None

print(f"[*] Fetching 70B profiles for {len(full_vocab)} ingredients concurrently...")
start = time.time()

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_profile, ing): ing for ing in full_vocab}
    for count, future in enumerate(as_completed(futures), 1):
        ing, profile = future.result()
        if profile:
            profiles[ing] = profile
        if count % 20 == 0:
            print(f"  ... {count}/{len(full_vocab)} done")

print(f"[*] Generation complete in {time.time()-start:.1f}s. Extracted: {len(profiles)}")

if not profiles:
    print("[ERROR] No profiles fetched!")
    exit(1)

# Upload to DB
features_col.delete_many({})
features_col.insert_many(list(profiles.values()))
print("[*] Inserted raw features into MongoDB")

# Math
import numpy as np

def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

def texture_sim(va, vb):
    va, vb = (va + [3]*6)[:6], (vb + [3]*6)[:6]
    dist = float(np.linalg.norm(np.array(va, float) - np.array(vb, float)))
    return max(0.0, 1.0 - dist / 9.798)

def composite_score(orig, cand):
    fl = jaccard(orig.get('flavor_molecules', []), cand.get('flavor_molecules', []))
    tx = texture_sim(orig.get('texture', [3]*6), cand.get('texture', [3]*6))
    fn = 1.0 if orig.get('role') == cand.get('role') else 0.0
    score = round(0.3*fl + 0.4*tx + 0.3*fn, 4)
    return score, {'flavor_similarity': round(fl, 4), 'texture_similarity': round(tx, 4), 'functional_fit': round(fn, 4)}

all_profiles = {doc['_id']: doc for doc in features_col.find({})}
vegan_pool = {k: v for k, v in all_profiles.items() if v.get('is_vegan') is True}
non_vegan_pool = {k: v for k, v in all_profiles.items() if v.get('is_vegan') is False}

print(f"[*] Matching {len(non_vegan_pool)} non-vegan against {len(vegan_pool)} vegan...")

match_count = 0
for non_veg_name, orig_data in non_vegan_pool.items():
    scored = []
    for veg_name, veg_data in vegan_pool.items():
        score, breakdown = composite_score(orig_data, veg_data)
        scored.append((veg_name, score, breakdown, veg_data))
    if not scored: continue
    scored.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score, best_breakdown, best_data = scored[0]
    
    # Simple blueprint
    om, sm = orig_data.get('macros', {}), best_data.get('macros', {})
    additions, techniques = [], []
    if (om.get('fat', 0) - sm.get('fat', 0)) > 0.10:
        additions.append({'name': 'coconut oil', 'amount': '1-2 tsp', 'purpose': 'fat deficit'})
    
    match_doc = {
        '_id': non_veg_name,
        'original_ingredient': non_veg_name,
        'best_vegan_substitute': best_name,
        'match_score': best_score,
        'score_breakdown': best_breakdown,
        'top5_alternatives': [{'substitute': s[0], 'score': s[1], 'breakdown': s[2]} for s in scored[:5]],
        'compensation_blueprint': {'auxiliary_additions': additions, 'techniques': techniques, 'spice_bridge': []}
    }
    alternatives_col.update_one({'_id': non_veg_name}, {'$set': match_doc}, upsert=True)
    match_count += 1

print(f"[*] SUCCESS! {match_count} matches pushed to ratatouille.vegan_alternatives")
