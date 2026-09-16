import os, pymongo
from dotenv import load_dotenv

load_dotenv()
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client.ratatouille

MEATS_AND_DAIRY = {
    "chicken", "mutton", "beef", "pork", "lamb", "veal", "duck", "turkey", "venison", "rabbit",
    "catfish", "hilsa", "salmon", "tuna", "pomfret", "prawn", "shrimp", "crab", "lobster", "fish", "squid", "octopus",
    "egg", "butter", "ghee", "milk", "cheese", "curd", "yogurt", "cream", "honey", "lard", "suet", "gelatin", "paneer", "mayonnaise"
}

def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

print("[*] Downloading existing chemical features...")
features = list(db.chemical_features.find({}))
for f in features:
    if f["_id"].lower() in MEATS_AND_DAIRY:
        f["is_vegan"] = False
        
db.chemical_features.delete_many({})
db.chemical_features.insert_many(features)
print("[*] Repaired 'is_vegan' flags in DB.")

vegan_pool = [f for f in features if f.get("is_vegan") and not f.get("is_spice")]
non_vegan_pool = [f for f in features if not f.get("is_vegan")]

print(f"[*] Re-calculating alternatives (Vegan candidates: {len(vegan_pool)})")

def score_candidates(orig):
    results = []
    for cand in vegan_pool:
        fl = jaccard(orig.get('flavor_molecules', []), cand.get('flavor_molecules', []))
        fn = 1.0 if orig.get('role') == cand.get('role') else 0.0
        # Discarding texture since Qwen3 hallucinated identical textures for all
        score = round(0.5*fl + 0.5*fn, 4)
        results.append({
            'substitute': cand['_id'],
            'score': score,
            'breakdown': { 'flavor_similarity': round(fl, 4), 'texture_similarity': 0.0, 'functional_fit': round(fn, 4) },
            'data': cand
        })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def get_spice_bridge(orig, sub):
    gap = set(orig.get('flavor_molecules', [])) - set(sub.get('flavor_molecules', []))
    if not gap: return []
    bridges = []
    for cand in vegan_pool:
        if cand.get('role') == 'seasoning':
            covered = set(cand.get('flavor_molecules', [])) & gap
            if covered:
                bridges.append({'spice': cand['_id'], 'ratio': round(len(covered)/len(gap), 2), 'supplies': list(covered)})
    bridges.sort(key=lambda x: x['ratio'], reverse=True)
    return bridges[:3]

alternatives = []
for nv in non_vegan_pool:
    scored = score_candidates(nv)
    if not scored: continue
    best = scored[0]
    
    om = nv.get('macros', {})
    sm = best['data'].get('macros', {})
    additions, techniques = [], []
    if (om.get('fat', 0) - sm.get('fat', 0)) > 0.10:
        additions.append({'name': 'coconut oil or neutral vegetable oil', 'amount': '1-2 tsp', 'purpose': 'compensates fat deficit'})
    
    # Simple moisture check (since the arrays are flawed, macros are sometimes decent)
    dw = om.get('water', 0) - sm.get('water', 0)
    if dw < -0.10: techniques.append(f"Press {best['substitute']} for 15 min to remove excess moisture.")
    elif dw > 0.20: techniques.append(f"Rehydrate {best['substitute']} in hot water before using.")
    
    alternatives.append({
        '_id': nv['_id'],
        'original_ingredient': nv['_id'],
        'best_vegan_substitute': best['substitute'],
        'match_score': best['score'],
        'score_breakdown': best['breakdown'],
        'top5_alternatives': [{'substitute': s['substitute'], 'score': s['score']} for s in scored[:5]],
        'compensation_blueprint': {
            'auxiliary_additions': additions,
            'techniques': techniques,
            'spice_bridge': get_spice_bridge(nv, best['data'])
        }
    })

db.vegan_alternatives.delete_many({})
db.vegan_alternatives.insert_many(alternatives)
print("[*] Success! Repaired vegan alternatives pushed to MongoDB.")
