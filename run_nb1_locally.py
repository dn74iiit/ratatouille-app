# ── Cell 2: Engine functions (mirrors vegan_engine.py exactly) ──────────────────
import json, numpy as np, csv
from datetime import datetime

DB_PATH = 'chemical_features.json'

def load_features():
    with open(DB_PATH) as f: return json.load(f)

def jaccard_similarity(a, b):
    if not a or not b: return 0.0
    i = len(a & b); u = len(a | b)
    return i / u if u > 0 else 0.0

def texture_similarity(va, vb):
    if not va or not vb: return 0.5
    dist = float(np.linalg.norm(np.array(va, float) - np.array(vb, float)))
    return 1.0 - dist / 9.798  # max Euclidean dist for 6D [1-5] space

def functional_overlap(ra, rb): return 1.0 if ra == rb else 0.0

def composite_score(od, cd, a=0.3, b=0.4, g=0.3):
    fl = jaccard_similarity(set(od.get('flavor_molecules',[])), set(cd.get('flavor_molecules',[])))
    tx = texture_similarity(od.get('texture',[]), cd.get('texture',[]))
    fn = functional_overlap(od.get('role',''), cd.get('role',''))
    total = round(a*fl + b*tx + g*fn, 4)
    return total, {'flavor_similarity': round(fl,4), 'texture_similarity': round(tx,4), 'functional_fit': round(fn,4)}

def spice_bridge(orig_mols, sub_mols, feats, k=3):
    gap = set(orig_mols) - set(sub_mols)
    if not gap: return []
    res = []
    for nm, d in feats.items():
        if d.get('is_spice') and d.get('is_vegan'):
            sm = set(d.get('flavor_molecules',[]))
            sc = len(sm & gap) / len(gap)
            if sc > 0: res.append({'spice': nm, 'fills_gap_ratio': round(sc,3)})
    res.sort(key=lambda x: x['fills_gap_ratio'], reverse=True)
    return res[:k]

def culinary_note(sub_name, od, cd):
    dw = cd.get('macros',{}).get('water',0) - od.get('macros',{}).get('water',0)
    ot = od.get('texture',[0]*6); st = cd.get('texture',[0]*6)
    notes = []
    if dw < -0.1: notes.append('Press ' + sub_name + ' 15 min to remove excess moisture.')
    elif dw > 0.2: notes.append('Soak ' + sub_name + ' in hot water 15 min and squeeze dry.')
    if sub_name == 'jackfruit': notes.append('Shred with two forks after simmering.')
    elif sub_name == 'tofu' and len(ot)>2 and len(st)>2 and (ot[2]-st[2]>1 or ot[1]-st[1]>1):
        notes.append('Freeze/thaw then pan-sear until golden for better texture.')
    elif sub_name == 'soya chunks' and len(ot)>2 and len(st)>2 and ot[2]-st[2]>2:
        notes.append('Soak in boiling water 15 min, squeeze dry, pan-fry before adding to sauce.')
    elif sub_name == 'king oyster mushroom':
        notes.append('Score stalks in cross-hatch pattern, saute in oil to mimic shrimp bite.')
    if not notes: notes.append('Use as direct replacement with standard cooking method.')
    return ' '.join(notes)

print('Engine ready.')


# ── Cell 3: Score every non-veg ingredient against all vegan candidates ─────────
TOP_K = 5
feats = load_features()
print('Loaded ' + str(len(feats)) + ' ingredients')

non_veg  = {n: d for n,d in feats.items() if not d.get('is_vegan',True) and not d.get('is_spice',False)}
veg_pool = [(n,d) for n,d in feats.items()  if d.get('is_vegan',False)  and not d.get('is_spice',False)]
print('Non-veg: ' + str(len(non_veg)) + '  |  Vegan pool: ' + str(len(veg_pool)))

vegan_db = {}
for orig, od in non_veg.items():
    scored = []
    for cand, cd in veg_pool:
        sc, bk = composite_score(od, cd)
        sb = spice_bridge(od.get('flavor_molecules',[]), cd.get('flavor_molecules',[]), feats)
        df = round(float(od.get('macros',{}).get('fat',0)   - cd.get('macros',{}).get('fat',0)),   3)
        dw = round(float(cd.get('macros',{}).get('water',0) - od.get('macros',{}).get('water',0)), 3)
        scored.append({'substitute': cand, 'composite_score': sc, 'score_breakdown': bk,
                       'chemical_delta': {'lipid_deficit': df, 'moisture_excess': dw},
                       'spice_bridge': sb, 'culinary_notes': culinary_note(cand, od, cd)})
    scored.sort(key=lambda x: x['composite_score'], reverse=True)
    for r, item in enumerate(scored[:TOP_K], 1): item['rank'] = r
    vegan_db[orig] = {'original_role': od.get('role','unknown'), 'alternatives': scored[:TOP_K]}
    print('  ' + orig.ljust(22) + ' -> ' + scored[0]['substitute'] + ' (score=' + str(scored[0]['composite_score']) + ')')

print('Done. Processed ' + str(len(vegan_db)) + ' ingredients.')


# ── Cell 4: Save JSON ────────────────────────────────────────────────────────────
out = {'_meta': {'generated_at': datetime.now().isoformat(),
                 'scoring_weights': {'alpha_flavor': 0.3, 'beta_texture': 0.4, 'gamma_role': 0.3},
                 'top_k': TOP_K, 'total_non_veg': len(vegan_db)},
       'alternatives': vegan_db}
with open('vegan_alternatives_db.json','w',encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('Saved vegan_alternatives_db.json')


# ── Cell 5: Save CSV + download both files ───────────────────────────────────────
rows = []
for orig, entry in vegan_db.items():
    for alt in entry['alternatives']:
        rows.append({'original_ingredient': orig, 'original_role': entry['original_role'],
                     'rank': alt['rank'], 'vegan_substitute': alt['substitute'],
                     'composite_score': alt['composite_score'],
                     'flavor_similarity': alt['score_breakdown']['flavor_similarity'],
                     'texture_similarity': alt['score_breakdown']['texture_similarity'],
                     'functional_fit': alt['score_breakdown']['functional_fit'],
                     'lipid_deficit': alt['chemical_delta']['lipid_deficit'],
                     'moisture_excess': alt['chemical_delta']['moisture_excess'],
                     'spice_bridge': ', '.join(s['spice'] for s in alt['spice_bridge']),
                     'culinary_notes': alt['culinary_notes']})
with open('vegan_alternatives_db.csv','w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('Saved vegan_alternatives_db.csv (' + str(len(rows)) + ' rows)')
from google.colab import files
files.download('vegan_alternatives_db.json')
files.download('vegan_alternatives_db.csv')


