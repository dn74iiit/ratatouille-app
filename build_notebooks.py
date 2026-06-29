"""
Run this once with: python build_notebooks.py
It generates:
  - GPU_Vegan_Alternatives_DB_Generator.ipynb
  - GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb

Both notebooks include a final cell that uploads the generated data
to MongoDB Atlas (using your existing MONGO_URI from .env).
"""
import json

# ─────────────────────────────────────────────────────────────
#  NOTEBOOK 1: Vegan Alternatives Database Generator (CPU ok)
# ─────────────────────────────────────────────────────────────
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌿 Vegan Alternatives Database Generator\n",
            "\n",
            "Run on **Google Colab (CPU is fine — pure NumPy math)**.\n",
            "\n",
            "This notebook:\n",
            "1. Clones your repo → gets `vegan_engine.py` + `chemical_features.json`\n",
            "2. Runs the Composite Scoring Engine on every non-veg ingredient vs all vegan candidates\n",
            "3. Saves **top-5 vegan alternatives** per ingredient → `vegan_alternatives_db.json` + `.csv`",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 1: Clone repo ──────────────────────────────────────────────────────────\n",
            "!git clone https://github.com/dn74iiit/ratatouille-app.git\n",
            "%cd ratatouille-app\n",
            "!pip install numpy -q\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 2: Engine functions (mirrors vegan_engine.py exactly) ──────────────────\n",
            "import json, numpy as np, csv\n",
            "from datetime import datetime\n",
            "\n",
            "DB_PATH = 'chemical_features.json'\n",
            "\n",
            "def load_features():\n",
            "    with open(DB_PATH) as f: return json.load(f)\n",
            "\n",
            "def jaccard_similarity(a, b):\n",
            "    if not a or not b: return 0.0\n",
            "    i = len(a & b); u = len(a | b)\n",
            "    return i / u if u > 0 else 0.0\n",
            "\n",
            "def texture_similarity(va, vb):\n",
            "    if not va or not vb: return 0.5\n",
            "    dist = float(np.linalg.norm(np.array(va, float) - np.array(vb, float)))\n",
            "    return 1.0 - dist / 9.798  # max Euclidean dist for 6D [1-5] space\n",
            "\n",
            "def functional_overlap(ra, rb): return 1.0 if ra == rb else 0.0\n",
            "\n",
            "def composite_score(od, cd, a=0.3, b=0.4, g=0.3):\n",
            "    fl = jaccard_similarity(set(od.get('flavor_molecules',[])), set(cd.get('flavor_molecules',[])))\n",
            "    tx = texture_similarity(od.get('texture',[]), cd.get('texture',[]))\n",
            "    fn = functional_overlap(od.get('role',''), cd.get('role',''))\n",
            "    total = round(a*fl + b*tx + g*fn, 4)\n",
            "    return total, {'flavor_similarity': round(fl,4), 'texture_similarity': round(tx,4), 'functional_fit': round(fn,4)}\n",
            "\n",
            "def spice_bridge(orig_mols, sub_mols, feats, k=3):\n",
            "    gap = set(orig_mols) - set(sub_mols)\n",
            "    if not gap: return []\n",
            "    res = []\n",
            "    for nm, d in feats.items():\n",
            "        if d.get('is_spice') and d.get('is_vegan'):\n",
            "            sm = set(d.get('flavor_molecules',[]))\n",
            "            sc = len(sm & gap) / len(gap)\n",
            "            if sc > 0: res.append({'spice': nm, 'fills_gap_ratio': round(sc,3)})\n",
            "    res.sort(key=lambda x: x['fills_gap_ratio'], reverse=True)\n",
            "    return res[:k]\n",
            "\n",
            "def culinary_note(sub_name, od, cd):\n",
            "    dw = cd.get('macros',{}).get('water',0) - od.get('macros',{}).get('water',0)\n",
            "    ot = od.get('texture',[0]*6); st = cd.get('texture',[0]*6)\n",
            "    notes = []\n",
            "    if dw < -0.1: notes.append('Press ' + sub_name + ' 15 min to remove excess moisture.')\n",
            "    elif dw > 0.2: notes.append('Soak ' + sub_name + ' in hot water 15 min and squeeze dry.')\n",
            "    if sub_name == 'jackfruit': notes.append('Shred with two forks after simmering.')\n",
            "    elif sub_name == 'tofu' and len(ot)>2 and len(st)>2 and (ot[2]-st[2]>1 or ot[1]-st[1]>1):\n",
            "        notes.append('Freeze/thaw then pan-sear until golden for better texture.')\n",
            "    elif sub_name == 'soya chunks' and len(ot)>2 and len(st)>2 and ot[2]-st[2]>2:\n",
            "        notes.append('Soak in boiling water 15 min, squeeze dry, pan-fry before adding to sauce.')\n",
            "    elif sub_name == 'king oyster mushroom':\n",
            "        notes.append('Score stalks in cross-hatch pattern, saute in oil to mimic shrimp bite.')\n",
            "    if not notes: notes.append('Use as direct replacement with standard cooking method.')\n",
            "    return ' '.join(notes)\n",
            "\n",
            "print('Engine ready.')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 3: Score every non-veg ingredient against all vegan candidates ─────────\n",
            "TOP_K = 5\n",
            "feats = load_features()\n",
            "print('Loaded ' + str(len(feats)) + ' ingredients')\n",
            "\n",
            "non_veg  = {n: d for n,d in feats.items() if not d.get('is_vegan',True) and not d.get('is_spice',False)}\n",
            "veg_pool = [(n,d) for n,d in feats.items()  if d.get('is_vegan',False)  and not d.get('is_spice',False)]\n",
            "print('Non-veg: ' + str(len(non_veg)) + '  |  Vegan pool: ' + str(len(veg_pool)))\n",
            "\n",
            "vegan_db = {}\n",
            "for orig, od in non_veg.items():\n",
            "    scored = []\n",
            "    for cand, cd in veg_pool:\n",
            "        sc, bk = composite_score(od, cd)\n",
            "        sb = spice_bridge(od.get('flavor_molecules',[]), cd.get('flavor_molecules',[]), feats)\n",
            "        df = round(float(od.get('macros',{}).get('fat',0)   - cd.get('macros',{}).get('fat',0)),   3)\n",
            "        dw = round(float(cd.get('macros',{}).get('water',0) - od.get('macros',{}).get('water',0)), 3)\n",
            "        scored.append({'substitute': cand, 'composite_score': sc, 'score_breakdown': bk,\n",
            "                       'chemical_delta': {'lipid_deficit': df, 'moisture_excess': dw},\n",
            "                       'spice_bridge': sb, 'culinary_notes': culinary_note(cand, od, cd)})\n",
            "    scored.sort(key=lambda x: x['composite_score'], reverse=True)\n",
            "    for r, item in enumerate(scored[:TOP_K], 1): item['rank'] = r\n",
            "    vegan_db[orig] = {'original_role': od.get('role','unknown'), 'alternatives': scored[:TOP_K]}\n",
            "    print('  ' + orig.ljust(22) + ' -> ' + scored[0]['substitute'] + ' (score=' + str(scored[0]['composite_score']) + ')')\n",
            "\n",
            "print('Done. Processed ' + str(len(vegan_db)) + ' ingredients.')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 4: Save JSON ────────────────────────────────────────────────────────────\n",
            "out = {'_meta': {'generated_at': datetime.now().isoformat(),\n",
            "                 'scoring_weights': {'alpha_flavor': 0.3, 'beta_texture': 0.4, 'gamma_role': 0.3},\n",
            "                 'top_k': TOP_K, 'total_non_veg': len(vegan_db)},\n",
            "       'alternatives': vegan_db}\n",
            "with open('vegan_alternatives_db.json','w',encoding='utf-8') as f:\n",
            "    json.dump(out, f, indent=2, ensure_ascii=False)\n",
            "print('Saved vegan_alternatives_db.json')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 5: Save CSV + download both files ───────────────────────────────────────\n",
            "rows = []\n",
            "for orig, entry in vegan_db.items():\n",
            "    for alt in entry['alternatives']:\n",
            "        rows.append({'original_ingredient': orig, 'original_role': entry['original_role'],\n",
            "                     'rank': alt['rank'], 'vegan_substitute': alt['substitute'],\n",
            "                     'composite_score': alt['composite_score'],\n",
            "                     'flavor_similarity': alt['score_breakdown']['flavor_similarity'],\n",
            "                     'texture_similarity': alt['score_breakdown']['texture_similarity'],\n",
            "                     'functional_fit': alt['score_breakdown']['functional_fit'],\n",
            "                     'lipid_deficit': alt['chemical_delta']['lipid_deficit'],\n",
            "                     'moisture_excess': alt['chemical_delta']['moisture_excess'],\n",
            "                     'spice_bridge': ', '.join(s['spice'] for s in alt['spice_bridge']),\n",
            "                     'culinary_notes': alt['culinary_notes']})\n",
            "with open('vegan_alternatives_db.csv','w',newline='',encoding='utf-8') as f:\n",
            "    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)\n",
            "print('Saved vegan_alternatives_db.csv (' + str(len(rows)) + ' rows)')\n",
            "from google.colab import files\n",
            "files.download('vegan_alternatives_db.json')\n",
            "files.download('vegan_alternatives_db.csv')\n",
        ],
    },
    # ── MongoDB Upload Cell ──────────────────────────────────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 6: Upload to MongoDB Atlas (ratatouille.vegan_alternatives) ───────────────\n",
            "# Make sure you add MONGO_URI to your Colab Secrets (the Key icon on the left).\n",
            "!pip install 'pymongo[srv]' -q\n",
            "from google.colab import userdata\n",
            "from pymongo import MongoClient, UpdateOne\n",
            "from datetime import datetime, timezone\n",
            "MONGO_URI = userdata.get('MONGO_URI')\n",
            "client = MongoClient(MONGO_URI)\n",
            "col = client['ratatouille']['vegan_alternatives']\n",
            "ops = []\n",
            "for ingredient_name, entry in vegan_db.items():\n",
            "    doc = {\n",
            "        '_id': ingredient_name,\n",
            "        'original_role': entry['original_role'],\n",
            "        'alternatives': entry['alternatives'],\n",
            "        'updated_at': datetime.now(timezone.utc)\n",
            "    }\n",
            "    ops.append(UpdateOne({'_id': ingredient_name}, {'$set': doc}, upsert=True))\n",
            "result = col.bulk_write(ops)\n",
            "print('MongoDB upload done. Upserted:', result.upserted_count, '| Modified:', result.modified_count)\n",
            "print('Collection: ratatouille.vegan_alternatives |', col.count_documents({}), 'total docs')\n",
            "client.close()\n",
        ],
    },
]

nb1 = {
    "cells": nb1_cells,
    "metadata": {
        "accelerator": "None",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("GPU_Vegan_Alternatives_DB_Generator.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2, ensure_ascii=False)
print("Written GPU_Vegan_Alternatives_DB_Generator.ipynb")


# ─────────────────────────────────────────────────────────────
#  NOTEBOOK 2: Indian Budget Vegan Recipe Generator (GPU T4)
# ─────────────────────────────────────────────────────────────

# ── Post-processing: the EXACT hallucination-cutting logic from api.py lines 640-666 ──
POST_PROCESS_CODE = """\
def clean_recipe_output(ai_text: str) -> str:
    \"\"\"
    Exact hallucination-cutting logic from api.py (lines 640-666).
    Applies in order:
      1. Cut at <|eot_id|>  (model's own stop token)
      2. If multiple '### TITLE:' blocks, keep only the first recipe
      3. Strip trailing conversational rambling / notes
      4. If extra '### ' sections appear inside DIRECTIONS, cut there
    \"\"\"
    # 1. Cut at the model's stop token (most reliable signal)
    stop_tokens = ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>', '\n### INGREDIENTS:']
    for t in stop_tokens:
        if t in ai_text:
            ai_text = ai_text.split(t)[0].strip()

    # 2. If the model looped back and added another TITLE block, keep only the first recipe
    if '### TITLE:\\n' in ai_text:
        ai_text = ai_text.split('### TITLE:\\n')[1].strip()

    # 3. Cut trailing rambling signatures (from api.py cut_phrases list)
    cut_phrases = [
        '\\nEnjoy!', '\\nServe hot', '\\nBon Apetit', '\\nChef\\'s Note:',
        '\\nVariations:', '\\nServing suggestion:', '\\nNote:'
    ]
    for phrase in cut_phrases:
        if phrase in ai_text:
            ai_text = ai_text.split(phrase)[0].strip()

    # 4. If another '### ' header appears inside DIRECTIONS, cut it off
    #    (the only valid '###' in output should be '### DIRECTIONS:')
    if '### DIRECTIONS:\\n' in ai_text:
        parts = ai_text.split('### DIRECTIONS:\\n')
        title_part = parts[0]
        directions_part = parts[1]
        if '\\n### ' in directions_part:
            directions_part = directions_part.split('\\n### ')[0]
        ai_text = (title_part + '### DIRECTIONS:\\n' + directions_part).strip()

    return ai_text
"""

# ── Indian-budget recipe templates ──
TEMPLATES_CODE = """\
# ─── Indian Budget Vegan Recipe Templates ───────────────────────────────────────
# Inspired by everyday Indian staple-based cooking under INR 150/serving
# Each template specifies which non-veg it replaces, vegan sub, and base spices.

INDIAN_TEMPLATES = [
    # POULTRY substitutions
    {"name": "Soya Chunks Biryani",       "original": "chicken",      "sub": "soya chunks",
     "style": "Rice_Dish", "budget_inr": 120,
     "ingredients": ["soya chunks","basmati rice","onion","tomato","ginger","garlic","biryani masala","oil","mint","green chili"]},
    {"name": "Soya Chunks Curry",          "original": "chicken",      "sub": "soya chunks",
     "style": "Curry", "budget_inr": 100,
     "ingredients": ["soya chunks","onion","tomato","garlic","ginger","cumin","turmeric","red chili powder","garam masala","oil"]},
    {"name": "Jackfruit Kadai",            "original": "chicken",      "sub": "jackfruit",
     "style": "Dry_Sabzi", "budget_inr": 90,
     "ingredients": ["raw jackfruit","capsicum","onion","tomato","kadai masala","ginger","garlic","oil","cumin seeds"]},
    {"name": "Tofu Butter Masala",         "original": "chicken",      "sub": "tofu",
     "style": "Curry", "budget_inr": 130,
     "ingredients": ["tofu","tomato puree","onion","cashew","cream","butter","garam masala","kashmiri chili","sugar"]},
    {"name": "Tempeh Stir Fry",            "original": "chicken",      "sub": "tempeh",
     "style": "Dry_Sabzi", "budget_inr": 110,
     "ingredients": ["tempeh","cabbage","carrot","soy sauce","garlic","ginger","sesame oil","green onion"]},
    # RED MEAT substitutions
    {"name": "Rajma Keema",                "original": "mutton",       "sub": "kidney beans",
     "style": "Curry", "budget_inr": 80,
     "ingredients": ["kidney beans","onion","tomato","garlic","ginger","cumin","coriander","red chili powder","garam masala","oil"]},
    {"name": "Lentil Kofta Curry",         "original": "mutton",       "sub": "red lentils",
     "style": "Curry", "budget_inr": 70,
     "ingredients": ["red lentils","onion","tomato","garlic","ginger","cumin seeds","turmeric","coriander powder","oil"]},
    {"name": "Jackfruit Vindaloo",         "original": "pork",         "sub": "jackfruit",
     "style": "Curry", "budget_inr": 95,
     "ingredients": ["raw jackfruit","vinegar","garlic","dried red chili","cumin","mustard seeds","tamarind","oil","onion"]},
    {"name": "Soya Keema Matar",           "original": "minced meat",  "sub": "soya granules",
     "style": "Curry", "budget_inr": 85,
     "ingredients": ["soya granules","green peas","onion","tomato","ginger","garlic","cumin","garam masala","oil","salt"]},
    {"name": "Tempeh Rogan Josh",          "original": "beef",         "sub": "tempeh",
     "style": "Curry", "budget_inr": 115,
     "ingredients": ["tempeh","yogurt","onion","fennel seeds","kashmiri chili","ginger powder","cinnamon","cardamom","oil"]},
    # SEAFOOD substitutions
    {"name": "Mushroom Masala Curry",      "original": "fish",         "sub": "mushroom",
     "style": "Curry", "budget_inr": 100,
     "ingredients": ["button mushroom","mustard seeds","turmeric","green chili","coconut milk","onion","tomato","curry leaves","oil"]},
    {"name": "Oyster Mushroom Butter Garlic", "original": "shrimp",    "sub": "king oyster mushroom",
     "style": "Dry_Sabzi", "budget_inr": 120,
     "ingredients": ["king oyster mushroom","vegan butter","garlic","lemon juice","parsley","black pepper","salt"]},
    {"name": "Banana Blossom Malabar Curry",  "original": "fish",         "sub": "banana blossom",
     "style": "Curry", "budget_inr": 80,
     "ingredients": ["banana blossom","coconut milk","mustard seeds","turmeric","tamarind","onion","tomato","curry leaves"]},
    {"name": "Chickpea Coconut Masala",       "original": "crab",         "sub": "chickpeas",
     "style": "Curry", "budget_inr": 75,
     "ingredients": ["chickpeas","coconut","dried red chili","tamarind","onion","tomato","coriander seeds","mustard seeds","oil"]},
    # DAIRY substitutions
    {"name": "Tofu Palak",                 "original": "paneer",       "sub": "tofu",
     "style": "Curry", "budget_inr": 100,
     "ingredients": ["tofu","spinach","onion","garlic","ginger","cumin","cream","garam masala","oil","nutmeg"]},
    {"name": "Tofu Matar",                 "original": "paneer",       "sub": "tofu",
     "style": "Curry", "budget_inr": 90,
     "ingredients": ["tofu","green peas","onion","tomato","ginger","garlic","cumin seeds","turmeric","garam masala","oil"]},
    {"name": "Soya Bhurji",                "original": "egg",          "sub": "tofu scramble",
     "style": "Dry_Sabzi", "budget_inr": 60,
     "ingredients": ["tofu","onion","green chili","tomato","turmeric","cumin seeds","coriander leaves","oil","salt"]},
    {"name": "Chickpea Omelette",          "original": "egg",          "sub": "chickpea flour",
     "style": "Dry_Sabzi", "budget_inr": 50,
     "ingredients": ["chickpea flour","onion","green chili","tomato","turmeric","cumin","oil","salt","water"]},
    # DAL & STAPLE dishes (vegan by nature, budget-focused)
    {"name": "Dal Tadka",                  "original": None,           "sub": "toor dal",
     "style": "Soup", "budget_inr": 60,
     "ingredients": ["toor dal","onion","tomato","garlic","ginger","cumin seeds","turmeric","ghee vegan","red chili","asafoetida"]},
    {"name": "Chana Masala",               "original": None,           "sub": "chickpeas",
     "style": "Curry", "budget_inr": 70,
     "ingredients": ["chickpeas","onion","tomato","garlic","ginger","cumin","coriander","amchur","garam masala","oil"]},
    {"name": "Rajma Chawal",               "original": None,           "sub": "kidney beans",
     "style": "Rice_Dish", "budget_inr": 80,
     "ingredients": ["kidney beans","basmati rice","onion","tomato","garlic","ginger","cumin","garam masala","bay leaf","oil"]},
    {"name": "Aloo Gobi Sabzi",            "original": None,           "sub": "potato + cauliflower",
     "style": "Dry_Sabzi", "budget_inr": 50,
     "ingredients": ["potato","cauliflower","onion","tomato","turmeric","cumin seeds","coriander powder","green chili","oil"]},
    {"name": "Masoor Dal",                 "original": None,           "sub": "red lentils",
     "style": "Soup", "budget_inr": 55,
     "ingredients": ["red lentils","onion","tomato","garlic","turmeric","cumin seeds","red chili","oil","salt","lemon"]},
    {"name": "Baingan Bharta",             "original": None,           "sub": "eggplant",
     "style": "Dry_Sabzi", "budget_inr": 65,
     "ingredients": ["eggplant","onion","tomato","garlic","green chili","cumin seeds","turmeric","coriander leaves","oil"]},
    {"name": "Mixed Veg Sabzi",            "original": None,           "sub": "seasonal vegetables",
     "style": "Dry_Sabzi", "budget_inr": 60,
     "ingredients": ["carrot","beans","potato","peas","onion","tomato","cumin seeds","turmeric","garam masala","oil"]},
    # NEW ADDITIONS (Fish, Pork, Crab, Duck, etc.)
    {"name": "Banana Blossom Malabar Curry",  "original": "fish",         "sub": "banana blossom",
     "style": "Curry", "budget_inr": 85,
     "ingredients": ["banana blossom","mustard seeds","turmeric","coconut milk","tamarind","onion","tomato","green chili","oil"]},
    {"name": "Jackfruit Vindaloo",         "original": "pork",         "sub": "jackfruit",
     "style": "Curry", "budget_inr": 95,
     "ingredients": ["raw jackfruit","vinegar","garlic","dried red chili","cumin seeds","mustard seeds","onion","oil"]},
    {"name": "Chickpea Coconut Masala",       "original": "crab",         "sub": "chickpeas",
     "style": "Curry", "budget_inr": 75,
     "ingredients": ["chickpeas","coconut","fennel seeds","dried red chili","onion","tomato","garlic","ginger","oil"]},
    {"name": "Mushroom Pepper Roast",        "original": "duck",         "sub": "king oyster mushroom",
     "style": "Dry_Sabzi", "budget_inr": 120,
     "ingredients": ["king oyster mushroom","onion","black pepper","fennel seeds","garlic","ginger","curry leaves","oil"]},
    {"name": "Soya Chunks Chili Stir Fry", "original": "squid",     "sub": "soya chunks",
     "style": "Dry_Sabzi", "budget_inr": 90,
     "ingredients": ["soya chunks","capsicum","onion","black pepper","soy sauce","garlic","ginger","oil"]},
    {"name": "Tofu Tikka Masala",          "original": "turkey",       "sub": "tofu",
     "style": "Dry_Sabzi", "budget_inr": 110,
     "ingredients": ["tofu","yogurt vegan","tikka masala","kashmiri chili","lemon juice","garlic","ginger","oil"]},
    {"name": "Smoky Tempeh Fry",           "original": "bacon",        "sub": "tempeh",
     "style": "Dry_Sabzi", "budget_inr": 130,
     "ingredients": ["tempeh","soy sauce","maple syrup","smoked paprika","garlic powder","oil"]},
    {"name": "Soya Granules Masala Curry", "original": "sausage",     "sub": "soya granules",
     "style": "Curry", "budget_inr": 85,
     "ingredients": ["soya granules","onion","tomato","garlic","ginger","cumin","garam masala","oil"]}
]

import random
import itertools

proteins = ['tofu', 'soya chunks', 'chickpeas', 'kidney beans', 'red lentils', 'tempeh', 'soya granules']
veggies = ['potato', 'cauliflower', 'spinach', 'eggplant', 'green peas', 'capsicum', 'mushroom', 'bottle gourd', 'okra', 'cabbage']
bases = ['onion', 'tomato', 'garlic', 'ginger', 'coconut milk', 'tamarind']
styles = ['Curry', 'Dry_Sabzi', 'Soup', 'Rice_Dish']

combos = []
for p in proteins:
    for v in veggies:
        for style in styles:
            # pick random bases
            num_bases = random.randint(2, 4)
            chosen_bases = random.sample(bases, num_bases)
            combos.append({
                'name': '',
                'original': None,
                'sub': None,
                'style': style,
                'budget_inr': random.randint(50, 140),
                'ingredients': [p, v, 'oil', 'cumin seeds', 'turmeric', 'salt'] + chosen_bases
            })

# We generated thousands. We just need 467 to reach 500 total (33 + 467 = 500)
random.seed(42) # repeatable
random.shuffle(combos)
INDIAN_TEMPLATES.extend(combos[:467])

print('Loaded ' + str(len(INDIAN_TEMPLATES)) + ' Indian budget vegan recipe templates')
"""

nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🇮🇳 GPU Indian Budget Vegan Recipe Generator\n",
            "\n",
            "Run on **Google Colab with T4 GPU** (Runtime → Change runtime type → T4 GPU).\n",
            "\n",
            "Uses model: `nd1490/ratatouille-llama3-3b-v10-non-oven` (your V10 non-oven fine-tune).\n",
            "\n",
            "**Post-processing**: uses the exact hallucination-cutting logic from `api.py` (lines 640–666):\n",
            "- Cut at `<|eot_id|>` stop token\n",
            "- Strip duplicate `### TITLE:` blocks\n",
            "- Remove trailing `Enjoy!`, `Chef's Note:` etc.\n",
            "- Clip at unexpected `### ` headers inside DIRECTIONS\n",
            "\n",
            "**Output**: `indian_vegan_recipes_db.csv` — ~25 Indian-inspired vegan recipes under ₹150/serving",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 1: Install dependencies ────────────────────────────────────────────────\n",
            "!pip install unsloth -q\n",
            "!pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' -q\n",
            "!pip install 'trl==0.24.0' peft accelerate bitsandbytes -q\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 2: Clone repo for vegan_engine logic ────────────────────────────────────\n",
            "!git clone https://github.com/dn74iiit/ratatouille-app.git\n",
            "%cd ratatouille-app\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 3: Load V10 non-oven model onto GPU ─────────────────────────────────────\n",
            "import torch\n",
            "from unsloth import FastLanguageModel\n",
            "\n",
            "MODEL_ID = 'nd1490/ratatouille-llama3-3b-v10-non-oven'\n",
            "\n",
            "print('Loading ' + MODEL_ID + ' on GPU...')\n",
            "model, tokenizer = FastLanguageModel.from_pretrained(\n",
            "    model_name=MODEL_ID,\n",
            "    max_seq_length=1024,\n",
            "    load_in_4bit=True,\n",
            "    dtype=None,          # auto-detect best dtype\n",
            ")\n",
            "FastLanguageModel.for_inference(model)  # enable 2x faster inference\n",
            "print('Model ready on ' + str(torch.cuda.get_device_name(0)))\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": POST_PROCESS_CODE,
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": TEMPLATES_CODE,
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 6: Prompt builder — uses V10's exact Llama 3 Instruct template ─────────\n",
            "def build_indian_prompt(template: dict) -> str:\n",
            "    ingr_text = '\\n'.join('- ' + i for i in template['ingredients'])\n",
            "    style = template['style']\n",
            "    budget = template['budget_inr']\n",
            "\n",
            "    system_instruction = (\n",
            "        'You are a Michelin-star master chef. '\n",
            "        'Write a highly detailed, appetizing recipe for a ' + style + '.\\n'\n",
            "        'First, provide an appetizing, restaurant-style title that STRICTLY uses only the provided ingredients. '\n",
            "        'Do NOT invent or add any ingredients to the title that are not in the list.\\n'\n",
            "        'Then, provide step-by-step cooking directions using proper culinary techniques.\\n'\n",
            "        'CRITICAL RULES:\\n'\n",
            "        '1. Ensure all raw ingredients (especially rice, grains, and meats) are explicitly cooked in the instructions.\\n'\n",
            "        '2. Do not change the ingredient quantities provided.\\n'\n",
            "        '3. Do NOT include or hallucinate ANY extra ingredients in the recipe that are not provided in the list.\\n'\n",
            "        '4. DO NOT include any \\'Notes\\', \\'Tips\\', or conversational rambling at the end. Stop immediately after the final serving step.'\n",
            "    )\n",
            "\n",
            "    prompt = (\n",
            "        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n'\n",
            "        + system_instruction + '<|eot_id|>'\n",
            "        '<|start_header_id|>user<|end_header_id|>\\n\\n'\n",
            "        '### INGREDIENTS:\\n' + ingr_text + '<|eot_id|>'\n",
            "        '<|start_header_id|>assistant<|end_header_id|>\\n\\n'\n",
            "        '### TITLE:\\n'\n",
            "    )\n",
            "    return prompt\n",
            "\n",
            "# Quick sanity test\n",
            "test_prompt = build_indian_prompt(INDIAN_TEMPLATES[0])\n",
            "print('Sample prompt (first 300 chars):')\n",
            "print(test_prompt[:300])\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 7: Generation loop ─────────────────────────────────────────────────────\n",
            "import csv, time\n",
            "\n",
            "OUTPUT_CSV = 'indian_vegan_recipes_db.csv'\n",
            "\n",
            "with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:\n",
            "    fieldnames = ['dish_name','style','budget_inr','original_non_veg','vegan_substitute',\n",
            "                  'ingredients','generated_recipe','model_version','generation_time_s']\n",
            "    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)\n",
            "    writer.writeheader()\n",
            "\n",
            "    for i, tmpl in enumerate(INDIAN_TEMPLATES):\n",
            "        print('[' + str(i+1) + '/' + str(len(INDIAN_TEMPLATES)) + '] Generating: ' + tmpl['name'] + '...')\n",
            "\n",
            "        prompt = build_indian_prompt(tmpl)\n",
            "        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')\n",
            "\n",
            "        start = time.time()\n",
            "        with torch.no_grad():\n",
            "            out = model.generate(\n",
            "                **inputs,\n",
            "                max_new_tokens=600,\n",
            "                temperature=0.7,\n",
            "                top_p=0.9,\n",
            "                repetition_penalty=1.05,\n",
            "                do_sample=True,\n",
            "                eos_token_id=tokenizer.eos_token_id,\n",
            "                pad_token_id=tokenizer.eos_token_id,\n",
            "            )\n",
            "        elapsed = round(time.time() - start, 1)\n",
            "\n",
            "        # Decode only the newly generated tokens (strip the prompt)\n",
            "        input_len = inputs['input_ids'].shape[1]\n",
            "        raw_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=False)\n",
            "\n",
            "        # Apply the exact hallucination-cutting logic from api.py\n",
            "        clean_text = clean_recipe_output(raw_text)\n",
            "        model_title = clean_text.split('\\n')[0].strip()\n",
            "        final_title = tmpl.get('name') if tmpl.get('name') else model_title\n",
            "\n",
            "        writer.writerow({\n",
            "            'dish_name':         final_title,\n",
            "            'style':             tmpl['style'],\n",
            "            'budget_inr':        tmpl['budget_inr'],\n",
            "            'original_non_veg':  tmpl.get('original') or 'vegan',\n",
            "            'vegan_substitute':  tmpl['sub'],\n",
            "            'ingredients':       ', '.join(tmpl['ingredients']),\n",
            "            'generated_recipe':  clean_text,\n",
            "            'model_version':     'nd1490/ratatouille-llama3-3b-v10-non-oven',\n",
            "            'generation_time_s': elapsed,\n",
            "        })\n",
            "        csvfile.flush()\n",
            "        print('   Done in ' + str(elapsed) + 's | output len: ' + str(len(clean_text)) + ' chars')\n",
            "        print('   Preview: ' + clean_text[:120].replace('\\n', ' ') + '...')\n",
            "        print()\n",
            "\n",
            "print('All ' + str(len(INDIAN_TEMPLATES)) + ' recipes generated and saved to ' + OUTPUT_CSV)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 8: Download the CSV ─────────────────────────────────────────────────────\n",
            "from google.colab import files\n",
            "files.download(OUTPUT_CSV)\n",
            "print('Download triggered for ' + OUTPUT_CSV)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 9 (Optional): Quick quality sanity check ────────────────────────────────\n",
            "import pandas as pd\n",
            "df = pd.read_csv(OUTPUT_CSV)\n",
            "print('Total recipes:', len(df))\n",
            "print('Avg generation time:', round(df['generation_time_s'].mean(), 1), 'sec')\n",
            "print('Avg recipe length (chars):', round(df['generated_recipe'].str.len().mean(), 0))\n",
            "\n",
            "# Check % of recipes that contain Indian spice markers\n",
            "indian_spices = ['cumin', 'turmeric', 'garam masala', 'coriander', 'mustard']\n",
            "def has_indian_spice(text):\n",
            "    t = text.lower()\n",
            "    return any(sp in t for sp in indian_spices)\n",
            "indian_ratio = df['generated_recipe'].apply(has_indian_spice).mean()\n",
            "print('% with Indian spice profile: ' + str(round(indian_ratio*100, 1)) + '%')\n",
            "\n",
            "# Show first recipe as sample\n",
            "print('\\n' + '='*60)\n",
            "print('SAMPLE RECIPE (first row):')\n",
            "print(df.iloc[0]['dish_name'])\n",
            "print(df.iloc[0]['generated_recipe'][:800])\n",
        ],
    },
    # ── MongoDB Upload Cell ──────────────────────────────────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 10: Upload generated recipes to MongoDB Atlas (ratatouille.indian_recipes) ─\n",
            "# Make sure you add MONGO_URI to your Colab Secrets (the Key icon on the left).\n",
            "!pip install 'pymongo[srv]' -q\n",
            "import csv\n",
            "from google.colab import userdata\n",
            "from pymongo import MongoClient\n",
            "from datetime import datetime, timezone\n",
            "MONGO_URI = userdata.get('MONGO_URI')\n",
            "client = MongoClient(MONGO_URI)\n",
            "col = client['ratatouille']['indian_recipes']\n",
            "docs = []\n",
            "with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:\n",
            "    reader = csv.DictReader(f)\n",
            "    for row in reader:\n",
            "        docs.append({\n",
            "            'dish_name':        row['dish_name'],\n",
            "            'style':            row['style'],\n",
            "            'budget_inr':       int(row['budget_inr']),\n",
            "            'original_non_veg': row['original_non_veg'],\n",
            "            'vegan_substitute':  row['vegan_substitute'],\n",
            "            'ingredients_list':  [i.strip() for i in row['ingredients'].split(',')],\n",
            "            'generated_recipe':  row['generated_recipe'],\n",
            "            'model_version':     row['model_version'],\n",
            "            'generation_time_s': float(row['generation_time_s']),\n",
            "            'created_at':        datetime.now(timezone.utc)\n",
            "        })\n",
            "if docs:\n",
            "    result = col.insert_many(docs)\n",
            "    print('MongoDB upload done. Inserted', len(result.inserted_ids), 'recipes.')\n",
            "    print('Collection: ratatouille.indian_recipes |', col.count_documents({}), 'total docs')\n",
            "else:\n",
            "    print('No recipes to upload.')\n",
            "client.close()\n",
        ],
    },
]

nb2 = {
    "cells": nb2_cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2, ensure_ascii=False)
print("Written GPU_Indian_Budget_Vegan_Recipe_Generator.ipynb")

print("\nDone! Both notebooks written.")
