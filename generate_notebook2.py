import json

with open('GPU_Open_World_Vegan_DB_Builder.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

# Keep 0, 1, 2, 3 (Intro, Install, Mongo, Vocab)
new_cells.extend(nb['cells'][0:4])

# API cell
api_source = [
    '# Cell 4 — Generate Profiles using HF Serverless API (Llama-3.3-70B)\n',
    'import os, json, requests, time, re\n',
    'from google.colab import userdata\n',
    'try:\n',
    '    HF_TOKEN = userdata.get("HF_TOKEN")\n',
    'except:\n',
    '    HF_TOKEN = input("Paste your HF_TOKEN: ")\n',
    'API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions"\n',
    'HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}\n',
    '\n',
    'PROMPT_TEMPLATE = """You are a food chemistry and culinary database. Output only valid JSON. No explanations.\n',
    'Generate a chemical and physical profile for the ingredient \'{ingredient}\' matching the specified JSON format.\n',
    'Determine if it is vegan (true/false).\n',
    'Specify macros (fat, protein, carb, water as ratios summing to 1.0).\n',
    'Rate its texture on a 1-5 scale: [hardness, chewiness, fibrousness, moisture, elasticity, granularity].\n',
    'List 3-5 primary flavor volatile compounds.\n',
    'Assign a culinary role: [bulk_protein, fat_source, binder, creamy_liquid, sweetener, seasoning, veggie, starch].\n',
    '\n',
    'Return ONLY valid JSON matching this exact structure:\n',
    '{{\n',
    '  "is_vegan": false,\n',
    '  "macros": {{"fat": 0.20, "protein": 0.22, "carb": 0.0, "water": 0.58}},\n',
    '  "texture": [4, 4, 4, 2, 2, 2],\n',
    '  "flavor_molecules": ["methanethiol", "dimethyl_sulfide", "pyrazines"],\n',
    '  "role": "bulk_protein"\n',
    '}}"""\n',
    'profiles = {}\n',
    'print(f"[*] Generating accurate 70B profiles for {len(full_vocab)} items...")\n',
    'start = time.time()\n',
    'for i, ing in enumerate(full_vocab):\n',
    '    payload = {"model": "meta-llama/Llama-3.3-70B-Instruct", "messages": [{"role": "system", "content": "You are a food chemistry expert. Output JSON only."}, {"role": "user", "content": PROMPT_TEMPLATE.format(ingredient=ing)}], "max_tokens": 250, "temperature": 0.1, "response_format": {"type": "json_object"}}\n',
    '    for attempt in range(3):\n',
    '        try:\n',
    '            resp = requests.post(API_URL, headers=HEADERS, json=payload)\n',
    '            if resp.status_code == 200:\n',
    '                text = resp.json()["choices"][0]["message"]["content"]\n',
    '                match = re.search(r"\\{.*\\}", text, re.DOTALL)\n',
    '                if match:\n',
    '                    profile = json.loads(match.group(0))\n',
    '                    profile["_id"] = ing\n',
    '                    profiles[ing] = profile\n',
    '                    break\n',
    '        except: pass\n',
    '        time.sleep(1)\n',
    '    if (i+1) % 10 == 0:\n',
    '        print(f"  ... {i+1}/{len(full_vocab)} completed")\n',
    'feats = profiles\n',
    'print(f"✅ High-Quality Database Generation Complete! ({len(profiles)} extracted)")\n'
]
new_cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'source': api_source, 'execution_count': None})

# Append cells 7, 8, 9 from orig_nb
new_cells.extend(nb['cells'][7:10])

# Strip pickle
for cell in new_cells[-3:]:
    cell['source'] = [line for line in cell['source'] if 'pickle.load' not in line and 'with open' not in line and 'pickle' not in line]

nb['cells'] = new_cells
with open('c:/Users/dhanu/OneDrive/Desktop/Capstone Proj CB/Rat-Model2V/RAT V3/V8/repo/Serverless_Vegan_DB_Builder.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
