import json

with open('vocab_list.txt', 'r', encoding='utf-8') as f:
    vocab = [line.strip() for line in f if line.strip()]

cells = []

# Cell 1: Vocab
vocab_source = ['# Curated Vocabulary List\n', f'full_vocab = {json.dumps(vocab, indent=2)}\n', 'print(f"Loaded {len(full_vocab)} ingredients.")']
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'source': vocab_source, 'execution_count': None})

# Cell 2: Serverless Fetch
serverless_source = [
    '# Generate Profiles using HF Serverless API (Llama-3.3-70B)\n',
    'import os, json, requests, time\n',
    'from google.colab import userdata\n',
    '\n',
    '# Try to get HF_TOKEN from colab secrets, else prompt user\n',
    'try:\n',
    '    HF_TOKEN = userdata.get("HF_TOKEN")\n',
    'except:\n',
    '    HF_TOKEN = input("Paste your HF_TOKEN: ")\n',
    '\n',
    'API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions"\n',
    'HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}\n',
    '\n',
    'PROMPT_TEMPLATE = """You are a food chemistry and culinary database. Output only valid JSON. No explanations.\n',
    'Generate a chemical and physical profile for the ingredient \'{ingredient}\' matching the specified JSON format.\n',
    'Determine if it is vegan (true/false).\n',
    'Specify macros (fat, protein, carb, water as ratios summing to 1.0).\n',
    'Rate its texture on a 1-5 scale: [hardness, chewiness, fibrousness(0-5), moisture, elasticity, granularity].\n',
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
    '\n',
    'profiles = {}\n',
    'print(f"[*] Generating accurate 70B profiles for {len(full_vocab)} items...")\n',
    'start = time.time()\n',
    'import re\n',
    'for i, ing in enumerate(full_vocab):\n',
    '    payload = {\n',
    '        "model": "meta-llama/Llama-3.3-70B-Instruct",\n',
    '        "messages": [\n',
    '            {"role": "system", "content": "You are a food chemistry expert. Output JSON only."},\n',
    '            {"role": "user", "content": PROMPT_TEMPLATE.format(ingredient=ing)}\n',
    '        ],\n',
    '        "max_tokens": 250,\n',
    '        "temperature": 0.1,\n',
    '        "response_format": {"type": "json_object"}\n',
    '    }\n',
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
    '        except Exception:\n',
    '            pass\n',
    '        time.sleep(1)\n',
    '    if (i+1) % 10 == 0:\n',
    '        print(f"  ... {i+1}/{len(full_vocab)} completed ({(time.time()-start):.1f}s)")\n',
    '\n',
    'print(f"\\n✅ High-Quality Database Generation Complete! ({len(profiles)} extracted)")\n',
    'feats = profiles\n'
]
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'source': serverless_source, 'execution_count': None})

# Cell 3: Math (Read from previous notebook)
with open('GPU_Open_World_Vegan_DB_Builder.ipynb', 'r', encoding='utf-8') as f:
    orig_nb = json.load(f)

for cell in orig_nb['cells']:
    src = ''.join(cell['source'])
    if 'def jaccard' in src:
        cells.append(cell)
    elif 'pymongo' in src and 'MONGO_URI' in src:
        new_src = [s.replace('import pymongo', 'import pymongo\\n!pip install pymongo python-dotenv -q\\n') for s in cell['source']]
        cell['source'] = new_src
        cells.append(cell)

nb_out = {
    'cells': cells,
    'metadata': {'colab': {'name': 'Serverless_Vegan_DB_Builder.ipynb'}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
    'nbformat': 4,
    'nbformat_minor': 0
}
with open('Serverless_Vegan_DB_Builder.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_out, f, indent=2)
print('Notebook generated!')
