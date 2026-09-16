import json

path = r'c:\Users\dhanu\OneDrive\Desktop\Capstone Proj CB\Rat-Model2V\RAT V3\V8\repo\RAT_V10_GENERATE_100_RECIPES.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    # Add DECONSTRUCT_URL to Cell 2
    if cell['cell_type'] == 'code' and 'PANTRY_URL' in ''.join(cell['source']):
        source = cell['source']
        for i, line in enumerate(source):
            if 'PANTRY_URL' in line:
                source.insert(i + 1, 'DECONSTRUCT_URL= f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/deconstruction_map.json"\n')
                break

    # Fetch it in Cell 3
    if cell['cell_type'] == 'code' and 'pantry_prices' in ''.join(cell['source']):
        source = cell['source']
        # Find end of cell
        source.append('\n')
        source.append('print("[*] Fetching offline deconstruction map...")\n')
        source.append('deconstruct_resp = requests.get(DECONSTRUCT_URL, headers=_hdrs)\n')
        source.append('offline_deconstruct_map = deconstruct_resp.json() if deconstruct_resp.status_code == 200 else {}\n')
        source.append('print(f"[OK] Deconstruction map: {len(offline_deconstruct_map)} entries")\n')
        break

# Now update the get_dynamic_price logic
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def get_dynamic_price' in ''.join(cell['source']):
        source = cell['source']
        
        # We want to replace the static dictionary and the keyword logic, with checking the offline map.
        # But we'll keep the keyword logic as the absolute final fallback.
        # First, remove STATIC_DECONSTRUCTION
        start_static = -1
        end_static = -1
        for i, line in enumerate(source):
            if 'STATIC_DECONSTRUCTION = {' in line:
                start_static = i
            if start_static != -1 and '}' in line:
                end_static = i
                break
        if start_static != -1:
            source[start_static:end_static+1] = []
            
        # Replace the check
        for i, line in enumerate(source):
            if 'if clean_ingredient in STATIC_DECONSTRUCTION:' in line:
                source[i] = '    if clean_ingredient in offline_deconstruct_map:\n'
                source[i+1] = '        print(f"      [price] \'{clean_ingredient}\' found in offline deconstruction map.")\n'
                source[i+2] = '        deconstructed = offline_deconstruct_map[clean_ingredient]\n'
                # Remove the else block that was there if it existed
                # Wait, the previous patch had an else block. I will just reconstruct the entire block carefully.
                break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
