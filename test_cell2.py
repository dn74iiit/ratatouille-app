# Generate Profiles using HF Serverless API (Llama-3.3-70B)
import os, json, requests, time
from google.colab import userdata

# Try to get HF_TOKEN from colab secrets, else prompt user
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except:
    HF_TOKEN = input("Paste your HF_TOKEN: ")

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

PROMPT_TEMPLATE = """You are a food chemistry and culinary database. Output only valid JSON. No explanations.
Generate a chemical and physical profile for the ingredient '{ingredient}' matching the specified JSON format.
Determine if it is vegan (true/false).
Specify macros (fat, protein, carb, water as ratios summing to 1.0).
Rate its texture on a 1-5 scale: [hardness, chewiness, fibrousness(0-5), moisture, elasticity, granularity].
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
print(f"[*] Generating accurate 70B profiles for {len(full_vocab)} items...")
start = time.time()
import re
for i, ing in enumerate(full_vocab):
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a food chemistry expert. Output JSON only."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(ingredient=ing)}
        ],
        "max_tokens": 250,
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload)
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    profile = json.loads(match.group(0))
                    profile["_id"] = ing
                    profiles[ing] = profile
                    break
        except Exception:
            pass
        time.sleep(1)
    if (i+1) % 10 == 0:
        print(f"  ... {i+1}/{len(full_vocab)} completed ({(time.time()-start):.1f}s)")

print(f"\n✅ High-Quality Database Generation Complete! ({len(profiles)} extracted)")
feats = profiles
