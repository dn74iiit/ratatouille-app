# ============================================================
# CELL 0 — INSTALL DEPENDENCIES
# ============================================================
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install",
    "unsloth", "--quiet"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install",
    "git+https://github.com/unslothai/unsloth.git",
    "--quiet"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install",
    "scipy", "nltk", "pandas", "numpy",
    "requests",
    "--quiet"], check=True)

import torch
print("=" * 50)
print(f"GPU available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 50)
# ============================================================
# CELL 1 — CONFIGURATION & AUTH
# ============================================================
from google.colab import userdata
from huggingface_hub import login

HF_TOKEN    = userdata.get("HF_TOKEN")
GITHUB_PAT  = userdata.get("GITHUB_PAT")

V10_REPO    = "nd1490/ratatouille-llama3-3b-v10-non-oven"
MAX_SEQ_LEN = 512


MANDI_URL  = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/daily_mandi_data.csv"
BOUNDS_URL = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/v8_lemmaized_ingredient_bounds.json"
PANTRY_URL = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/pantry_prices.json"
DECONSTRUCT_URL = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/deconstruction_map.json"

NUM_RECIPES         = 100
MAX_NEW_TOKENS      = 400
TEMPERATURE         = 0.7
TOP_P               = 0.9
REPETITION_PENALTY  = 1.15
MIN_INGREDIENTS     = 2

login(token=HF_TOKEN)
print(f"[OK] Authenticated to HF")
print(f"[OK] V10 model : {V10_REPO}")

# ============================================================
# CELL 2 — LOAD MARKET DATA & NLP TOOLS
# ============================================================
import io, re, json, requests
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from scipy.optimize import linprog

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

print("[*] Loading Mandi data...")
_hdrs = {"Authorization": f"token {GITHUB_PAT}"}

mandi_resp = requests.get(MANDI_URL, headers=_hdrs)
df_mandi = pd.read_csv(
    io.StringIO(mandi_resp.text),
    header=None,
    names=['State','District','Market','Commodity','Variety','Grade',
           'Arrival_Date','Min_Price','Max_Price','Modal_Price']
)
latest_date = df_mandi['Arrival_Date'].max()
current_mandi = df_mandi[df_mandi['Arrival_Date'] == latest_date].copy()
current_mandi['Processed_Ingredient'] = current_mandi['Commodity'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(w.lower()) for w in str(x).split()])
)
current_mandi['Modal_Price'] = pd.to_numeric(current_mandi['Modal_Price'], errors='coerce')
current_mandi = current_mandi.dropna(subset=['Modal_Price'])
current_mandi['Price_per_Gram'] = current_mandi['Modal_Price'] / 100000

bounds_resp = requests.get(BOUNDS_URL, headers=_hdrs)
bounds_dict = bounds_resp.json() if bounds_resp.status_code == 200 else {}

pantry_resp = requests.get(PANTRY_URL, headers=_hdrs)
DECONSTRUCT_URL = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/deconstruction_map.json"
pantry_prices = pantry_resp.json() if pantry_resp.status_code == 200 else {
    "vanilla extract": 4.50, "dark chocolate": 1.20, "soy sauce": 0.40,
    "olive oil": 0.80, "sugar": 0.05, "salt": 0.02
}

print(f"[OK] Mandi data: {len(current_mandi):,} rows (date: {latest_date})")
print(f"[OK] Bounds dict: {len(bounds_dict)} entries")
print(f"[OK] Pantry prices: {len(pantry_prices)} entries")
print("[*] Fetching offline deconstruction map...")
deconstruct_resp = requests.get(DECONSTRUCT_URL, headers=_hdrs)
offline_deconstruct_map = deconstruct_resp.json() if deconstruct_resp.status_code == 200 else {}
print(f"[OK] Deconstruction map: {len(offline_deconstruct_map)} entries")

def get_dynamic_price(clean_ingredient, user_state):
    """
    Priority: state Mandi -> national Mandi -> pantry -> offline_deconstruction_map -> keyword heuristic -> 0.2
    """
    lem = " ".join([lemmatizer.lemmatize(w) for w in clean_ingredient.split()])
    
    state_data = current_mandi[
        (current_mandi['State'].str.lower() == user_state.lower()) &
        (current_mandi['Processed_Ingredient'] == lem)
    ]
    if not state_data.empty:
        return state_data['Price_per_Gram'].median()

    nat_data = current_mandi[current_mandi['Processed_Ingredient'] == lem]
    if not nat_data.empty:
        return nat_data['Price_per_Gram'].median()

    if clean_ingredient in pantry_prices:
        return pantry_prices[clean_ingredient]

    if clean_ingredient in offline_deconstruct_map:
        deconstructed = offline_deconstruct_map[clean_ingredient]
        
        # Safely handle if LLM outputted a float or string instead of a dict
        if isinstance(deconstructed, dict) and deconstructed:
            print(f"      [price] '{clean_ingredient}' found in offline map.")
            if not all(isinstance(v, (int, float)) for v in deconstructed.values()):
                pass # Fall through to heuristic
            else:
                avg_price = 0
                total_weight = sum(deconstructed.values())
                if total_weight > 0:
                    for sub_ing, weight in deconstructed.items():
                        sub_lem = " ".join([lemmatizer.lemmatize(w) for w in sub_ing.split()])
                        sub_price = current_mandi[current_mandi['Processed_Ingredient'] == sub_lem]['Price_per_Gram'].median()
                        if pd.isna(sub_price):
                            sub_price = pantry_prices.get(sub_ing, 0.5)
                        avg_price += sub_price * (weight / total_weight)
                        
                    result_price = avg_price * 1.3
                    print(f"      [price] {deconstructed} -> {result_price:.4f} Rs/g")
                    return result_price

    # --- HEURISTIC KEYWORD FALLBACK ---
    print(f"      [price] '{clean_ingredient}' falling back to keyword heuristic...")
    name = clean_ingredient.lower()
    if 'oil' in name or 'fat' in name or 'ghee' in name or 'butter' in name:
        fallback_price = 0.15
    elif 'meat' in name or 'chicken' in name or 'fish' in name or 'beef' in name or 'pork' in name or 'mutton' in name:
        fallback_price = 0.30
    elif 'cheese' in name or 'paneer' in name:
        fallback_price = 0.50
    elif 'spice' in name or 'powder' in name or 'extract' in name or 'seed' in name or 'paste' in name:
        fallback_price = 0.80
    elif 'flour' in name or 'wheat' in name or 'rice' in name or 'grain' in name or 'dal' in name or 'lentil' in name:
        fallback_price = 0.05
    else:
        fallback_price = 0.20

    print(f"      [price] Keyword heuristic assigned: {fallback_price:.4f} Rs/g")
    return fallback_price

# ============================================================
# OPTIMIZATION AND BUDGET RECOVERY LOGIC
# ============================================================
import numpy as np
from scipy.optimize import linprog
import re

def parse_ingredient_input(raw_str):
    # Strip leading quantities
    clean_name = re.sub(r'^-?\s*[\d\./]+\s*(g|kg|ml|cup|cups|tsp|tbsp|oz)?\s+', '', raw_str.lower()).strip()
    return None, clean_name

def tag_ingredient(ing_name):
    proteins = ['paneer', 'chicken', 'soya', 'tofu', 'dal', 'lentil', 'egg', 'meat', 'fish']
    bases = ['onion', 'tomato', 'garlic', 'ginger', 'puree']
    sweets = ['sugar', 'jaggery', 'chocolate', 'vanilla', 'syrup']
    carbs = ['rice', 'flour', 'wheat', 'bread', 'noodle', 'pasta', 'potato']
    if any(p in ing_name for p in proteins): return 'protein'
    if any(b in ing_name for b in bases): return 'base'
    if any(s in ing_name for s in sweets): return 'sweet'
    if any(c in ing_name for c in carbs): return 'neutral'
    return 'veggie'

def get_recipe_archetype(ingredients: list) -> str:
    s = " ".join(ingredients).lower()
    if any(w in s for w in ["rice", "biryani", "pulao", "fried rice"]): return "Rice_Dish"
    if any(w in s for w in ["pasta", "noodle", "macaroni", "spaghetti", "basil"]): return "Rice_Dish"
    if any(w in s for w in ["oat", "banana", "sugar", "flour", "chocolate", "cake", "honey", "cream", "milk", "vanilla"]): return "Dessert"
    if any(w in s for w in ["mushroom", "soup", "broth", "carrot", "ginger", "coconut milk"]): return "Soup"
    if any(w in s for w in ["lettuce", "salad", "cucumber", "olive"]): return "Salad"
    if any(w in s for w in ["bread", "dough", "yeast"]): return "Bread"
    return "Curry"

def optimize_recipe_v2(raw_user_ingredients, total_budget, servings, user_state, archetype=None, skip_llm=True):
    if archetype is None:
        archetype = get_recipe_archetype(raw_user_ingredients)
    n = len(raw_user_ingredients)
    c = [-1] * n
    prices, bounds, tags = [], [], []

    for raw_ing in raw_user_ingredients:
        user_qty, clean_name = parse_ingredient_input(raw_ing)
        prices.append(get_dynamic_price(clean_name, user_state))
        tags.append(tag_ingredient(clean_name))

        if user_qty is not None:
            bounds.append((user_qty / servings, user_qty / servings))
        else:
            lower, upper = bounds_dict.get(lemmatizer.lemmatize(clean_name.split()[-1]), (10, 400))
            if clean_name not in ['garlic', 'ginger', 'chili', 'salt', 'pepper']: lower = max(lower, 15.0)
            else: lower = max(lower, 2.0)
            upper = min(upper, 800.0) 
            if lower > upper: lower = upper
            bounds.append((lower, upper))

    A_ub = [prices]
    b_ub = [total_budget / servings]

    if archetype == "Curry":
        if 'protein' in tags and 'base' in tags:
            A_ub.append([0.8 if t == 'protein' else -1.0 if t == 'base' else 0.0 for t in tags])
            b_ub.append(0)
            A_ub.append([-3.0 if t == 'protein' else 1.0 if t == 'base' else 0.0 for t in tags])
            b_ub.append(0)
    elif archetype == "Dry_Sabzi":
        if 'protein' in tags and 'base' in tags:
            A_ub.append([-0.8 if t == 'protein' else 1.0 if t == 'base' else 0.0 for t in tags])
            b_ub.append(0)
    elif archetype == "Salad":
        if ('protein' in tags or 'neutral' in tags) and 'veggie' in tags:
            A_ub.append([2.0 if t in ['protein', 'neutral'] else -1.0 if t == 'veggie' else 0.0 for t in tags])
            b_ub.append(0)
    elif archetype == "Rice_Dish":
        if 'neutral' in tags and 'protein' in tags:
            A_ub.append([1.0 if t == 'protein' else -1.0 if t == 'neutral' else 0.0 for t in tags])
            b_ub.append(0)
    elif archetype == "Soup":
        if 'protein' in tags and 'base' in tags:
            A_ub.append([4.0 if t == 'protein' else -1.0 if t == 'base' else 0.0 for t in tags])
            b_ub.append(0)
    elif archetype == "Dessert":
        if 'sweet' in tags:
            A_ub.append([-0.4 if t in ['neutral', 'protein', 'base'] else 1.0 if t == 'sweet' else 0.0 for t in tags])
            b_ub.append(0)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        A_ub_fallback = [prices]
        b_ub_fallback = [total_budget / servings]
        result = linprog(c, A_ub=A_ub_fallback, b_ub=b_ub_fallback, bounds=bounds, method='highs')

    if not result.success:
        emergency_bounds = [(5.0, max(b[1], 5.0)) for b in bounds]
        result = linprog(c, A_ub=A_ub_fallback, b_ub=b_ub_fallback, bounds=emergency_bounds, method='highs')

    if not result.success:
        nuclear_bounds = [(1.0, 1000.0) for _ in bounds]
        result = linprog(c, A_ub=A_ub_fallback, b_ub=b_ub_fallback, bounds=nuclear_bounds, method='highs')

    if result.success:
        estimated_grams_per_serving = np.round(result.x, 1)
        final_quantities = estimated_grams_per_serving * servings
        return [f"{qty}g {parse_ingredient_input(r)[1]}" for qty, r in zip(final_quantities, raw_user_ingredients)], archetype
        
    return None, archetype

def optimize_with_budget_recovery(ingredients, budget, servings, state, min_ingredients):
    original_ingredients = list(ingredients)
    dropped = []
    current_ingredients = list(ingredients)
    
    while len(current_ingredients) >= min_ingredients:
        calc_ingredients, archetype = optimize_recipe_v2(current_ingredients, budget, servings, state, skip_llm=True)
        if calc_ingredients is not None:
            return calc_ingredients, archetype, current_ingredients, dropped
            
        prices = [(ing, get_dynamic_price(parse_ingredient_input(ing)[1], state)) for ing in current_ingredients]
        prices.sort(key=lambda x: x[1], reverse=True)
        dropped_ing = prices[0][0]
        
        current_ingredients.remove(dropped_ing)
        dropped.append(dropped_ing)
        
    return None, None, current_ingredients, dropped


# ============================================================
# CELL 4 — 100 RANDOM INPUT SCENARIOS
# ============================================================
import random

INGREDIENT_POOL = {
    "proteins": [
        "paneer", "chicken", "egg", "tofu", "lentil", "dal", "chickpea", "fish",
        "mutton", "prawn", "soya", "kidney bean", "black bean", "moong dal"
    ],
    "bases": [
        "onion", "tomato", "garlic", "ginger", "tomato puree", "green onion"
    ],
    "veggies": [
        "spinach", "peas", "capsicum", "cauliflower", "broccoli", "carrot",
        "mushroom", "zucchini", "eggplant", "cabbage", "bitter gourd",
        "bottle gourd", "drumstick", "fenugreek leaf", "corn", "pumpkin"
    ],
    "spices": [
        "cumin", "turmeric", "coriander", "chili", "garam masala", "bay leaf",
        "mustard seed", "cardamom", "cinnamon", "pepper", "fennel"
    ],
    "carbs": [
        "rice", "flour", "potato", "pasta", "noodle", "bread",
        "oat", "macaroni", "wheat"
    ],
    "fats_dairy": [
        "milk", "cream", "butter", "yogurt", "coconut milk", "ghee", "oil"
    ],
    "fruits_sweets": [
        "banana", "lemon", "orange", "mango", "apple", "sugar",
        "honey", "chocolate", "vanilla", "coconut"
    ]
}

STATES = [
    "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "West Bengal",
    "Punjab", "Rajasthan", "Gujarat", "Uttar Pradesh", "Kerala",
    "Madhya Pradesh", "Haryana", "Andhra Pradesh", "Telangana", "Bihar"
]

BUDGET_RANGE    = (50, 500)
SERVING_OPTIONS = [1, 2, 3, 4]

random.seed(42)

def make_random_inputs(n: int) -> list:
    scenarios = []
    all_categories = list(INGREDIENT_POOL.keys())

    for i in range(n):
        num_ingredients = random.randint(3, 6)
        ingredients = []

        if random.random() < 0.75:
            ingredients.append(random.choice(INGREDIENT_POOL["proteins"]))
        ingredients.append(random.choice(INGREDIENT_POOL["bases"]))

        remaining = num_ingredients - len(ingredients)
        for _ in range(remaining):
            cat = random.choice(all_categories)
            ing = random.choice(INGREDIENT_POOL[cat])
            if ing not in ingredients:
                ingredients.append(ing)

        scenarios.append({
            "recipe_id":       i + 1,
            "ingredients_raw": ingredients,
            "budget_inr":      random.randint(BUDGET_RANGE[0], BUDGET_RANGE[1]),
            "servings":        random.choice(SERVING_OPTIONS),
            "state":           random.choice(STATES),
        })

    return scenarios


SCENARIOS = make_random_inputs(NUM_RECIPES)
print(f"[OK] Generated {len(SCENARIOS)} scenarios")
print("\nSample:")
for s in SCENARIOS[:5]:
    print(f"  #{s['recipe_id']:3d} | Rs.{s['budget_inr']:4d} | "
          f"{s['servings']} serving(s) | {s['state']}")
    print(f"        {', '.join(s['ingredients_raw'])}")
# ============================================================
# CELL 5 — LOAD V10 MODEL (4-bit, Unsloth fast inference)
# ============================================================
import os, torch
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
from unsloth import FastLanguageModel

print(f"Loading {V10_REPO} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=V10_REPO,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    token=HF_TOKEN,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

FastLanguageModel.for_inference(model)

print("[OK] V10 model ready for inference.")
print(f"[OK] eos_token_id: {tokenizer.eos_token_id}")
print("[OK] deconstruct_ingredient_local() is now live.")
# ============================================================
# CELL 6 — GENERATE 100 RECIPES  (full pipeline, no server)
# ============================================================
import time



def apply_post_processing(text: str) -> str:
    # Clean up any lingering EOS tokens or extra markdown from the LLM output
    return text.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()

def build_prompt(calc_ingredients: list, archetype: str) -> str:
    ingr_text = "\n".join(f"- {i}" for i in calc_ingredients)
    return f"### INGREDIENTS:\n{ingr_text}\n### TITLE:\n"


def generate_recipe_local(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


results = []
t_start = time.time()

for idx, scenario in enumerate(SCENARIOS, 1):
    recipe_id   = scenario["recipe_id"]
    ingredients = scenario["ingredients_raw"]
    budget      = scenario["budget_inr"]
    servings    = scenario["servings"]
    state       = scenario["state"]

    print(f"[{idx:3d}/{NUM_RECIPES}] #{recipe_id} | {', '.join(ingredients)} | "
          f"Rs.{budget} | {servings} serving(s) | {state}")

    t_opt = time.time()
    calc_ingredients, archetype, used_ingredients, dropped_ingredients = \
        optimize_with_budget_recovery(ingredients, budget, servings, state, MIN_INGREDIENTS)
    opt_ms = round((time.time() - t_opt) * 1000)

    if calc_ingredients is None:
        print(f"         Truly infeasible after dropping {len(dropped_ingredients)} ingredient(s).")
        results.append({
            "recipe_id":             recipe_id,
            "input_ingredients":     ", ".join(ingredients),
            "budget_inr":            budget,
            "servings":              servings,
            "state":                 state,
            "archetype":             archetype,
            "used_ingredients":      ", ".join(ingredients),
            "dropped_ingredients":   ", ".join(dropped_ingredients),
            "optimized_ingredients": "",
            "status":                "infeasible",
            "recipe_title":          "",
            "recipe_full":           "",
            "gen_time_s":            0,
        })
        continue

    if dropped_ingredients:
        print(f"         Budget recovery: dropped {dropped_ingredients}")
    print(f"         Archetype: {archetype} | Opt: {opt_ms}ms")
    print(f"         Optimized: {', '.join(calc_ingredients)}")

    prompt = build_prompt(calc_ingredients, archetype)

    t_gen = time.time()
    raw_output = generate_recipe_local(prompt)
    gen_s = round(time.time() - t_gen, 1)

    recipe_text = apply_post_processing(raw_output)
    title_line  = re.sub(r"^#+\s*", "", recipe_text.split("\n")[0].strip()).strip()

    status = "success_partial" if dropped_ingredients else "success"
    print(f"         [{status}] {gen_s}s | {title_line[:65]}")
    preview = recipe_text.replace("\n", " | ")[:500]
    print(f"         Preview: {preview}...")

    results.append({
        "recipe_id":             recipe_id,
        "input_ingredients":     ", ".join(ingredients),
        "budget_inr":            budget,
        "servings":              servings,
        "state":                 state,
        "archetype":             archetype,
        "used_ingredients":      ", ".join(used_ingredients),
        "dropped_ingredients":   ", ".join(dropped_ingredients),
        "optimized_ingredients": " | ".join(calc_ingredients),
        "status":                status,
        "recipe_title":          title_line,
        "recipe_full":           recipe_text,
        "gen_time_s":            gen_s,
    })


elapsed_total  = round(time.time() - t_start, 1)
success_full   = sum(1 for r in results if r["status"] == "success")
success_part   = sum(1 for r in results if r["status"] == "success_partial")
fail_count     = sum(1 for r in results if r["status"] == "infeasible")

print("\n" + "=" * 60)
print(f"Full success    : {success_full}")
print(f"Partial (drops) : {success_part}")
print(f"Truly infeasible: {fail_count}")
print(f"Total time      : {elapsed_total}s ({elapsed_total/60:.1f} min)")
print("=" * 60)
# ============================================================
# CELL 7 — BUILD DATAFRAME & PREVIEW
# ============================================================
df = pd.DataFrame(results)

print(f"Shape: {df.shape}")
print(f"\nStatus:\n{df['status'].value_counts().to_string()}")
print(f"\nArchetype:\n{df['archetype'].value_counts().to_string()}")
print(f"\nTop states:\n{df['state'].value_counts().head(8).to_string()}")
print(f"\nBudget stats:\n{df['budget_inr'].describe().to_string()}")

partial_df = df[df['status'] == 'success_partial'][['recipe_id','input_ingredients','dropped_ingredients','budget_inr']]
if len(partial_df) > 0:
    print(f"\nBudget recovery cases ({len(partial_df)}):")
    for _, row in partial_df.iterrows():
        print(f"  #{row['recipe_id']:3d} | Rs.{row['budget_inr']} | dropped: {row['dropped_ingredients']}")

print("\n--- Sample successful recipe ---")
sample = df[df['status'].isin(['success','success_partial'])].iloc[0]
print(f"Title    : {sample['recipe_title']}")
print(f"Inputs   : {sample['input_ingredients']}")
if sample['dropped_ingredients']:
    print(f"Dropped  : {sample['dropped_ingredients']}  (budget recovery)")
print(f"Budget   : Rs.{sample['budget_inr']} | Servings: {sample['servings']} | {sample['state']}")
print(f"Optimized: {sample['optimized_ingredients']}")
print(f"\nRecipe:\n{sample['recipe_full'][:800]}...")
# ============================================================
# CELL 8 — DOWNLOAD CSV FROM COLAB
# ============================================================
from google.colab import files

LOCAL_CSV = "/content/v10_100_recipes.csv"
df.to_csv(LOCAL_CSV, index=False)
files.download(LOCAL_CSV)
print(f"[OK] Download triggered.")