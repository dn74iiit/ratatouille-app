# ==========================================
# CELL 1: ENVIRONMENT SETUP & INSTALLATIONS
# ==========================================
import os
import sys

print("⚙️ Installing core libraries...")
!pip install transformers accelerate bitsandbytes peft scipy pandas numpy -q

print("📚 Downloading NLP dictionaries...")
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import torch
print("\n" + "="*40)
if torch.cuda.is_available():
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print("✅ Environment is ready for Ratatouille Inference.")
else:
    print("❌ NO GPU FOUND! Go to Runtime -> Change runtime type -> Select T4 GPU")
print("="*40)
# ==========================================
# CELL 2: DATA INGESTION
# ==========================================
import pandas as pd
import requests
import io
import numpy as np
import re
import json
from nltk.stem import WordNetLemmatizer
from scipy.optimize import linprog

lemmatizer = WordNetLemmatizer()

# URLs
mandi_url = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/daily_mandi_data.csv"
bounds_url = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/v8_lemmaized_ingredient_bounds.json"
pantry_url = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/pantry_prices.json"

# Replace with your actual PAT if making private repo requests
from google.colab import userdata
GITHUB_PAT = userdata.get('GITHUB_PAT') # Securely loads from Colab Secrets
headers = {"Authorization": f"token {GITHUB_PAT}", "Accept": "application/vnd.github.v3.raw"}

print("Downloading datasets securely...")

# 1. Fetch Mandi Data
mandi_response = requests.get(mandi_url, headers=headers)
MANDI_COLUMNS = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price']
df_mandi = pd.read_csv(io.StringIO(mandi_response.text), header=None, names=MANDI_COLUMNS)

latest_date = df_mandi['Arrival_Date'].max()
current_mandi = df_mandi[df_mandi['Arrival_Date'] == latest_date].copy()
current_mandi['Processed_Ingredient'] = current_mandi['Commodity'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word.lower()) for word in str(x).split()])
)
current_mandi['Modal_Price'] = pd.to_numeric(current_mandi['Modal_Price'], errors='coerce')
current_mandi = current_mandi.dropna(subset=['Modal_Price'])
current_mandi['Price_per_Gram'] = current_mandi['Modal_Price'] / 100000

# 2. Fetch Bounds
bounds_response = requests.get(bounds_url, headers=headers)
bounds_dict = bounds_response.json() if bounds_response.status_code == 200 else {}

# 3. Fetch Pantry Prices (Assuming you will upload this later)
pantry_response = requests.get(pantry_url, headers=headers)
if pantry_response.status_code == 200:
    pantry_prices = pantry_response.json()
else:
    print("⚠️ Pantry JSON not found on GitHub. Using offline default fallback dictionary.")
    pantry_prices = {
        "vanilla extract": 4.50, "dark chocolate": 1.20, "soy sauce": 0.40,
        "olive oil": 0.80, "macaroni": 0.30, "vegan cashew mozzarella": 2.50,
        "sugar": 0.05, "salt": 0.02
    }

print("✅ Data Loaded and Prepared")
# ==========================================
# CELL 3: V8 NATIVE INFERENCE (BASE + PEFT)
# ==========================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
ADAPTER_REPO = "nd1490/ratatouille-llama3-3b-v8-50k"

print("📥 Phase 1: Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"

print("📥 Phase 2: Loading Base Model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
)

print("📥 Phase 3: Attaching V8 Adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
print("✅ AI Engine Online.")
# ==========================================
# CELL 4: THE MASTER LOGIC ENGINE
# ==========================================

def extract_json_from_llm(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return None

def deconstruct_ingredient(ingredient):
    prompt = f"""Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops.
    Assign approximate weight percentages. Return ONLY valid JSON.
    Example for 'tomato ketchup': {{"tomato": 0.8, "sugar": 0.1, "onion": 0.1}}

    ### INGREDIENT:
    {ingredient}
    ### JSON:
    """
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50, use_cache=True, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return extract_json_from_llm(text.split("### JSON:\n")[-1])

def get_recipe_archetype(ingredients_list):
    ingr_str = ", ".join(ingredients_list)
    prompt = f"""Classify the dish structure based on these ingredients: [{ingr_str}].
    Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish].
    Return ONLY the word.

    ### INGREDIENTS:
    {ingr_str}
    ### ARCHETYPE:
    """
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=5, use_cache=True, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    result = text.split("### ARCHETYPE:\n")[-1].strip()
    valid_archetypes = ["Curry", "Dry_Sabzi", "Salad", "Dessert", "Bread", "Soup", "Rice_Dish"]
    return result if result in valid_archetypes else "Curry"

def parse_ingredient_input(raw_string):
    match = re.match(r"^([\d\.]+)\s*(g|kg|ml)?\s+(.*)$", raw_string.strip().lower())
    if match:
        qty = float(match.group(1))
        unit = match.group(2)
        name = match.group(3)
        if unit == 'kg': qty *= 1000
        return qty, name
    return None, raw_string.lower().strip()

def get_dynamic_price(clean_ingredient, user_state):
    lemmatized_ing = " ".join([lemmatizer.lemmatize(w) for w in clean_ingredient.split()])

    state_data = current_mandi[(current_mandi['Processed_Ingredient'] == lemmatized_ing) & (current_mandi['State'].str.lower() == user_state.lower())]
    if not state_data.empty: return state_data['Price_per_Gram'].median()

    nat_data = current_mandi[current_mandi['Processed_Ingredient'] == lemmatized_ing]
    if not nat_data.empty: return nat_data['Price_per_Gram'].median()

    if clean_ingredient in pantry_prices: return pantry_prices[clean_ingredient]

    deconstructed = deconstruct_ingredient(clean_ingredient)
    if deconstructed:
        avg_price = 0
        total_weight = sum(deconstructed.values())
        for sub_ing, weight in deconstructed.items():
            sub_lem = lemmatizer.lemmatize(sub_ing.lower())
            sub_price = current_mandi[current_mandi['Processed_Ingredient'] == sub_lem]['Price_per_Gram'].median()
            if pd.isna(sub_price): sub_price = pantry_prices.get(sub_ing, 0.5)
            avg_price += sub_price * (weight / total_weight)
        return avg_price * 1.3

    return 0.5

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

def optimize_recipe_v2(raw_user_ingredients, total_budget, servings, user_state):
    archetype = get_recipe_archetype(raw_user_ingredients)
    n = len(raw_user_ingredients)
    c = [-1] * n

    prices = []
    bounds = []
    tags = []

    for raw_ing in raw_user_ingredients:
        user_qty, clean_name = parse_ingredient_input(raw_ing)
        prices.append(get_dynamic_price(clean_name, user_state))
        tags.append(tag_ingredient(clean_name))

        if user_qty is not None:
            bounds.append((user_qty / servings, user_qty / servings))
        else:
            lower, upper = bounds_dict.get(lemmatizer.lemmatize(clean_name.split()[-1]), (15, 150))
            if clean_name not in ['garlic', 'ginger', 'chili', 'salt', 'pepper']: lower = max(lower, 30.0)
            else: lower = max(lower, 2.0)
            upper = min(upper, 250.0)
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

    if result.success:
        estimated_grams_per_serving = np.round(result.x, 1)
        final_quantities = estimated_grams_per_serving * servings
        return [f"{qty}g {ing}" for qty, ing in zip(final_quantities, [parse_ingredient_input(r)[1] for r in raw_user_ingredients])], archetype
    return None, archetype

def generate_final_recipe(calculated_ingredients, archetype):
    ingr_text = "\n".join(f"- {i}" for i in calculated_ingredients)
    system_instruction = f"""You are a master chef. Write a highly detailed, appetizing recipe for a {archetype}.
    First, provide a creative title. Then, provide step-by-step cooking directions using proper culinary techniques.
    Do not change the ingredient quantities provided."""

    prompt = f"{system_instruction}\n\n### INGREDIENTS:\n{ingr_text}\n### TITLE:\n"

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=350, use_cache=True, do_sample=True,
        temperature=0.7, top_p=0.9, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id
    )

    full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    if "### TITLE:\n" in full_text:
        return full_text.split("### TITLE:\n")[1].strip()
    return full_text.strip()
# ==========================================
# CELL 5: INTERACTIVE RUNNER
# ==========================================

def run_recipe_engine():
    print("========================================")
    print("👨‍🍳 Ratatouille: Culinary AI Engine")
    print("========================================\n")

    raw_ing_input = input("Ingredients (comma separated, e.g., '200g paneer, tomatoes, ginger garlic paste'): ")
    user_ingredients = [i.strip() for i in raw_ing_input.split(',')]

    try:
        user_budget = float(input("Total budget in INR (e.g., 150): ₹"))
        user_servings = int(input("Number of servings (e.g., 2): "))
    except ValueError:
        print("❌ Invalid input for budget or servings.")
        return

    user_state = input("State for market pricing (e.g., Delhi): ")

    print("\n🔄 Initializing Pre-Processor Oracles...")
    print("⚖️ Calculating Optimal Quantities and Cost Constraints...")

    calculated_ingredients, archetype = optimize_recipe_v2(user_ingredients, user_budget, user_servings, user_state)

    print(f"🧠 AI Classified Archetype: {archetype}")

    if not calculated_ingredients:
        print("\n❌ Error: The provided budget is mathematically impossible for these ingredients at current market/pantry prices.")
        return

    print("\n✅ Budget Approved! Quantities Locked:")
    for ing in calculated_ingredients:
        print(f"  - {ing}")

    print("\n🔥 Firing up the ovens (Generating Instructions)...")
    final_recipe = generate_final_recipe(calculated_ingredients, archetype)

    print("\n" + "="*50)
    print(final_recipe)
    print("="*50)

# Run it!
run_recipe_engine()