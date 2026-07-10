import os
import re
import json
import time
import requests
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from nltk.stem import WordNetLemmatizer
from scipy.optimize import linprog
from dotenv import load_dotenv
from gradio_client import Client
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================
# APP INIT
# ============================================================
app = FastAPI(title="Ratatouille V10 API — Cloud Inference")

# Allow the React frontend (dev + production) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ENV / SECRETS
# ============================================================
load_dotenv()
HF_TOKEN    = os.getenv("HF_TOKEN")
GITHUB_PAT  = os.getenv("GITHUB_PAT")
MONGO_URI   = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not HF_TOKEN:
    raise RuntimeError("[!] HF_TOKEN not found in .env — cannot reach Hugging Face Inference API.")

# ============================================================
# DATABASE INIT
# ============================================================
if MONGO_URI:
    print("[*] Connecting to MongoDB Atlas...")
    db_client = AsyncIOMotorClient(MONGO_URI)
    db = db_client.ratatouille
    recipes_collection             = db.recipes
    vegan_alternatives_collection  = db.vegan_alternatives
    indian_recipes_collection      = db.indian_recipes
    print("[OK] Connected to MongoDB Atlas (async Motor client).")
    # Separate sync client for vegan lookup inside sync endpoints
    try:
        import pymongo as _pymongo
        _sync_mongo = _pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        _sync_mongo.server_info()  # validate connection
        vegan_alternatives_sync = _sync_mongo.ratatouille.vegan_alternatives
        generation_logs_sync = _sync_mongo.ratatouille.generation_logs
        llm_cache_sync = _sync_mongo.ratatouille.llm_cache
        print("[OK] Sync MongoDB client ready for vegan lookups, logs, and LLM cache.")
    except Exception as _e:
        vegan_alternatives_sync = None
        generation_logs_sync = None
        llm_cache_sync = None
        print(f"[WARN] Sync MongoDB client failed: {_e}")
else:
    recipes_collection             = None
    vegan_alternatives_collection  = None
    indian_recipes_collection      = None
    vegan_alternatives_sync        = None
    generation_logs_sync           = None
    llm_cache_sync                 = None
    print("[!] MONGO_URI not found. Database features will be disabled.")

# ============================================================
# HUGGING FACE SPACES — DUAL GRADIO CLIENTS (V8 + V10)
# ============================================================
HF_SPACE_V8 = os.getenv("HF_SPACE_URL",    "nd1490/ratatouille-inference")
HF_SPACE_V10 = os.getenv("HF_SPACE_URL_V10", "nd1490/ratatouille-inference-v10-q8")

gradio_client_v8 = None
gradio_client_v10 = None

def _get_client(version: str):
    """Return the Gradio client for the requested version.
    Retries up to 3 times with 30s delay to handle sleeping HF Spaces.
    Raises HTTPException 503 if connection fails after all retries.
    """
    from fastapi import HTTPException
    global gradio_client_v8, gradio_client_v10

    space_id  = HF_SPACE_V10 if version == "v10" else HF_SPACE_V8
    current   = gradio_client_v10 if version == "v10" else gradio_client_v8

    if current is not None:
        return current

    # Space is not connected — try up to 3 times
    for attempt in range(1, 4):
        try:
            print(f">> Connecting to {version} Space (attempt {attempt}/3): {space_id}")
            client = Client(space_id, token=HF_TOKEN)
            if version == "v10":
                gradio_client_v10 = client
            else:
                gradio_client_v8 = client
            print(f"[OK] Connected to {version} Space.")
            return client
        except Exception as e:
            print(f"[WARN] Attempt {attempt}/3 failed for {version}: {e}")
            if attempt < 3:
                time.sleep(30)

    # All retries failed — return a clean 503 instead of crashing
    raise HTTPException(
        status_code=503,
        detail=f"HF Space ({version}) is still waking up. Please retry in 1-2 minutes."
    )

# Lookup helper — used in endpoints
_clients = {"v8": lambda: _get_client("v8"), "v10": lambda: _get_client("v10")}

# ============================================================
# HUGGING FACE SERVERLESS INFERENCE CLIENT (for metadata tasks)
# ============================================================
inference_client = None

def get_inference_client():
    global inference_client
    if inference_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("[!] GROQ_API_KEY not found in .env — cannot reach Groq API.")
        inference_client = Groq(api_key=GROQ_API_KEY)
    return inference_client

def query_serverless_llm(messages: list, max_tokens: int = 250, temperature: float = 0.1) -> str:
    """Queries Groq API (llama-3.3-70b-versatile) for instantaneous serverless inference."""
    try:
        client = get_inference_client()
        res = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] Groq API query failed: {e}")
        return ""

def query_hf_model(prompt: str, max_new_tokens: int = 350,
                   temperature: float = 0.7, top_p: float = 0.9,
                   repetition_penalty: float = 1.1,
                   do_sample: bool = True,
                   retries: int = 3,
                   client=None) -> str:
    """
    Central gateway — calls the Gradio Space's /api/predict endpoint.
    Pass `client=gradio_client_v10` to use the new V9 model.
    Defaults to V8 if no client is specified.
    Retries automatically on transient failures (cold starts, timeouts).
    """
    if client is None:
        client = _get_client("v8")
    for attempt in range(1, retries + 1):
        try:
            result = client.predict(
                prompt,               # Textbox: prompt
                max_new_tokens,       # Slider: max_new_tokens
                temperature,          # Slider: temperature
                top_p,                # Slider: top_p
                repetition_penalty,   # Slider: repetition_penalty
                do_sample,            # Checkbox: do_sample
                api_name="/generate",
            )
            if isinstance(result, str) and result.strip():
                return result.strip()
            print(f"[!] Empty response from Space (attempt {attempt}/{retries})")
        except Exception as e:
            print(f"[*] Space not ready — {e} (attempt {attempt}/{retries})")
            time.sleep(15)

    return ""

# ============================================================
# NLP TOOLS
# ============================================================
lemmatizer = WordNetLemmatizer()

# ============================================================
# LOAD MARKET DATA (from GitHub)
# ============================================================
headers = {"Authorization": f"token {GITHUB_PAT}", "Accept": "application/vnd.github.v3.raw"} if GITHUB_PAT else {}
mandi_url  = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/daily_mandi_data.csv"
bounds_url = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/v8_lemmaized_ingredient_bounds.json"
pantry_url = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/pantry_prices.json"

print("[*] Fetching Mandi Market Data...")
mandi_response = requests.get(mandi_url, headers=headers)
df_mandi = pd.read_csv(
    io.StringIO(mandi_response.text),
    header=None,
    names=['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade',
           'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price']
)
latest_date = df_mandi['Arrival_Date'].max()
current_mandi = df_mandi[df_mandi['Arrival_Date'] == latest_date].copy()
current_mandi['Processed_Ingredient'] = current_mandi['Commodity'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word.lower()) for word in str(x).split()])
)
current_mandi['Modal_Price'] = pd.to_numeric(current_mandi['Modal_Price'], errors='coerce')
current_mandi = current_mandi.dropna(subset=['Modal_Price'])
current_mandi['Price_per_Gram'] = current_mandi['Modal_Price'] / 100000

bounds_response = requests.get(bounds_url, headers=headers)
bounds_dict = bounds_response.json() if bounds_response.status_code == 200 else {}

pantry_prices = {
    "vanilla extract": 4.50, "dark chocolate": 1.20, "soy sauce": 0.40,
    "olive oil": 0.80, "macaroni": 0.30, "vegan cashew mozzarella": 2.50,
    "sugar": 0.05, "salt": 0.02, "egg": 0.12, "chicken": 0.25, "milk": 0.06,
    "mutton": 0.80, "fish": 0.30, "paneer": 0.40, "pork": 0.30, "beef": 0.30,
    "cheese": 0.50, "butter": 0.60, "yogurt": 0.10, "curd": 0.10
}
pantry_response = requests.get(pantry_url, headers=headers)
if pantry_response.status_code == 200:
    try:
        pantry_prices.update(pantry_response.json())
    except Exception:
        pass
print("[OK] Market Data Loaded.")

# ============================================================
# CANONICAL INGREDIENT NORMALIZATION MAP  (Priority 1)
# Maps variant names → canonical base so "chicken breast",
# "chicken thigh" etc. all hit the same MongoDB/engine entry
# as "chicken". Safe to extend — just add more lines.
# ============================================================
CANONICAL_MAP: dict[str, str] = {
    # ── Poultry ─────────────────────────────────────────────
    "chicken breast": "chicken",   "chicken thigh": "chicken",
    "chicken thighs": "chicken",   "chicken wing": "chicken",
    "chicken wings": "chicken",    "chicken leg": "chicken",
    "chicken legs": "chicken",     "chicken tikka": "chicken",
    "minced chicken": "chicken",   "ground chicken": "chicken",
    "chicken mince": "chicken",    "boneless chicken": "chicken",
    "chicken pieces": "chicken",   "chicken fillet": "chicken",
    "chicken fillets": "chicken",  "chicken strips": "chicken",
    "shredded chicken": "chicken", "chicken cubes": "chicken",
    # ── Egg ────────────────────────────────────────────────
    "eggs": "egg",         "egg yolk": "egg",   "egg yolks": "egg",
    "egg white": "egg",    "egg whites": "egg",
    "whole egg": "egg",    "whole eggs": "egg",
    "large egg": "egg",    "large eggs": "egg",
    # ── Beef ─────────────────────────────────────────────
    "ground beef": "beef",  "beef mince": "beef",
    "minced beef": "beef",  "beef steak": "beef",
    "beef strips": "beef",  "beef cubes": "beef",
    "chuck steak": "beef",  "sirloin": "beef",
    "ribeye": "beef",       "beef tenderloin": "beef",
    "stewing beef": "beef",
    # ── Mutton / Lamb ──────────────────────────────────────
    "lamb chop": "mutton",  "lamb chops": "mutton",
    "lamb mince": "mutton", "minced lamb": "mutton",
    "ground lamb": "mutton","lamb leg": "mutton",
    "rack of lamb": "mutton","mutton chop": "mutton",
    "mutton pieces": "mutton","goat meat": "mutton",
    # ── Pork ──────────────────────────────────────────────
    "pork belly": "pork",   "pork chop": "pork",
    "pork chops": "pork",   "pork ribs": "pork",
    "pork mince": "pork",   "ground pork": "pork",
    "pork loin": "pork",    "pork shoulder": "pork",
    # ── Fish ──────────────────────────────────────────────
    "salmon fillet": "fish", "salmon fillets": "fish",
    "tuna steak": "fish",    "cod fillet": "fish",
    "fish fillet": "fish",   "fish fillets": "fish",
    "tilapia": "fish",  "basa": "fish",  "rohu": "fish",
    "catla": "fish",    "pomfret": "fish","hilsa": "fish",
    # ── Shrimp / Prawn ──────────────────────────────────────
    "king prawn": "shrimp",  "king prawns": "shrimp",
    "tiger prawn": "shrimp", "tiger prawns": "shrimp",
    "jumbo shrimp": "shrimp",
    # ── Dairy fat ──────────────────────────────────────────
    "clarified butter": "ghee",
    "unsalted butter": "butter",  "salted butter": "butter",
    # ── Dairy liquid ───────────────────────────────────────
    "whole milk": "milk",    "skim milk": "milk",
    "skimmed milk": "milk",  "full fat milk": "milk",
    "low fat milk": "milk",  "evaporated milk": "milk",
    "buttermilk": "milk",
    "heavy cream": "cream",  "heavy whipping cream": "cream",
    "double cream": "cream", "whipping cream": "cream",
    "single cream": "cream", "fresh cream": "cream",
    "cooking cream": "cream",
    "sour cream": "yogurt",  "creme fraiche": "yogurt",
    "cottage cheese": "paneer",
    # ── Cheese ────────────────────────────────────────────
    "mozzarella": "cheese",  "cheddar": "cheese",
    "parmesan": "cheese",    "processed cheese": "cheese",
}

def canonicalize_ingredient(name_lower: str) -> str:
    """Return canonical base ingredient name for a variant.
    e.g. 'chicken breast' -> 'chicken', 'egg yolk' -> 'egg'.
    Falls back to the original string if no mapping exists."""
    return CANONICAL_MAP.get(name_lower, name_lower)


# ============================================================
# INGREDIENT / PRICE LOGIC  (unchanged from V10)
# ============================================================
def extract_json_from_llm(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return None

def deconstruct_ingredient(ingredient):
    """Use Serverless Llama 3.3 70B to break a processed ingredient into raw crops."""
    cache_id = f"deconstruct_{ingredient.strip().lower()}"
    if llm_cache_sync is not None:
        try:
            cached = llm_cache_sync.find_one({"_id": cache_id})
            if cached and "result" in cached:
                print(f"[CACHE HIT] deconstruct_ingredient('{ingredient}') -> {cached['result']}")
                return cached["result"]
        except Exception as e:
            print(f"[WARN] Cache lookup failed: {e}")

    messages = [
        {"role": "system", "content": "You are a food chemistry and agricultural database. Output only valid JSON. Do not write any explanations or conversational text outside the JSON."},
        {"role": "user", "content": f"Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops with approximate weight percentages (total summing to 1.0).\nExample: for 'tomato ketchup', return exactly: {{\"tomato\": 0.8, \"sugar\": 0.1, \"onion\": 0.1}}"}
    ]
    text = query_serverless_llm(messages, max_tokens=100, temperature=0.1)
    # If serverless query fails, fall back to deconstruction using Gradio Llama 3B space
    if not text:
        print(f"[WARN] Serverless deconstruct failed for '{ingredient}'. Falling back to Gradio Llama 3B...")
        prompt = (
            f"<|begin_of_text|>Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops.\n"
            f"Assign approximate weight percentages. Return ONLY valid JSON.\n"
            f"Example for 'tomato ketchup': {{\"tomato\": 0.8, \"sugar\": 0.1, \"onion\": 0.1}}\n\n"
            f"### INGREDIENT:\n{ingredient}\n### JSON:\n"
        )
        text = query_hf_model(prompt, max_new_tokens=50, temperature=0.0, do_sample=False)
        
    result = extract_json_from_llm(text)
    
    if result and llm_cache_sync is not None:
        try:
            llm_cache_sync.update_one(
                {"_id": cache_id},
                {"$set": {"result": result, "timestamp": time.time()}},
                upsert=True
            )
            print(f"[CACHE SET] Saved deconstruction for '{ingredient}'.")
        except Exception as e:
            print(f"[WARN] Cache save failed: {e}")
            
    return result

def get_recipe_archetype(ingredients_list):
    """Classify the dish archetype using Serverless Llama 3.3 70B."""
    ingr_str = ", ".join(ingredients_list)
    sorted_ingrs = ",".join(sorted([i.strip().lower() for i in ingredients_list]))
    cache_id = f"archetype_{sorted_ingrs}"
    
    if llm_cache_sync is not None:
        try:
            cached = llm_cache_sync.find_one({"_id": cache_id})
            if cached and "result" in cached:
                print(f"[CACHE HIT] get_recipe_archetype -> {cached['result']}")
                return cached["result"]
        except Exception as e:
            print(f"[WARN] Cache lookup failed: {e}")

    messages = [
        {"role": "system", "content": "You are a recipe classification assistant. Output ONLY a single word classification from the permitted list. No explanation."},
        {"role": "user", "content": f"Classify the dish structure based on these ingredients: [{ingr_str}].\nChoose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish]."}
    ]
    text = query_serverless_llm(messages, max_tokens=15, temperature=0.1)
    
    # Fallback to Gradio Llama 3B
    if not text:
        print("[WARN] Serverless archetype classification failed. Falling back to Gradio Llama 3B...")
        prompt = (
            f"<|begin_of_text|>Classify the dish structure based on these ingredients: [{ingr_str}].\n"
            f"Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish].\n"
            f"Return ONLY the word.\n\n"
            f"### INGREDIENTS:\n{ingr_str}\n### ARCHETYPE:\n"
        )
        text = query_hf_model(prompt, max_new_tokens=10, temperature=0.0, do_sample=False)

    valid_archetypes = ["Curry", "Dry_Sabzi", "Salad", "Dessert", "Bread", "Soup", "Rice_Dish"]
    result = "Curry"
    if text:
        for v in valid_archetypes:
            if v.lower() in text.lower():
                result = v
                break
                
    if llm_cache_sync is not None:
        try:
            llm_cache_sync.update_one(
                {"_id": cache_id},
                {"$set": {"result": result, "timestamp": time.time(), "original_ingredients": ingredients_list}},
                upsert=True
            )
            print(f"[CACHE SET] Saved archetype '{result}' for {ingredients_list}.")
        except Exception as e:
            print(f"[WARN] Cache save failed: {e}")
            
    return result

def parse_ingredient_input(raw_string):
    match = re.match(r"^([\d\.]+)\s*(g|kg|ml)?\s+(.*)$", raw_string.strip().lower())
    if match:
        qty = float(match.group(1))
        unit = match.group(2)
        name = match.group(3)
        if unit == 'kg': qty *= 1000
        return qty, name
    return None, raw_string.lower().strip()

def get_dynamic_price(clean_ingredient, user_state, skip_llm=False):
    lemmatized_ing = " ".join([lemmatizer.lemmatize(w) for w in clean_ingredient.split()])
    state_data = current_mandi[(current_mandi['Processed_Ingredient'] == lemmatized_ing) & (current_mandi['State'].str.lower() == user_state.lower())]
    if not state_data.empty: return state_data['Price_per_Gram'].median()
    nat_data = current_mandi[current_mandi['Processed_Ingredient'] == lemmatized_ing]
    if not nat_data.empty: return nat_data['Price_per_Gram'].median()
    if clean_ingredient in pantry_prices: return pantry_prices[clean_ingredient]
    if skip_llm:
        return 0.5  # default fallback price — avoids LLM call in /optimize-only
    deconstructed = deconstruct_ingredient(clean_ingredient)
    if deconstructed and isinstance(deconstructed, dict):
        # Prevent crash if the AI hallucinates nested dictionaries
        if not all(isinstance(v, (int, float)) for v in deconstructed.values()):
            return 0.5

        avg_price = 0
        total_weight = sum(deconstructed.values())
        if total_weight == 0:
            return 0.5  # Fallback if AI gives invalid weights
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

# ============================================================
# FAST KEYWORD-BASED ARCHETYPE CLASSIFIER (no LLM)
# ============================================================
def _classify_archetype_fast(ingredients: list) -> str:
    """Classify dish archetype using keyword matching — instant, no LLM needed."""
    s = " ".join(ingredients).lower()
    if any(w in s for w in ["rice", "biryani", "pulao", "fried rice"]):
        return "Rice_Dish"
    if any(w in s for w in ["pasta", "noodle", "macaroni", "spaghetti", "basil"]):
        return "Rice_Dish"  # similar ratio constraints
    if any(w in s for w in ["oat", "banana", "sugar", "flour", "chocolate", "cake", "honey", "cream", "milk", "vanilla"]):
        return "Dessert"
    if any(w in s for w in ["mushroom", "soup", "broth", "carrot", "ginger", "coconut milk"]):
        return "Soup"
    if any(w in s for w in ["lettuce", "salad", "cucumber", "olive"]):
        return "Salad"
    if any(w in s for w in ["bread", "dough", "yeast"]):
        return "Bread"
    return "Curry"  # default for most Indian dishes

# ============================================================
# SCIPY LINEAR PROGRAMMING OPTIMIZER  (unchanged from V10)
# ============================================================
def optimize_recipe_v2(raw_user_ingredients, total_budget, servings, user_state, archetype=None, skip_llm=False):
    if archetype is None:
        archetype = get_recipe_archetype(raw_user_ingredients)  # LLM call (used by /generate-recipe)
    n = len(raw_user_ingredients)
    c = [-1] * n
    prices, bounds, tags = [], [], []

    for raw_ing in raw_user_ingredients:
        user_qty, clean_name = parse_ingredient_input(raw_ing)
        prices.append(get_dynamic_price(clean_name, user_state, skip_llm=skip_llm))
        tags.append(tag_ingredient(clean_name))

        if user_qty is not None:
            bounds.append((user_qty / servings, user_qty / servings))
        else:
            # RELAXED BOUNDS: Increased the max limit so the solver has more breathing room.
            lower, upper = bounds_dict.get(lemmatizer.lemmatize(clean_name.split()[-1]), (10, 400))
            if clean_name not in ['garlic', 'ginger', 'chili', 'salt', 'pepper']: lower = max(lower, 15.0)
            else: lower = max(lower, 2.0)
            upper = min(upper, 250.0) # Allow up to 250g per ingredient per serving max
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

    # First Attempt: With Archetype Constraints
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Fallback: If the constraints are too strict (infeasible), drop the archetype constraints and just optimize for budget
    if not result.success:
        print(f"[*] Strict archetype constraints failed ({result.message}). Relaxing constraints and retrying...")
        A_ub_fallback = [prices]
        b_ub_fallback = [total_budget / servings]
        result = linprog(c, A_ub=A_ub_fallback, b_ub=b_ub_fallback, bounds=bounds, method='highs')

    # Super Fallback: If it STILL fails, the minimum gram bounds are too high for the user's budget.
    # Force lower bounds to 5g to guarantee a mathematical solution.
    if not result.success:
        print(f"[*] Budget mathematically too low for default bounds! Forcing lower bounds to 5g...")
        emergency_bounds = [(5.0, max(b[1], 5.0)) for b in bounds]
        result = linprog(c, A_ub=A_ub_fallback, b_ub=b_ub_fallback, bounds=emergency_bounds, method='highs')

    # Nuclear Fallback: If it fails due to some impossible matrix constraint, remove ALL limits.
    if not result.success:
        print(f"[*] Nuclear Fallback activated: Overriding all bounds to (1g, 300g)...")
        nuclear_bounds = [(1.0, 300.0) for _ in bounds]
        result = linprog(c, A_ub=A_ub_fallback, b_ub=b_ub_fallback, bounds=nuclear_bounds, method='highs')

    if result.success:
        estimated_grams_per_serving = np.round(result.x, 1)
        final_quantities = estimated_grams_per_serving * servings
        return [f"{qty}g {parse_ingredient_input(r)[1]}" for qty, r in zip(final_quantities, raw_user_ingredients)], archetype
    else:
        print(f"[!] linprog failed even without constraints! {result.message}")
        print(f"[*] prices={prices}, bounds={bounds}")
        
    return None, archetype

# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class RecipeRequest(BaseModel):
    ingredients: list[str]
    budget: float
    servings: int = 1
    state: str = "Delhi"
    model_version: str = "v10"   # "v8" = old Space | "v10" = new retrained Space
    is_vegan: bool = False

class SaveRecipeRequest(BaseModel):
    username: str
    recipe_data: dict

class VeganRequest(BaseModel):
    ingredients: list[str] = []
    ingredient: str = ""
    archetype: str = "Curry"

# ============================================================

# ENDPOINTS
# ============================================================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "spaces": {"v8": HF_SPACE_V8, "v9": HF_SPACE_V10},
        "db_connected": recipes_collection is not None,
    }

@app.post("/optimize-only")
def optimize_only(request: RecipeRequest):
    """Lightweight endpoint: runs SciPy budget optimization only — no LLM call.
    Returns the optimized ingredient weights and archetype.
    Used by the eval script to get ingredients before calling HF Spaces directly.
    """
    clean_ingredients = []
    for item in request.ingredients:
        clean_ingredients.extend([i.strip() for i in item.split(',') if i.strip()])

    archetype_fast = _classify_archetype_fast(clean_ingredients)
    calculated_ingredients, archetype = optimize_recipe_v2(
        clean_ingredients, request.budget, request.servings, request.state,
        archetype=archetype_fast,  # skip LLM for archetype
        skip_llm=True             # skip LLM for unknown ingredient prices
    )

    if not calculated_ingredients:
        return {
            "status": "error",
            "message": "Budget is mathematically infeasible for these ingredients at current market prices.",
            "archetype": archetype,
            "calculated_ingredients": []
        }

    return {
        "status": "success",
        "archetype": archetype,
        "calculated_ingredients": calculated_ingredients,
    }

@app.post("/save-recipe")
async def save_recipe_db(request: SaveRecipeRequest):
    if recipes_collection is None:
        return {"status": "error", "message": "Database not connected on the server."}
    
    document = {
        "username": request.username.lower(),
        "recipe": request.recipe_data,
        "created_at": time.time()
    }
    await recipes_collection.insert_one(document)
    return {"status": "success", "message": "Recipe saved to your profile!"}

@app.get("/my-recipes/{username}")
async def get_my_recipes(username: str):
    if recipes_collection is None:
        return {"status": "error", "message": "Database not connected on the server."}
    
    cursor = recipes_collection.find({"username": username.lower()}).sort("created_at", -1)
    recipes = await cursor.to_list(length=50)
    
    # Convert MongoDB ObjectId to string for JSON serialization
    for r in recipes:
        r["_id"] = str(r["_id"])
        
    return {"status": "success", "recipes": recipes}


# ============================================================
# VEGAN DATABASE ENDPOINTS  (serve GPU-generated MongoDB data)
# ============================================================

@app.get("/vegan-alternatives/{ingredient}")
async def get_vegan_alternatives(ingredient: str):
    """
    Return top-5 vegan alternatives for an ingredient.
    Checks MongoDB first (GPU-generated cached data).
    Falls back to live vegan engine if not found in DB.
    """
    clean = ingredient.strip().lower()

    # 1. Try MongoDB cache
    if vegan_alternatives_collection is not None:
        doc = await vegan_alternatives_collection.find_one({"_id": clean})
        if doc:
            doc.pop("_id", None)  # remove MongoDB _id before returning
            return {
                "status": "success",
                "ingredient": clean,
                "source": "db",
                "original_role": doc.get("original_role", "unknown"),
                "alternatives": doc.get("alternatives", [])
            }

    # 2. Live engine fallback (existing behaviour)
    import vegan_engine
    blueprint = vegan_engine.generate_vegan_blueprint(clean, archetype="Curry")
    if blueprint.get("status") == "success":
        return {
            "status": "success",
            "ingredient": clean,
            "source": "live_engine",
            "original_role": "unknown",
            "alternatives": [{
                "rank": 1,
                "substitute": blueprint["best_vegan_substitute"],
                "composite_score": blueprint["match_score"],
                "score_breakdown": blueprint.get("score_breakdown", {}),
                "chemical_delta": blueprint.get("chemical_delta", {}),
                "spice_bridge": blueprint.get("compensation_blueprint", {}).get("spice_bridge", []),
                "culinary_notes": "; ".join(blueprint.get("compensation_blueprint", {}).get("techniques", []))
            }]
        }
    elif blueprint.get("status") == "already_vegan":
        return {"status": "already_vegan", "ingredient": clean, "message": blueprint["message"]}
    else:
        return {"status": "not_found", "ingredient": clean,
                "message": f"'{ingredient}' not found in database or vegan engine."}


@app.get("/indian-recipes/styles")
async def get_indian_recipe_styles():
    """Return distinct recipe styles and their counts from the indian_recipes collection."""
    if indian_recipes_collection is None:
        return {"status": "error", "message": "Database not connected on the server."}
    pipeline = [{"$group": {"_id": "$style", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}]
    cursor = indian_recipes_collection.aggregate(pipeline)
    styles = [{"name": doc["_id"], "count": doc["count"]} async for doc in cursor]
    return {"status": "success", "styles": styles}


@app.get("/indian-recipes")
async def get_indian_recipes(style: str = "", limit: int = 12, skip: int = 0):
    """
    Return paginated Indian budget vegan recipes from MongoDB.
    Optional ?style=Curry filter. Default page size 12.
    """
    if indian_recipes_collection is None:
        return {"status": "error", "message": "Database not connected on the server."}
    query = {"style": style} if style else {}
    total = await indian_recipes_collection.count_documents(query)
    cursor = indian_recipes_collection.find(query).skip(skip).limit(limit).sort("created_at", -1)
    recipes = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        recipes.append(doc)
    return {"status": "success", "total": total, "skip": skip, "limit": limit, "recipes": recipes}

def bootstrap_ingredient_profile(ingredient_name):
    """Query Serverless Llama 3.3 70B to generate a physical-chemical JSON profile for an unseen ingredient."""
    messages = [
        {"role": "system", "content": "You are a food chemistry and culinary database. Output only valid JSON. No explanations."},
        {"role": "user", "content": (
            f"Generate a chemical and physical profile for the ingredient '{ingredient_name}' matching the specified JSON format.\n"
            f"Determine if it is vegan (true/false).\n"
            f"Specify macros (proteins, fats, carbs in grams per 100g, summing up to at most 100).\n"
            f"Rate its texture on a 1.0-10.0 scale: [hardness, chewiness, moisture, fat_mouthfeel, elasticity].\n"
            f"List 3-5 primary flavor volatile compounds (e.g. aldehydes, pyrazines, esters, terpenes, or specific molecules).\n"
            f"Assign a culinary role: [base_protein, fat_source, flavor_enhancer, thickener, sweetener, aromatic, dairy, binding_agent].\n\n"
            f"Return ONLY valid JSON matching this exact structure:\n"
            f"{{\n"
            f"  \"is_vegan\": false,\n"
            f"  \"macros\": {{\"proteins\": 22.0, \"fats\": 20.0, \"carbs\": 0.0}},\n"
            f"  \"texture_profile\": [4.5, 4.0, 4.0, 6.0, 2.0],\n"
            f"  \"flavor_molecules\": [\"methanethiol\", \"dimethyl sulfide\", \"pyrazines\"],\n"
            f"  \"culinary_role\": \"base_protein\"\n"
            f"}}\n"
        )}
    ]
    raw_text = query_serverless_llm(messages, max_tokens=250, temperature=0.1)
    
    # Fallback to Gradio Llama 3B V10
    if not raw_text:
        print(f"[WARN] Serverless bootstrapping failed for '{ingredient_name}'. Falling back to Gradio V10 Space...")
        prompt = (
            f"<|begin_of_text|>You are a food chemistry and culinary database. Generate a chemical and physical profile for the ingredient '{ingredient_name}' matching the specified JSON format.\n"
            f"Determine if it is vegan (true/false).\n"
            f"Specify macros (proteins, fats, carbs in grams per 100g, summing up to at most 100).\n"
            f"Rate its texture on a 1.0-10.0 scale: [hardness, chewiness, moisture, fat_mouthfeel, elasticity].\n"
            f"List 3-5 primary flavor volatile compounds (e.g. aldehydes, pyrazines, esters, terpenes, or specific molecules).\n"
            f"Assign a culinary role: [base_protein, fat_source, flavor_enhancer, thickener, sweetener, aromatic, dairy, binding_agent].\n\n"
            f"Return ONLY valid JSON matching this example:\n"
            f"{{\n"
            f"  \"is_vegan\": false,\n"
            f"  \"macros\": {{\"proteins\": 22.0, \"fats\": 20.0, \"carbs\": 0.0}},\n"
            f"  \"texture_profile\": [4.5, 4.0, 4.0, 6.0, 2.0],\n"
            f"  \"flavor_molecules\": [\"methanethiol\", \"dimethyl sulfide\", \"pyrazines\"],\n"
            f"  \"culinary_role\": \"base_protein\"\n"
            f"}}\n\n"
            f"### INGREDIENT: {ingredient_name}\n"
            f"### JSON:\n"
        )
        try:
            client = _get_client("v10")
            raw_text = query_hf_model(prompt, max_new_tokens=250, temperature=0.0, do_sample=False, client=client)
        except Exception as e:
            print(f"[WARN] Gradio V10 Space backup bootstrapping failed: {e}")
            return None

    try:
        profile = extract_json_from_llm(raw_text)
        if profile and isinstance(profile, dict) and "is_vegan" in profile and "macros" in profile and "texture_profile" in profile:
            if len(profile["texture_profile"]) == 5 and all(isinstance(v, (int, float)) for v in profile["texture_profile"]):
                return profile
    except Exception as e:
        print(f"[WARN] Error parsing bootstrapped profile JSON: {e}")
    return None

@app.post("/get-vegan-blueprint")
def get_vegan_blueprint(request: VeganRequest):
    import vegan_engine
    target_ingredients = []
    if request.ingredient:
        target_ingredients.append(request.ingredient)
    if request.ingredients:
        target_ingredients.extend(request.ingredients)
        
    # Remove duplicates while preserving order
    seen = set()
    ingredients_to_process = []
    for x in target_ingredients:
        cleaned = x.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ingredients_to_process.append(cleaned)
            
    results = []
    for ing in ingredients_to_process:
        # ── STEP 0: Canonical normalization (Priority 1) ──────────────────
        canonical = canonicalize_ingredient(ing)
        if canonical != ing:
            print(f"[CANON] '{ing}' → '{canonical}'")

        # 1. Check if it's already in the database
        try:
            features = vegan_engine.load_features()
        except Exception:
            features = {}

        if canonical not in features:
            # 2. Try to bootstrap using LLM
            profile = bootstrap_ingredient_profile(canonical)
            if profile:
                # 3. Save it to cache JSON database
                vegan_engine.save_new_feature(canonical, profile)
                print(f"[OK] Dynamically bootstrapped and cached ingredient: {canonical}")

        blueprint = vegan_engine.generate_vegan_blueprint(canonical, archetype=request.archetype)
        results.append(blueprint)
        
    return {
        "status": "success",
        "results": results
    }

@app.post("/generate-recipe")
def generate_recipe(request: RecipeRequest):
    def event_stream():
        yield f"data: {json.dumps({'step': 'starting', 'message': 'Initializing...'})}\n\n"
        
        start_time = time.time()
        times = {}

        # Automatically split by comma in case the user sends one giant string
        clean_ingredients = []
        for item in request.ingredients:
            clean_ingredients.extend([i.strip() for i in item.split(',') if i.strip()])

        if request.is_vegan:
            yield f"data: {json.dumps({'step': 'veganizing', 'message': 'Running Vegan Substitution Engine...'})}\n\n"
            step_start = time.time()
            
            import vegan_engine
            archetype_fast = _classify_archetype_fast([parse_ingredient_input(i)[1] for i in clean_ingredients])
            veganized_ingredients = []
            for raw_ing in clean_ingredients:
                qty, name = parse_ingredient_input(raw_ing)
                name_lower = name.lower().strip()
                # ── STEP 0: Canonical normalization (Priority 1) ──────────────────
                # "chicken breast" → "chicken", "egg yolk" → "egg", etc.
                # This means your 20 MongoDB entries now cover 80+ variant names.
                canonical_name = canonicalize_ingredient(name_lower)
                if canonical_name != name_lower:
                    print(f"[CANON] '{name_lower}' → '{canonical_name}'")
                blueprint = None

                # ── STEP 1: Check pre-built MongoDB match table (fast, ~1ms) ──────
                if vegan_alternatives_sync is not None:
                    try:
                        db_doc = vegan_alternatives_sync.find_one({"_id": canonical_name})
                        if db_doc and db_doc.get("best_vegan_substitute"):
                            blueprint = {
                                "status": "success",
                                "original_ingredient": name,
                                "best_vegan_substitute": db_doc["best_vegan_substitute"],
                                "match_score": db_doc.get("match_score", 0),
                                "compensation_blueprint": db_doc.get("compensation_blueprint", {
                                    "auxiliary_additions": [],
                                    "techniques": [],
                                    "spice_bridge": []
                                })
                            }
                            print(f"[DB HIT] '{canonical_name}' → '{db_doc['best_vegan_substitute']}' (score: {db_doc.get('match_score', '?')})")
                    except Exception as e:
                        print(f"[DB WARN] vegan_alternatives lookup failed for '{canonical_name}': {e}")

                # ── STEP 2: Fallback to local vegan_engine (no LLM bootstrapping) ──
                # Only runs if ingredient was NOT found in the MongoDB match table.
                # We skip bootstrap_ingredient_profile() — it spends 1-60s per ingredient
                # calling an LLM. If profile exists locally → instant math. If unknown → keep as-is.
                if blueprint is None:
                    try:
                        features = vegan_engine.load_features()
                        # Proceed if it's explicitly in DB OR if it matches a static fallback
                        if canonical_name in features or vegan_engine.classify_by_keyword(canonical_name) is not None:
                            blueprint = vegan_engine.generate_vegan_blueprint(canonical_name, archetype=archetype_fast)
                            print(f"[LOCAL HIT] '{canonical_name}' → '{blueprint.get('best_vegan_substitute', 'N/A')}'")
                        else:
                            # Truly unknown ingredient — keep as-is, no LLM call
                            print(f"[SKIP] '{canonical_name}' not in DB and no static fallback — keeping as-is")
                            blueprint = {"status": "unknown"}
                    except Exception as e:
                        print(f"[ENGINE ERROR] vegan_engine failed for '{canonical_name}': {e}")
                        blueprint = {"status": "error"}


                # ── STEP 3: Apply substitution + compensation ──────────────────────────
                if blueprint and blueprint.get("status") == "success" and \
                   blueprint.get("original_ingredient") != blueprint.get("best_vegan_substitute"):
                    substitute = blueprint["best_vegan_substitute"]
                    veganized_ingredients.append(f"{qty}g {substitute}" if qty is not None else substitute)
                    for add in blueprint.get("compensation_blueprint", {}).get("auxiliary_additions", []):
                        veganized_ingredients.append(f"{add['amount']} {add['name']}")
                    for spice in blueprint.get("compensation_blueprint", {}).get("spice_bridge", []):
                        veganized_ingredients.append(spice["spice"])
                else:
                    veganized_ingredients.append(raw_ing)

            # Override the ingredients list with the newly veganized list
            clean_ingredients = veganized_ingredients
            times['veganization_sec'] = time.time() - step_start

        yield f"data: {json.dumps({'step': 'optimizing', 'message': f'Running Cost Constraint Optimization (Budget: ₹{request.budget})...'})}\n\n"
        step_start = time.time()
        print(f"[] Running Cost Constraint Optimization for Budget: INR {request.budget}...")
        calculated_ingredients, archetype = optimize_recipe_v2(clean_ingredients, request.budget, request.servings, request.state)

        if not calculated_ingredients:
            yield f"data: {json.dumps({'step': 'error', 'message': 'The provided budget is mathematically impossible for these ingredients at current market prices.'})}\n\n"
            return

        times['optimization_sec'] = time.time() - step_start

        ingr_text = "\n".join(f"- {i}" for i in calculated_ingredients)

        # EXACT V10 PROMPT STRUCTURE
        prompt = (
            f"### INGREDIENTS:\n"
            f"{ingr_text}\n"
            f"### TITLE:\n"
        )
        print(f"[AI] Generating final recipe for archetype: {archetype} using model_version={request.model_version}")

        yield f"data: {json.dumps({'step': 'generating', 'message': f'Generating AI Recipe (Model: {request.model_version})...'})}\n\n"
        step_start = time.time()

        chosen_client = _clients.get(request.model_version, _clients["v8"])()
        ai_text = query_hf_model(
            prompt,
            max_new_tokens=500,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            client=chosen_client,
        )
        
        times['generation_sec'] = time.time() - step_start

        print(f"DEBUG RAW AI_TEXT: {repr(ai_text)}")

        # Post-process to remove Llama 3 hallucinations/rambling
        stop_tokens = ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>', '\n### INGREDIENTS:']
        for t in stop_tokens:
            if t in ai_text:
                ai_text = ai_text.split(t)[0].strip()

        # Sometimes it doesn't emit eot_id and just loops back to Step 1 or adds another Title block
        if "### TITLE:\n" in ai_text:
            ai_text = ai_text.split("### TITLE:\n")[1].strip()
            
        # Cut off standard looping signatures and Notes
        cut_phrases = [
            "\nEnjoy!", "\nServe hot", "\nBon Apetit", "\nChef's Note:", 
            "\nVariations:", "\nServing suggestion:", "\nNote:"
        ]
        for phrase in cut_phrases:
            if phrase in ai_text:
                ai_text = ai_text.split(phrase)[0].strip()

        # If the AI tries to start a new section with "###" (like "### Serving suggestion:" or "### Note:"), cut it off.
        # The only valid "###" in the AI output is "### DIRECTIONS:"
        if "### DIRECTIONS:\n" in ai_text:
            parts = ai_text.split("### DIRECTIONS:\n")
            title_part = parts[0]
            directions_part = parts[1]
            
            # If there's another "### " in the directions, cut it
            if "\n### " in directions_part:
                directions_part = directions_part.split("\n### ")[0]
                
            ai_text = f"{title_part}### DIRECTIONS:\n{directions_part}".strip()
            
        total_time = time.time() - start_time
        
        # Log to MongoDB
        if generation_logs_sync is not None:
            try:
                log_entry = {
                    "timestamp": time.time(),
                    "model_version": request.model_version,
                    "is_vegan": request.is_vegan,
                    "archetype": archetype,
                    "budget": request.budget,
                    "servings": request.servings,
                    "state": request.state,
                    "times_sec": times,
                    "total_time_sec": total_time,
                    "ingredient_count": len(clean_ingredients)
                }
                generation_logs_sync.insert_one(log_entry)
                print(f"[OK] Generation log saved to DB. Total time: {total_time:.2f}s")
            except Exception as e:
                print(f"[WARN] Failed to save generation log: {e}")

        final_result = {
            "status": "success",
            "archetype": archetype,
            "calculated_ingredients": calculated_ingredients,
            "recipe": ai_text
        }
        yield f"data: {json.dumps({'step': 'complete', 'result': final_result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
