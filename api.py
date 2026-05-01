import os
import re
import json
import time
import requests
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from nltk.stem import WordNetLemmatizer
from scipy.optimize import linprog
from dotenv import load_dotenv
from gradio_client import Client
from motor.motor_asyncio import AsyncIOMotorClient

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================
# APP INIT
# ============================================================
app = FastAPI(title="Ratatouille V9 API — Cloud Inference")

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

if not HF_TOKEN:
    raise RuntimeError("[!] HF_TOKEN not found in .env — cannot reach Hugging Face Inference API.")

# ============================================================
# DATABASE INIT
# ============================================================
if MONGO_URI:
    print("[*] Connecting to MongoDB Atlas...")
    db_client = AsyncIOMotorClient(MONGO_URI)
    db = db_client.ratatouille
    recipes_collection = db.recipes
    print("[OK] Connected to MongoDB Atlas.")
else:
    recipes_collection = None
    print("[!] MONGO_URI not found. Database features will be disabled.")

# ============================================================
# HUGGING FACE SPACE — GRADIO CLIENT
# ============================================================
# After you create the HF Space, set this in .env or replace below.
# Example: HF_SPACE_URL=nd1490/ratatouille-inference
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "nd1490/ratatouille-inference")

print(f">> Connecting to HF Space: {HF_SPACE_URL} ...")
gradio_client = Client(HF_SPACE_URL, token=HF_TOKEN)
print("[OK] Connected to inference Space.")

def query_hf_model(prompt: str, max_new_tokens: int = 350,
                   temperature: float = 0.7, top_p: float = 0.9,
                   repetition_penalty: float = 1.1,
                   do_sample: bool = True,
                   retries: int = 3) -> str:
    """
    Central gateway — calls the Gradio Space's /api/predict endpoint.
    The Space loads the full model on a free T4/ZeroGPU and generates text.
    Retries automatically on transient failures (cold starts, timeouts).
    """
    for attempt in range(1, retries + 1):
        try:
            result = gradio_client.predict(
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

pantry_response = requests.get(pantry_url, headers=headers)
pantry_prices = pantry_response.json() if pantry_response.status_code == 200 else {
    "vanilla extract": 4.50, "dark chocolate": 1.20, "soy sauce": 0.40,
    "olive oil": 0.80, "macaroni": 0.30, "vegan cashew mozzarella": 2.50,
    "sugar": 0.05, "salt": 0.02
}
print("[OK] Market Data Loaded.")

# ============================================================
# INGREDIENT / PRICE LOGIC  (unchanged from V9)
# ============================================================
def extract_json_from_llm(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return None

def deconstruct_ingredient(ingredient):
    """Use HF cloud model to break a processed ingredient into raw crops."""
    prompt = (
        f"<|begin_of_text|>Deconstruct the processed culinary ingredient '{ingredient}' into its primary raw agricultural crops.\n"
        f"Assign approximate weight percentages. Return ONLY valid JSON.\n"
        f"Example for 'tomato ketchup': {{\"tomato\": 0.8, \"sugar\": 0.1, \"onion\": 0.1}}\n\n"
        f"### INGREDIENT:\n{ingredient}\n### JSON:\n"
    )
    text = query_hf_model(prompt, max_new_tokens=50, temperature=0.0, do_sample=False)
    return extract_json_from_llm(text)

def get_recipe_archetype(ingredients_list):
    """Classify the dish archetype via HF cloud model."""
    ingr_str = ", ".join(ingredients_list)
    prompt = (
        f"<|begin_of_text|>Classify the dish structure based on these ingredients: [{ingr_str}].\n"
        f"Choose exactly ONE from this list: [Curry, Dry_Sabzi, Salad, Dessert, Bread, Soup, Rice_Dish].\n"
        f"Return ONLY the word.\n\n"
        f"### INGREDIENTS:\n{ingr_str}\n### ARCHETYPE:\n"
    )
    text = query_hf_model(prompt, max_new_tokens=10, temperature=0.0, do_sample=False)
    valid_archetypes = ["Curry", "Dry_Sabzi", "Salad", "Dessert", "Bread", "Soup", "Rice_Dish"]
    for v in valid_archetypes:
        if v.lower() in text.lower(): return v
    return "Curry"

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
# SCIPY LINEAR PROGRAMMING OPTIMIZER  (unchanged from V9)
# ============================================================
def optimize_recipe_v2(raw_user_ingredients, total_budget, servings, user_state):
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
            # RELAXED BOUNDS: Increased the max limit so the solver has more breathing room.
            lower, upper = bounds_dict.get(lemmatizer.lemmatize(clean_name.split()[-1]), (10, 400))
            if clean_name not in ['garlic', 'ginger', 'chili', 'salt', 'pepper']: lower = max(lower, 15.0)
            else: lower = max(lower, 2.0)
            upper = min(upper, 800.0) # Allow up to 800g per ingredient if budget permits
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
        print(f"[*] Nuclear Fallback activated: Overriding all bounds to (1g, 1000g)...")
        nuclear_bounds = [(1.0, 1000.0) for _ in bounds]
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

class SaveRecipeRequest(BaseModel):
    username: str
    recipe_data: dict

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/health")
def health_check():
    return {"status": "ok", "model": HF_SPACE_URL, "db_connected": recipes_collection is not None}

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

@app.post("/generate-recipe")
def generate_recipe(request: RecipeRequest):
    # Automatically split by comma in case the user sends one giant string
    clean_ingredients = []
    for item in request.ingredients:
        clean_ingredients.extend([i.strip() for i in item.split(',') if i.strip()])

    print(f"[] Running Cost Constraint Optimization for Budget: ₹{request.budget}...")
    calculated_ingredients, archetype = optimize_recipe_v2(clean_ingredients, request.budget, request.servings, request.state)

    if not calculated_ingredients:
        return {"status": "error", "message": "The provided budget is mathematically impossible for these ingredients at current market prices."}

    ingr_text = "\n".join(f"- {i}" for i in calculated_ingredients)

    # EXACT V9 PROMPT STRUCTURE with Llama 3 BOS Token
    system_instruction = (
        f"You are a master chef. Write a highly detailed, appetizing recipe for a {archetype}.\n"
        f"First, provide a creative title. Then, provide step-by-step cooking directions using proper culinary techniques.\n"
        f"CRITICAL RULES:\n"
        f"1. Ensure all raw ingredients (especially rice, grains, and meats) are explicitly cooked in the instructions.\n"
        f"2. Do not change the ingredient quantities provided.\n"
        f"3. DO NOT include any 'Notes', 'Tips', or conversational rambling at the end. Stop immediately after the final serving step."
    )
    prompt = f"<|begin_of_text|>{system_instruction}\n\n### INGREDIENTS:\n{ingr_text}\n### TITLE:\n"

    print(f"[AI] Generating final recipe for archetype: {archetype}")

    ai_text = query_hf_model(
        prompt,
        max_new_tokens=400,
        temperature=0.6,  # Lowered temperature slightly to make it less chatty/random
        top_p=0.9,
        repetition_penalty=1.15, # Increased slightly to prevent looping
    )

    print(f"DEBUG RAW AI_TEXT: {repr(ai_text)}")

    # Post-process to remove Llama 3 hallucinations/rambling
    ai_text = ai_text.split("<|eot_id|>")[0].strip()
    
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
        
    return {
        "status": "success",
        "archetype": archetype,
        "calculated_ingredients": calculated_ingredients,
        "recipe": ai_text
    }
