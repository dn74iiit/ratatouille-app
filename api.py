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

# Agentic Modules
try:
    from graph_retriever import GraphRetriever
    from llm_judge import RecipeJudge
except ImportError:
    GraphRetriever = None
    RecipeJudge = None

import os
nltk_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_dir, exist_ok=True)
nltk.data.path.append(nltk_dir)
nltk.download('wordnet', download_dir=nltk_dir, quiet=True)
nltk.download('omw-1.4', download_dir=nltk_dir, quiet=True)

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
    """Queries Groq API (qwen/qwen3.8-27b) for instantaneous serverless inference."""
    try:
        client = get_inference_client()
        res = client.chat.completions.create(
            messages=messages,
            model="qwen/qwen3.8-27b",
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] Groq API query failed: {e}")
        return ""

def get_groq_recipe_model(client) -> str:
    default_model = "qwen/qwen3.8-27b"
    try:
        models = [m.id for m in client.models.list().data]
        for m in models:
            if 'qwen' in m.lower():
                return m
        for fb in ["openai/gpt-oss-20b", "allam-2-7b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            if fb in models:
                return fb
    except Exception as e:
        print(f"[WARN] Failed to list Groq models: {e}")
    return default_model

def call_groq_with_fallback(client, messages, preferred_model=None, max_tokens=500, temperature=0.7):
    if not preferred_model:
        preferred_model = get_groq_recipe_model(client)
    
    candidate_models = [preferred_model, "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "allam-2-7b"]
    seen = set()
    models_to_try = []
    for m in candidate_models:
        if m and m not in seen:
            seen.add(m)
            models_to_try.append(m)
            
    last_err = None
    for model_name in models_to_try:
        try:
            res = client.chat.completions.create(
                messages=messages,
                model=model_name,
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
            return res, model_name
        except Exception as e:
            last_err = e
            print(f"[WARN] Groq model '{model_name}' failed: {e}. Trying fallback...")
            continue
    raise last_err or RuntimeError("All Groq models failed")

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
# NLP TOOLS & AGENTIC GLOBALS
# ============================================================
lemmatizer = WordNetLemmatizer()

try:
    print("[*] Initializing Agentic Pipeline (Retriever & Judge)...")
    graph_retriever = GraphRetriever()
    recipe_judge = RecipeJudge()
    print("[OK] Agentic Pipeline Ready.")
except Exception as e:
    print(f"[WARN] Failed to initialize Agentic Pipeline: {e}")
    graph_retriever = None
    recipe_judge = None

# ============================================================
# LOAD MARKET DATA (from GitHub)
# ============================================================
headers = {"Authorization": f"token {GITHUB_PAT}", "Accept": "application/vnd.github.v3.raw"} if GITHUB_PAT else {}
mandi_url  = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/daily_mandi_data.csv"
bounds_url = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/v8_lemmaized_ingredient_bounds.json"
pantry_url = "https://raw.githubusercontent.com/dn74iiit/recipe-data-automation/main/pantry_prices.json"

print("[*] Fetching Mandi Market Data...")
# Stream directly into Pandas with usecols to eliminate a ~200MB memory spike
df_mandi = pd.read_csv(
    mandi_url,
    storage_options=headers,
    header=None,
    names=['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade',
           'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price'],
    usecols=['State', 'Commodity', 'Arrival_Date', 'Modal_Price']
)
latest_date = df_mandi['Arrival_Date'].max()
current_mandi = df_mandi[df_mandi['Arrival_Date'] == latest_date].copy()
current_mandi['Processed_Ingredient'] = current_mandi['Commodity'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word.lower()) for word in str(x).split()])
)
current_mandi['Modal_Price'] = pd.to_numeric(current_mandi['Modal_Price'], errors='coerce')
current_mandi = current_mandi.dropna(subset=['Modal_Price'])
current_mandi['Price_per_Gram'] = current_mandi['Modal_Price'] / 100000

# Free df_mandi memory
del df_mandi

bounds_response = requests.get(bounds_url, headers=headers)
bounds_dict = bounds_response.json() if bounds_response.status_code == 200 else {}

pantry_prices = {
    "vanilla extract": 4.50, "dark chocolate": 1.20, "soy sauce": 0.40,
    "olive oil": 0.80, "vegetable oil": 0.25, "oil": 0.25, "macaroni": 0.30,
    "vegan cashew mozzarella": 2.50, "sugar": 0.05, "salt": 0.02, "egg": 0.12,
    "chicken": 0.25, "milk": 0.06, "mutton": 0.80, "fish": 0.30, "paneer": 0.40,
    "pork": 0.30, "beef": 0.30, "cheese": 0.50, "butter": 0.60, "yogurt": 0.10, "curd": 0.10,
    # Common seasonings, spices & pantry staples
    "coconut milk": 0.15, "coconut cream": 0.18, "garlic powder": 0.20, "onion powder": 0.18, "ginger powder": 0.25,
    "cocoa powder": 0.40, "mustard": 0.08, "black pepper": 0.80, "white pepper": 0.70,
    "chili powder": 0.30, "turmeric": 0.25, "cumin": 0.40, "coriander": 0.25,
    "garam masala": 0.50, "curry powder": 0.45, "paprika": 0.35, "cinnamon": 0.50,
    "cardamom": 1.50, "clove": 1.20, "basil": 0.35, "oregano": 0.35, "thyme": 0.35,
    "rosemary": 0.40, "parsley": 0.30, "cilantro": 0.08, "baking powder": 0.15,
    "baking soda": 0.08, "ketchup": 0.15, "vinegar": 0.08, "zucchini": 0.06, "mushroom": 0.15
}
pantry_response = requests.get(pantry_url, headers=headers)
if pantry_response.status_code == 200:
    try:
        pantry_prices.update(pantry_response.json())
    except Exception:
        pass
print("[OK] Market Data Loaded.")

DECONSTRUCTION_MAP = {}
try:
    map_file = os.path.join(os.path.dirname(__file__), "deconstruction_map.json")
    if os.path.exists(map_file):
        with open(map_file, "r") as f:
            DECONSTRUCTION_MAP = json.load(f)
        print(f"[OK] Loaded local deconstruction_map.json ({len(DECONSTRUCTION_MAP)} entries).")
except Exception as e:
    print(f"[WARN] Could not load deconstruction_map.json: {e}")

# ============================================================
# UNSPLASH IMAGE CACHE (By Archetype)
# ============================================================
import random
UNSPLASH_CACHE = {
    "Curry": [
        "1585937421612-70a008356fbe",  # Indian masala curry in handi
        "1603894584373-5ac82b2ae398",  # Rich gravy curry dish
        "1565557623262-b51c2513a641",  # Dal makhani & curry spread
        "1618160702438-9b02ab6515c9",  # Warm Indian spiced curry
        "1631452180519-c014fe946bc7"   # Paneer butter masala
    ],
    "Chicken": [
        "1598515214211-89d3c73ae83b",
        "1604908176997-125f25cc6f3d",
        "1626082927389-6cd097cdc6ec",
        "1562967914-608f82629710"
    ],
    "Egg": [
        "1525351484163-7529414344d8",
        "1565299585323-38d6b0865b47",
        "1587314168485-3236d6710814"
    ],
    "Biryani": [
        "1563379091339-03b21ab4a4f8",
        "1541832676-9b763b0239ab",
        "1589302168068-964664d93dc0"
    ],
    "Salad": [
        "1512621776951-a57141f2eefd",  # Fresh green vegetable salad bowl
        "1490645935967-10de6ba17061",  # Fresh plant-based avocado salad
        "1540420773420-3366772f4999"   # Vibrant rainbow salad bowl
    ],
    "Dessert": [
        "1488477181946-6428a0291777",
        "1550617931-e17a7b70dce2",
        "1551024601-bec78aea704b",
        "1563805042-7684c019e1cb"
    ],
    "Bread": [
        "1509440159596-0249088772ff",
        "1608198093002-ad4e005484ec"
    ],
    "Soup": [
        "1547592166-23ac45744acd",  # Warm vegetable stew / soup pot
        "1547592180-85f173990554",  # Rich vegetable stew pot
        "1604152135912-04a022e23696",  # Savory vegetable soup bowl
        "1578020190125-f4f7c18bc9cb"   # Hearty stew dish
    ],
    "Rice_Dish": [
        "1596797038530-2c107229654b",  # Cilantro lime veggie rice bowl
        "1536304929831-ee1ca9d44906",  # Aromatic basmati herb rice bowl
        "1505253716362-afaea1d3d1af"   # Plant-based vegetable grain bowl
    ],
    "Dry_Sabzi": [
        "1546833999-b9f581a1996d",  # Spiced vegetable roast / dry sabzi
        "1565557623262-b51c2513a641",
        "1604152135912-04a022e23696"
    ],
    "NonVegan": [
        "1598515214211-89d3c73ae83b",
        "1604908176997-125f25cc6f3d",
        "1626082927389-6cd097cdc6ec",
        "1565299585323-38d6b0865b47",
        "1541832676-9b763b0239ab"
    ],
    "Vegan": [
        "1512621776951-a57141f2eefd",  # Fresh vegetable salad
        "1490645935967-10de6ba17061",  # Fresh plant-based salad
        "1547592166-23ac45744acd",  # Warm vegetable stew pot
        "1546833999-b9f581a1996d",  # Spiced vegetable kadai
        "1596797038530-2c107229654b",  # Veggie rice bowl
        "1536304929831-ee1ca9d44906"   # Herb basmati rice bowl
    ]
}

SAVORY_WORDS_LIST = [
    'curry', 'gravy', 'masala', 'korma', 'dal', 'dhal', 'daal', 'sambar', 'rasam', 
    'sabzi', 'subzi', 'roast', 'fry', 'stir-fry', 'sukka', 'poriyal', 'bhaji', 
    'rice', 'pulao', 'pilaf', 'biryani', 'khichdi', 'skillet', 'bowl',
    'stew', 'soup', 'broth', 'shorba', 'chowder', 'kadhi', 'karhi', 'pakora', 'pakoras',
    'kofta', 'paneer', 'tofu', 'mushroom', 'potato', 'aloo', 'gobi', 'chana', 'besan',
    'chaap', 'tikka', 'kebab', 'roti', 'naan', 'paratha', 'bread', 'salad', 'chaat'
]

def get_random_banner(archetype, recipe_title=None, is_vegan=False):
    arch = archetype
    if recipe_title:
        t = recipe_title.lower().replace('#', '').replace('*', '').replace('"', '').strip()
        words = [w for w in re.split(r'[\s\-_,]+', t) if w]
        last_1 = words[-1] if words else ''
        last_2 = ' '.join(words[-2:]) if len(words) >= 2 else last_1
        last_3 = ' '.join(words[-3:]) if len(words) >= 3 else last_2
        
        # 1. End-Term / Suffix Culinary Classification (last 1-3 words)
        if any(re.search(r'\b(stew|soup|broth|shorba|chowder|potage|laksa)\b', x) for x in [last_1, last_2]):
            arch = "Soup"
        elif any(re.search(r'\b(rice bowl|rice skillet|fried rice|pulao|pulav|pilaf|biryani|khichdi|rice|skillet)\b', x) for x in [last_2, last_3, last_1]):
            if ('biryani' in last_2 or 'biryani' in last_1) and not is_vegan:
                arch = "Biryani"
            else:
                arch = "Rice_Dish"
        elif any(re.search(r'\b(gravy|curry|korma|masala|saag|makhani|dal|dhal|daal|sambar|rasam|kofta|kadhi|karhi|salan|jalfrezi|handi)\b', x) for x in [last_1, last_2]):
            arch = "Curry"
        elif any(re.search(r'\b(roast|fry|stir-fry|sukka|poriyal|bhaji|thoran|dry|pakora|pakoras|pakoda|pakodas|fritter|fritters|tikka|kebab|tikki|chaap|sabzi|subzi)\b', x) for x in [last_1, last_2]):
            arch = "Dry_Sabzi"
        elif any(re.search(r'\b(roti|naan|paratha|kulcha|puri|poori|bhatura|chapati|phulka|bread|flatbread|wrap|roll)\b', x) for x in [last_1, last_2]):
            arch = "Bread"
        elif any(re.search(r'\b(salad|slaw|chaat|raita|kachumber)\b', x) for x in [last_1, last_2]):
            arch = "Salad"
        else:
            # 2. Whole Title Search Fallback
            is_savory = any(sw in t for sw in SAVORY_WORDS_LIST)
            if not is_vegan and (re.search(r'\b(chicken|mutton|meat|lamb)\b', t) and not re.search(r'\b(rice|biryani)\b', t)):
                arch = "Chicken"
            elif not is_vegan and re.search(r'\b(egg|omelette|anda)\b', t) and not re.search(r'\b(paneer)\b', t):
                arch = "Egg"
            elif not is_vegan and re.search(r'\b(fish|catfish|prawn|prawns|shrimp|shrimps|seafood|pork|beef|bacon|sausage)\b', t):
                arch = "NonVegan"
            elif not is_vegan and re.search(r'\b(biryani)\b', t):
                arch = "Biryani"
            elif re.search(r'\b(stew|soup|broth|shorba)\b', t):
                arch = "Soup"
            elif re.search(r'\b(rice|pulao|khichdi|skillet)\b', t):
                arch = "Rice_Dish"
            elif re.search(r'\b(curry|gravy|masala|korma|dal|kadhi|kofta|paneer)\b', t):
                arch = "Curry"
            elif re.search(r'\b(roast|sukka|sabzi|pakora|tikka|chaap)\b', t):
                arch = "Dry_Sabzi"
            elif re.search(r'\b(bread|roti|naan|paratha)\b', t):
                arch = "Bread"
            elif re.search(r'\b(salad|chaat)\b', t):
                arch = "Salad"
            elif not is_savory and any(re.search(r'\b(cake|kheer|halwa|payasam|pudding|cookie|cookies|brownie|brownies|pastry|muffin|ladoo|laddu|barfi|dessert)\b', x) for x in [last_1, last_2, t]):
                arch = "Dessert"
            
    # Strict Savory vs. Dessert Barrier: A savory dish must NEVER have a dessert banner
    is_savory_dish = any(sw in (recipe_title or '').lower() for sw in SAVORY_WORDS_LIST)
    if arch == "Dessert" and is_savory_dish:
        arch = "Curry" if not is_vegan else "Vegan"
            
    # Strict Vegan Protection: A vegan/plant dish must NEVER have meat/egg/biryani banners
    if is_vegan and arch in ["Chicken", "Egg", "Biryani", "NonVegan"]:
        arch = "Rice_Dish" if ("rice" in (recipe_title or "").lower() or "biryani" in (recipe_title or "").lower() or "pulao" in (recipe_title or "").lower()) else "Vegan"
        
    arch = arch if arch in UNSPLASH_CACHE else ("Vegan" if is_vegan else "Curry")
    photo_id = random.choice(UNSPLASH_CACHE[arch])
    return f"https://images.unsplash.com/photo-{photo_id}?q=80&w=1632&auto=format&fit=crop"

def normalize_recipe_text(ai_text: str) -> str:
    """
    Cleans raw model recipe output across all endpoints:
    1. Removes LLM stop tokens & markdown artifacts
    2. Strips leading/trailing asterisks, quotes, and hashes from title
    3. Strips duplicated ingredients block / ingredients headings from directions
    4. Formats as '### TITLE: Dish Name\\n### DIRECTIONS:\\n1. ...'
    """
    if not ai_text:
        return ""
    import re
    text = ai_text.strip()
    
    stop_tokens = ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>', '\n### INGREDIENTS:']
    for t in stop_tokens:
        if t in text:
            text = text.split(t)[0].strip()
            
    if "### TITLE:\n" in text:
        text = text.split("### TITLE:\n")[1].strip()
    elif text.startswith("### TITLE:"):
        text = text[len("### TITLE:"):].strip()

    # Cut off standard looping signatures and Notes
    text = re.split(r'\n\d+\.\s*(?:Variation|Note|To serve|Serve|Enjoy)', text, flags=re.IGNORECASE)[0]
    cut_phrases = [
        "\nEnjoy!", "\nServe hot", "\nBon Apetit", "\nChef's Note:", 
        "\nVariations:", "\nServing suggestion:", "\nNote:",
        "\nVariation #", "\nTo serve"
    ]
    for phrase in cut_phrases:
        if phrase.lower() in text.lower():
            text = re.split(re.escape(phrase), text, flags=re.IGNORECASE)[0].strip()

    # Separate title and directions
    raw_lines = text.split("\n")
    first_line = raw_lines[0].strip() if raw_lines else ""
    clean_title = re.sub(r'^(###\s*(TITLE:)?|TITLE:|\*+|#+|["\'])+', '', first_line)
    clean_title = re.sub(r'(\*+|["\'])+$', '', clean_title).strip()
    if not clean_title:
        clean_title = "Delicious Dish"

    # Directions extraction
    if "### DIRECTIONS:\n" in text or "### DIRECTIONS:" in text:
        dir_part = text.split("### DIRECTIONS:")[1]
    elif "**Directions**" in text:
        dir_part = text.split("**Directions**")[1]
    elif "**Instructions**" in text:
        dir_part = text.split("**Instructions**")[1]
    elif "### INSTRUCTIONS:" in text:
        dir_part = text.split("### INSTRUCTIONS:")[1]
    else:
        dir_part = "\n".join(raw_lines[1:])

    # Clean lines in dir_part: discard ingredients headings and bulleted ingredient items
    cleaned_steps = []
    in_ing_block = False
    for line in dir_part.split("\n"):
        s = line.strip()
        if not s:
            continue
        # Check if line is ingredients heading
        if re.match(r'^(\*\*|###\s*)?(ingredients|ingredients to use):?', s, re.IGNORECASE):
            in_ing_block = True
            continue
        if re.match(r'^(\*\*|###\s*)?(directions|instructions|steps|method):?', s, re.IGNORECASE):
            in_ing_block = False
            continue
        if in_ing_block:
            if re.match(r'^[-*•]\s+', s) or re.match(r'^\d+(\.\d+)?\s*(g|kg|ml|tsp|tbsp|cup)', s, re.IGNORECASE):
                continue
            if re.match(r'^\d+\.\s+', s):
                in_ing_block = False
            else:
                continue
        # Skip bullet points of ingredients
        if re.match(r'^[-*•]\s+\d+(\.\d+)?\s*(g|kg|ml|tsp|tbsp|cup|piece)', s, re.IGNORECASE):
            continue
        # Skip markdown asterisks headings like "**Directions**"
        if re.match(r'^\*\*(directions|instructions|method|ingredients)\*\*$', s, re.IGNORECASE):
            continue
        # If there's another "### " in the directions, cut it
        if s.startswith("### "):
            break
        cleaned_steps.append(s)

    return f"### TITLE: {clean_title}\n### DIRECTIONS:\n" + "\n".join(cleaned_steps)

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
    
    # Check local precomputed deconstruction map
    if clean_ingredient in DECONSTRUCTION_MAP:
        decomp = DECONSTRUCTION_MAP[clean_ingredient]
        if isinstance(decomp, dict):
            avg_price = 0
            tot_w = sum(decomp.values())
            if tot_w > 0:
                for sub_ing, weight in decomp.items():
                    sub_lem = lemmatizer.lemmatize(sub_ing.lower())
                    sub_p = current_mandi[current_mandi['Processed_Ingredient'] == sub_lem]['Price_per_Gram'].median()
                    if pd.isna(sub_p): sub_p = pantry_prices.get(sub_ing, 0.2)
                    avg_price += sub_p * (weight / tot_w)
                return round(avg_price * 1.2, 4)

    if skip_llm:
        # Check cache before generic fallback
        cache_id = f"deconstruct_{clean_ingredient.strip().lower()}"
        if llm_cache_sync is not None:
            try:
                cached = llm_cache_sync.find_one({"_id": cache_id})
                if cached and "result" in cached and isinstance(cached["result"], dict):
                    deconstructed = cached["result"]
                    avg_price = 0
                    total_weight = sum(deconstructed.values())
                    if total_weight > 0:
                        for sub_ing, weight in deconstructed.items():
                            sub_lem = lemmatizer.lemmatize(sub_ing.lower())
                            sub_price = current_mandi[current_mandi['Processed_Ingredient'] == sub_lem]['Price_per_Gram'].median()
                            if pd.isna(sub_price): sub_price = pantry_prices.get(sub_ing, 0.2)
                            avg_price += sub_price * (weight / total_weight)
                        return round(avg_price * 1.2, 4)
            except Exception:
                pass
        return pantry_prices.get(clean_ingredient, 0.25)
        
    deconstructed = deconstruct_ingredient(clean_ingredient)
    if deconstructed and isinstance(deconstructed, dict):
        if not all(isinstance(v, (int, float)) for v in deconstructed.values()):
            return 0.25

        avg_price = 0
        total_weight = sum(deconstructed.values())
        if total_weight == 0:
            return 0.25
        for sub_ing, weight in deconstructed.items():
            sub_lem = lemmatizer.lemmatize(sub_ing.lower())
            sub_price = current_mandi[current_mandi['Processed_Ingredient'] == sub_lem]['Price_per_Gram'].median()
            if pd.isna(sub_price): sub_price = pantry_prices.get(sub_ing, 0.2)
            avg_price += sub_price * (weight / total_weight)
        return round(avg_price * 1.2, 4)
    return 0.25

# ============================================================
# CULINARY CONDIMENT CATEGORIZATION & DYNAMIC DOSAGE
# ============================================================
CONDIMENT_CLASSES = {
    "intense_spice": {
        "items": [
            "salt", "black pepper", "white pepper", "chili powder", "cayenne pepper",
            "red pepper flakes", "clove", "cloves", "cinnamon", "cardamom", "cardamoms",
            "nutmeg", "saffron", "saffron strands", "black peppercorns", "peppercorns",
            "red chilli powder", "red chili powder", "chilli powder", "hing", "asafoetida", "star anise"
        ],
        "base_lower": 0.5,
        "base_upper": 2.5
    },
    "aromatic_spice": {
        "items": [
            "garlic powder", "onion powder", "ginger powder", "cumin", "cumin seeds", "jeera",
            "turmeric", "turmeric powder", "haldi", "coriander", "coriander powder", "coriander seeds",
            "paprika", "garam masala", "curry powder", "oregano", "basil", "thyme", "rosemary",
            "dill", "fennel", "fennel seeds", "saunf", "allspice", "mint", "mint leaves", "fresh mint",
            "fenugreek seeds", "methi seeds", "kasuri methi", "curry leaves", "fresh curry leaves",
            "chaat masala", "chana masala", "biryani masala", "kitchen king", "amchur", "dried mango powder",
            "mustard seeds", "rai", "ajwain", "carom seeds", "bay leaf", "bay leaves", "tejpatta",
            "cilantro", "fresh cilantro", "coriander leaves"
        ],
        "base_lower": 1.0,
        "base_upper": 4.0
    },
    "flavor_powder": {
        "items": ["cocoa powder", "baking powder", "baking soda", "vanilla extract"],
        "base_lower": 1.0,
        "base_upper": 5.0
    },
    "wet_condiment": {
        "items": [
            "mustard", "ketchup", "soy sauce", "vinegar", "balsamic vinegar",
            "apple cider vinegar", "red wine vinegar", "lemon juice", "lime juice"
        ],
        "base_lower": 3.0,
        "base_upper": 10.0
    }
}

ARCHETYPE_CONDIMENT_MODIFIERS = {
    "Curry": {
        "intense_spice": 1.0,
        "aromatic_spice": 1.2,
        "flavor_powder": 0.8,
        "wet_condiment": 0.8
    },
    "Soup": {  # Covers stews, soups, broths
        "intense_spice": 1.1,
        "aromatic_spice": 0.8,
        "flavor_powder": 0.8,
        "wet_condiment": 0.9
    },
    "Dry_Sabzi": {
        "intense_spice": 0.85,
        "aromatic_spice": 1.0,
        "flavor_powder": 0.5,
        "wet_condiment": 0.7
    },
    "Salad": {
        "intense_spice": 0.8,
        "aromatic_spice": 0.5,
        "flavor_powder": 0.5,
        "wet_condiment": 1.3
    },
    "Rice_Dish": {
        "intense_spice": 1.1,
        "aromatic_spice": 1.0,
        "flavor_powder": 0.5,
        "wet_condiment": 0.6
    },
    "Dessert": {
        "intense_spice": 0.8,
        "aromatic_spice": 0.3,
        "flavor_powder": 1.4,
        "wet_condiment": 0.3
    }
}

def get_condiment_category(clean_name: str):
    clean = clean_name.lower().strip()
    for cat, data in CONDIMENT_CLASSES.items():
        for item in data["items"]:
            if item == clean or clean.startswith(item + " ") or clean.endswith(" " + item) or f" {item} " in f" {clean} ":
                return cat
    return None

def get_condiment_bounds(clean_name: str, archetype: str = "Curry", servings: int = 1):
    category = get_condiment_category(clean_name)
    if not category:
        return None
    cfg = CONDIMENT_CLASSES[category]
    arch_mod = ARCHETYPE_CONDIMENT_MODIFIERS.get(archetype, {}).get(category, 1.0)
    
    # Sub-linear serving scaling: seasonings scale with batch size, not pure linear
    serving_scale = float(servings) ** 0.82
    lower = round(cfg["base_lower"] * (float(servings) ** 0.75), 1)
    upper = round(cfg["base_upper"] * serving_scale * arch_mod, 1)
    if lower > upper:
        lower = upper
    return (max(0.5, lower), max(1.0, upper))

def tag_ingredient(ing_name):
    # Check if it's a condiment or seasoning first!
    if get_condiment_category(ing_name) is not None:
        return 'condiment'
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
# SMART MULTI-FACTOR ARCHETYPE CLASSIFIER (no LLM)
# ============================================================
def _classify_archetype_fast(ingredients: list) -> str:
    """Classify dish archetype using robust multi-factor token semantics."""
    cleaned = [parse_ingredient_input(i)[1].lower().strip() for i in ingredients]
    all_text = " " + " ".join(cleaned) + " "
    
    # 1. Detect Compound tokens & avoid substring collisions
    has_coconut_milk = "coconut milk" in all_text or "coconut cream" in all_text
    has_sweet_potato = "sweet potato" in all_text
    
    # 2. Check for savory staples (vegetables, proteins, bases)
    savory_veg = ["potato", "zucchini", "mushroom", "tomato", "onion", "garlic", "spinach", 
                  "cabbage", "carrot", "bell pepper", "eggplant", "cauliflower", "broccoli", "radish"]
    has_savory_veg = any(re.search(r'\b' + re.escape(v) + r'\b', all_text) for v in savory_veg) or has_sweet_potato
    
    proteins = ["chicken", "paneer", "tofu", "egg", "fish", "meat", "mutton", "pork", "beef", "shrimp", "dal", "lentil", "chickpea"]
    has_protein = any(re.search(r'\b' + re.escape(p) + r'\b', all_text) for p in proteins)
    
    # 3. Rice Dishes
    if any(re.search(r'\b' + re.escape(r) + r'\b', all_text) for r in ["rice", "biryani", "pulao", "fried rice", "pasta", "noodle", "macaroni"]):
        return "Rice_Dish"
        
    # 4. Stews / Soups
    # If coconut milk is paired with savory veggies or if soup/broth words are present:
    if any(re.search(r'\b' + re.escape(w) + r'\b', all_text) for w in ["soup", "broth", "stew"]):
        return "Soup"
    if has_coconut_milk and has_savory_veg:
        return "Soup"
        
    # 5. Salads
    if any(re.search(r'\b' + re.escape(w) + r'\b', all_text) for w in ["lettuce", "salad", "cucumber", "olive"]):
        return "Salad"
        
    # 6. Desserts (STRICT: only if savory veg and proteins are absent!)
    has_sweet = any(re.search(r'\b' + re.escape(w) + r'\b', all_text) for w in ["sugar", "chocolate", "cocoa powder", "cocoa", "honey", "jaggery", "vanilla", "cake", "kheer", "halwa"])
    if has_sweet and not (has_savory_veg or has_protein):
        return "Dessert"

    # 7. Breads
    if any(re.search(r'\b' + re.escape(w) + r'\b', all_text) for w in ["bread", "dough", "yeast", "flour", "atta", "maida"]) and not (has_savory_veg or has_protein):
        return "Bread"
        
    if not (has_savory_veg or has_protein):
        # Exclude coconut milk from milk check
        text_without_coconut = all_text.replace("coconut milk", "").replace("coconut cream", "")
        if any(re.search(r'\b' + re.escape(w) + r'\b', text_without_coconut) for w in ["oat", "banana", "milk"]):
            return "Dessert"
            
    # 8. Dry Sabzi
    if has_savory_veg and not has_coconut_milk and not any(w in all_text for w in ["curry", "gravy"]):
        return "Dry_Sabzi"
        
    return "Curry"

# ============================================================
# SCIPY LINEAR PROGRAMMING OPTIMIZER WITH DYNAMIC HEADROOM
# ============================================================
def optimize_recipe_v2(raw_user_ingredients, total_budget, servings, user_state, archetype=None, skip_llm=False, return_prices=False):
    if archetype is None:
        archetype = _classify_archetype_fast(raw_user_ingredients)
        
    servings_int = max(1, int(servings))
    total_budget_float = float(total_budget)
    
    # 1. Parse and price all ingredients
    parsed_items = []
    prices = []
    tags = []
    unit_prices = {}
    
    bulk_indices = []
    condiment_indices = []
    
    for idx, raw_ing in enumerate(raw_user_ingredients):
        user_qty, clean_name = parse_ingredient_input(raw_ing)
        p = get_dynamic_price(clean_name, user_state, skip_llm=skip_llm)
        t = tag_ingredient(clean_name)
        
        parsed_items.append((user_qty, clean_name, raw_ing))
        prices.append(p)
        tags.append(t)
        unit_prices[clean_name] = p
        
        if t == 'condiment':
            condiment_indices.append(idx)
        else:
            bulk_indices.append(idx)

    # 2. Dynamic Condiment Sizing & Headroom Calculation
    condiment_results = {}
    condiment_headroom = 0.0
    
    for c_idx in condiment_indices:
        user_qty, clean_name, _ = parsed_items[c_idx]
        p = prices[c_idx]
        cond_b = get_condiment_bounds(clean_name, archetype=archetype, servings=servings_int)
        
        if user_qty is not None:
            assigned_qty = float(user_qty)
        elif cond_b is not None:
            assigned_qty = round((cond_b[0] + cond_b[1]) / 2.0, 1)
        else:
            assigned_qty = round(3.0 * (servings_int ** 0.8), 1)
            
        max_b = cond_b[1] if cond_b else assigned_qty * 1.5
        condiment_headroom += max_b * p
        condiment_results[c_idx] = assigned_qty

    # 3. Reserve Condiment Headroom from Total Budget
    reserved_headroom = round(min(condiment_headroom, total_budget_float * 0.35), 2)
    available_bulk_budget = max(10.0, total_budget_float - reserved_headroom)
    
    # 4. Optimize Bulk Ingredients with SciPy linprog
    final_quantities_by_idx = {}
    
    if bulk_indices:
        bulk_prices = [prices[i] for i in bulk_indices]
        bulk_tags = [tags[i] for i in bulk_indices]
        bulk_bounds = []
        
        for b_idx in bulk_indices:
            user_qty, clean_name, _ = parsed_items[b_idx]
            if user_qty is not None:
                bulk_bounds.append((user_qty / servings_int, user_qty / servings_int))
            else:
                lower, upper = bounds_dict.get(lemmatizer.lemmatize(clean_name.split()[-1]), (15, 300))
                lower = max(lower, 15.0)
                upper = min(upper, 250.0)
                if lower > upper:
                    lower = upper
                bulk_bounds.append((lower, upper))
                
        c_obj = [-1] * len(bulk_indices)
        A_ub = [bulk_prices]
        b_ub = [available_bulk_budget / servings_int]
        
        if archetype == "Curry":
            if 'protein' in bulk_tags and 'base' in bulk_tags:
                A_ub.append([0.8 if t == 'protein' else -1.0 if t == 'base' else 0.0 for t in bulk_tags])
                b_ub.append(0)
                A_ub.append([-3.0 if t == 'protein' else 1.0 if t == 'base' else 0.0 for t in bulk_tags])
                b_ub.append(0)
        elif archetype == "Dry_Sabzi":
            if 'protein' in bulk_tags and 'base' in bulk_tags:
                A_ub.append([-0.8 if t == 'protein' else 1.0 if t == 'base' else 0.0 for t in bulk_tags])
                b_ub.append(0)
        elif archetype == "Salad":
            if ('protein' in bulk_tags or 'neutral' in bulk_tags) and 'veggie' in bulk_tags:
                A_ub.append([2.0 if t in ['protein', 'neutral'] else -1.0 if t == 'veggie' else 0.0 for t in bulk_tags])
                b_ub.append(0)
        elif archetype == "Rice_Dish":
            if 'neutral' in bulk_tags and 'protein' in bulk_tags:
                A_ub.append([1.0 if t == 'protein' else -1.0 if t == 'neutral' else 0.0 for t in bulk_tags])
                b_ub.append(0)
        elif archetype == "Soup":
            if 'protein' in bulk_tags and 'base' in bulk_tags:
                A_ub.append([4.0 if t == 'protein' else -1.0 if t == 'base' else 0.0 for t in bulk_tags])
                b_ub.append(0)
        elif archetype == "Dessert":
            if 'sweet' in bulk_tags:
                A_ub.append([-0.4 if t in ['neutral', 'protein', 'base'] else 1.0 if t == 'sweet' else 0.0 for t in bulk_tags])
                b_ub.append(0)
                
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bulk_bounds, method='highs')
        
        if not res.success:
            A_ub_fb = [bulk_prices]
            b_ub_fb = [available_bulk_budget / servings_int]
            res = linprog(c_obj, A_ub=A_ub_fb, b_ub=b_ub_fb, bounds=bulk_bounds, method='highs')
            
        if not res.success:
            em_bounds = [(5.0, max(b[1], 10.0)) for b in bulk_bounds]
            A_ub_fb = [bulk_prices]
            b_ub_fb = [total_budget_float / servings_int]
            res = linprog(c_obj, A_ub=A_ub_fb, b_ub=b_ub_fb, bounds=em_bounds, method='highs')
            
        if res.success:
            grams_per_serving = np.round(res.x, 1)
            for idx_in_bulk, b_idx in enumerate(bulk_indices):
                final_quantities_by_idx[b_idx] = grams_per_serving[idx_in_bulk] * servings_int
        else:
            for b_idx in bulk_indices:
                final_quantities_by_idx[b_idx] = 100.0 * servings_int
    
    # 5. Fine-Tune Condiment Quantities to fit remaining headroom
    bulk_cost = sum(final_quantities_by_idx.get(b_idx, 0.0) * prices[b_idx] for b_idx in bulk_indices)
    remaining_for_condiments = max(0.0, total_budget_float - bulk_cost)
    
    cur_condiment_cost = sum(condiment_results.get(c_idx, 0.0) * prices[c_idx] for c_idx in condiment_indices)
    if cur_condiment_cost > remaining_for_condiments and cur_condiment_cost > 0:
        scale_down = max(0.4, remaining_for_condiments / cur_condiment_cost)
        for c_idx in condiment_indices:
            condiment_results[c_idx] = round(condiment_results[c_idx] * scale_down, 1)
            
    for c_idx, q in condiment_results.items():
        final_quantities_by_idx[c_idx] = q

    # 6. Recombine all ingredients in original order
    final_ingredients = []
    for idx, (user_qty, clean_name, raw_ing) in enumerate(parsed_items):
        final_g = round(final_quantities_by_idx.get(idx, 10.0), 1)
        final_ingredients.append(f"{final_g}g {clean_name}")
        
    optimize_recipe_v2.last_unit_prices = unit_prices
    
    if return_prices:
        return final_ingredients, archetype, unit_prices
    return final_ingredients, archetype

def calculate_cost_breakdown(calculated_ingredients, user_state="Delhi", servings=1, total_budget=150.0, unit_prices=None, **kwargs):
    """
    Computes an itemized cost breakdown for each calculated ingredient.
    """
    if "budget" in kwargs and kwargs["budget"] is not None:
        total_budget = kwargs["budget"]
    if not calculated_ingredients:
        return None
        
    if unit_prices is None and hasattr(optimize_recipe_v2, "last_unit_prices"):
        unit_prices = getattr(optimize_recipe_v2, "last_unit_prices", {}) or {}
        
    items = []
    total_cost = 0.0
    for raw_ing in calculated_ingredients:
        qty, name = parse_ingredient_input(raw_ing)
        clean_name = canonicalize_ingredient(name.lower().strip())
        
        if unit_prices and clean_name in unit_prices:
            unit_price_per_gram = unit_prices[clean_name]
        else:
            unit_price_per_gram = get_dynamic_price(clean_name, user_state, skip_llm=True)
            
        grams = float(qty) if qty is not None else 100.0
        cost = round(grams * unit_price_per_gram, 2)
        total_cost += cost
        items.append({
            "name": clean_name,
            "grams": round(grams, 1),
            "rate_per_100g": round(unit_price_per_gram * 100.0, 2),
            "cost": cost
        })
        
    total_cost = round(total_cost, 2)
    budget_val = float(total_budget) if total_budget is not None else 0.0
    
    if budget_val > 0 and total_cost > budget_val and (total_cost - budget_val) < 1.0:
        total_cost = budget_val
        
    headroom = round(max(0.0, budget_val - total_cost), 2)
    return {
        "items": items,
        "total_cost": total_cost,
        "total_spent": total_cost,
        "budget": budget_val,
        "headroom": headroom,
        "servings": int(servings)
    }

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
    valid_recipes = []
    for r in recipes:
        r["_id"] = str(r["_id"])
        rec_data = r.get("recipe", {})
        if isinstance(rec_data, dict):
            cost_info = rec_data.get("cost_breakdown", {})
            spent = cost_info.get("total_spent") or cost_info.get("total_cost")
            budget = rec_data.get("budget")
            if spent is not None and budget is not None:
                try:
                    if float(spent) > float(budget) + 0.5:
                        continue
                except Exception:
                    pass
        valid_recipes.append(r)
        
    return {"status": "success", "recipes": valid_recipes}

@app.get("/all-recipes")
async def get_all_recipes():
    if recipes_collection is None:
        return {"status": "error", "message": "Database not connected on the server."}
    
    # Fetch all community recipes sorted by newest first
    cursor = recipes_collection.find({}).sort("created_at", -1)
    recipes = await cursor.to_list(length=200)
    
    valid_recipes = []
    for r in recipes:
        r["_id"] = str(r["_id"])
        # Guard: Filter out recipes where cost exceeded target budget
        rec_data = r.get("recipe", {})
        if isinstance(rec_data, dict):
            cost_info = rec_data.get("cost_breakdown", {})
            spent = cost_info.get("total_spent") or cost_info.get("total_cost")
            budget = rec_data.get("budget")
            if spent is not None and budget is not None:
                try:
                    if float(spent) > float(budget) + 0.5:
                        continue
                except Exception:
                    pass
        valid_recipes.append(r)
        
    return {"status": "success", "recipes": valid_recipes}


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
                            print(f"[LOCAL HIT] '{canonical_name}' -> '{blueprint.get('best_vegan_substitute', 'N/A')}'")
                        else:
                            # Truly unknown ingredient — keep as-is, no LLM call
                            print(f"[SKIP] '{canonical_name}' not in DB and no static fallback - keeping as-is")
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

        # ==========================================
        # AGENTIC PIPELINE: Phase 2 Context Injection
        # ==========================================
        # Retrieve Graph Context
        if graph_retriever is not None:
            yield f"data: {json.dumps({'step': 'retrieving_context', 'message': 'Retrieving Context from Hybrid Graph...'})}\n\n"
            step_start = time.time()
            # Extract just the ingredient names for retrieval, ignoring quantities
            retrieval_ings = [parse_ingredient_input(r)[1] for r in clean_ingredients]
            graph_context = graph_retriever.retrieve_context(retrieval_ings)
            injected_context = graph_retriever.generate_prompt_injection(graph_context)
            
            # Phase 3: Hybrid Search Few-Shot Injection
            few_shot_examples = graph_retriever.retrieve_few_shot_examples(graph_context, archetype, k=2)
            if few_shot_examples:
                injected_context += "\n<FEW_SHOT_EXAMPLES>\n"
                injected_context += "Here are historically accurate examples of how this archetype is prepared:\n\n"
                print(f"[FAISS] Successfully retrieved {len(few_shot_examples)} Semantic Vector Examples!")
                for idx, ex in enumerate(few_shot_examples):
                    # Extract just the title for the terminal log to avoid spamming the console
                    ex_title = ex.split('\n')[0].replace('TITLE: ', '')
                    print(f"   -> Example {idx+1}: {ex_title}")
                    injected_context += f"Example {idx+1}:\n{ex}\n\n"
                injected_context += "</FEW_SHOT_EXAMPLES>\n"
                
            times['retrieval_sec'] = time.time() - step_start
        else:
            injected_context = ""
            graph_context = {}

        # EXACT V10 PROMPT STRUCTURE (With Agentic Context)
        base_prompt = (
            f"<|begin_of_text|>System: You are a strict chef. You MUST explicitly use EVERY single ingredient provided in the list below in your recipe directions.\n"
            f"CRITICAL RULES:\n"
            f"1. DO NOT copy the FEW_SHOT_EXAMPLES. They are ONLY for structural reference.\n"
            f"2. DO NOT output variations, alternative fillings, personal notes, or reviews.\n"
            f"3. Output ONLY the TITLE and the numbered DIRECTIONS.\n\n"
            f"{injected_context}\n\n"
            f"### INGREDIENTS:\n"
            f"{ingr_text}\n"
            f"### TITLE:\n"
        )
        print(f"[AI] Generating final recipe for archetype: {archetype} using model_version={request.model_version}")

        # ==========================================
        # AGENTIC PIPELINE: Iterative Self-Correction Loop
        # ==========================================
        max_retries = 3
        ai_text = ""
        current_prompt = base_prompt
        chosen_client = _clients.get(request.model_version, _clients["v8"])()
        
        initial_cvs_score = None
        final_cvs_score = None
        
        for attempt in range(1, max_retries + 1):
            yield f"data: {json.dumps({'step': 'generating', 'message': f'Generating AI Recipe (Attempt {attempt}/{max_retries})...'})}\n\n"
            step_start = time.time()
            
            ai_text = query_hf_model(
                current_prompt,
                max_new_tokens=500,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.05,
                client=chosen_client,
            )
            
            times[f'generation_sec_attempt_{attempt}'] = time.time() - step_start
            
            if recipe_judge is None:
                break # Fallback if judge is missing
                
            yield f"data: {json.dumps({'step': 'judging', 'message': f'Micro-Validation: Judging Recipe...'})}\n\n"
            judge_start = time.time()
            judge_result = recipe_judge.evaluate_recipe(ai_text, graph_context)
            times[f'judge_sec_attempt_{attempt}'] = time.time() - judge_start
            
            if judge_result.get("is_valid"):
                print(f"[JUDGE] Passed on attempt {attempt}: {judge_result.get('critique')}")
                if initial_cvs_score is None: initial_cvs_score = judge_result.get("score")
                final_cvs_score = judge_result.get("score")
                break
            else:
                print(f"[JUDGE] Failed on attempt {attempt}: {judge_result.get('critique')}")
                if initial_cvs_score is None: initial_cvs_score = judge_result.get("score")
                final_cvs_score = judge_result.get("score")
                if attempt < max_retries:
                    yield f"data: {json.dumps({'step': 'correcting', 'message': f'Critique received. Regenerating Recipe...'})}\n\n"
                    # Feed the critique back to the model as an instruction block
                    current_prompt = (
                        f"{base_prompt}\n"
                        f"[PREVIOUS DRAFT REJECTED BY JUDGE]:\n{ai_text}\n"
                        f"[JUDGE CRITIQUE TO FIX]: {judge_result.get('critique')}\n"
                        f"Rewrite the recipe, strictly following the critique.\n"
                        f"### TITLE:\n"
                    )

        print(f"DEBUG RAW AI_TEXT: {repr(ai_text)}")

        ai_text = normalize_recipe_text(ai_text)
            
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
                    "ingredient_count": len(clean_ingredients),
                    "initial_cvs_score": initial_cvs_score,
                    "final_cvs_score": final_cvs_score,
                    "self_correction_attempts": attempt, "latency": total_time, "model": request.model_version, "budget": request.budget, "city": request.state
                }
                generation_logs_sync.insert_one(log_entry)
                print(f"[OK] Generation log saved to DB. Total time: {total_time:.2f}s")
            except Exception as e:
                print(f"[WARN] Failed to save generation log: {e}")

        clean_title = ""
        title_match = re.search(r'### TITLE:\s*(.+)', ai_text)
        if title_match:
            clean_title = re.sub(r'[*_#"`]', '', title_match.group(1)).strip()

        final_result = {
            "status": "success",
            "archetype": archetype,
            "calculated_ingredients": calculated_ingredients,
            "recipe": ai_text,
            "image_url": get_random_banner(archetype, recipe_title=clean_title, is_vegan=request.is_vegan),
            "is_vegan": request.is_vegan,
            "initial_cvs_score": initial_cvs_score,
            "final_cvs_score": final_cvs_score,
            "self_correction_attempts": attempt, "latency": total_time, "model": request.model_version, "budget": request.budget, "city": request.state,
            "cost_breakdown": calculate_cost_breakdown(calculated_ingredients, request.state, request.servings, request.budget)
        }
        yield f"data: {json.dumps({'step': 'complete', 'result': final_result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ============================================================
# VEGAN AGENTIC PIPELINE (GROQ)
# ============================================================
class VeganGraphRetriever:
    """Simulates a specialized Vegan Knowledge Graph using Agentic LLM-Based Retrieval."""
    def __init__(self):
        self.map_path = os.path.join(os.path.dirname(__file__), "data", "vegan_knowledge_map.json")
        self.matrix_path = os.path.join(os.path.dirname(__file__), "data", "vegan_functional_matrix.json")
        self.knowledge_map = []
        self.functional_matrix = {}
        
        # Load Knowledge Map (The RAG chunks from scraped + curated database)
        try:
            if os.path.exists(self.map_path):
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.knowledge_map = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load vegan_knowledge_map.json: {e}")
            
        # Load JSON Matrix
        try:
            if os.path.exists(self.matrix_path):
                with open(self.matrix_path, "r", encoding="utf-8") as f:
                    self.functional_matrix = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load vegan_functional_matrix.json: {e}")

    def get_context(self, ingredients):
        injected_rules = []
        
        # 1. Structural Mapping Injection (Fast dictionary lookup)
        for ing in ingredients:
            ing_lower = ing.lower()
            for key, data in self.functional_matrix.items():
                if key in ing_lower:
                    rule = f"CRITICAL MAPPING FOR {key.upper()}: Substitute with {data['substitute']}. {data['context']} Function: {data['function']}."
                    if rule not in injected_rules:
                        injected_rules.append(rule)
        
        # 2. Scraped Knowledge Candidate Matching + LLM Selection
        if self.knowledge_map and ingredients:
            matched_chunks = []
            ing_tokens = set()
            for ing in ingredients:
                ing_tokens.update(re.findall(r'\b\w+\b', ing.lower()))
            
            for chunk in self.knowledge_map:
                chunk_lower = chunk.lower()
                if any(t in chunk_lower for t in ing_tokens if len(t) > 2):
                    matched_chunks.append(chunk)
            
            candidates = matched_chunks[:10] if matched_chunks else self.knowledge_map[:10]
            
            try:
                client = get_inference_client()
                techniques_text = "\n".join([f"[{i}] {chunk}" for i, chunk in enumerate(candidates)])
                system_prompt = "You are a Vegan Culinary Retrieval Agent. Return ONLY the indices of the 2 most relevant cooking techniques/recipes for the ingredients."
                user_prompt = f"Ingredients: {', '.join(ingredients)}\n\nTechniques:\n{techniques_text}\n\nReturn ONLY a comma-separated list of the 2 best integer indices (e.g., '1, 3'). Do not write any other text."
                
                res, _ = call_groq_with_fallback(
                    client=client,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=20,
                    temperature=0.0
                )
                output = res.choices[0].message.content.strip()
                indices = [int(x) for x in re.findall(r'\d+', output)]
                for idx in indices[:2]:
                    if 0 <= idx < len(candidates):
                        rule = f"CULINARY RAG TECHNIQUE: {candidates[idx]}"
                        if rule not in injected_rules:
                            injected_rules.append(rule)
            except Exception as e:
                print(f"Agentic Retrieval LLM selection failed: {e}. Falling back to top candidate.")
                if candidates:
                    injected_rules.append(f"CULINARY RAG TECHNIQUE: {candidates[0]}")

        injected_rules.append("GENERAL RULE: Restrict all substitutions to easily accessible Indian ingredients (e.g., Tofu, Soy Chunks/Nutrela, Besan, Cashews, Coconut, Jackfruit). Avoid expensive or rare Western vegan products.")

        return "<VEGAN_STRUCTURAL_CONTEXT>\n" + "\n".join(injected_rules) + "\n</VEGAN_STRUCTURAL_CONTEXT>"

class VeganRecipeJudge:
    def __init__(self):
        self.non_vegan_culprits = [
            'butter', 'ghee', 'cheese', 'paneer', 'egg', 'eggs', 'honey', 'milk', 
            'cream', 'chicken', 'meat', 'beef', 'pork', 'mutton', 'fish', 'curd', 
            'yogurt', 'gelatin', 'whey'
        ]
        self.safe_vegan_patterns = [
            r'\b(?:coconut|almond|cashew|soy|oat|rice|hemp|plant)\s+milk\b',
            r'\b(?:peanut|almond|cashew|vegan|nut)\s+butter\b',
            r'\b(?:cashew|coconut|vegan|soy|plant)\s+cream\b',
            r'\b(?:vegan|dairy-free|plant-based)\s+(?:cheese|paneer|ghee|butter|curd|yogurt|meat|chicken)\b',
            r'\b(?:soy|tofu)\s+curd\b',
            r'\bsoya\s+chunks\b',
            r'\bsoya\s+keema\b',
            r'\bflaxseed\s+egg\b',
            r'\bbesan\s+egg\b'
        ]
        
    def evaluate_recipe(self, recipe_text):
        recipe_lower = recipe_text.lower()
        
        # 1. Zero-Tolerance Hallucination Check (Pass 1)
        # Pre-mask safe vegan compound phrases so "coconut milk" isn't flagged as animal "milk"
        masked_text = recipe_lower
        for pattern in self.safe_vegan_patterns:
            masked_text = re.sub(pattern, ' __SAFE_VEGAN__ ', masked_text, flags=re.IGNORECASE)
            
        hallucinated = [
            c for c in self.non_vegan_culprits 
            if re.search(r'\b' + c + r'\b', masked_text)
        ]
        
        if hallucinated:
            return {
                "score": 0.2,
                "is_valid": False,
                "critique": f"FATAL ERROR: Found non-vegan animal ingredient(s): {', '.join(set(hallucinated))}. You MUST remove them immediately and substitute with plant-based alternatives (e.g. tofu for paneer, soya chunks for chicken/meat, coconut milk for cream/milk, mustard or coconut oil for ghee/butter)."
            }
            
        # 2. Structural Physics & Cultural Accessibility Check (Pass 2 - LLM Judge)
        prompt = (
            "You are a strict Vegan Culinary Judge specializing in Indian cuisine.\n"
            "Evaluate the following recipe draft based on these criteria:\n"
            "1. NO ANIMAL PRODUCTS: Are there any hidden animal products or animal fats? (If yes, score < 0.4)\n"
            "2. ACCESSIBILITY: Did they use hard-to-find Western vegan products (like Aquafaba, Seitan, Nutritional Yeast) instead of common Indian ones (like Besan, Soy Chunks, Cashews, Mustard Oil)? (If yes, score < 0.75)\n"
            "3. STRUCTURAL PHYSICS: Did the substitutions maintain the fat and moisture balance? (e.g., replacing ghee with water is bad; replacing paneer with tofu is good if pressed; rehydrating soya chunks properly).\n\n"
            f"[RECIPE DRAFT]:\n{recipe_text}\n\n"
            "Provide a 'Culinary Validity Score' from 0.0 to 1.0.\n"
            "If the score is less than 0.85, provide a strict 1-sentence CRITIQUE explaining what must be fixed.\n"
            "Output format:\n"
            "SCORE: [score]\n"
            "CRITIQUE: [critique if any]"
        )
        
        try:
            client = get_inference_client()
            res, _ = call_groq_with_fallback(
                client=client,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            response = res.choices[0].message.content.strip()
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response)
            critique_match = re.search(r'CRITIQUE:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            critique = critique_match.group(1).strip() if critique_match else "No critique provided."
            is_valid = score >= 0.85
            return {"score": score, "critique": critique, "is_valid": is_valid}
        except Exception as e:
            print(f"[WARN] Vegan LLM Judge fallback: {e}")
            return {"score": 0.9, "critique": "Passed heuristic checks. LLM judge unavailable.", "is_valid": True}

vegan_retriever = VeganGraphRetriever()
vegan_judge = VeganRecipeJudge()

@app.post("/generate-agentic-vegan")
async def generate_agentic_vegan(request: RecipeRequest):
    start_time = time.time()
    
    clean_ingredients = []
    for item in request.ingredients:
        clean_ingredients.extend([i.strip() for i in item.split(',') if i.strip()])

    # Run Cost Constraint Optimization if budget provided
    cost_breakdown = None
    calculated_ingredients = clean_ingredients
    
    if request.budget and request.budget > 0:
        opt_ingredients = []
        for raw_ing in clean_ingredients:
            qty, name = parse_ingredient_input(raw_ing)
            name_lower = name.lower().strip()
            sub_found = None
            for key, data in vegan_retriever.functional_matrix.items():
                if key in name_lower:
                    sub_found = data['substitute'].split('(')[0].split('or')[0].strip()
                    break
            final_sub = sub_found if sub_found else name
            opt_ingredients.append(f"{qty}g {final_sub}" if qty else final_sub)

        opt_res, arch = optimize_recipe_v2(opt_ingredients, request.budget, request.servings, request.state, archetype="Curry", skip_llm=True)
        if opt_res:
            calculated_ingredients = opt_res
            cost_breakdown = calculate_cost_breakdown(calculated_ingredients, request.state, request.servings, request.budget)
            
    ingr_text = "\n".join(f"- {i}" for i in calculated_ingredients)
    vegan_context = vegan_retriever.get_context(clean_ingredients)
    
    base_prompt = (
        f"System: You are an expert Indian vegan chef. The user wants to cook a dish using the ingredients below.\n"
        f"YOUR MISSION: You MUST ensure 100% plant-based compliance and replace any animal products using the exact RAG rules below. DO NOT use animal products (dairy, ghee, meat, eggs) in the final recipe.\n"
        f"CRITICAL RULES:\n"
        f"1. DO NOT output variations, reviews, or personal notes.\n"
        f"2. Output ONLY the TITLE on the first line (e.g. '### TITLE: Dish Name') and the numbered DIRECTIONS under '### DIRECTIONS:'.\n"
        f"3. DO NOT repeat or print the ingredients list in your directions output. Jump straight into the numbered cooking directions.\n\n"
        f"{vegan_context}\n\n"
        f"### INGREDIENTS TO USE:\n{ingr_text}\n\n### TITLE:\n"
    )
    
    current_prompt = base_prompt
    final_recipe = ""
    attempts = 0
    final_judge_result = {"score": 1.0, "critique": "Passed."}
    
    client = get_inference_client()
    target_model = get_groq_recipe_model(client)
    actual_model = target_model
            
    for attempt in range(1, 4):
        res, actual_model = call_groq_with_fallback(
            client=client,
            messages=[{"role": "user", "content": current_prompt}],
            preferred_model=target_model,
            max_tokens=500,
            temperature=0.6
        )
        recipe_text = res.choices[0].message.content.strip()
        
        # Format and normalize recipe output
        clean_text = normalize_recipe_text(recipe_text)
            
        judge_result = vegan_judge.evaluate_recipe(clean_text)
        final_judge_result = judge_result
        attempts = attempt
        final_recipe = clean_text
        
        if judge_result.get('score', 0) >= 0.85:
            break
        else:
            current_prompt = base_prompt + f"\n\n[JUDGE CRITIQUE TO FIX]: {judge_result.get('critique')}\nRewrite the recipe to fix this error immediately."
            
    latency = time.time() - start_time
    
    def generate():
        yield f"data: {json.dumps({'step': 'retrieving_context', 'message': 'Querying Vegan Structural Knowledge Graph...'})}\n\n"
        time.sleep(0.3)
        if request.budget and request.budget > 0:
            yield f"data: {json.dumps({'step': 'optimizing', 'message': f'Optimizing Ingredients for Budget: ₹{request.budget}...'})}\n\n"
            time.sleep(0.3)
        if attempts > 1:
            j_score = final_judge_result.get('score', 0.9)
            yield f"data: {json.dumps({'step': 'judging', 'message': f'Vegan Judge Self-Corrected in {attempts} iterations (Score: {j_score})...'})}\n\n"
            time.sleep(0.3)
            
        yield f"data: {json.dumps({'step': 'final_recipe', 'recipe': final_recipe, 'calculated_ingredients': calculated_ingredients, 'cost_breakdown': cost_breakdown, 'archetype': 'Agentic Vegan', 'is_vegan': True, 'attempts': attempts, 'judge_score': final_judge_result.get('score', 0.9), 'judge_critique': final_judge_result.get('critique', ''), 'latency': latency, 'model': actual_model, 'budget': request.budget, 'city': request.state})}\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/generate-fast-qwen")
def generate_fast_qwen(request: RecipeRequest):
    def event_stream():
        yield f"data: {json.dumps({'step': 'starting', 'message': 'Initializing Groq Agentic Pipeline...'})}\n\n"
        
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
                canonical_name = canonicalize_ingredient(name_lower)
                if canonical_name != name_lower:
                    print(f"[CANON] '{name_lower}' → '{canonical_name}'")
                blueprint = None

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

                if blueprint is None:
                    try:
                        features = vegan_engine.load_features()
                        if canonical_name in features or vegan_engine.classify_by_keyword(canonical_name) is not None:
                            blueprint = vegan_engine.generate_vegan_blueprint(canonical_name, archetype=archetype_fast)
                            print(f"[LOCAL HIT] '{canonical_name}' -> '{blueprint.get('best_vegan_substitute', 'N/A')}'")
                        else:
                            print(f"[SKIP] '{canonical_name}' not in DB and no static fallback - keeping as-is")
                            blueprint = {"status": "unknown"}
                    except Exception as e:
                        print(f"[ENGINE ERROR] vegan_engine failed for '{canonical_name}': {e}")
                        blueprint = {"status": "error"}


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

        # Retrieve Graph Context
        if graph_retriever is not None:
            yield f"data: {json.dumps({'step': 'retrieving_context', 'message': 'Retrieving Context from Hybrid Graph...'})}\n\n"
            step_start = time.time()
            retrieval_ings = [parse_ingredient_input(r)[1] for r in clean_ingredients]
            graph_context = graph_retriever.retrieve_context(retrieval_ings)
            injected_context = graph_retriever.generate_prompt_injection(graph_context)
            
            few_shot_examples = graph_retriever.retrieve_few_shot_examples(graph_context, archetype, k=2)
            if few_shot_examples:
                injected_context += "\n<FEW_SHOT_EXAMPLES>\n"
                injected_context += "Here are historically accurate examples of how this archetype is prepared:\n\n"
                print(f"[FAISS] Successfully retrieved {len(few_shot_examples)} Semantic Vector Examples!")
                for idx, ex in enumerate(few_shot_examples):
                    ex_title = ex.split('\n')[0].replace('TITLE: ', '')
                    print(f"   -> Example {idx+1}: {ex_title}")
                    injected_context += f"Example {idx+1}:\n{ex}\n\n"
                injected_context += "</FEW_SHOT_EXAMPLES>\n"
                
            times['retrieval_sec'] = time.time() - step_start
        else:
            injected_context = ""
            graph_context = {}

        base_prompt = (
            f"<|begin_of_text|>System: You are a strict chef. You MUST explicitly use EVERY single ingredient provided in the list below in your recipe directions.\n"
            f"CRITICAL RULES:\n"
            f"1. DO NOT copy the FEW_SHOT_EXAMPLES. They are ONLY for structural reference.\n"
            f"2. DO NOT output variations, alternative fillings, personal notes, or reviews.\n"
            f"3. Output ONLY the TITLE and the numbered DIRECTIONS.\n\n"
            f"{injected_context}\n\n"
            f"### INGREDIENTS:\n"
            f"{ingr_text}\n"
            f"### TITLE:\n"
        )
        print(f"[AI-GROQ] Generating final recipe for archetype: {archetype}")

        max_retries = 3
        ai_text = ""
        current_prompt = base_prompt
        
        initial_cvs_score = None
        final_cvs_score = None
        all_critiques = []
        
        client = get_inference_client()
        qwen_model = get_groq_recipe_model(client)
        actual_model = qwen_model
        
        for attempt in range(1, max_retries + 1):
            yield f"data: {json.dumps({'step': 'generating', 'message': f'Generating AI Recipe (Attempt {attempt}/{max_retries})...'})}\n\n"
            step_start = time.time()
            
            res, actual_model = call_groq_with_fallback(
                client=client,
                messages=[{"role": "user", "content": current_prompt}],
                preferred_model=qwen_model,
                max_tokens=500,
                temperature=0.6
            )
            
            ai_text = res.choices[0].message.content.strip()
                
            times[f'generation_sec_attempt_{attempt}'] = time.time() - step_start
            
            if recipe_judge is None:
                break
                
            yield f"data: {json.dumps({'step': 'judging', 'message': f'Micro-Validation: Judging Recipe...'})}\n\n"
            judge_start = time.time()
            judge_result = recipe_judge.evaluate_recipe(ai_text, graph_context)
            times[f'judge_sec_attempt_{attempt}'] = time.time() - judge_start
            
            all_critiques.append(f"Attempt {attempt}: {judge_result.get('critique')}")
            
            if judge_result.get("is_valid"):
                print(f"[JUDGE] Passed on attempt {attempt}: {judge_result.get('critique')}")
                if initial_cvs_score is None: initial_cvs_score = judge_result.get("score")
                final_cvs_score = judge_result.get("score")
                break
            else:
                print(f"[JUDGE] Failed on attempt {attempt}: {judge_result.get('critique')}")
                if initial_cvs_score is None: initial_cvs_score = judge_result.get("score")
                final_cvs_score = judge_result.get("score")
                if attempt < max_retries:
                    yield f"data: {json.dumps({'step': 'correcting', 'message': f'Critique received. Regenerating Recipe...'})}\n\n"
                    current_prompt = (
                        f"{base_prompt}\n"
                        f"[PREVIOUS DRAFT REJECTED BY JUDGE]:\n{ai_text}\n"
                        f"[JUDGE CRITIQUE TO FIX]: {judge_result.get('critique')}\n"
                        f"Rewrite the recipe, strictly following the critique.\n"
                        f"### TITLE:\n"
                    )

        print(f"DEBUG RAW AI_TEXT: {repr(ai_text)}")

        ai_text = normalize_recipe_text(ai_text)
            
        total_time = time.time() - start_time
        
        if generation_logs_sync is not None:
            try:
                log_entry = {
                    "timestamp": time.time(),
                    "model_version": "groq_fast",
                    "is_vegan": request.is_vegan,
                    "archetype": archetype,
                    "budget": request.budget,
                    "servings": request.servings,
                    "state": request.state,
                    "times_sec": times,
                    "total_time_sec": total_time,
                    "ingredient_count": len(clean_ingredients),
                    "initial_cvs_score": initial_cvs_score,
                    "final_cvs_score": final_cvs_score,
                    "self_correction_attempts": attempt, "latency": total_time, "model": actual_model, "budget": request.budget, "city": request.state
                }
                generation_logs_sync.insert_one(log_entry)
                print(f"[OK] Generation log saved to DB. Total time: {total_time:.2f}s")
            except Exception as e:
                print(f"[WARN] Failed to save generation log: {e}")

        clean_title = ""
        title_match = re.search(r'### TITLE:\s*(.+)', ai_text)
        if title_match:
            clean_title = re.sub(r'[*_#"`]', '', title_match.group(1)).strip()

        final_result = {
            "status": "success",
            "archetype": archetype,
            "calculated_ingredients": calculated_ingredients,
            "recipe": ai_text,
            "image_url": get_random_banner(archetype, recipe_title=clean_title, is_vegan=request.is_vegan),
            "is_vegan": request.is_vegan,
            "initial_cvs_score": initial_cvs_score,
            "final_cvs_score": final_cvs_score,
            "self_correction_attempts": attempt,
            "latency": total_time,
            "model": actual_model,
            "budget": request.budget,
            "city": request.state,
            "judge_critiques": all_critiques,
            "cost_breakdown": calculate_cost_breakdown(calculated_ingredients, request.state, request.servings, request.budget),
            "graph_context": graph_context if 'graph_context' in locals() else {},
            "few_shot_examples": few_shot_examples if 'few_shot_examples' in locals() else [],
            "step_times": times
        }
        yield f"data: {json.dumps({'step': 'complete', 'result': final_result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/generate-fast-bare")
def generate_fast_bare(request: RecipeRequest):
    """
    Groq (Bare) 🚀 Pipeline — Zero agentic features.
    No graph, no FAISS, no judge, no SciPy. Just raw LLM generation.
    """
    def generate():
        try:
            start_time = time.time()
            ingr_text = "\n".join(f"- {i}" for i in request.ingredients)
            
            client = get_inference_client()
            chosen_model = get_groq_recipe_model(client)
                
            vegan_directive = "YOU MUST ENSURE THIS RECIPE IS 100% STRICTLY VEGAN. DO NOT USE ANY ANIMAL PRODUCTS." if request.is_vegan else ""
            
            prompt = (
                f"You are a master chef. Create a delicious recipe using these ingredients:\n"
                f"{ingr_text}\n\n"
                f"{vegan_directive}\n"
                f"Output ONLY the title on the first line starting with '### TITLE:' and the numbered instructions.\n"
            )
            
            yield f"data: {json.dumps({'step': 'generating', 'message': f'Turbo Generating Bare with {chosen_model}...'})}\n\n"
            
            res, actual_model = call_groq_with_fallback(
                client=client,
                messages=[{"role": "user", "content": prompt}],
                preferred_model=chosen_model,
                max_tokens=500,
                temperature=0.7
            )
            
            recipe_text = res.choices[0].message.content.strip()
            latency = time.time() - start_time
            yield f"data: {json.dumps({'step': 'final_recipe', 'recipe': recipe_text, 'calculated_ingredients': request.ingredients, 'archetype': 'Bare Qwen', 'is_vegan': request.is_vegan, 'attempts': 0, 'latency': latency, 'model': actual_model, 'budget': request.budget, 'city': request.state})}\n\n"
        except Exception as e:
            print(f"[ERROR in generate_fast_bare] {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'message': f'Bare generation failed: {str(e)}'})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/generate-fast-budget")
def generate_fast_budget(request: RecipeRequest):
    """
    Groq (Budget) 💰 Pipeline — SciPy optimization only.
    No graph, no FAISS, no judge. Just budget optimization + LLM generation.
    """
    clean_ingredients = []
    for item in request.ingredients:
        clean_ingredients.extend([i.strip() for i in item.split(',') if i.strip()])
        
    def generate():
        try:
            start_time = time.time()
            active_ingredients = list(clean_ingredients)
            yield f"data: {json.dumps({'step': 'starting', 'message': 'Initializing Budget Pipeline...'})}\n\n"
            time.sleep(0.3)
            
            if request.is_vegan:
                yield f"data: {json.dumps({'step': 'veganizing', 'message': 'Running Vegan Substitution Engine...'})}\n\n"
                import vegan_engine
                archetype_fast = _classify_archetype_fast([parse_ingredient_input(i)[1] for i in active_ingredients])
                veganized_ingredients = []
                for raw_ing in active_ingredients:
                    qty, name = parse_ingredient_input(raw_ing)
                    name_lower = name.lower().strip()
                    canonical_name = canonicalize_ingredient(name_lower)
                    blueprint = None
                    if vegan_alternatives_sync is not None:
                        try:
                            db_doc = vegan_alternatives_sync.find_one({"_id": canonical_name})
                            if db_doc and db_doc.get("best_vegan_substitute"):
                                blueprint = {
                                    "status": "success",
                                    "original_ingredient": name,
                                    "best_vegan_substitute": db_doc["best_vegan_substitute"],
                                    "compensation_blueprint": db_doc.get("compensation_blueprint", {
                                        "auxiliary_additions": [], "techniques": [], "spice_bridge": []
                                    })
                                }
                        except Exception as e:
                            pass
                    if blueprint is None:
                        try:
                            features = vegan_engine.load_features()
                            if canonical_name in features or vegan_engine.classify_by_keyword(canonical_name) is not None:
                                blueprint = vegan_engine.generate_vegan_blueprint(canonical_name, archetype=archetype_fast)
                        except Exception as e:
                            blueprint = {"status": "error"}

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
                active_ingredients = veganized_ingredients

            yield f"data: {json.dumps({'step': 'optimizing', 'message': f'Running Cost Constraint Optimization (Budget: ₹{request.budget})...'})}\n\n"
            
            calculated_ingredients, archetype = optimize_recipe_v2(active_ingredients, request.budget, request.servings, request.state)
            
            if not calculated_ingredients:
                yield f"data: {json.dumps({'step': 'error', 'message': 'The provided budget is mathematically impossible for these ingredients at current market prices.'})}\n\n"
                return
                
            ingr_text = "\n".join(f"- {i}" for i in calculated_ingredients)
            
            client = get_inference_client()
            chosen_model = get_groq_recipe_model(client)
                
            vegan_directive = "YOU MUST ENSURE THIS RECIPE IS 100% STRICTLY VEGAN. DO NOT USE ANY ANIMAL PRODUCTS." if request.is_vegan else ""
            
            prompt = (
                f"You are a master chef. Create a delicious budget recipe using these calculated ingredient quantities:\n"
                f"{ingr_text}\n\n"
                f"{vegan_directive}\n"
                f"Output ONLY the title on the first line starting with '### TITLE:' and the numbered instructions.\n"
            )
            
            yield f"data: {json.dumps({'step': 'generating', 'message': f'Generating Budget Recipe with {chosen_model}...'})}\n\n"
            
            res, actual_model = call_groq_with_fallback(
                client=client,
                messages=[{"role": "user", "content": prompt}],
                preferred_model=chosen_model,
                max_tokens=500,
                temperature=0.7
            )
            
            recipe_text = res.choices[0].message.content.strip()
            latency = time.time() - start_time
            final_archetype = "Agentic Vegan" if request.is_vegan else (archetype or "Budget Qwen")
            cost_breakdown = calculate_cost_breakdown(calculated_ingredients, request.state, request.servings, request.budget)
            yield f"data: {json.dumps({'step': 'final_recipe', 'recipe': recipe_text, 'calculated_ingredients': calculated_ingredients, 'cost_breakdown': cost_breakdown, 'archetype': final_archetype, 'is_vegan': request.is_vegan, 'attempts': 0, 'latency': latency, 'model': actual_model, 'budget': request.budget, 'city': request.state})}\n\n"
        except Exception as e:
            print(f"[ERROR in generate_fast_budget] {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'message': f'Budget recipe generation error: {str(e)}'})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/benchmark-reports")
async def get_benchmark_reports():
    """Fetches benchmark run data for the analysis tab."""
    if db is None:
        return {"status": "error", "message": "MongoDB not connected"}
    try:
        cursor = db.benchmark_runs.find().sort("Timestamp", -1)
        runs = await cursor.to_list(length=1000)
        for run in runs:
            run["_id"] = str(run["_id"])
        return {"status": "success", "data": runs}
    except Exception as e:
        return {"status": "error", "message": str(e)}
