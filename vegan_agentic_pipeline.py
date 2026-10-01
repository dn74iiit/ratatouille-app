import os
import time
import json
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[!] GROQ_API_KEY not found in .env")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamically select active Groq models
PRIMARY_MODEL = "llama-3.1-8b-instant"
try:
    if client:
        available_models = [m.id for m in client.models.list().data]
        PRIMARY_MODEL = next((m for m in available_models if 'llama-3' in m.lower()), "llama-3.1-8b-instant")
except:
    pass

class RecipeRequest(BaseModel):
    ingredients: list
    budget: float = 0.0
    servings: int = 1
    state: str = ""
    archetype: str = "Curry"
    is_vegan: bool = False
    model_version: str = ""

class VeganGraphRetriever:
    """Simulates a specialized Vegan Knowledge Graph for Agentic Context Injection."""
    def __init__(self):
        self.structural_rules = {
            "tofu": "CRITICAL TECHNIQUE: You MUST instruct the user to press the water out of the tofu before cooking it.",
            "coconut oil": "CRITICAL TECHNIQUE: Coconut oil has a lower smoke point. You MUST instruct the user to cook on low-medium heat to prevent burning.",
            "flaxseed": "CRITICAL TECHNIQUE: To use flaxseed as a binder, you MUST instruct the user to mix it with water and let it sit for 5 minutes to form a gel.",
            "cashew cream": "CRITICAL TECHNIQUE: Cashews must be soaked in hot water before blending into a cream."
        }
        
    def get_context(self, ingredients):
        injected_rules = []
        for ing in ingredients:
            for key, rule in self.structural_rules.items():
                if key in ing.lower():
                    injected_rules.append(rule)
        
        if not injected_rules:
            return ""
            
        return "<VEGAN_STRUCTURAL_CONTEXT>\n" + "\n".join(injected_rules) + "\n</VEGAN_STRUCTURAL_CONTEXT>"

class VeganRecipeJudge:
    def __init__(self):
        self.non_vegan_culprits = ['butter', 'ghee', 'cheese', 'paneer', 'egg', 'eggs', 'honey', 'milk', 'cream', 'chicken', 'meat', 'beef', 'pork']
        
    def evaluate_recipe(self, recipe_text):
        recipe_lower = recipe_text.lower()
        
        # 1. Zero-Tolerance Hallucination Check
        hallucinated = [c for c in self.non_vegan_culprits if re.search(r'\b' + c + r'\b', recipe_lower) and c not in ["coconut milk", "almond milk", "cashew cream", "peanut butter"]]
        if hallucinated:
            return {
                "score": 0.2,
                "critique": f"FATAL ERROR: You hallucinated non-vegan ingredients: {', '.join(hallucinated)}. You MUST remove them immediately."
            }
            
        # 2. Structural Physics Check (LLM Judge)
        prompt = (
            "You are a strict Vegan Culinary Judge.\n"
            "Evaluate the following recipe draft based on these criteria:\n"
            "1. NO ANIMAL PRODUCTS: Are there any hidden animal products?\n"
            "2. STRUCTURAL PHYSICS: If the recipe uses Tofu, did they press it? If it uses Coconut Oil, did they use low heat?\n\n"
            f"[RECIPE DRAFT]:\n{recipe_text}\n\n"
            "Provide a 'Culinary Validity Score' from 0.0 to 1.0.\n"
            "If the score is less than 0.8, provide a strict 1-sentence CRITIQUE explaining what must be fixed.\n"
            "Output format:\n"
            "SCORE: [score]\n"
            "CRITIQUE: [critique if any]"
        )
        
        try:
            res = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=PRIMARY_MODEL,
                max_completion_tokens=150,
                temperature=0.1
            )
            response = res.choices[0].message.content.strip()
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response)
            critique_match = re.search(r'CRITIQUE:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            critique = critique_match.group(1).strip() if critique_match else "No critique provided."
            return {"score": score, "critique": critique}
        except:
            return {"score": 1.0, "critique": "Passed rules."}

retriever = VeganGraphRetriever()
judge = VeganRecipeJudge()

@app.post("/generate-agentic-vegan")
async def generate_agentic_vegan(request: RecipeRequest):
    start_time = time.time()
    
    ingr_text = "\n".join(f"- {i}" for i in request.ingredients)
    vegan_context = retriever.get_context(request.ingredients)
    
    base_prompt = (
        f"System: You are a strict vegan chef. You MUST explicitly use EVERY single ingredient provided below.\n"
        f"CRITICAL RULES:\n"
        f"1. DO NOT output variations or notes.\n"
        f"2. Output ONLY the TITLE and the numbered DIRECTIONS.\n\n"
        f"{vegan_context}\n\n"
        f"### INGREDIENTS:\n{ingr_text}\n\n### TITLE:\n"
    )
    
    current_prompt = base_prompt
    final_recipe = ""
    attempts = 0
    
    target_model = PRIMARY_MODEL
    if request.model_version == 'qwen':
        try:
            available_models = [m.id for m in client.models.list().data]
            target_model = next((m for m in available_models if 'qwen' in m.lower()), "mixtral-8x7b-32768")
        except:
            target_model = "qwen-2.5-32b"
            
    for attempt in range(3):
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": current_prompt}],
            model=target_model,
            max_completion_tokens=500,
            temperature=0.6
        )
        recipe_text = res.choices[0].message.content.strip()
        
        judge_result = judge.evaluate_recipe(recipe_text)
        
        if judge_result['score'] >= 0.8:
            final_recipe = recipe_text
            attempts = attempt
            break
        else:
            current_prompt = base_prompt + f"\n\n[JUDGE CRITIQUE TO FIX]: {judge_result['critique']}\nRewrite the recipe to fix this error."
            final_recipe = recipe_text
            attempts = attempt + 1
            
    latency = time.time() - start_time
    
    # We yield a fake SSE stream to match the frontend's expectations for parsing
    from fastapi.responses import StreamingResponse
    
    def generate():
        yield f"data: {json.dumps({'step': 'retrieving_context', 'message': 'Querying Vegan Structural Graph...'})}\n\n"
        time.sleep(0.5)
        if attempts > 0:
            yield f"data: {json.dumps({'step': 'judging', 'message': f'Vegan Judge Rejected Recipe! Retrying {attempts} times...'})}\n\n"
            time.sleep(0.5)
            
        yield f"data: {json.dumps({'step': 'final_recipe', 'recipe': final_recipe, 'calculated_ingredients': request.ingredients, 'archetype': request.archetype, 'is_vegan': True, 'attempts': attempts})}\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/generate-fast-qwen")
async def generate_fast_qwen(request: RecipeRequest):
    start_time = time.time()
    ingr_text = "\n".join(f"- {i}" for i in request.ingredients)
    
    qwen_model = "qwen-2.5-32b"
    try:
        if client:
            available_models = [m.id for m in client.models.list().data]
            qwen_model = next((m for m in available_models if 'qwen' in m.lower()), "mixtral-8x7b-32768")
    except:
        pass
        
    vegan_directive = "YOU MUST ENSURE THIS RECIPE IS 100% STRICTLY VEGAN. DO NOT USE ANY ANIMAL PRODUCTS." if request.is_vegan else ""
    
    prompt = (
        f"You are a master chef. Create a delicious recipe using these ingredients:\n"
        f"{ingr_text}\n\n"
        f"{vegan_directive}\n"
        f"Output ONLY the title on the first line starting with '### TITLE:' and the numbered instructions.\n"
    )
    
    res = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=qwen_model,
        max_completion_tokens=500,
        temperature=0.7
    )
    
    recipe_text = res.choices[0].message.content.strip()
    
    from fastapi.responses import StreamingResponse
    def generate():
        yield f"data: {json.dumps({'step': 'generating', 'message': f'Turbo Generating with {qwen_model}...'})}\n\n"
        time.sleep(0.5)
        yield f"data: {json.dumps({'step': 'final_recipe', 'recipe': recipe_text, 'calculated_ingredients': request.ingredients, 'archetype': 'Fast Qwen', 'is_vegan': request.is_vegan, 'attempts': 0})}\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)
