import os
import time
import pandas as pd
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[!] GROQ_API_KEY not found in .env")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

# Import the exact Agentic Pipeline classes from your codebase
from graph_retriever import GraphRetriever
from llm_judge import RecipeJudge

print("[*] Initializing Agentic Pipeline for Qwen...")
retriever = GraphRetriever()
judge = RecipeJudge()
print("[OK] Pipeline Ready.")

# Dynamically find available models for this specific API key
try:
    available_models = [m.id for m in client.models.list().data if 'whisper' not in m.id.lower() and 'vision' not in m.id.lower() and 'guard' not in m.id.lower()]
    PRIMARY_MODEL = next((m for m in available_models if 'llama-3' in m.lower()), available_models[0] if available_models else "llama-3.1-8b-instant")
    FALLBACK_MODEL = next((m for m in available_models if 'mixtral' in m.lower() or 'gemma' in m.lower()), available_models[1] if len(available_models) > 1 else PRIMARY_MODEL)
    print(f"[*] Dynamically selected Groq models: Primary='{PRIMARY_MODEL}', Fallback='{FALLBACK_MODEL}'")
except Exception as e:
    print(f"[WARN] Failed to fetch Groq models list: {e}")
    PRIMARY_MODEL = "llama-3.1-8b-instant"
    FALLBACK_MODEL = "mixtral-8x7b-32768"

def query_qwen(messages: list, max_tokens: int = 500, temperature: float = 0.5) -> str:
    try:
        res = client.chat.completions.create(
            messages=messages,
            model=PRIMARY_MODEL,
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        try:
            res = client.chat.completions.create(
                messages=messages,
                model=FALLBACK_MODEL,
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
            return res.choices[0].message.content.strip()
        except Exception as e2:
            print(f"[WARN] Groq API failed on both models. Primary Error: {e} | Fallback Error: {e2}")
            return ""

# Monkeypatch api.py in memory so we don't have to modify the user's actual files!
# RecipeJudge imports query_serverless_llm from api.py dynamically. This forces it to use our new dynamic models.
import api
api.query_serverless_llm = query_qwen

def generate_qwen_with_agentic_pipeline(title, ingredients_str, archetype):
    start_time = time.time()
    
    # 1. Parse raw ingredients to pass to GraphRetriever
    # Ingredients str is like: "82.0g raspberry jam, 240.0g bisquick"
    raw_ings = [i.split('g ')[-1].strip() if 'g ' in i else i.strip() for i in ingredients_str.split(',')]
    
    # 2. Retrieve Hybrid Graph Context & Few Shot Examples
    graph_context = retriever.retrieve_context(raw_ings)
    injected_context = retriever.generate_prompt_injection(graph_context)
    
    few_shot_examples = retriever.retrieve_few_shot_examples(graph_context, archetype, k=2)
    if few_shot_examples:
        injected_context += "\n<FEW_SHOT_EXAMPLES>\n"
        for idx, ex in enumerate(few_shot_examples):
            injected_context += f"Example {idx+1}:\n{ex}\n\n"
        injected_context += "</FEW_SHOT_EXAMPLES>\n"
        
    ingr_text = "\n".join(f"- {i.strip()}" for i in ingredients_str.split(','))
    
    # 3. Base Prompt identical to Llama 3
    base_prompt = (
        f"System: You are a strict chef. You MUST explicitly use EVERY single ingredient provided in the list below in your recipe directions.\n"
        f"CRITICAL RULES:\n"
        f"1. DO NOT copy the FEW_SHOT_EXAMPLES. They are ONLY for structural reference.\n"
        f"2. DO NOT output variations, alternative fillings, personal notes, or reviews.\n"
        f"3. Output ONLY the TITLE and the numbered DIRECTIONS.\n\n"
        f"{injected_context}\n\n"
        f"### INGREDIENTS:\n"
        f"{ingr_text}\n"
        f"### TITLE:\n"
        f"{title}\n"
    )
    
    # 4. Iterative Self-Correction Loop (Up to 3 attempts)
    current_prompt = base_prompt
    final_recipe = ""
    final_score = 0
    attempts = 0
    
    for attempt in range(3):
        messages = [{"role": "user", "content": current_prompt}]
        recipe_text = query_qwen(messages, max_tokens=500, temperature=0.6)
        
        if not recipe_text.startswith(title):
            recipe_text = f"{title}\n### DIRECTIONS:\n{recipe_text}"
            
        # Judge the recipe
        judge_result = judge.evaluate_recipe(recipe_text, graph_context)
        final_score = judge_result['score']
        
        if final_score >= 0.8:
            final_recipe = recipe_text
            attempts = attempt
            break
        else:
            print(f"      [Judge] Attempt {attempt+1} failed with score {final_score}. Critique: {judge_result['critique']}")
            current_prompt = base_prompt + f"\n\n[JUDGE CRITIQUE TO FIX]: {judge_result['critique']}\nRewrite the recipe to fix this error."
            final_recipe = recipe_text
            attempts = attempt + 1
            time.sleep(2) # rate limit
            
    latency = time.time() - start_time
    time.sleep(2) # rate limit
    
    return final_recipe, latency, final_score, attempts

def run_comparison():
    input_csv = "data/Reference_Catalog_v3_clean.csv"
    output_csv = "data/Reference_Catalog_v4_qwen_comparison.csv"
    
    if not os.path.exists(input_csv):
        print(f"[!] Input file {input_csv} not found.")
        return
        
    print(f"[*] Loading generated Llama 3 recipes from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Check if we can resume from a previous run
    if os.path.exists(output_csv):
        print(f"[*] Found existing output file {output_csv}, attempting to resume...")
        df_out = pd.read_csv(output_csv)
        # Map existing qwen recipes to the main dataframe
        if 'qwen_recipe' in df_out.columns:
            for c in ['qwen_recipe', 'qwen_latency_sec', 'qwen_final_cvs_score', 'qwen_self_correction_attempts']:
                if c in df_out.columns:
                    df[c] = df_out[c]
    
    # Initialize columns if they don't exist
    for c in ['qwen_recipe', 'qwen_latency_sec', 'qwen_final_cvs_score', 'qwen_self_correction_attempts']:
        if c not in df.columns:
            df[c] = None
    
    total = len(df)
    
    for idx, row in df.iterrows():
        # Check if already processed
        if pd.notna(row.get('qwen_recipe')):
            print(f"[{idx+1}/{total}] Skipping already processed recipe: {row['original_title']}")
            continue
            
        print(f"[{idx+1}/{total}] Generating Qwen Agentic pipeline for: {row['original_title']}...")
        
        ings = row['ingredients'] if pd.notna(row['ingredients']) else ""
        archetype = row['archetype'] if pd.notna(row['archetype']) else "Unknown"
            
        # Generate via full pipeline
        recipe, latency, score, attempts = generate_qwen_with_agentic_pipeline(row['original_title'], ings, archetype)
        
        df.at[idx, 'qwen_recipe'] = recipe
        df.at[idx, 'qwen_latency_sec'] = round(latency, 2)
        df.at[idx, 'qwen_final_cvs_score'] = score
        df.at[idx, 'qwen_self_correction_attempts'] = attempts
        
        print(f"  -> Done in {latency:.2f}s | Score: {score} | Attempts: {attempts}")
        
        # Save after EVERY recipe to ensure no data loss on crash
        df.to_csv(output_csv, index=False)
        
    print(f"[*] Complete! Data fully saved to {output_csv}")

if __name__ == "__main__":
    run_comparison()
