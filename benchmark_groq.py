import requests
import json
import time
import pandas as pd
import os
from dotenv import load_dotenv
import pymongo

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    mongo_client = pymongo.MongoClient(MONGO_URI)
    benchmark_db = mongo_client.ratatouille.benchmark_runs
    print("✅ Connected to MongoDB. Results will be saved to the 'benchmark_runs' collection.")
else:
    benchmark_db = None
    print("⚠️ MONGO_URI not found. MongoDB logging is disabled.")

# The three endpoints we want to test
ENDPOINTS = {
    "Groq Bare": "http://localhost:8000/generate-fast-bare",
    "Groq Budget": "http://localhost:8000/generate-fast-budget",
    "Groq Full (Agentic)": "http://localhost:8000/generate-fast-qwen"
}

import ast
import re

def generate_test_cases(num_cases=100):
    print(f"Loading {num_cases} random test cases from the 50k recipe dataset...")
    try:
        df_source = pd.read_csv("final_clean_50k_recipes_grams.csv")
        sample_size = min(num_cases, len(df_source))
        df_sampled = df_source.sample(n=sample_size, random_state=42)
        
        test_cases = []
        for _, row in df_sampled.iterrows():
            ing_str = row['ingredients']
            original_title = row.get('title', 'Unknown Recipe')
            if isinstance(ing_str, str):
                try:
                    ings = ast.literal_eval(ing_str)
                    clean_ings = []
                    for item in ings:
                        cleaned = re.sub(r'\([^)]*\)', '', item)
                        cleaned = re.sub(r'[\d\.\s]+(g|ml|cups|tbsp|tsp|oz|lb|pound|can|cans|cloves|clove|slice|slices)\s+', '', cleaned, flags=re.IGNORECASE)
                        cleaned = re.sub(r'[\d\.\/]+', '', cleaned)
                        cleaned = cleaned.split(',')[0].strip()
                        if cleaned:
                            clean_ings.append(cleaned)
                    if len(clean_ings) >= 3:
                        test_cases.append({
                            "name": original_title,
                            "ingredients": clean_ings,
                            "budget": 300,  # default static budget for testing
                            "servings": 2,
                            "state": "Delhi"
                        })
                except Exception as e:
                    pass
        print(f"Successfully generated {len(test_cases)} valid test cases.")
        return test_cases
    except Exception as e:
        print(f"[!] Could not load dataset: {e}. Falling back to default.")
        return [
            {"name": "Simple Staples", "ingredients": ["chicken", "rice", "onion", "garlic"], "budget": 300, "servings": 2, "state": "Delhi"},
            {"name": "Weird Combo", "ingredients": ["pineapple", "beef", "mint", "soy sauce"], "budget": 500, "servings": 2, "state": "Maharashtra"}
        ]

def parse_sse(response_iter):
    """Helper to parse Server Sent Events (SSE) stream to get the final recipe data."""
    final_data = None
    for line in response_iter.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: '):
                try:
                    data = json.loads(decoded[6:])
                    if data.get('step') == 'final_recipe' or data.get('step') == 'complete':
                        final_data = data
                    elif data.get('step') == 'error':
                        final_data = data
                except json.JSONDecodeError:
                    pass
    return final_data

def run_benchmark():
    print("🚀 Starting Groq Pipeline Benchmark...")
    results = []
    
    test_cases = generate_test_cases(100)
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Testing Scenario {i+1}/100: {case['name']} ---")
        
        for pipeline_name, url in ENDPOINTS.items():
            print(f"Running {pipeline_name}...")
            
            payload = {
                "ingredients": case["ingredients"],
                "budget": case["budget"],
                "servings": case["servings"],
                "state": case["state"],
                "is_vegan": False
            }
            
            start_req = time.time()
            try:
                # We use stream=True because the endpoints return Server-Sent Events
                response = requests.post(url, json=payload, stream=True)
                final_event = parse_sse(response)
                end_req = time.time()
                
                latency = round(end_req - start_req, 2)
                
                # Default failure states
                status = "Success"
                cvs_score = "N/A"
                final_ingredients = []
                
                if not final_event:
                    status = "Failed (No Event)"
                elif final_event.get("step") == "error":
                    status = f"Failed: {final_event.get('message')}"
                else:
                    # Extract the nested result if it's the Groq Full pipeline
                    result_data = final_event.get('result', final_event)
                    cvs_score = result_data.get('final_cvs_score', 'N/A')
                    final_ingredients = result_data.get('calculated_ingredients', [])
                
                run_data = {
                    "Scenario": case["name"],
                    "Pipeline": pipeline_name,
                    "Latency (sec)": latency,
                    "Status": status,
                    "CVS Score": cvs_score,
                    "Budget Handled": "Yes" if final_ingredients else "No",
                    "Input_Ingredients": case["ingredients"],
                    "Calculated_Ingredients_Weights": final_ingredients,
                    "Budget": case["budget"],
                    "State": case["state"],
                    "Generated_Recipe": result_data.get("recipe", "") if final_event and "recipe" in result_data else "",
                    "Self_Correction_Attempts": result_data.get("self_correction_attempts", 0) if final_event else 0,
                    "Judge_Critiques": result_data.get("judge_critiques", []) if final_event else [],
                    "Graph_Context": result_data.get("graph_context", {}) if final_event else {},
                    "Few_Shot_Examples": result_data.get("few_shot_examples", []) if final_event else [],
                    "Timestamp": time.time()
                }
                results.append(run_data)
                
                if benchmark_db is not None:
                    try:
                        benchmark_db.insert_one(run_data)
                    except Exception as e:
                        print(f"  [WARN] Failed to log to DB: {e}")
                        
                print(f"  -> Latency: {latency}s | CVS: {cvs_score} | Status: {status}")
                
            except Exception as e:
                print(f"  -> ERROR: {e}")
                results.append({
                    "Scenario": case["name"],
                    "Pipeline": pipeline_name,
                    "Latency (sec)": 0,
                    "Status": f"Crash: {str(e)}",
                    "CVS Score": "N/A",
                    "Budget Handled": "N/A"
                })
                
    # Generate Report
    print("\n\n📊 BENCHMARK RESULTS 📊")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    df.to_csv("groq_benchmark_results.csv", index=False)
    print("\n✅ Results saved to 'groq_benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark()
