import os
import json
import time
import requests
import pandas as pd

API_URL = "http://localhost:8000/generate-recipe"

def generate_recipe(ingredients, budget=5.0):
    payload = {
        "ingredients": ingredients,
        "budget": budget,
        "servings": 2,
        "state": "Delhi",
        "model_version": "v10",
        "is_vegan": True
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, stream=True)
    
    final_result = None
    
    # Process SSE stream
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                try:
                    data = json.loads(decoded_line[6:])
                    if data.get("step") == "complete":
                        final_result = data.get("result")
                except Exception as e:
                    print(f"Error parsing SSE data: {e}")
                    
    latency = time.time() - start_time
    
    # We retrieve the logs directly from MongoDB if needed, 
    # but the API response could also be updated to return these.
    # Since the API logs to generation_logs_sync, the massive scale analytics
    # are automatically captured in MongoDB!
    # For this script, we'll extract what we can from the API response 
    # and save successful recipes to the Reference Catalog.
    
    return final_result, latency

def run_batch_generation(num_recipes=5):
    print(f"[*] Starting massive scale batch generation of {num_recipes} recipes...")
    
    test_cases = [
        ["chicken", "rice", "garlic", "onion"],
        ["paneer", "tomato", "spinach", "garlic"],
        ["potato", "cauliflower", "cumin", "turmeric"],
        ["lentil", "onion", "chili", "mustard seed"],
        ["mushroom", "coconut milk", "ginger", "curry leaf"]
    ]
    
    # If we need more, we loop over the test cases
    extended_test_cases = (test_cases * (num_recipes // len(test_cases) + 1))[:num_recipes]
    
    catalog = []
    
    for idx, ingredients in enumerate(extended_test_cases):
        print(f"\n[{idx+1}/{num_recipes}] Generating recipe for: {ingredients}")
        result, latency = generate_recipe(ingredients)
        
        if result and result.get("status") == "success":
            print(f"  -> Generated {result.get('archetype')} in {latency:.2f}s")
            catalog.append({
                "archetype": result.get("archetype"),
                "ingredients": ", ".join(result.get("calculated_ingredients", [])),
                "recipe": result.get("recipe"),
                "is_vegan": result.get("is_vegan")
            })
        else:
            print("  -> Generation failed or was rejected by budget constraints.")
            
    if catalog:
        df = pd.DataFrame(catalog)
        os.makedirs("data", exist_ok=True)
        catalog_path = "data/Reference_Catalog.csv"
        df.to_csv(catalog_path, index=False)
        print(f"\n[*] Batch generation complete! Saved {len(catalog)} recipes to {catalog_path}.")
        print("[*] Note: Detailed analytics (CVS scores, latencies, self-correction attempts) are logged in the 'generation_logs' MongoDB collection.")
    else:
        print("\n[!] Batch generation failed to produce any valid recipes.")

if __name__ == "__main__":
    # For testing, we run a small batch. 
    # For full scale, change to 1000.
    run_batch_generation(num_recipes=5)
