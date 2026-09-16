import os
import json
import time
import requests
import pandas as pd

API_URL = "https://ratatouille-backend.onrender.com/generate-recipe"

def generate_recipe(ingredients, budget=200.0):
    payload = {
        "ingredients": ingredients,
        "budget": budget,
        "servings": 2,
        "state": "Delhi",
        "model_version": "v10",
        "is_vegan": False
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

def run_batch_generation(num_recipes=150):
    print(f"[*] Starting massive scale batch generation of {num_recipes} recipes...")
    
    # Load unique ingredient combinations from the STANDARD dataset to test the Vegan Translation engine!
    try:
        import ast
        df_source = pd.read_csv("final_clean_50k_recipes_grams.csv")
        # Ensure we don't sample more than exists
        sample_size = min(num_recipes, len(df_source))
        df_sampled = df_source.sample(n=sample_size, random_state=42)
        
        # Convert ingredient strings "['80.0g flour', 'chicken']" into clean lists ["flour", "chicken"]
        import re
        test_cases = []
        for _, row in df_sampled.iterrows():
            ing_str = row['ingredients']
            original_title = row.get('title', 'Unknown Recipe')
            if isinstance(ing_str, str):
                try:
                    ings = ast.literal_eval(ing_str)
                    clean_ings = []
                    for item in ings:
                        # Remove numbers, units (g, ml, oz, etc), and text inside parentheses
                        cleaned = re.sub(r'\([^)]*\)', '', item) # remove (453.6g)
                        cleaned = re.sub(r'[\d\.\s]+(g|ml|cups|tbsp|tsp|oz|lb|pound|can|cans|cloves|clove|slice|slices)\s+', '', cleaned, flags=re.IGNORECASE)
                        cleaned = re.sub(r'[\d\.\/]+', '', cleaned) # remove remaining numbers like "1/2"
                        cleaned = cleaned.split(',')[0].strip() # keep base ingredient before comma (e.g. "celery" from "celery, chopped")
                        if cleaned:
                            clean_ings.append(cleaned)
                    if len(clean_ings) >= 5:
                        test_cases.append({"title": original_title, "ingredients": clean_ings})
                except Exception as e:
                    print(f"Parse error: {e}")
                    pass
    except Exception as e:
        print(f"[!] Could not load dataset: {e}. Falling back to default test cases.")
        test_cases = [
            {"title": "Test Chicken Recipe", "ingredients": ["chicken", "rice", "garlic", "onion"]},
            {"title": "Test Beef Recipe", "ingredients": ["beef", "tomato", "spinach", "garlic"]},
            {"title": "Test Pork Recipe", "ingredients": ["pork", "cauliflower", "cumin", "turmeric"]},
            {"title": "Test Lamb Recipe", "ingredients": ["lamb", "onion", "chili", "mustard seed"]},
            {"title": "Test Egg Recipe", "ingredients": ["egg", "coconut milk", "ginger", "curry leaf"]}
        ]
    
    extended_test_cases = (test_cases * (num_recipes // len(test_cases) + 1))[:num_recipes]
    
    catalog = []
    
    for idx, case in enumerate(extended_test_cases):
        original_title = case["title"]
        ingredients = case["ingredients"]
        print(f"\n[{idx+1}/{num_recipes}] Generating recipe for '{original_title}' ({len(ingredients)} ingredients)")
        result, latency = generate_recipe(ingredients)
        
        if result and result.get("status") == "success":
            recipe_text = result.get("recipe", "")
            # Try to extract the generated title from the first line of the markdown
            generated_title_match = re.search(r'^\s*#\s+(.+)', recipe_text, re.MULTILINE)
            generated_title = generated_title_match.group(1).strip() if generated_title_match else original_title

            print(f"  -> Generated '{generated_title}' ({result.get('archetype')}) in {latency:.2f}s")
            catalog.append({
                "original_title": original_title,
                "generated_title": generated_title,
                "archetype": result.get("archetype"),
                "ingredients": ", ".join(result.get("calculated_ingredients", [])),
                "recipe": recipe_text,
                "is_vegan": result.get("is_vegan"),
                "latency_sec": round(latency, 2),
                "initial_cvs_score": result.get("initial_cvs_score"),
                "final_cvs_score": result.get("final_cvs_score"),
                "self_correction_attempts": result.get("self_correction_attempts")
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
    # Run the 150-recipe batch overnight
    run_batch_generation(num_recipes=150)
