import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
import re

load_dotenv()

def generate_vegan_mappings():
    print("Initializing Groq Client for Offline Mapping Generation...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment.")
        return

    client = Groq(api_key=api_key)
    
    categories = [
        "Dairy (Milk, Butter, Ghee, Paneer, Curd, Cream, Condensed Milk, Khoa/Mawa)",
        "Meat & Poultry (Chicken, Mutton, Lamb, Beef, Pork, Bacon, Keema/Minced Meat)",
        "Seafood (Fish, Shrimp/Prawns, Crab, Anchovies, Fish Sauce, Oyster Sauce)",
        "Eggs & Binders (Eggs for baking, Eggs for scrambling, Eggs for binding/fritters, Gelatin)",
        "Sweeteners & Additives (Honey, Carmine, Whey, Casein, Milk Powder)"
    ]
    
    all_mappings = {}
    
    system_prompt = (
        "You are an expert Indian Vegan Culinary Scientist. "
        "Your job is to provide structural, mathematically sound vegan substitutions for non-vegan ingredients. "
        "CRITICAL RULES:\n"
        "1. ALL substitutions MUST be highly affordable, accessible Indian ingredients (e.g., Soya Chunks/Nutrela, Besan, Jackfruit/Kathal, Peanuts, Cashews, Tofu, Coconut, Flaxseed/Alsi, Mashed Banana).\n"
        "2. DO NOT suggest expensive/rare Western items like Seitan, Aquafaba, Nutritional Yeast, or Commercial Mock Meats.\n"
        "3. Output strictly valid JSON. Do not include markdown code blocks. Just output a JSON object where the keys are the non-vegan ingredients (lowercase) and the values are dictionaries containing 'function', 'substitute', and 'context'."
    )

    for category in categories:
        print(f"Generating mappings for category: {category}...")
        
        user_prompt = f"""
Generate an exhaustive list of affordable Indian vegan substitutions for the following category of non-vegan ingredients: {category}.
Think deeply about the different culinary functions (e.g. eggs for baking vs eggs for scrambling).

Provide the output strictly as a JSON object matching this schema:
{{
  "ingredient_name_lowercase": {{
    "function": "Brief description of its culinary purpose (e.g., Fat & Richness, Binder)",
    "substitute": "The affordable Indian vegan substitute",
    "context": "1-2 sentences on exactly how to prep/cook this substitute structurally."
  }},
  ...
}}
Output ONLY valid JSON.
"""
        
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_completion_tokens=4000
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # Extract JSON from potential markdown formatting
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                try:
                    category_mappings = json.loads(json_text)
                    all_mappings.update(category_mappings)
                    print(f"Successfully added {len(category_mappings)} mappings from {category}.")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON for {category}: {e}")
            else:
                print(f"Could not find JSON block in response for {category}.")
                
            time.sleep(2) # Respect rate limits
            
        except Exception as e:
            print(f"Error calling Groq API for {category}: {e}")
            
    # Save the combined mappings to a file
    output_path = os.path.join("data", "vegan_functional_matrix.json")
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_mappings, f, indent=2)
        
    print(f"\nTotal mappings generated and saved: {len(all_mappings)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_vegan_mappings()
