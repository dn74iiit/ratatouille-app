import pandas as pd
import ast
import itertools
from collections import defaultdict
import math
import json
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_ingredient(ing):
    """Basic cleaning to extract the core ingredient name."""
    # Remove quantities (e.g., '80.0g flour' -> 'flour')
    ing = re.sub(r'^\d+(\.\d+)?[a-zA-Z]*\s+', '', ing)
    ing = re.sub(r'^\d+\s*(cup|tsp|tbsp|teaspoon|tablespoon|oz|ounce|pound|lb|kg|g|ml)s?\s+', '', ing, flags=re.IGNORECASE)
    # Remove contents inside parentheses
    ing = re.sub(r'\(.*?\)', '', ing)
    # Get the part before a comma (e.g., 'celery, chopped' -> 'celery')
    ing = ing.split(',')[0].strip().lower()
    # Lemmatize to handle plurals (tomatoes -> tomato)
    ing = " ".join([lemmatizer.lemmatize(w) for w in ing.split()])
    return ing

def calculate_pmi(csv_path, output_path, min_co_occurrence=5):
    print(f"Loading recipes from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract ingredient lists
    print("Extracting and cleaning ingredients...")
    recipe_ingredients = []
    for idx, row in df.iterrows():
        try:
            # Parse the string representation of list
            ings = ast.literal_eval(row['ingredients'])
            # Clean and filter empty
            cleaned = set([clean_ingredient(i) for i in ings if i.strip()])
            recipe_ingredients.append(list(cleaned))
        except Exception as e:
            continue
            
    total_recipes = len(recipe_ingredients)
    print(f"Total valid recipes processed: {total_recipes}")
    
    # Count frequencies
    ing_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    
    print("Counting frequencies and co-occurrences...")
    for ings in recipe_ingredients:
        for ing in ings:
            ing_counts[ing] += 1
            
        # Get all unique pairs in this recipe
        pairs = list(itertools.combinations(sorted(ings), 2))
        for pair in pairs:
            pair_counts[pair] += 1
            
    # Calculate PMI
    print("Calculating PMI...")
    pmi_scores = []
    for (ing1, ing2), co_count in pair_counts.items():
        if co_count >= min_co_occurrence:
            p_ing1 = ing_counts[ing1] / total_recipes
            p_ing2 = ing_counts[ing2] / total_recipes
            p_pair = co_count / total_recipes
            
            pmi = math.log2(p_pair / (p_ing1 * p_ing2))
            pmi_scores.append({
                "source": ing1,
                "target": ing2,
                "co_occurrence": co_count,
                "pmi": pmi
            })
            
    # Sort by PMI descending
    pmi_scores.sort(key=lambda x: x['pmi'], reverse=True)
    
    print(f"Saving {len(pmi_scores)} pairs to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pmi_scores, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    csv_file = "final_clean_50k_recipes_grams.csv"
    out_file = "data/ingredient_pmi.json"
    calculate_pmi(csv_file, out_file)
