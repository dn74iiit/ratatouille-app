import pandas as pd
import ast
import json
import re
from collections import defaultdict
import math
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Using the same clean_ingredient function from calculate_pmi.py
def clean_ingredient(ing):
    ing = re.sub(r'^\d+(\.\d+)?[a-zA-Z]*\s+', '', ing)
    ing = re.sub(r'^\d+\s*(cup|tsp|tbsp|teaspoon|tablespoon|oz|ounce|pound|lb|kg|g|ml)s?\s+', '', ing, flags=re.IGNORECASE)
    ing = re.sub(r'\(.*?\)', '', ing)
    ing = ing.split(',')[0].strip().lower()
    ing = " ".join([lemmatizer.lemmatize(w) for w in ing.split()])
    return ing

TECHNIQUES = [
    "bake", "roast", "grill", "broil", "fry", "saute", "sauté", "sear", "pan-fry", 
    "deep-fry", "stir-fry", "boil", "simmer", "poach", "steam", "blanch", "braise", 
    "stew", "marinate", "baste", "deglaze", "fold", "whip", "whisk", "beat", "blend", 
    "puree", "purée", "chop", "dice", "mince", "julienne", "slice", "grate", "shred", 
    "peel", "core", "pit", "crush", "mash", "knead", "roll", "proof", "glaze", 
    "flambe", "flambé", "smoke", "cure", "pickle", "ferment", "toast", "temper", 
    "caramelize", "reduce", "strain", "sift", "dredge", "bread", "coat", "score", 
    "butterfly", "truss", "stuff", "garnish", "zest", "sweat", "render", "clarify", 
    "coddle", "parboil", "microwave", "barbecue", "bbq", "brown", "thicken", 
    "thin", "dilute", "steep", "infuse", "muddle", "toss", "mix", "stir", 
    "scramble", "shuck", "debone", "carve", "tenderize", "pound"
]

def extract_prepared_by(csv_path, output_path, min_co_occurrence=5):
    print(f"Loading recipes from {csv_path} for technique extraction...")
    df = pd.read_csv(csv_path)
    
    ing_counts = defaultdict(int)
    tech_counts = defaultdict(int)
    co_counts = defaultdict(int)
    
    total_valid = 0
    
    for idx, row in df.iterrows():
        try:
            ings = ast.literal_eval(row['ingredients'])
            dirs = ast.literal_eval(row['directions'])
            
            cleaned_ings = set([clean_ingredient(i) for i in ings if i.strip()])
            
            # Combine all directions into one lowercase string
            full_dirs = " ".join(dirs).lower()
            # Extract words and lemmatize them as verbs to match "fries", "baked", etc.
            dir_words = set([lemmatizer.lemmatize(w, pos='v') for w in re.findall(r'\b\w+\b', full_dirs)])
            
            # Find which techniques are mentioned in the directions
            found_techs = set()
            for tech in TECHNIQUES:
                t_base = "saute" if tech == "sauté" else tech
                t_lemma = lemmatizer.lemmatize(t_base, pos='v')
                if t_lemma in dir_words:
                    found_techs.add(t_base)
                    
            if not found_techs or not cleaned_ings:
                continue
                
            total_valid += 1
            
            # Update counts
            for ing in cleaned_ings:
                ing_counts[ing] += 1
                for tech in found_techs:
                    co_counts[(ing, tech)] += 1
                    
            for tech in found_techs:
                tech_counts[tech] += 1
                
        except Exception as e:
            continue
            
    print(f"Total valid recipes with identifiable techniques: {total_valid}")
    
    prepared_by_edges = []
    
    # Calculate PMI for ingredient-technique pairings
    for (ing, tech), co_count in co_counts.items():
        if co_count >= min_co_occurrence:
            p_ing = ing_counts[ing] / total_valid
            p_tech = tech_counts[tech] / total_valid
            p_co = co_count / total_valid
            
            # Basic PMI
            pmi = math.log2(p_co / (p_ing * p_tech))
            
            # We want positive correlation
            if pmi > 0:
                prepared_by_edges.append({
                    "ingredient": ing,
                    "technique": tech,
                    "co_occurrence": co_count,
                    "pmi": pmi
                })
                
    prepared_by_edges.sort(key=lambda x: x['pmi'], reverse=True)
    
    print(f"Saving {len(prepared_by_edges)} PREPARED_BY edges to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prepared_by_edges, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    csv_file = "final_clean_50k_recipes_grams.csv"
    out_file = "data/prepared_by_edges.json"
    extract_prepared_by(csv_file, out_file)
