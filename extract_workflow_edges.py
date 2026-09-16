import pandas as pd
import ast
import json
import re
from collections import defaultdict
import math
from nltk.stem import WordNetLemmatizer
import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

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

def extract_followed_by(csv_path, output_path, min_transition_count=5):
    print(f"Loading recipes from {csv_path} for workflow extraction...")
    df = pd.read_csv(csv_path)
    
    # Track transitions from A to B: A -> B
    transitions = defaultdict(int)
    # Track total occurrences of technique A that are followed by ANY technique
    tech_sources = defaultdict(int)
    
    total_valid = 0
    
    for idx, row in df.iterrows():
        try:
            dirs = ast.literal_eval(row['directions'])
            
            recipe_techniques_sequence = []
            
            for direction in dirs:
                direction = direction.lower()
                dir_words = [lemmatizer.lemmatize(w, pos='v') for w in re.findall(r'\b\w+\b', direction)]
                
                # To maintain order within a single direction step, we can find the indices
                # But a simpler approximation is just picking up techniques as they appear
                found_in_step = []
                for word in dir_words:
                    if word in TECHNIQUES:
                        found_in_step.append(word)
                    elif word == "sauté": # Handle sauté edge case
                        found_in_step.append("saute")
                
                if found_in_step:
                    # Remove consecutive duplicates (e.g. if they say "chop and chop")
                    deduped = []
                    for t in found_in_step:
                        if not deduped or deduped[-1] != t:
                            deduped.append(t)
                    recipe_techniques_sequence.extend(deduped)
            
            if not recipe_techniques_sequence:
                continue
                
            total_valid += 1
            
            # Record transitions
            for i in range(len(recipe_techniques_sequence) - 1):
                tech_a = recipe_techniques_sequence[i]
                tech_b = recipe_techniques_sequence[i + 1]
                
                # We don't want self-transitions (chop -> chop) for workflow archetypes
                if tech_a != tech_b:
                    transitions[(tech_a, tech_b)] += 1
                    tech_sources[tech_a] += 1
                
        except Exception as e:
            continue
            
    print(f"Total valid recipes with workflow sequences: {total_valid}")
    
    followed_by_edges = []
    
    for (tech_a, tech_b), count in transitions.items():
        if count >= min_transition_count:
            # Probability that given you just did A, you will do B next
            prob = count / tech_sources[tech_a]
            
            followed_by_edges.append({
                "source_technique": tech_a,
                "target_technique": tech_b,
                "count": count,
                "probability": prob
            })
            
    # Sort by count and probability
    followed_by_edges.sort(key=lambda x: (x['count'], x['probability']), reverse=True)
    
    print(f"Saving {len(followed_by_edges)} FOLLOWED_BY edges to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(followed_by_edges, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    # We will use the large clean dataset
    csv_file = "final_clean_50k_recipes_grams.csv"
    out_file = "data/followed_by_edges.json"
    extract_followed_by(csv_file, out_file)
