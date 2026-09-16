import json

# A basic deterministic rule-based engine to map ingredient keywords to archetypes
class ArchetypeClassifier:
    def __init__(self, seed_path="data/archetypes_seed.json"):
        with open(seed_path, 'r') as f:
            self.archetypes = json.load(f)
            
        # Basic keyword mapping (this should be expanded with a more comprehensive ruleset)
        self.rules = {
            "Biryani": ["rice", "chicken", "mutton", "biryani masala", "saffron", "yogurt", "mint"],
            "Dal": ["lentil", "split peas", "dal", "turmeric", "cumin", "mustard seeds"],
            "Curry": ["curry powder", "coconut milk", "garam masala", "chicken", "paneer", "gravy"],
            "Soup": ["broth", "stock", "water", "carrots", "celery", "onion", "simmer"],
            "Salad": ["lettuce", "greens", "cucumber", "tomato", "vinaigrette", "dressing"],
            "Dessert": ["sugar", "chocolate", "vanilla", "sweet", "cream", "cake", "cookie"],
            "Flatbread": ["flour", "water", "knead", "dough", "roll", "skillet", "roti", "naan"],
            "Pasta": ["pasta", "spaghetti", "macaroni", "tomato sauce", "cheese", "boil"],
            "Stir-fry": ["soy sauce", "wok", "oil", "stir", "fry", "vegetables", "ginger", "garlic"]
        }

    def classify(self, ingredients, directions):
        """
        Classifies a recipe into an archetype based on a simple scoring mechanism.
        Returns the archetype with the highest score, or 'Unknown' if no match.
        """
        text_corpus = (" ".join(ingredients) + " " + " ".join(directions)).lower()
        
        scores = {arch: 0 for arch in self.archetypes}
        
        for arch, keywords in self.rules.items():
            for keyword in keywords:
                if keyword in text_corpus:
                    scores[arch] += 1
                    
        # Find the max score
        max_arch = max(scores, key=scores.get)
        if scores[max_arch] > 0:
            return max_arch
        return "Unknown"

if __name__ == "__main__":
    classifier = ArchetypeClassifier()
    # Test with a dummy recipe
    ing = ["1 cup rice", "500g chicken", "1 tbsp biryani masala", "mint leaves"]
    dirs = ["boil rice", "cook chicken", "mix together"]
    print(f"Classification result: {classifier.classify(ing, dirs)}")
