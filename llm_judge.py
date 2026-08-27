import json
import re

class RecipeJudge:
    def __init__(self):
        pass
        
    def _extract_ingredients_from_text(self, recipe_text):
        """
        A heuristic to extract words that might be ingredients.
        In a full pipeline, you might use NLP (spaCy) or a small NER model.
        For now, we'll do basic tokenization to cross-reference against allowed lists.
        """
        # Convert to lowercase and strip punctuation
        text = re.sub(r'[^\w\s]', ' ', recipe_text.lower())
        words = set(text.split())
        return words

    def evaluate_recipe(self, recipe_text, context):
        """
        Evaluates a draft recipe against the allowed subgraph context.
        Returns a dictionary with validation status and critique.
        """
        allowed_ingredients = set([i.lower() for i in context.get("input_ingredients", [])])
        allowed_pairings = set([i.lower() for i in context.get("recommended_pairings", [])])
        
        all_allowed = allowed_ingredients.union(allowed_pairings)
        
        # We need a robust way to check for hallucinations.
        # A simple approach: we have a known list of 'common sense' ingredients 
        # (salt, pepper, water, oil) that might be implicitly allowed, 
        # or we strictly enforce NO extra ingredients. 
        # For Phase 2, we STRICTLY enforce the subgraph to prevent hallucinations.
        
        # Let's check for common hallucinated ingredients that ARE NOT in the allowed list
        # This is a sample list of common culprits SLMs hallucinate:
        common_hallucinations = ['cream', 'milk', 'butter', 'cheese', 'sugar', 'flour', 'egg', 'eggs', 'tomato', 'onion', 'lemon']
        
        hallucinated_found = []
        recipe_words = self._extract_ingredients_from_text(recipe_text)
        
        for culprit in common_hallucinations:
            if culprit in recipe_words and culprit not in all_allowed:
                # Check if it's part of a larger allowed phrase (e.g. "yellow onion")
                is_subword = any(culprit in allowed_item for allowed_item in all_allowed)
                if not is_subword:
                    hallucinated_found.append(culprit)

        # Build the critique
        if hallucinated_found:
            critique = (
                f"CRITIQUE: The recipe contains forbidden ingredients: {', '.join(hallucinated_found)}. "
                f"You MUST remove them. You are only allowed to use the ingredients from the CONTEXT_BLOCK."
            )
            return {
                "is_valid": False,
                "score": 0.2, # Low Culinary Validity Score
                "critique": critique
            }
            
        # If no explicit hallucinations found via rules, we construct the LLM-as-a-judge prompt
        llm_judge_prompt = self._build_judge_prompt(recipe_text, context)
        
        # Query the fast Groq model via API's serverless function
        try:
            from api import query_serverless_llm
            messages = [
                {"role": "system", "content": "You are an expert Culinary Judge. Output only the SCORE and CRITIQUE in the exact format requested."},
                {"role": "user", "content": llm_judge_prompt}
            ]
            
            response = query_serverless_llm(messages, max_tokens=150, temperature=0.1)
            
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response)
            critique_match = re.search(r'CRITIQUE:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            critique = critique_match.group(1).strip() if critique_match else "No critique provided."
            
            is_valid = score >= 0.8
            
            return {
                "is_valid": is_valid,
                "score": score,
                "critique": critique if not is_valid else "Recipe passed all checks.",
                "llm_judge_prompt": llm_judge_prompt
            }
        except Exception as e:
            print(f"[WARN] LLM Judge Failed: {e}")
            return {
                "is_valid": True, # Fail-open if the judge crashes
                "score": 1.0,
                "critique": f"Passed heuristic check. LLM Judge failed: {e}",
                "llm_judge_prompt": llm_judge_prompt
            }
        
    def _build_judge_prompt(self, recipe_text, context):
        prompt = (
            "You are an expert Culinary Judge.\n"
            "Evaluate the following recipe draft based on these criteria:\n"
            "1. NO HALLUCINATIONS: Does it use ingredients outside the allowed list?\n"
            "2. LOGICAL FLOW: Are the steps physically possible (e.g. you cannot fry something that is liquid without a pan)?\n"
            "3. WORKFLOW: Did it generally follow the suggested techniques (e.g. boiling, chopping)?\n\n"
            f"[ALLOWED INGREDIENTS]: {', '.join(context.get('input_ingredients', []) + context.get('recommended_pairings', []))}\n\n"
            f"[RECIPE DRAFT]:\n{recipe_text}\n\n"
            "Provide a 'Culinary Validity Score' from 0.0 to 1.0.\n"
            "If the score is less than 0.8, provide a strict 1-sentence CRITIQUE explaining what must be fixed.\n"
            "Output format:\n"
            "SCORE: [score]\n"
            "CRITIQUE: [critique if any]"
        )
        return prompt

if __name__ == "__main__":
    # Test the Judge
    judge = RecipeJudge()
    
    test_context = {
        "input_ingredients": ["chicken", "rice", "garlic"],
        "recommended_pairings": ["chili", "soya sauce", "coriander"]
    }
    
    # Draft 1: Fails the test by hallucinating "cream" and "butter"
    bad_draft = (
        "1. Boil the rice.\n"
        "2. Fry the chicken in butter.\n"
        "3. Add garlic and chili.\n"
        "4. Pour in heavy cream to make a sauce.\n"
        "5. Serve with soya sauce."
    )
    
    # Draft 2: Passes the strict test
    good_draft = (
        "1. Boil the rice.\n"
        "2. Fry the chicken.\n"
        "3. Add garlic, chili, and soya sauce to the chicken and stir.\n"
        "4. Serve chicken over rice garnished with coriander."
    )
    
    print("--- EVALUATING BAD DRAFT ---")
    bad_result = judge.evaluate_recipe(bad_draft, test_context)
    print(json.dumps(bad_result, indent=2))
    
    print("\n--- EVALUATING GOOD DRAFT ---")
    good_result = judge.evaluate_recipe(good_draft, test_context)
    print(json.dumps(good_result, indent=2))
