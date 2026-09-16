"""
Ratatouille Capstone — Evaluation Script (v2)
===============================================
Architecture:
  - Task 1 (Model Comparison):
      Step 1: Call Render /optimize-only  → gets SciPy ingredient weights (fast, no LLM)
      Step 2: Call HF Spaces DIRECTLY     → generates recipe (bypasses Render memory limits)

  - Task 2 (Budget Verification):
      Call Render /optimize-only only     → checks if budget is feasible (no LLM at all)

This approach avoids Render memory exhaustion from long-running LLM requests.

Usage:
    python eval/compare_models.py

Output files (written to eval/results/):
    - model_comparison.md
    - budget_check.md
"""

import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
API_BASE  = "https://ratatouille-backend.onrender.com"
HF_TOKEN  = os.getenv("HF_TOKEN")
HF_SPACE_V8 = "nd1490/ratatouille-inference"
HF_SPACE_V9 = "nd1490/ratatouille-inference-v9"

RENDER_TIMEOUT = 90    # /optimize-only is pure math but Render may take time to wake between calls
HF_TIMEOUT     = 600   # 10 min max for HF Space LLM generation

os.makedirs("eval/results", exist_ok=True)

# ============================================================
# 15 FIXED TEST CASES
# ============================================================
TEST_CASES = [
    # (ingredients, budget, servings, label)
    (["paneer", "tomato", "onion", "garlic"],               150, 2, "Paneer Curry"),
    (["chicken", "yogurt", "ginger", "cumin"],              200, 2, "Chicken Masala"),
    (["potato", "peas", "cumin", "coriander"],              80,  2, "Aloo Matar"),
    (["lentils", "spinach", "garlic", "tomato"],            60,  2, "Dal Palak"),
    (["rice", "egg", "onion", "soy sauce"],                 100, 2, "Egg Fried Rice"),
    (["mushroom", "butter", "garlic", "cream"],             180, 2, "Mushroom Soup"),
    (["banana", "sugar", "flour", "milk"],                  70,  4, "Banana Cake"),
    (["chickpea", "tomato", "onion", "chili"],              90,  2, "Chana Masala"),
    (["fish", "coconut", "turmeric", "green chili"],        220, 3, "Fish Curry"),
    (["carrot", "ginger", "orange", "honey"],               120, 2, "Carrot Soup"),
    (["tofu", "broccoli", "sesame", "soy sauce"],           130, 2, "Tofu Stir Fry"),
    (["mutton", "onion", "potato", "bay leaf"],             350, 4, "Mutton Stew"),
    (["oats", "banana", "milk", "honey"],                   50,  1, "Overnight Oats"),
    (["cauliflower", "potato", "cumin", "turmeric"],        75,  2, "Aloo Gobi"),
    (["pasta", "tomato", "basil", "garlic"],                110, 2, "Tomato Pasta"),
]

BUDGET_TEST_CASES = [
    (["potato", "onion", "tomato"],                         [50, 100, 200, 500], [1, 2, 4], "Simple Sabzi"),
    (["chicken", "onion", "garlic", "ginger", "tomato"],   [100, 200, 350, 600], [2, 4],   "Chicken Curry"),
    (["paneer", "spinach", "cream", "garlic"],              [80, 150, 300],       [2],       "Palak Paneer"),
]

# ============================================================
# HELPERS
# ============================================================
def optimize_only(ingredients, budget, servings, state="Delhi"):
    """Call Render /optimize-only — just SciPy math, no LLM. Very fast."""
    try:
        resp = requests.post(
            f"{API_BASE}/optimize-only",
            json={"ingredients": ingredients, "budget": budget,
                  "servings": servings, "state": state, "model_version": "v8"},
            timeout=RENDER_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "calculated_ingredients": [], "archetype": "N/A",
                "message": str(e)}


def apply_slop_filter(ai_text: str) -> str:
    """Apply the exact same post-processing as api.py /generate-recipe endpoint."""
    # 1. Cut at Llama 3 end-of-turn token
    ai_text = ai_text.split("<|eot_id|>")[0].strip()

    # 2. If model echoed the prompt and added another TITLE block, take from second title onward
    if "### TITLE:\n" in ai_text:
        ai_text = ai_text.split("### TITLE:\n")[1].strip()

    # 3. Cut standard looping / notes signatures
    cut_phrases = [
        "\nEnjoy!", "\nServe hot", "\nBon Apetit", "\nChef's Note:",
        "\nVariations:", "\nServing suggestion:", "\nNote:"
    ]
    for phrase in cut_phrases:
        if phrase in ai_text:
            ai_text = ai_text.split(phrase)[0].strip()

    # 4. If the model starts a new ### section inside DIRECTIONS, cut it off
    if "### DIRECTIONS:\n" in ai_text:
        parts = ai_text.split("### DIRECTIONS:\n")
        title_part = parts[0]
        directions_part = parts[1]
        if "\n### " in directions_part:
            directions_part = directions_part.split("\n### ")[0]
        ai_text = f"{title_part}### DIRECTIONS:\n{directions_part}".strip()

    return ai_text


def generate_recipe_direct(calc_ingredients, archetype, hf_space):
    """Call HF Space directly via gradio_client — bypasses Render entirely.
    Applies the same slop filter as api.py /generate-recipe endpoint.
    """
    try:
        from gradio_client import Client
        ingr_text = "\n".join(f"- {i}" for i in calc_ingredients)
        system_instruction = (
            f"You are a master chef. Write a highly detailed, appetizing recipe for a {archetype}.\n"
            f"First, provide a creative title. Then, provide step-by-step cooking directions using proper culinary techniques.\n"
            f"CRITICAL RULES:\n"
            f"1. Ensure all raw ingredients are explicitly cooked in the instructions.\n"
            f"2. Do not change the ingredient quantities provided.\n"
            f"3. DO NOT include any 'Notes', 'Tips', or conversational rambling. Stop after the final serving step."
        )
        prompt = f"<|begin_of_text|>{system_instruction}\n\n### INGREDIENTS:\n{ingr_text}\n### TITLE:\n"

        client = Client(hf_space, token=HF_TOKEN)
        result = client.predict(
            prompt, 400, 0.6, 0.9, 1.15, True,
            api_name="/generate"
        )
        raw = result.strip() if isinstance(result, str) and result.strip() else "[Empty response]"
        return apply_slop_filter(raw)   # ← same filter as the live app
    except Exception as e:
        return f"[ERROR: {e}]"


# ============================================================
# TASK 1 — MODEL COMPARISON
# ============================================================
def run_model_comparison():
    print("\n" + "="*60)
    print("TASK 1: Model Comparison (15 prompts x 2 models)")
    print("  Step 1: Render /optimize-only  (SciPy math)")
    print("  Step 2: HF Space direct call   (LLM generation)")
    print("="*60)

    lines = [
        "# Ratatouille — Model Comparison Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Models:** RecipeDB3-trained (v8) vs RecipeDB1.1-updated (v9)  ",
        "",
        "---", "",
    ]

    spaces = {
        "RecipeDB3-trained (v8)": HF_SPACE_V8,
        "RecipeDB1.1-updated (v9)": HF_SPACE_V9,
    }

    for i, (ingredients, budget, servings, label) in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/15] {label} | Budget: Rs.{budget} | Servings: {servings}")

        # Step 1: Get optimized ingredients from Render (same for both models)
        print("  -> Optimizing ingredients via Render...", end=" ", flush=True)
        opt = optimize_only(ingredients, budget, servings)
        archetype     = opt.get("archetype", "N/A")
        calc_ingr     = opt.get("calculated_ingredients", [])
        opt_status    = opt.get("status", "error")
        print(f"done (archetype={archetype})")

        lines += [
            f"## Prompt {i}: {label}",
            f"**Ingredients:** {', '.join(ingredients)}  ",
            f"**Budget:** Rs.{budget} | **Servings:** {servings}  ",
            f"**Archetype detected:** `{archetype}`  ",
            "",
            "**Optimized ingredients (same for both models):**",
        ]
        for ci in calc_ingr:
            lines.append(f"- {ci}")
        lines.append("")

        if opt_status == "error" or not calc_ingr:
            lines += [
                "> Budget infeasible — skipping recipe generation.",
                "", "---", "",
            ]
            print("  [SKIP] Budget infeasible.")
            continue

        # Step 2: Generate recipe from each HF Space directly
        for display_name, space_id in spaces.items():
            print(f"  -> Calling {display_name} directly...", end=" ", flush=True)
            t0 = time.time()
            recipe_text = generate_recipe_direct(calc_ingr, archetype, space_id)
            elapsed = round(time.time() - t0, 1)
            print(f"done ({elapsed}s)")

            lines += [
                f"### {display_name}",
                f"- **Response time:** {elapsed}s  ",
                "",
                "**Generated Recipe:**",
                "",
                "```",
                recipe_text,
                "```",
                "",
            ]

        lines += ["---", ""]

    out_path = "eval/results/model_comparison.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDone! Model comparison saved -> {out_path}")


# ============================================================
# TASK 2 — BUDGET CONSTRAINT VERIFICATION
# ============================================================
def run_budget_check():
    print("\n" + "="*60)
    print("TASK 2: Budget Constraint Verification")
    print("  (Render /optimize-only only — no LLM calls)")
    print("="*60)

    lines = [
        "# Ratatouille — Budget Constraint Verification Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        "",
        "> For each test, the optimizer should return ingredient weights within the given budget.",
        "> 'Infeasible' means the budget is mathematically too low for those ingredients.",
        "",
        "---", "",
    ]

    for ingredients, budget_levels, servings_levels, display in BUDGET_TEST_CASES:
        print(f"\n  -> {display}")
        lines += [f"## {display}", f"**Ingredients:** {', '.join(ingredients)}  ", ""]

        for servings in servings_levels:
            lines += [
                f"### Servings: {servings}", "",
                "| Budget (Rs.) | Status | Archetype | Optimized Ingredients |",
                "|---|---|---|---|",
            ]

            for budget in budget_levels:
                print(f"     Budget Rs.{budget}, {servings} serving(s)...", end=" ", flush=True)
                result = optimize_only(ingredients, budget, servings)
                status    = result.get("status", "error")
                archetype = result.get("archetype", "N/A")
                calc      = result.get("calculated_ingredients", [])
                calc_str  = ", ".join(calc) if calc else "—"

                if status == "error" or not calc:
                    row = f"| Rs.{budget} | Infeasible | {archetype} | {result.get('message','N/A')[:60]} |"
                else:
                    row = f"| Rs.{budget} | Feasible | {archetype} | {calc_str} |"
                lines.append(row)
                print("done")

            lines.append("")

        lines += ["---", ""]

    out_path = "eval/results/budget_check.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDone! Budget check saved -> {out_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\nRatatouille Evaluation Script v2")
    print(f"API: {API_BASE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Wake up Render
    print("\n[*] Checking API health...")
    api_ok = False
    for attempt in range(1, 6):
        try:
            health = requests.get(f"{API_BASE}/health", timeout=90)
            print(f"[OK] API reachable — {health.json()}")
            api_ok = True
            break
        except requests.exceptions.Timeout:
            print(f"[{attempt}/5] Render waking up, waiting 20s...")
            time.sleep(20)
        except Exception as e:
            print(f"[{attempt}/5] Error: {e} — waiting 20s...")
            time.sleep(20)

    if not api_ok:
        print("[ERROR] API did not respond after 5 attempts.")
        exit(1)

    run_budget_check()      # Fast — no LLM. Run first.
    run_model_comparison()  # Slow — direct HF Space calls. Run second.

    print(f"\n\nAll done! Results saved to eval/results/")
