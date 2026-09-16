"""
Apply slop filter to already-generated model_comparison.md
Runs instantly — no LLM calls needed.
"""
import re

def apply_slop_filter(ai_text):
    ai_text = ai_text.split("<|eot_id|>")[0].strip()
    if "### TITLE:\n" in ai_text:
        ai_text = ai_text.split("### TITLE:\n")[1].strip()
    cut_phrases = [
        "\nEnjoy!", "\nServe hot", "\nBon Apetit", "\nChef's Note:",
        "\nVariations:", "\nServing suggestion:", "\nNote:"
    ]
    for phrase in cut_phrases:
        if phrase in ai_text:
            ai_text = ai_text.split(phrase)[0].strip()
    if "### DIRECTIONS:\n" in ai_text:
        parts = ai_text.split("### DIRECTIONS:\n")
        title_part = parts[0]
        directions_part = parts[1]
        if "\n### " in directions_part:
            directions_part = directions_part.split("\n### ")[0]
        ai_text = f"{title_part}### DIRECTIONS:\n{directions_part}".strip()
    return ai_text

with open("eval/results/model_comparison.md", "r", encoding="utf-8") as f:
    content = f.read()

def filter_block(match):
    raw = match.group(1)
    filtered = apply_slop_filter(raw)
    return f"```\n{filtered}\n```"

filtered_content = re.sub(r"```\n(.*?)\n```", filter_block, content, flags=re.DOTALL)

with open("eval/results/model_comparison_filtered.md", "w", encoding="utf-8") as f:
    f.write(filtered_content)

print("Done! Saved to eval/results/model_comparison_filtered.md")
