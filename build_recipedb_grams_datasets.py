"""
Build RecipeDB datasets in the same schema/format as final_clean_50k_recipes_grams.csv,
with ingredient quantities normalized to grams where possible.

Outputs:
  - recipedb_full_grams.csv      — all recipes from new data recipeDB 1
  - recipedb_oven_grams.csv      — subset where RecipeDB_general Utensils mentions oven
"""

from __future__ import annotations

import ast
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "new data recipeDB 1"
GENERAL_PATH = REPO / "RecipeDB_general - RecipeDB_general.csv"
TARGET_COLUMNS = [
    "Recipe_ID",
    "title",
    "Cuisine",
    "Category",
    "Prep Time",
    "Cook Time",
    "Total Time",
    "Servings",
    "ingredients",
    "directions",
    "Nutrients",
    "Keywords",
    "Source",
    "URL",
    "Image_ID",
    "Image_URL",
    "NA_Image",
    "Language",
    "Ratings",
    "Ratings_Count",
    "Description",
]

UNIT_TO_G = {
    "cup": 240.0,
    "cups": 240.0,
    "tablespoon": 14.79,
    "tablespoons": 14.79,
    "tbsp": 14.79,
    "tbs": 14.79,
    "teaspoon": 4.93,
    "teaspoons": 4.93,
    "tsp": 4.93,
    "fluid ounce": 29.57,
    "fluid ounces": 29.57,
    "fl oz": 29.57,
    "pint": 473.18,
    "pints": 473.18,
    "quart": 946.35,
    "quarts": 946.35,
    "gallon": 3785.41,
    "gallons": 3785.41,
    "liter": 1000.0,
    "liters": 1000.0,
    "litre": 1000.0,
    "litres": 1000.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "ml": 1.0,
    " l": 1.0,
    "pound": 453.59,
    "pounds": 453.59,
    "lb": 453.59,
    "lbs": 453.59,
    " lb": 453.59,
    "ounce": 28.35,
    "ounces": 28.35,
    "oz": 28.35,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "kg": 1000.0,
    "gram": 1.0,
    "grams": 1.0,
    "g": 1.0,
    " g": 1.0,
    "milligram": 0.001,
    "milligrams": 0.001,
    "mg": 0.001,
    "pinch": 0.3,
    "pinches": 0.3,
    "dash": 0.6,
    "dashes": 0.6,
    "drop": 0.05,
    "drops": 0.05,
    "handful": 40.0,
    "handfuls": 40.0,
    "stick": 113.0,
    "sticks": 113.0,
    "clove": 5.0,
    "cloves": 5.0,
    "slice": 25.0,
    "slices": 25.0,
    "piece": 30.0,
    "pieces": 30.0,
    "can": 400.0,
    "cans": 400.0,
    "package": 250.0,
    "packages": 250.0,
    "pkg": 250.0,
    "bunch": 100.0,
    "bunches": 100.0,
    "sprig": 2.0,
    "sprigs": 2.0,
    "jar": 350.0,
    "jars": 350.0,
    "container": 300.0,
    "containers": 300.0,
    "bag": 200.0,
    "bags": 200.0,
}

# Units kept as count-style strings (like final_clean_50k), not forced to X.Xg prefix.
COUNT_STYLE_UNITS = {
    "clove",
    "cloves",
    "slice",
    "slices",
    "piece",
    "pieces",
    "egg",
    "eggs",
    "leaf",
    "leaves",
    "bay",
    "stalk",
    "stalks",
    "sprig",
    "sprigs",
    "inch",
    "inches",
    "strip",
    "strips",
    "breast",
    "breasts",
    "banana",
    "bananas",
    "head",
    "heads",
    "fillet",
    "fillets",
    "boneless",
    "skinless",
    "halved",
    "whole",
    "large",
    "small",
    "medium",
}

PAREN_G_UNITS = {"can", "cans", "jar", "jars", "package", "packages", "pkg", "container", "containers"}


def parse_quantity(raw) -> float | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if "/" in s:
            return float(Fraction(s))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def format_qty(qty: float) -> str:
    if abs(qty - round(qty)) < 1e-6:
        return str(int(round(qty)))
    return f"{qty:g}"


def normalize_unit(raw) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    u = str(raw).strip().lower()
    return u or None


def build_ingredient_name(row: pd.Series) -> str:
    base = str(row.get("ingredient", "")).strip()
    state = row.get("state")
    size = row.get("size")
    parts = [base]
    if isinstance(state, str) and state.strip():
        parts.append(state.strip())
    if isinstance(size, str) and size.strip() and size.strip().lower() not in parts[0].lower():
        parts.insert(0, size.strip())
    return ", ".join(parts) if len(parts) > 1 else parts[0]


def phrase_to_grams(phrase: str) -> str:
    """Parse leading quantity+unit from ingredient_Phrase when structured fields fail."""
    if not isinstance(phrase, str) or not phrase.strip():
        return phrase
    m = re.match(
        r"^(?P<qty>\d+\s*/\s*\d+|\d+\.\d+|\d+)\s+"
        r"(?P<unit>[a-zA-Z][a-zA-Z.\s]{0,20}?)\s+"
        r"(?P<rest>.+)$",
        phrase.strip(),
    )
    if not m:
        return phrase
    qty = parse_quantity(m.group("qty"))
    unit = normalize_unit(m.group("unit"))
    rest = m.group("rest").strip()
    if qty is None or unit is None:
        return phrase
    row = pd.Series({"quantity": qty, "unit": unit, "ingredient": rest, "state": np.nan, "size": np.nan})
    return format_ingredient_row(row, fallback_phrase=phrase)


def format_ingredient_row(row: pd.Series, fallback_phrase: str | None = None) -> str:
    phrase = fallback_phrase
    if phrase is None:
        phrase = str(row.get("ingredient_Phrase", "")).strip()

    qty = parse_quantity(row.get("quantity"))
    unit = normalize_unit(row.get("unit"))
    name = build_ingredient_name(row)

    if qty is None or unit is None:
        if phrase:
            return phrase_to_grams(phrase)
        return name or phrase or ""

    if unit in COUNT_STYLE_UNITS or unit not in UNIT_TO_G:
        if phrase:
            return phrase
        return f"{format_qty(qty)} {unit} {name}".strip()

    grams = qty * UNIT_TO_G[unit]

    if unit in PAREN_G_UNITS:
        return f"{format_qty(qty)} ({grams:.1f}g) {unit} {name}".strip()

    if unit in COUNT_STYLE_UNITS:
        return f"{format_qty(qty)} {unit} {name}".strip()

    return f"{grams:.1f}g {name}".strip()


def process_steps(step_str: str) -> str:
    if not isinstance(step_str, str):
        return str([])
    parts = [p.strip() for p in step_str.split(".") if p.strip()]
    return str(parts)


def format_servings(x) -> str:
    if pd.isna(x):
        return ""
    x_str = str(x).strip()
    if x_str.isdigit():
        return f"{x_str} servings"
    try:
        val = float(x_str)
        if val.is_integer():
            return f"{int(val)} servings"
    except ValueError:
        pass
    return x_str


def build_nutrients_dict(row: pd.Series) -> str:
    keys = {
        "calories": "Calories",
        "carbs": "Carbohydrate, by difference (g)",
        "protein": "Protein (g)",
        "fat": "Total lipid (fat) (g)",
        "energy": "Energy (kcal)",
    }
    out = {}
    for label, col in keys.items():
        if col in row.index and pd.notna(row[col]):
            out[label] = row[col]
    return str(out)


def load_ingredients_grouped() -> pd.DataFrame:
    print("Loading and converting ingredients to grams...")
    path = DATA_DIR / "RecipeDB_ingredient_phrase.csv"
    chunks = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        chunk["formatted"] = chunk.apply(format_ingredient_row, axis=1)
        grouped = (
            chunk.groupby("recipe_no")["formatted"]
            .apply(list)
            .reset_index()
            .rename(columns={"recipe_no": "Recipe_id", "formatted": "ingredients"})
        )
        chunks.append(grouped)

    full = pd.concat(chunks, ignore_index=True)
    # Some recipes span chunk boundaries — regroup if duplicate ids (shouldn't happen per chunk)
    full = (
        full.groupby("Recipe_id")["ingredients"]
        .apply(lambda lists: sum(lists, []))
        .reset_index()
    )
    full["ingredients"] = full["ingredients"].apply(str)
    return full


def build_base_dataframe() -> pd.DataFrame:
    print("Loading metadata and instructions...")
    df_main = pd.read_csv(DATA_DIR / "RecipeDB1_ID_Title_Serving.csv")
    df_main = df_main.rename(columns={"Recipe_title": "title"})

    df_gen = pd.read_csv(
        GENERAL_PATH,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    )
    df_gen["Recipe_id"] = df_gen["Recipe_id"].astype(int)
    df_gen["has_oven"] = (
        df_gen["Utensils"].astype(str).str.contains("oven", case=False, na=False)
    )

    with open(DATA_DIR / "RecipeDB_instructions.json", encoding="utf-8") as f:
        instructions_data = json.load(f)
    df_inst = pd.DataFrame(instructions_data)
    df_inst["recipe_id"] = df_inst["recipe_id"].astype(int)
    df_inst["directions"] = df_inst["steps"].apply(process_steps)

    ing_grouped = load_ingredients_grouped()

    print("Merging...")
    df = df_main.merge(ing_grouped, on="Recipe_id", how="left")
    df = df.merge(df_inst, left_on="Recipe_id", right_on="recipe_id", how="left")
    df = df.merge(
        df_gen[
            [
                "Recipe_id",
                "Region",
                "Sub_region",
                "prep_time",
                "cook_time",
                "total_time",
                "url",
                "Source",
                "img_url",
                "Calories",
                "Carbohydrate, by difference (g)",
                "Protein (g)",
                "Total lipid (fat) (g)",
                "Energy (kcal)",
                "Utensils",
                "has_oven",
            ]
        ],
        on="Recipe_id",
        how="left",
    )

    df["Recipe_ID"] = df["Recipe_id"]
    df["Cuisine"] = df["Region"].fillna("Unknown")
    df["Category"] = df["Sub_region"].fillna("Unknown")
    df["Prep Time"] = df["prep_time"]
    df["Cook Time"] = df["cook_time"]
    df["Total Time"] = df["total_time"]
    df["Servings"] = df["servings"].apply(format_servings)
    df["Nutrients"] = df.apply(build_nutrients_dict, axis=1)
    df["Keywords"] = str([])
    df["Source"] = df["Source"].fillna("RecipeDB")
    df["URL"] = df["url"].fillna("")
    df["Image_ID"] = -1
    df["Image_URL"] = df["img_url"].fillna("")
    df["NA_Image"] = -1
    df["Language"] = "en"
    df["Ratings"] = np.nan
    df["Ratings_Count"] = np.nan
    df["Description"] = df["title"].apply(
        lambda x: f"Recipe for {x}" if pd.notna(x) else ""
    )

    df["ingredients"] = df["ingredients"].fillna(str([]))
    df["directions"] = df["directions"].fillna(str([]))

    return df


def validate_sample(df: pd.DataFrame, reference_path: Path, n: int = 200) -> None:
    """Quick sanity check against 50k ingredient style."""
    ref = pd.read_csv(reference_path, nrows=n)
    ref_g = 0
    for ing_str in ref["ingredients"]:
        for item in ast.literal_eval(ing_str):
            if re.match(r"^\d+\.?\d*g ", item) or re.search(r"\(\d+\.?\d*g\)", item):
                ref_g += 1
    new_g = 0
    sample = df.sample(min(n, len(df)), random_state=42)
    for ing_str in sample["ingredients"]:
        for item in ast.literal_eval(ing_str):
            if re.match(r"^\d+\.?\d*g ", item) or re.search(r"\(\d+\.?\d*g\)", item):
                new_g += 1
    print(f"Gram-style ingredients (sample n={n}): reference ~{ref_g}, new ~{new_g}")


def main() -> None:
    df = build_base_dataframe()
    df_out = df[TARGET_COLUMNS].copy()

    full_path = REPO / "recipedb_full_grams.csv"
    oven_path = REPO / "recipedb_oven_grams.csv"

    print(f"Saving full dataset ({len(df_out):,} rows) -> {full_path.name}")
    df_out.to_csv(full_path, index=False)

    df_oven = df_out[df["has_oven"].values].copy()
    print(f"Saving oven subset ({len(df_oven):,} rows) -> {oven_path.name}")
    df_oven.to_csv(oven_path, index=False)

    validate_sample(df_out, REPO / "final_clean_50k_recipes_grams.csv")
    print("Done.")


if __name__ == "__main__":
    main()
