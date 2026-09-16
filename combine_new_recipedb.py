import pandas as pd
import json
import numpy as np
import re

def process_steps(step_str):
    if not isinstance(step_str, str):
        return str([])
    # Split by '.' and strip, but avoid empty strings
    parts = [p.strip() for p in step_str.split('.') if p.strip()]
    # Add back the period if you want, but looking at directions, usually it's just the sentence
    return str(parts)

def main():
    target_columns = [
        'Recipe_ID', 'title', 'Cuisine', 'Category', 'Prep Time', 'Cook Time', 
        'Total Time', 'Servings', 'ingredients', 'directions', 'Nutrients', 
        'Keywords', 'Source', 'URL', 'Image_ID', 'Image_URL', 'NA_Image', 
        'Language', 'Ratings', 'Ratings_Count', 'Description'
    ]

    print("Loading data...")
    # Load main metadata
    df_main = pd.read_csv('new data recipeDB 1/RecipeDB1_ID_Title_Serving.csv')
    df_main.rename(columns={'Recipe_title': 'title'}, inplace=True)
    
    def format_serving(x):
        if pd.isna(x):
            return ""
        x_str = str(x).strip()
        if x_str.isdigit():
            return f"{x_str} servings"
        # some decimals might come in like 4.0
        try:
            val = float(x_str)
            if val.is_integer():
                return f"{int(val)} servings"
        except ValueError:
            pass
        return x_str

    # Format servings
    df_main['Servings'] = df_main['servings'].apply(format_serving)

    # Load ingredients
    df_ing = pd.read_csv('new data recipeDB 1/RecipeDB_ingredient_phrase.csv')
    # Group ingredients into list of strings, then convert to string representation of list
    ing_grouped = df_ing.groupby('recipe_no')['ingredient_Phrase'].apply(list).reset_index()
    ing_grouped.rename(columns={'recipe_no': 'Recipe_id', 'ingredient_Phrase': 'ingredients'}, inplace=True)
    ing_grouped['ingredients'] = ing_grouped['ingredients'].apply(str)

    # Load instructions
    with open('new data recipeDB 1/RecipeDB_instructions.json', 'r', encoding='utf-8') as f:
        instructions_data = json.load(f)
    df_inst = pd.DataFrame(instructions_data)
    df_inst['recipe_id'] = df_inst['recipe_id'].astype(int)
    # Convert 'steps' string into a stringified list of sentences
    df_inst['directions'] = df_inst['steps'].apply(process_steps)

    print("Merging data...")
    # Merge
    df_final = df_main.merge(ing_grouped, left_on='Recipe_id', right_on='Recipe_id', how='left')
    df_final = df_final.merge(df_inst, left_on='Recipe_id', right_on='recipe_id', how='left')

    # Assign mapped columns
    df_final['Recipe_ID'] = df_final['Recipe_id']
    
    # Fill in missing columns to match target dataset
    df_final['Cuisine'] = 'Unknown'
    df_final['Category'] = 'Unknown'
    df_final['Prep Time'] = np.nan
    df_final['Cook Time'] = np.nan
    df_final['Total Time'] = np.nan
    df_final['Nutrients'] = str({})
    df_final['Keywords'] = str([])
    df_final['Source'] = 'RecipeDB'
    df_final['URL'] = ''
    df_final['Image_ID'] = -1
    df_final['Image_URL'] = ''
    df_final['NA_Image'] = -1
    df_final['Language'] = 'en'
    df_final['Ratings'] = np.nan
    df_final['Ratings_Count'] = np.nan
    df_final['Description'] = df_final['title'].apply(lambda x: f"Recipe for {x}" if pd.notna(x) else "")

    print("Filtering and organizing columns...")
    # Fill NaN for stringified lists where missing
    df_final['ingredients'] = df_final['ingredients'].fillna(str([]))
    df_final['directions'] = df_final['directions'].fillna(str([]))

    # Keep exactly the target columns
    df_final = df_final[target_columns]

    output_file = 'RecipeDB_formatted_like_50k.csv'
    print(f"Saving to {output_file}...")
    df_final.to_csv(output_file, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
