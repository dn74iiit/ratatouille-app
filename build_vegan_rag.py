import os
import json
import pandas as pd

def build_knowledge_layer():
    print("[1/3] Initializing Vegan RAG Knowledge Builder...")
    
    # 1. Base curated Indian vegan culinary rules
    knowledge_chunks = [
        "To replace paneer in Palak Paneer or Matar Paneer, use extra-firm tofu. You must press the tofu for 20 minutes to remove excess moisture, then pan-fry in oil until golden brown so it holds shape in gravy.",
        "When substituting ghee for tempering (tadka), use coconut oil or cold-pressed mustard oil. Ghee has a high smoke point and rich mouthfeel; coconut oil is the best structural lipid equivalent.",
        "Heavy cream is commonly used to thicken Dal Makhani and Shahi gravies. To veganize, soak cashews in hot water for 30 minutes, blend with a splash of water into a silky paste, and stir in at the end of cooking.",
        "Eggs used as binders in Indian fritters (pakoras) or baking: substitute with 1 tbsp ground flaxseed (alsi) whisked into 3 tbsp warm water, or use besan (gram flour) slurry.",
        "Chicken provides fibrous chewiness in curries and biryanis. Substitute with soy chunks (Nutrela). Boil soy chunks in salted water for 10 minutes, squeeze dry completely, and pan-sear before simmering.",
        "Honey in glazes and chutneys: substitute with melted jaggery (gud) or date syrup in a 1:1 ratio to preserve stickiness and caramel notes.",
        "To replace yogurt or curd in marinades (Tikka Masala), blend soaked cashews with lemon juice, a splash of water, and a pinch of amchur (dry mango powder) to replicate lactic fermentation tang.",
        "Mutton has a dense, fibrous texture. Substitute with raw young green jackfruit (kathal). Pressure cook the jackfruit with turmeric and salt for 2 whistles before simmering in the onion-tomato masala.",
        "Milk in kheer (rice pudding) or payasam: substitute with full-fat coconut milk or almond milk. Simmer gently and do not boil almond milk vigorously to prevent curdling.",
        "To replicate fish or seafood in coastal curries, use sliced raw green banana (kachha kela) or scored king oyster mushroom stalks simmered in kokum and coconut gravy.",
        "For minced meat (keema) dishes, use rehydrated textured soy granules (soya keema). Sauté with cumin, coriander, and garam masala until dry and crumbly.",
        "To replace khoa or mawa in traditional Indian sweets (mithai), reduce thick cashew paste and almond flour in a non-stick pan with a touch of cardamom and jaggery.",
        "For custard or sweet binders without eggs, whisk 2 tsp cornstarch (cornflour) or arrowroot powder with plant milk and simmer until glossy and thickened."
    ]

    base_dir = os.path.dirname(__file__)
    
    # 2. Ingest from vegan_alternatives_db.csv if present
    alt_db_path = os.path.join(base_dir, "vegan_alternatives_db.csv")
    if os.path.exists(alt_db_path):
        print("[2/3] Ingesting culinary techniques from vegan_alternatives_db.csv...")
        try:
            df_alt = pd.read_csv(alt_db_path)
            for _, row in df_alt.iterrows():
                orig = str(row.get('original_ingredient', '')).strip()
                sub = str(row.get('vegan_substitute', '')).strip()
                notes = str(row.get('culinary_notes', '')).strip()
                spice = str(row.get('spice_bridge', '')).strip()
                lipid = row.get('lipid_deficit', 0)
                
                if orig and sub and notes and notes != 'nan' and len(notes) > 10:
                    chunk = f"SUBSTITUTION TECHNIQUE ({orig.upper()} -> {sub}): {notes}."
                    if lipid and str(lipid) != 'nan' and float(lipid) > 5.0:
                        chunk += f" Add 1-2 tsp neutral or coconut oil to compensate for the {float(lipid):.1f}g fat deficit."
                    if spice and spice != 'nan' and spice != '[]':
                        chunk += f" Recommended aromatic bridge: {spice}."
                    knowledge_chunks.append(chunk)
            print(f"   -> Added alternatives techniques. Total chunks now: {len(knowledge_chunks)}")
        except Exception as e:
            print(f"   [WARN] Could not parse vegan_alternatives_db.csv: {e}")

    # 3. Ingest from indian_vegan_recipes_db.csv if present
    recipe_db_path = os.path.join(base_dir, "indian_vegan_recipes_db.csv")
    if os.path.exists(recipe_db_path):
        print("[3/3] Ingesting authentic recipe archetypes from indian_vegan_recipes_db.csv...")
        try:
            df_rec = pd.read_csv(recipe_db_path)
            seen_dishes = set()
            for _, row in df_rec.iterrows():
                dish = str(row.get('dish_name', '')).strip()
                style = str(row.get('style', '')).strip()
                orig = str(row.get('original_non_veg', '')).strip()
                sub = str(row.get('vegan_substitute', '')).strip()
                ingr = str(row.get('ingredients', '')).strip()
                
                if not dish or dish in seen_dishes or not ingr or ingr == 'nan':
                    continue
                seen_dishes.add(dish)
                
                if sub and sub.lower() != 'nan':
                    chunk = (
                        f"AUTHENTIC INDIAN VEGAN DISH ({style} - {dish}): Replaces {orig} with {sub}. "
                        f"Key ingredients: {ingr}. "
                        f"Culinary directive: Sauté the {sub} in hot oil with whole spices (jeera, cloves, cardamom) "
                        f"before blending into the gravy for authentic flavor absorption."
                    )
                else:
                    chunk = (
                        f"AUTHENTIC INDIAN VEGAN DISH ({style} - {dish}): Traditional plant-based preparation. "
                        f"Key ingredients: {ingr}. "
                        f"Culinary directive: Temper whole spices in oil and simmer gently to develop authentic regional flavor."
                    )
                knowledge_chunks.append(chunk)
            print(f"   -> Added recipe templates. Total chunks now: {len(knowledge_chunks)}")
        except Exception as e:
            print(f"   [WARN] Could not parse indian_vegan_recipes_db.csv: {e}")

    # Deduplicate while preserving order
    deduped_chunks = []
    seen = set()
    for c in knowledge_chunks:
        c_clean = c.strip()
        if c_clean and c_clean not in seen:
            seen.add(c_clean)
            deduped_chunks.append(c_clean)

    # Save Knowledge Map JSON
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    map_path = os.path.join(data_dir, 'vegan_knowledge_map.json')
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(deduped_chunks, f, indent=2)
    print(f"Success! Saved {len(deduped_chunks)} clean knowledge chunks to '{map_path}'.")

if __name__ == "__main__":
    build_knowledge_layer()
