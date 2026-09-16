import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_vector_db():
    csv_path = "final_clean_50k_recipes_grams.csv"
    index_path = "data/recipes.index"
    metadata_path = "data/recipes_metadata.json"
    
    print(f"[*] Loading dataset from {csv_path}...")
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] File {csv_path} not found. Please ensure it exists.")
        return
        
    print(f"[*] Original dataset size: {len(df)} recipes.")
    
    # We drop any NaNs in the required columns
    df = df.dropna(subset=['title', 'ingredients', 'directions'])
    
    # Randomly sample 12,000 recipes as per the proposal (or fewer if dataset is smaller)
    sample_size = min(12000, len(df))
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"[*] Sampled {len(df)} recipes for vector indexing.")
    
    print("[*] Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings combining title, ingredients, and directions for maximum semantic accuracy
    print("[*] Encoding recipes into vector space (this may take a few minutes)...")
    sentences = df.apply(lambda row: f"Title: {row['title']}. Ingredients: {row['ingredients']}. Directions: {row['directions']}", axis=1).tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    # Ensure they are float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    print("[*] Building FAISS Index...")
    dimension = embeddings.shape[1]
    
    # Use L2 distance
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs("data", exist_ok=True)
    
    # Save the FAISS index
    print(f"[*] Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    # Save the metadata (mapping from index to recipe text and ID)
    print(f"[*] Saving metadata to {metadata_path}...")
    metadata = []
    for idx, row in df.iterrows():
        metadata.append({
            "faiss_id": idx,
            "recipe_id": str(row['Recipe_ID']),
            "title": row['title'],
            "ingredients": row['ingredients'],
            "directions": row['directions']
        })
        
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print("[OK] Vector Database construction complete!")

if __name__ == "__main__":
    build_vector_db()
