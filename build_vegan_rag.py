import os
import pandas as pd
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def build_knowledge_layer():
    print("Initializing FAISS Vegan RAG Builder...")
    
    # 1. Load the Model
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Prepare the Ground-Truth Knowledge
    # In a real environment, you'd load the 6000+ Archana's Kitchen CSV here:
    # df = pd.read_csv('archanas_kitchen_vegan.csv')
    # For now, since downloading the massive CSV requires Kaggle API keys, 
    # we are seeding the FAISS index with highly detailed culinary chunks.
    
    knowledge_chunks = [
        "To replace paneer in Palak Paneer, use extra-firm tofu. You must press the tofu for 20 minutes to remove moisture, then pan-fry it until golden brown so it doesn't crumble in the spinach gravy.",
        "When substituting ghee for tempering (tadka), use coconut oil or mustard oil. Ghee has a high smoke point, so coconut oil is the best structural equivalent, though peanut oil works well for savory dals.",
        "Heavy cream is commonly used to thicken Dal Makhani. To veganize this, soak cashews in hot water for 30 minutes, then blend into a smooth paste. Stir it in at the very end of cooking.",
        "Eggs are used as binders in Indian baking and fritters. For every 1 egg, substitute with 1 tablespoon of ground flaxseed (alsi) mixed with 3 tablespoons of water. Let it sit for 5 minutes until gelatinous.",
        "Chicken provides chewiness in dry curries. Substitute with soy chunks (Nutrela). You must boil the soy chunks in salted water for 10 minutes, squeeze all the water out, and pan-fry them before adding to the curry.",
        "Honey is used as a sweetener or glaze. Substitute with jaggery (gud) melted with a tiny splash of warm water.",
        "To replace yogurt or curd in marinades (like Tikka Masala), blend soaked cashews with lemon juice and a pinch of dry mango powder (amchur) to replicate the lactic tanginess.",
        "Mutton has a fibrous, hearty texture. Substitute with raw young jackfruit (kathal). Pressure cook the jackfruit for 2 whistles before simmering it in the masala.",
        "Milk is used as a liquid base in kheer (rice pudding). Substitute with full-fat coconut milk or almond milk, but do not boil the almond milk on high heat as it may split."
    ]
    
    print(f"Embedding {len(knowledge_chunks)} culinary knowledge chunks into vectors...")
    
    # 3. Vectorize the Chunks
    embeddings = model.encode(knowledge_chunks)
    
    # 4. Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # 5. Save the FAISS Index and the Mapping
    os.makedirs('data', exist_ok=True)
    faiss.write_index(index, 'data/vegan_knowledge.index')
    
    with open('data/vegan_knowledge_map.json', 'w') as f:
        json.dump(knowledge_chunks, f)
        
    print("Success! Built FAISS Index at 'data/vegan_knowledge.index'.")
    print("Built Mapping JSON at 'data/vegan_knowledge_map.json'.")

if __name__ == "__main__":
    build_knowledge_layer()
