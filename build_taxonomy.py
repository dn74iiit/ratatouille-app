import json
import os
from owlready2 import *

def build_taxonomy(owl_path, output_path):
    print("Loading ontology...")
    # Load the ontology
    # owlready2 expects absolute path or proper URI
    abs_path = os.path.abspath(owl_path)
    onto = get_ontology(f"file://{abs_path}").load()
    print(f"Ontology loaded: {onto.base_iri}")
    
    # We want to find a root class for foods to extract taxonomy from.
    # In FoodOn, 'food product' is typically http://purl.obolibrary.org/obo/FOODON_00001002
    food_product = onto.search_one(iri="*FOODON_00001002*")
    if not food_product:
        # Fallback to searching by label if IRI is different
        food_product = onto.search_one(label="food product")
        if not food_product:
            food_product = onto.search_one(label="food material")
    
    if not food_product:
        print("Could not find 'food product' class. Extracting top level classes instead.")
        root_classes = list(Thing.subclasses())
    else:
        print(f"Found root class: {food_product.label}")
        root_classes = [food_product]
        
    taxonomy = []
    
    # A simple recursive function to build a tree
    def extract_hierarchy(cls, depth=0, max_depth=3):
        node = {
            "name": cls.label[0] if cls.label else cls.name,
            "iri": cls.iri,
            "children": []
        }
        if depth < max_depth:
            for sub_cls in cls.subclasses():
                # Avoid circular loops if any or huge graphs
                node["children"].append(extract_hierarchy(sub_cls, depth + 1, max_depth))
        return node
        
    for root in root_classes:
        taxonomy.append(extract_hierarchy(root, max_depth=3))
        
    print(f"Saving taxonomy to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    owl_file = "data/foodon.owl"
    out_file = "data/foodon_taxonomy.json"
    build_taxonomy(owl_file, out_file)
