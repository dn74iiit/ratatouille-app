import json
import networkx as nx
import os

def build_graph(taxonomy_path, pmi_path, techniques_path, workflow_path, output_path):
    print("Initializing Hybrid Culinary Knowledge Graph...")
    G = nx.DiGraph()

    # 1. Load Taxonomy (IS_A edges)
    if os.path.exists(taxonomy_path):
        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
            
        def add_taxonomy_edges(node, parent_name=None):
            current_name = node.get("name")
            if current_name:
                # Add node
                G.add_node(current_name, type="ingredient_class", iri=node.get("iri"))
                if parent_name:
                    # Directed edge from specific to general (Child IS_A Parent)
                    G.add_edge(current_name, parent_name, relation="IS_A")
                    
                for child in node.get("children", []):
                    add_taxonomy_edges(child, current_name)
                    
        for root in taxonomy:
            add_taxonomy_edges(root)
    else:
        print(f"Taxonomy file {taxonomy_path} not found. Skipping taxonomy edges.")

    # 2. Load PMI (PAIRS_WITH edges)
    if os.path.exists(pmi_path):
        print(f"Loading PMI data from {pmi_path}...")
        with open(pmi_path, 'r', encoding='utf-8') as f:
            pmi_data = json.load(f)
            
        for item in pmi_data:
            src = item["source"]
            tgt = item["target"]
            score = item["pmi"]
            
            # Add nodes if they don't exist
            if not G.has_node(src):
                G.add_node(src, type="ingredient")
            if not G.has_node(tgt):
                G.add_node(tgt, type="ingredient")
                
            # Undirected functional relationship for pairs
            # NetworkX DiGraph requires bidirectional for undirected
            G.add_edge(src, tgt, relation="PAIRS_WITH", weight=score)
            G.add_edge(tgt, src, relation="PAIRS_WITH", weight=score)
    else:
        print(f"PMI file {pmi_path} not found. Skipping PMI edges.")
        
    # 3. Load Techniques (PREPARED_BY edges)
    if os.path.exists(techniques_path):
        print(f"Loading Techniques data from {techniques_path}...")
        with open(techniques_path, 'r', encoding='utf-8') as f:
            tech_data = json.load(f)
            
        for item in tech_data:
            ing = item["ingredient"]
            tech = item["technique"]
            score = item["pmi"]
            
            if not G.has_node(ing):
                G.add_node(ing, type="ingredient")
            if not G.has_node(tech):
                G.add_node(tech, type="technique")
                
            # Directed relationship from ingredient to technique
            G.add_edge(ing, tech, relation="PREPARED_BY", weight=score)
    else:
        print(f"Techniques file {techniques_path} not found. Skipping PREPARED_BY edges.")

    # 4. Load Workflow Edges (FOLLOWED_BY edges)
    if os.path.exists(workflow_path):
        print(f"Loading Workflow data from {workflow_path}...")
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
            
        for item in workflow_data:
            src = item["source_technique"]
            tgt = item["target_technique"]
            prob = item["probability"]
            
            if not G.has_node(src):
                G.add_node(src, type="technique")
            if not G.has_node(tgt):
                G.add_node(tgt, type="technique")
                
            # Directed sequence relationship
            G.add_edge(src, tgt, relation="FOLLOWED_BY", weight=prob)
    else:
        print(f"Workflow file {workflow_path} not found. Skipping FOLLOWED_BY edges.")

    # 5. Save Graph
    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Saving graph to {output_path}...")
    
    # Save as node-link JSON which is easy to read and visualize
    data = nx.node_link_data(G)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    tax_path = "data/foodon_taxonomy.json"
    pmi_path = "data/ingredient_pmi.json"
    tech_path = "data/prepared_by_edges.json"
    workflow_path = "data/followed_by_edges.json"
    out_path = "data/hybrid_ckg.json"
    build_graph(tax_path, pmi_path, tech_path, workflow_path, out_path)
