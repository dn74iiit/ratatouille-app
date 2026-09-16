import json
import networkx as nx

def main():
    print("Loading graph...")
    with open('data/hybrid_ckg.json', 'r') as f:
        data = json.load(f)
        
    # The new version of networkx node_link_graph doesn't need to specify edges arg if we are careful,
    # but we can handle it safely:
    G = nx.node_link_graph(data)
    
    print("Graph loaded. Extracting subgraph around 'garlic'...")
    # Extract ego graph around garlic (radius 1)
    if 'garlic' in G:
        subG = nx.ego_graph(G, 'garlic', radius=1)
        # Sort edges by weight (PMI) if available
        edges = sorted(subG.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
        
        print("\n--- GARLIC SUBGRAPH (Top 15 Edges) ---")
        for u, v, d in edges[:15]:
            weight = f"{d.get('weight', 0):.2f}" if 'weight' in d else 'N/A'
            print(f"{u} --[{d.get('relation', 'UNKNOWN')}] (PMI: {weight})--> {v}")
    
    print("\nGraph loaded. Extracting subgraph around 'chicken'...")
    if 'chicken' in G:
        subG = nx.ego_graph(G, 'chicken', radius=1)
        # Sort edges by weight (PMI) if available
        edges = sorted(subG.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
        
        print("\n--- CHICKEN SUBGRAPH (Top 15 Edges) ---")
        for u, v, d in edges[:15]:
            weight = f"{d.get('weight', 0):.2f}" if 'weight' in d else 'N/A'
            print(f"{u} --[{d.get('relation', 'UNKNOWN')}] (PMI: {weight})--> {v}")

if __name__ == "__main__":
    main()
