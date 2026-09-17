import json
import os
import numpy as np
import networkx as nx

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class GraphRetriever:
    def __init__(self, graph_path="data/hybrid_ckg.json"):
        print(f"Loading knowledge graph from {graph_path}...")
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data, edges="links" if "links" in data else "edges")
        print(f"Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        
        # Load Vector DB for Hybrid RAG (Phase 3)
        self.faiss_index = None
        self.vector_metadata = []
        self.embedding_model = None
        
        index_path = "data/recipes.index"
        metadata_path = "data/recipes_metadata.json"
        
        if FAISS_AVAILABLE and os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"[*] Loading FAISS index from {index_path}...")
            self.faiss_index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.vector_metadata = json.load(f)
            print("[*] Using HuggingFace Inference API for Hybrid Search (bypassing local RAM).")
            print(f"[OK] FAISS Vector DB loaded with {len(self.vector_metadata)} recipes.")
        else:
            print("[WARN] FAISS Vector DB not found or dependencies missing. Few-shot injection will be disabled.")
        
    def retrieve_context(self, ingredients, max_pairs=20, max_techniques=15, max_workflow_steps=15):
        """
        Retrieves the relevant subgraph context for a given list of ingredients.
        """
        context = {
            "input_ingredients": ingredients,
            "recommended_pairings": [],
            "suggested_techniques": [],
            "logical_workflow": []
        }
        
        # Lowercase all input ingredients
        ingredients = [i.lower() for i in ingredients]
        
        # 1. Retrieve PAIRS_WITH (Flavor pair suggestions)
        pairings = []
        for ing in ingredients:
            if ing in self.G:
                for neighbor in self.G.neighbors(ing):
                    edge_data = self.G.get_edge_data(ing, neighbor)
                    if edge_data and edge_data.get('relation') == 'PAIRS_WITH':
                        # Avoid suggesting things already in the input
                        if neighbor not in ingredients:
                            pairings.append({
                                "ingredient": neighbor,
                                "pmi": edge_data.get('weight', 0)
                            })
                            
        # Sort by PMI score and take top N unique
        pairings.sort(key=lambda x: x['pmi'], reverse=True)
        seen_pairs = set()
        for p in pairings:
            if p['ingredient'] not in seen_pairs and len(seen_pairs) < max_pairs:
                seen_pairs.add(p['ingredient'])
                context["recommended_pairings"].append(p['ingredient'])
                
        # 2. Retrieve PREPARED_BY (Technique suggestions based on input + paired ingredients)
        all_relevant_ingredients = ingredients + context["recommended_pairings"]
        techniques = []
        for ing in all_relevant_ingredients:
            if ing in self.G:
                for neighbor in self.G.neighbors(ing):
                    edge_data = self.G.get_edge_data(ing, neighbor)
                    if edge_data and edge_data.get('relation') == 'PREPARED_BY':
                        techniques.append({
                            "technique": neighbor,
                            "pmi": edge_data.get('weight', 0)
                        })
                        
        techniques.sort(key=lambda x: x['pmi'], reverse=True)
        seen_techs = set()
        for t in techniques:
            if t['technique'] not in seen_techs and len(seen_techs) < max_techniques:
                seen_techs.add(t['technique'])
                context["suggested_techniques"].append(t['technique'])
                
        # 3. Retrieve FOLLOWED_BY (Workflow template based on selected techniques)
        # Find paths between our suggested techniques
        workflow_edges = []
        for tech in context["suggested_techniques"]:
            if tech in self.G:
                for neighbor in self.G.neighbors(tech):
                    edge_data = self.G.get_edge_data(tech, neighbor)
                    if edge_data and edge_data.get('relation') == 'FOLLOWED_BY':
                        # Only keep it if the neighbor is also a common technique or one of our suggestions
                        workflow_edges.append({
                            "from": tech,
                            "to": neighbor,
                            "prob": edge_data.get('weight', 0)
                        })
                        
        workflow_edges.sort(key=lambda x: x['prob'], reverse=True)
        
        # Build a simple sequence logic block
        for edge in workflow_edges[:max_workflow_steps]:
            context["logical_workflow"].append(f"{edge['from']} -> {edge['to']}")
            
        return context
        
    def generate_prompt_injection(self, context):
        """
        Formats the retrieved context into a strict prompt block.
        """
        prompt = "<CONTEXT_BLOCK>\n"
        prompt += "YOU ARE AN AI CHEF. YOU MUST STRICTLY ADHERE TO THE FOLLOWING INGREDIENT CONTEXT.\n"
        prompt += "DO NOT USE ANY INGREDIENTS NOT LISTED BELOW OR IN THE ORIGINAL REQUEST.\n\n"
        
        prompt += f"[ALLOWED BASE INGREDIENTS]\n{', '.join(context['input_ingredients'])}\n\n"
        
        if context['recommended_pairings']:
            prompt += f"[ALLOWED FLAVOR PAIRINGS]\n{', '.join(context['recommended_pairings'])}\n\n"
            
        prompt += "THE FOLLOWING TECHNIQUES AND WORKFLOWS ARE HIGHLY RECOMMENDED.\n"
        prompt += "You may use standard preparation techniques to bridge gaps, but aim to incorporate this core logic:\n\n"
        
        if context['suggested_techniques']:
            prompt += f"[SUGGESTED TECHNIQUES]\n{', '.join(context['suggested_techniques'])}\n\n"
            
        if context['logical_workflow']:
            prompt += "[RECOMMENDED WORKFLOW LOGIC]\n"
            for step in context['logical_workflow']:
                prompt += f"- {step}\n"
        
        prompt += "</CONTEXT_BLOCK>\n"
        return prompt
        
    def retrieve_few_shot_examples(self, context, archetype, k=2):
        """
        Phase 3: Hybrid Search over FAISS Vector DB using Graph context.
        We form a natural language query combining the archetype and ingredients,
        and retrieve top-k semantically similar historical recipes to inject as examples.
        """
        # Form the query
        ingredients = context.get('input_ingredients', [])
        techniques = context.get('suggested_techniques', [])
        
        query_text = f"A {archetype} recipe containing {', '.join(ingredients)}."
        if techniques:
            query_text += f" Prepared by {', '.join(techniques[:3])}."

        # Encode query using HF API to save RAM
        import requests
        import time
        api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        hf_token = os.environ.get("HF_TOKEN")
        
        if not hf_token:
            print("[WARN] HF_TOKEN not found in environment, cannot perform vector search.")
            return []
            
        headers = {"Authorization": f"Bearer {hf_token}"}
        query_vector = None
        
        for attempt in range(1, 4):
            try:
                response = requests.post(api_url, headers=headers, json={"inputs": [query_text], "options": {"wait_for_model": True}}, timeout=15)
                response.raise_for_status()
                query_vector = np.array(response.json()).astype('float32')
                if len(query_vector.shape) == 1:
                    query_vector = np.expand_dims(query_vector, axis=0)
                break # Success
            except Exception as e:
                print(f"[WARN] Failed to get embedding from HF API on attempt {attempt}: {e}")
                if attempt < 3:
                    time.sleep(2)
        
        if query_vector is None:
            return []
        
        # Search FAISS
        distances, indices = self.faiss_index.search(query_vector, k)
        
        examples = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.vector_metadata):
                meta = self.vector_metadata[idx]
                ex_str = f"TITLE: {meta['title']}\nINGREDIENTS: {meta['ingredients']}\nDIRECTIONS: {meta['directions']}"
                examples.append(ex_str)
                
        return examples

if __name__ == "__main__":
    # Test the retriever
    retriever = GraphRetriever()
    
    test_ingredients = ["chicken", "rice", "garlic"]
    print(f"\nRetrieving subgraph for: {test_ingredients}")
    
    ctx = retriever.retrieve_context(test_ingredients)
    print("\n--- RETRIEVED CONTEXT ---")
    print(json.dumps(ctx, indent=2))
    
    print("\n--- PROMPT INJECTION ---")
    print(retriever.generate_prompt_injection(ctx))
