import os
import glob
import networkx as nx
# from networkx import write_gpickle
import pickle
import numpy as np
import torch                              
from sentence_transformers import SentenceTransformer
from agents.base_agent import Agent

class GraphBuilderAgent(Agent):
    """
    Agent to construct a dense page-level graph from extracted images and text.
    Supports sequential and semantic inter-document edges.
    Uses config.dataset_name for naming the output gpickle.
    """
    def __init__(self, config, agent=None):
        # super().__init__(config, agent)
        self.config = config
        self.extract_path = config.extract_path
        embed_name = getattr(config, "embedding_model_name", None)
        if embed_name:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embed_model = SentenceTransformer(embed_name, device=device)
        else:
            self.embed_model = None
        self.sim_threshold = getattr(config, "semantic_edge_threshold", 0.8)

    def build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        docs = {}
        for txt_file in sorted(glob.glob(os.path.join(self.extract_path, "*.txt"))):
            fname = os.path.basename(txt_file)
            base, _ = os.path.splitext(fname)            
            doc_id, page_str = base.rsplit("_", 1) 
            page_idx = int(page_str)
            docs.setdefault(doc_id, []).append(page_idx)
            with open(txt_file) as f:
                text = f.read().replace("\n", " ")
            img_file = os.path.join(self.extract_path, f"{doc_id}_{page_idx}.png")
            node = f"{doc_id}_page{page_idx}"
            G.add_node(node, text=text, image=img_file, doc_id=doc_id)

        # sequential edges
        for doc_id, pages in docs.items():
            for idx in sorted(pages):
                prev = idx - 1
                if prev in pages:
                    G.add_edge(f"{doc_id}_page{prev}", f"{doc_id}_page{idx}", type="sequence")

        # semantic edges
        if self.embed_model:
            nodes = list(G.nodes)
            texts = [G.nodes[n]['text'] for n in nodes]
            embs = self.embed_model.encode(texts, convert_to_numpy=True)
            embs /= np.linalg.norm(embs, axis=1, keepdims=True)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    sim = float(np.dot(embs[i], embs[j]))
                    if sim >= self.sim_threshold:
                        G.add_edge(nodes[i], nodes[j], type="semantic", weight=sim)
                        G.add_edge(nodes[j], nodes[i], type="semantic", weight=sim)

        return G

    def run(self) -> nx.DiGraph:                  
        """
        Build and save the graph under:
           <extract_path>/<dataset_name>_page_graph.gpickle
        """
        graph = self.build_graph()
        ds_name = getattr(self.config, "dataset_name", None)
        fname   = f"{ds_name}_page_graph.gpickle" if ds_name else "page_graph.gpickle"
        out = os.path.join(self.extract_path, fname)

        with open(out, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Graph saved at {out}")
        return graph
