#!/usr/bin/env python3
import json
import pickle
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

GRAPH_PATH    = "tmp/MMLongBench/MMLongBench_page_graph.gpickle"
SAMPLE_IN     = "data/MMLongBench/samples.json"
SAMPLE_OUT    = "data/MMLongBench/samples_retrieval_graph.json"
EMBED_MODEL   = "sentence-transformers/all-mpnet-base-v2"
TOP_K         = 20

def main():
    # 1) load graph
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # 2) load or compute node embeddings
    node_ids = list(G.nodes)
    texts    = [G.nodes[n]["text"] for n in node_ids]

    embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    embs     = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # 3) load queries
    with open(SAMPLE_IN) as f:
        samples = json.load(f)

    output = []
    for sample in samples:
        query = sample["retrieval-query"]  # or "qwen_retrieval-query"
        q_emb  = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # 4) cosine sims
        sims   = embs @ q_emb

        # 5) top K
        idxs   = np.argsort(-sims)[:TOP_K]
        top_nodes  = [node_ids[i] for i in idxs]
        top_scores = [float(sims[i])    for i in idxs]

        # extract the page‐index part of each node_id
        page_idxs = [int(n.split("_page")[-1]) for n in top_nodes]

        # 6) extend the sample record
        rec = sample.copy()
        rec["graph-top-20-question"]         = page_idxs
        rec["graph-top-20-question_score"]   = top_scores
        rec["graph-top-20-question_node"]    = top_nodes

        output.append(rec)

    # 7) write out
    with open(SAMPLE_OUT, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Wrote {len(output)} entries to {SAMPLE_OUT}")

if __name__ == "__main__":
    main()
