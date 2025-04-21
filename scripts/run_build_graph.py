#!/usr/bin/env python3
import os
import sys
import yaml
from types import SimpleNamespace

# make sure imports of agents/ and mydatasets/ work:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.graph_builder_agent import GraphBuilderAgent

def main():
    cfg_path = os.path.join(os.path.dirname(__file__),
                            '..', 'config', 'agent', 'graph_builder_agent.yaml')
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)

    # wrap the dict so we can access cfg.extract_path, cfg.dataset_name, etc.
    config = SimpleNamespace(**cfg_dict)

    agent = GraphBuilderAgent(config)
    graph = agent.run()

    print(f"Done. Graph has {graph.number_of_nodes()} nodes "
          f"and {graph.number_of_edges()} edges.")

if __name__ == "__main__":
    main()
