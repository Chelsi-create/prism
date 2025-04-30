import ast
import yaml
from typing import List, Tuple
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class PlanAgent:
    """
    PlanAgent for generating a reasoning DAG (Plan*RAG) from a query.
    Loads model locally without server calls.
    """

    def __init__(self, config_path: str = "config/plan/decompose.yaml", device: str = "cuda"):
        # Load Plan prompt + model name
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.plan_prompt_template = config["plan_prompt"]
        self.model_name = config["model_name"]
        self.device = device

        # Load model + tokenizer
        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print(f"Model {self.model_name} loaded.")

    def generate_plan(self, query: str) -> List[Tuple[str, str]]:
        prompt = self.plan_prompt_template + f"\nFinal Query: {query}\nFinal DAG:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the output part
        text = decoded.split("Final DAG:")[-1]

        start_idx = text.find('[')
        end_idx = text.find(']', start_idx)

        if start_idx != -1 and end_idx != -1:
            list_text = text[start_idx:end_idx+1]
            try:
                dag = ast.literal_eval(list_text)
            except Exception as e:
                print("Failed to parse DAG:", e)
                dag = [(f"Q: {query}", f"Q: {query}")]
        else:
            print("No valid DAG found.")
            dag = [(f"Q: {query}", f"Q: {query})")]
        
        if isinstance(dag, tuple):
            dag = [dag]
        
        # print(plan)
        return dag

    def get_root_and_subqueries(self, plan: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
        parents = [p for p, c in plan]
        children = [c for p, c in plan]
        root = None
        for p in parents:
            if p not in children:
                root = p
                break
        subqueries = []
        for p, c in plan:
            if c != root and c not in subqueries:
                subqueries.append(c)
        return root, subqueries

# # Example usage:
if __name__ == "__main__":
    agent = PlanAgent()

    with open("data/MMLongBench/samples.json", "r") as f:
        samples = json.load(f)

    for sample in tqdm(samples, desc="Generating Reasoning DAGs"):
        query = sample.get("question", None)
        if query:
            # print(f"\nProcessing Query: {query}")
            plan = agent.generate_plan(query)
            sample["reasoning_dag"] = plan 

    with open("data/MMLongBench/samples_with_dag.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("\nFinished generating DAGs and saved to data/MMLongBench/samples_with_dag.json")

    # query = "Rumble Fish was a novel by the author of the coming-of-age novel published in what year by Viking Press?"
    # # query = "What is the distance between the locations that hosted the last two Men’s Cricket World Cup finals?"

    # plan = agent.generate_plan(query)
    # root, subqueries = agent.get_root_and_subqueries(plan)
    # print("\n=== Reasoning DAG ===")
    # print(plan)
    # print("\n=== Root Node ===")
    # print(root)
    # print("\n=== Subqueries ===")
    # for sq in subqueries:
    #     print("-", sq)
