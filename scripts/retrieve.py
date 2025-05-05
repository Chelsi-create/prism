import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from retrieval.base_retrieval import BaseRetrieval
from agents.mdoc_agent import MDocAgent
import hydra
import importlib
import torch
from tqdm import tqdm

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.retrieval.cuda_visible_devices
    print(cfg.retrieval.cuda_visible_devices)
    print(torch.cuda.device_count())
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Device Name:", torch.cuda.get_device_name(0))
    # torch.cuda.set_device(1)

    print(os.environ["CUDA_VISIBLE_DEVICES"])

        # ============ Expand agent and model configs ============
    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name=f"agent/{agent_name}", overrides=[]).agent
        model_cfg = hydra.compose(config_name=f"model/{model_name}", overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg

    cfg.mdoc_agent.sum_agent.agent = hydra.compose(config_name=f"agent/{cfg.mdoc_agent.sum_agent.agent}", overrides=[]).agent
    cfg.mdoc_agent.sum_agent.model = hydra.compose(config_name=f"model/{cfg.mdoc_agent.sum_agent.model}", overrides=[]).model

    # ------------ Load Dataset ------------
    print("-------------Loading dataset-------------")
    dataset = BaseDataset(cfg.dataset)
    samples = dataset.load_data(use_retreival=True)

    # # ------------ Load Single Retriever ------------
    # print("-------------Loading retrieval model-------------")
    # retrieval_class_path = cfg.retrieval.class_path
    # module_name, class_name = retrieval_class_path.rsplit('.', 1)
    # module = importlib.import_module(module_name)
    # retrieval_class = getattr(module, class_name)
    # retriever: BaseRetrieval = retrieval_class(cfg.retrieval)

    # # Attach retriever to dataset (either text_retriever or image_retriever depending on type)
    # if "text" in retrieval_class_path.lower():
    #     dataset.text_retriever = retriever

    # elif "image" in retrieval_class_path.lower():
    #     dataset.image_retriever = retriever
    # else:
    #     raise ValueError(f"Unknown retrieval type: {retrieval_class_path}")

    print("Loading retrieval")
    text_class_path = cfg.retrieval.text_retrieval.class_path
    image_class_path = cfg.retrieval.image_retrieval.class_path

    # Text retriever
    text_module_name, text_class_name = text_class_path.rsplit('.', 1)
    text_module = importlib.import_module(text_module_name)
    TextRetrieverClass = getattr(text_module, text_class_name)
    dataset.text_retriever = TextRetrieverClass(cfg.retrieval.text_retrieval)

    # Image retriever
    image_module_name, image_class_name = image_class_path.rsplit('.', 1)
    image_module = importlib.import_module(image_module_name)
    ImageRetrieverClass = getattr(image_module, image_class_name)
    dataset.image_retriever = ImageRetrieverClass(cfg.retrieval.image_retrieval)

    # ------------ Load Agent System ------------
    print("-------------Loading MDocAgent-------------")
    mdoc_agent = MDocAgent(cfg.mdoc_agent)

    # Preload only for image mode
    print("-------------Preloading Image Embeddings-------------")
    document_embeds = dataset.image_retriever.load_document_embeds(dataset)
    
    # ------------ Retrieve per Sub-Query ------------
    print("-------------Retrieving TopK for each sub-query in reasoning_dag-------------")
    updated_samples = []
    for sample in tqdm(samples):
        subquery_to_answer = {}
        if "reasoning_dag" not in sample:
            continue
        
        if "retrieval_info" not in sample:
            sample["retrieval_info"] = {}

        for parent, child in sample["reasoning_dag"]:
            subquery = child
            subquery_id = self_extract_subquery_id(child)

            if "<A" in subquery:  # fast check
                try:
                    final_subquery = replace_previous_answers(subquery, subquery_to_answer)
                except Exception as e:
                    print(f"[Warning] Failed to replace answers in subquery '{subquery_id}': {e}")
                    final_subquery = subquery 
            else:
                final_subquery = subquery

            # TODO: Use LLM to replace subquery answer (optional)

            top_page_indices, _ = dataset.text_retriever.find_top_k_for_subquery(
                sample=sample,
                subquery=final_subquery,
                top_k=cfg.retrieval.top_k,
                page_id_key=dataset.config.page_id_key
            )
            sample["retrieval_info"].setdefault(subquery_id, {})["text"] = top_page_indices

            ## We can further process the image (todo)
            doc_embed = document_embeds.get(sample[cfg.retrieval.doc_key])
            top_page_indices, _ = dataset.image_retriever.find_top_k_for_subquery(
                sample=sample,
                subquery=final_subquery,
                document_embed=doc_embed,
                top_k=cfg.retrieval.top_k,
                page_id_key=dataset.config.page_id_key
            )
            sample["retrieval_info"].setdefault(subquery_id, {})["image"] = top_page_indices

            final_answer, _ = mdoc_agent.predict_query(
                sample=sample,
                subquery=final_subquery,
                subquery_id=subquery_id,
                dataset=dataset
            )

            if "answers" not in sample:
                sample["answers"] = {}
            sample["answers"][subquery_id] = final_answer
            subquery_to_answer[subquery_id] = final_answer  # update for next hops

        prompt = (
            f"Original Question: {sample['question']}\n"
            f"Subquery: {sample['reasoning_dag']}\n"
            f"Subquery Answers: {sample["answers"]}"
        )

        ans1, _ = mdoc_agent.predict_final_ans_context(sample=sample, sub_ans=sample["answers"], reasoning_dag=sample["reasoning_dag"], dataset=dataset)
        ans2, _ = mdoc_agent.predict(prompt, None, None)

        sample["ans1"] = ans1
        sample["ans2"] = ans2
        print(ans2)

        updated_samples.append(sample)
        with open("sample_retrieval_dag_final.json", "w") as f:
            json.dump(updated_samples, f, indent=2)

    # Mdocagent = question + texts + images
    # (question+subquery+answers, texts ,images)
    # (question+subquery+answers)

    # ------------ Save Updated Samples ------------
    # save_path = dataset.dump_data(samples, use_retreival=True)
    # print(f"Saved updated retrieval_info at {save_path}.")

def self_extract_subquery_id(query: str):
    query = query.strip()
    if query.startswith("Q") and ":" in query:
        return query.split(":")[0].strip()
    return "unknown"

def replace_previous_answers(subquery: str, subquery_to_answer: dict):
    updated_query = subquery

    for sub_id, ans in subquery_to_answer.items():
        placeholder = f"<A{sub_id[1:]}>"
        if placeholder in updated_query:
            updated_query = updated_query.replace(placeholder, ans)
    return updated_query

# get the evaluation results
# go through more papers 
# get the experiment running at psu lab
# meeting notes
# implement the algorithm (today)
# batch inference (later)
# tower research start prep


if __name__ == "__main__":
    main()
