from models.base_model import BaseModel
from mydatasets.base_dataset import BaseDataset
import os
from typing import Dict, Union
import json
import pandas as pd
from tqdm import tqdm
import re
import importlib

class Agent:
    def __init__(self, config, model=None):
        self.config = config
        self.messages = None
        if model is not None:
            self.model:BaseModel = model
        else:
            try:
                module = importlib.import_module(self.config.model.module_name)
                model_class = getattr(module, self.config.model.class_name)
                print("Create model: ", self.config.model.class_name)
                self.model = model_class(self.config.model)
            except Exception:
                module = importlib.import_module(self.config.agent.module_name)
                model_class = getattr(module, self.config.agent.class_name)
                print("Create model: ", self.config.agent.class_name)
                self.model = model_class(self.config.agent)
    
    def clean_messages(self):
        self.messages = None
        
    def _predict(self, question, texts=None, images=None, add_to_message = False):
        if not self.config.agent.use_text:
            texts = None
        if not self.config.agent.use_image:
            images = None
        generated_ans, messages = self.model.predict(question, texts, images, self.messages)
        if add_to_message:
            self.messages = messages
        return generated_ans, messages
    
    def predict(self, question, texts=None, images=None, with_sys_prompt=True):
        print("base_agent.py")
        if with_sys_prompt:
            question = self.config.agent.system_prompt + question
        return self._predict(question, texts, images, add_to_message = True)
    
    def self_reflect(self, prompt=None, add_to_message = True):
        if prompt is None:
            self_reflect_prompt = self.config.agent.self_reflect_prompt
        else:
            self_reflect_prompt = prompt
        
        generated_ans, messages = self._predict(question = self_reflect_prompt)
        if add_to_message:
            self.messages = messages
        
        return generated_ans
    
    def eval(self, question, answer, gt):
        prompt = self.config.agent.eval_system_prompt.format(question=question, answer=answer, gt=gt)
        try:
            generated_ans, _ = self.model.predict(prompt)
            result = extract_evaluation_metrics(generated_ans)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {"binary_correctness": 0}
    
    def eval2_dataset(self, dataset: BaseDataset):
        sample_path = "./data/MMLongBench/sample_retrieval_dag4.json"
        ans_path = "./results/MMLongBench/agent_plan_rag/evaluation_plan_rag_full.json"

        with open(sample_path, "r") as f:
            try:
                samples = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Failed to load JSON file: {str(e)}")
                return

        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]

        binary_correctness_list_1 = []
        binary_correctness_list_2 = []

        os.makedirs(os.path.dirname(ans_path), exist_ok=True)
        with open(ans_path, "w") as out_file:
            out_file.write("[\n")  # Start of JSON array

            for idx, sample in enumerate(tqdm(samples)):
                try:
                    question = sample[dataset.config.question_key]
                    gt = sample[dataset.config.gt_key]

                    ans1 = sample.get("ans1", "")
                    ans2 = sample.get("ans2", "")

                    result1 = self.eval(question, ans1, gt)
                    result2 = self.eval(question, ans2, gt)

                    bc1 = result1.get('binary_correctness', None)
                    bc2 = result2.get('binary_correctness', None)

                    sample['binary_correctness_ans1'] = bc1
                    sample['binary_correctness_ans2'] = bc2

                    binary_correctness_list_1.append(bc1)
                    binary_correctness_list_2.append(bc2)

                    json.dump(sample, out_file)
                    if idx < len(samples) - 1:
                        out_file.write(",\n")
                    else:
                        out_file.write("\n")

                except Exception as e:
                    print(f"Error evaluating sample: {str(e)}")

            out_file.write("]\n")  # End of JSON array

        result_txt_path = os.path.join(os.path.dirname(ans_path), "results_dual.txt")
        with open(result_txt_path, "a") as file:
            file.write("\nEvaluation Results for ans1 and ans2:\n")
            file.write(f"Sample file: {sample_path}\n")
            file.write(f"Output file: {ans_path}\n")
            file.write(f"Average Binary Correctness (ans1): {pd.Series(binary_correctness_list_1).mean():.3f}\n")
            file.write(f"Average Binary Correctness (ans2): {pd.Series(binary_correctness_list_2).mean():.3f}\n")

        print(f"Saved dual evaluation results to {result_txt_path}")


    def eval_dataset(self, dataset: BaseDataset):
        samples, ans_path = dataset.load_latest_results()
        # samples = samples[:1]
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
        samples_with_answer = []
        for sample in tqdm(samples):
            try:
                question = sample[dataset.config.question_key]
                answer = sample[self.config.ans_key]
                gt = sample[dataset.config.gt_key]
                result = self.eval(question, answer, gt)
                sample['binary_correctness'] = result.get('binary_correctness', None)
                samples_with_answer.append(sample)
            except Exception as e:
                print(f"Error evaluating sample: {str(e)}")
                
        ans_file_path_name = ans_path[:-5]+"_results.json"
        with open(ans_file_path_name, "w") as file:
            json.dump(samples_with_answer, file, indent=4)
            
        samples_with_answer = pd.DataFrame(samples_with_answer)
        path = os.path.join(dataset.config.result_dir,"results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            file.write(f"Average Binary Correctness: {samples_with_answer['binary_correctness'].mean():.3f}\n")
        
        print(f"Save results to {path}.")

def extract_evaluation_metrics(eval_str: str) -> Dict[str, Union[float, int]]:
    try:
        start_index = eval_str.find('{') 
        end_index = eval_str.rfind('}') + 1 
        eval_str = eval_str[start_index:end_index]
        metrics = json.loads(eval_str)
        return {
            'binary_correctness': int(metrics.get('binary_correctness', 0))
        }
    except json.JSONDecodeError as e:
        return {
            'binary_correctness': 0
        }
    except Exception as e:
        return {
            'binary_correctness': 0
        }