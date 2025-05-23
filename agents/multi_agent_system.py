from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from tqdm import tqdm
import importlib
import json
import torch
from typing import List
import os
from itertools import islice

class MultiAgentSystem:
    def __init__(self, config):
        self.config = config
        self.agents:List[Agent] = []
        self.models:dict = {}
        for agent_config in self.config.agents:
            if agent_config.model.class_name not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[agent_config.model.class_name] = model_class(agent_config.model)
            self.add_agent(agent_config, self.models[agent_config.model.class_name])
            
        if config.sum_agent.model.class_name not in self.models:
            module = importlib.import_module(config.sum_agent.model.module_name)
            model_class = getattr(module, config.sum_agent.model.class_name)
            self.models[config.sum_agent.model.class_name] = model_class(config.sum_agent.model)
        self.sum_agent = Agent(config.sum_agent, self.models[config.sum_agent.model.class_name])
        
    def add_agent(self, agent_config, model):
        module = importlib.import_module(agent_config.agent.module_name)
        agent_class = getattr(module, agent_config.agent.class_name)
        agent:Agent = agent_class(agent_config, model)
        self.agents.append(agent)
        
    def predict(self, question, texts, images):
        '''Implement the method in the subclass'''
        pass
    
    def sum(self, sum_question):
        ans, all_messages = self.sum_agent.predict(sum_question)
        def extract_final_answer(agent_response):
            try:
                response_dict = json.loads(agent_response)
                answer = response_dict.get("Answer", None)
                return answer
            except:
                return agent_response
        final_ans = extract_final_answer(ans)
        return final_ans, all_messages

    def batch_iter(self, iterable, batch_size):
        it = iter(iterable)
        while batch := list(islice(it, batch_size)):
            yield batch


    def predict_dataset_con(self, dataset:BaseDataset, resume_path = None):
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
            
        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            question, texts, images = dataset.load_sample_retrieval_data(sample)

            try:
                final_ans, final_messages, text_reading_notes, image_reading_notes = self.predict2(question, texts, images)
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                final_ans, final_messages = None, None
                text_reading_notes, image_reading_notes = {}, {}
            sample[self.config.ans_key] = final_ans
            sample["text_reading_notes"] = text_reading_notes
            sample["image_reading_notes"] = image_reading_notes

            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = final_messages
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")

    ## 
    def predict_dataset(self, dataset:BaseDataset, resume_path = None):
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
            
        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            question, texts, images = dataset.load_sample_retrieval_data(sample)
            try:
                final_ans, final_messages = self.predict(question, texts, images)
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                final_ans, final_messages = None, None
            sample[self.config.ans_key] = final_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = final_messages
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")

    def predict_query(self, sample, subquery: str, subquery_id: str, dataset: BaseDataset):
        print("Implementing predict query method")
        subquery_question, texts, images = dataset.load_subquery_retrieval_data(sample, subquery_id)

        try:
            final_ans, final_messages = self.predict(subquery, texts, images)
        except RuntimeError as e:
            print(e)
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            final_ans, final_messages = None, None

        torch.cuda.empty_cache()
        self.clean_messages()

        return final_ans, final_messages
    
    def predict_final_ans_context(self, sample, sub_ans, reasoning_dag, dataset: BaseDataset):
        question, texts, images = dataset.load_sample_retrieval_data(sample)

        prompt = (
            f"Original Question: {question}\n"
            f"Subquery: {reasoning_dag}\n"
            f"Subquery Answers: {sub_ans}"
        )

        try:
            final_ans, final_messages = self.predict(prompt, texts, images)
            print(final_ans)

        except RuntimeError as e:
            print(e)
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            final_ans, final_messages = None, None

        torch.cuda.empty_cache()
        self.clean_messages()

        return final_ans, final_messages


    # def predict_dataset(self, dataset:BaseDataset, resume_path=None):
    #     samples = dataset.load_data(use_retreival=True)
    #     if resume_path:
    #         assert os.path.exists(resume_path)
    #         with open(resume_path, 'r') as f:
    #             samples = json.load(f)
    #     if self.config.truncate_len:
    #         samples = samples[:self.config.truncate_len]
        
    #     batch_size = 8
    #     sample_no = 0

    #     for batch in tqdm(self.batch_iter(samples, batch_size), total=(len(samples) + batch_size - 1) // batch_size):
    #         batch_inputs = []
    #         batch_samples = []
    #         for sample in batch:
    #             if resume_path and self.config.ans_key in sample:
    #                 continue
    #             question, texts, images = dataset.load_sample_retrieval_data(sample)
    #             batch_inputs.append((question, texts, images))
    #             batch_samples.append(sample)

    #         if not batch_inputs:
    #             continue

    #         try:
    #             batch_outputs = self.predict_batch(batch_inputs)  # <-- You need to implement this in MDocAgent
    #         except RuntimeError as e:
    #             print(e)
    #             if "out of memory" in str(e):
    #                 torch.cuda.empty_cache()
    #             batch_outputs = [(None, None)] * len(batch_inputs)

    #         for sample, (final_ans, final_messages) in zip(batch_samples, batch_outputs):
    #             sample[self.config.ans_key] = final_ans
    #             if self.config.save_message:
    #                 sample[self.config.ans_key + "_message"] = final_messages
    #             self.clean_messages()

    #         sample_no += len(batch_samples)
    #         torch.cuda.empty_cache()

    #         if sample_no % self.config.save_freq == 0:
    #             path = dataset.dump_reults(samples)
    #             print(f"Save {sample_no} results to {path}.")

    #     path = dataset.dump_reults(samples)
    #     print(f"Save final results to {path}.")
    
    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()
        self.sum_agent.clean_messages()

