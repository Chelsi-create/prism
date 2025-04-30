from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from agents.dag_controller import DAGController

class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)
    
    # def predict(self, question, texts, images):
    #     general_agent = self.agents[-1]
    #     general_response, messages = general_agent.predict(question, texts, images, with_sys_prompt=True)
    #     # print("### General Agent: "+ general_response)
    #     critical_info = general_agent.self_reflect(prompt = general_agent.config.agent.critical_prompt, add_to_message=False)
    #     # print("### General Critical Agent: " + critical_info)

    #     start_index = critical_info.find('{') 
    #     end_index = critical_info.find('}') + 1 
    #     critical_info = critical_info[start_index:end_index]
    #     text_reflection = ""
    #     image_reflection = ""
    #     try:
    #         critical_info = json.loads(critical_info)
    #         text_reflection = critical_info.get("text", "")
    #         image_reflection = critical_info.get("image", "")
    #     except Exception as e:
    #         print(e)

    #     text_agent = self.agents[1]
    #     image_agent = self.agents[0]
    #     all_messages = "General Agent:\n" + general_response + "\n"
        
    #     relect_prompt = "\nYou may use the given clue:\n"
    #     text_response, messages = text_agent.predict(question + relect_prompt +text_reflection, texts = texts, images = None, with_sys_prompt=True)
    #     all_messages += "Text Agent:\n" + text_response + "\n"
    #     image_response, messages = image_agent.predict(question + relect_prompt +image_reflection, texts = None, images = images, with_sys_prompt=True)
    #     all_messages += "Image Agent:\n" + image_response + "\n"
            
    #     # print("### Text Agent: " + text_response)
    #     # print("### Image Agent: " + image_response)
    #     final_ans, final_messages = self.sum(all_messages)
    #     # print("### Final Answer: "+final_ans)
        
    #     return final_ans, final_messages

    def predict(self, question, texts, images):
        def move_to_device(obj, device):
            if isinstance(obj, dict):
                return {k: v.to(device) if hasattr(v, "to") else v for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif hasattr(obj, "to"):
                return obj.to(device)
            return obj

        def get_model_device(agent):
            try:
                return next(agent.model.model.parameters()).device  # for Qwen2VL
            except AttributeError:
                try:
                    return agent.model.pipeline.device  # for Llama3
                except AttributeError:
                    return next(agent.model.parameters()).device  # fallback

        general_agent = self.agents[-1]
        # device_general = next(general_agent.model.model.parameters()).device
        device_general = get_model_device(general_agent)
        texts_general = move_to_device(texts, device_general)
        images_general = move_to_device(images, device_general)
        general_response, messages = general_agent.predict(question, texts_general, images_general, with_sys_prompt=True)

        critical_info = general_agent.self_reflect(prompt=general_agent.config.agent.critical_prompt, add_to_message=False)
        start_index = critical_info.find('{') 
        end_index = critical_info.find('}') + 1 
        critical_info = critical_info[start_index:end_index]
        text_reflection = ""
        image_reflection = ""
        try:
            critical_info = json.loads(critical_info)
            text_reflection = critical_info.get("text", "")
            image_reflection = critical_info.get("image", "")
        except Exception as e:
            print(e)

        text_agent = self.agents[1]
        device_text = get_model_device(text_agent)
        texts_text = move_to_device(texts, device_text)
        text_response, messages = text_agent.predict(question + "\nYou may use the given clue:\n" + text_reflection, texts=texts_text, images=None, with_sys_prompt=True)

        image_agent = self.agents[0]
        device_image = get_model_device(image_agent)
        images_image = move_to_device(images, device_image)
        image_response, messages = image_agent.predict(question + "\nYou may use the given clue:\n" + image_reflection, texts=None, images=images_image, with_sys_prompt=True)

        all_messages = "General Agent:\n" + general_response + "\n"
        all_messages += "Text Agent:\n" + text_response + "\n"
        all_messages += "Image Agent:\n" + image_response + "\n"

        final_ans, final_messages = self.sum(all_messages)
        return final_ans, final_messages

    def predict_batch(self, batch_inputs):
        results = []
        for question, texts, images in batch_inputs:
            try:
                result = self.predict(question, texts, images)
            except RuntimeError as e:
                print(e)
                torch.cuda.empty_cache()
                result = (None, None)
            results.append(result)
        return results
