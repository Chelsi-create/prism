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

    def predict(self, question, texts, images):
        print("mdocagent.py")
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

    
    def predict2(self, question, texts, images):
        print("mdocagent.py [predict2]")

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
                return next(agent.model.model.parameters()).device
            except AttributeError:
                try:
                    return agent.model.pipeline.device
                except AttributeError:
                    return next(agent.model.parameters()).device

        general_agent = self.agents[-1]
        device_general = get_model_device(general_agent)
        # texts_general = move_to_device(texts, device_general)
        # images_general = move_to_device(images, device_general)
        # general_response, messages = general_agent.predict(question, texts_general, images_general, with_sys_prompt=True)

        # critical_info = general_agent.self_reflect(prompt=general_agent.config.agent.critical_prompt, add_to_message=False)
        # start_index = critical_info.find('{') 
        # end_index = critical_info.find('}') + 1 
        # critical_info = critical_info[start_index:end_index]
        # text_reflection = ""
        # image_reflection = ""
        # try:
        #     critical_info = json.loads(critical_info)
        #     text_reflection = critical_info.get("text", "")
        #     image_reflection = critical_info.get("image", "")
        # except Exception as e:
        #     print("Reflection parsing error:", e)

        text_agent = self.agents[1]
        device_text = get_model_device(text_agent)
        text_reading_notes = {}

        text_con_prompt_template = """Task Description:
    1. Read the given question and the document content (either text or image).
    2. Write a reading note summarizing the key points from the document.
    3. Assess how relevant the document is to the question and explain briefly.
    4. If the document answers the question, say so clearly and extract the answer.
    5. If it gives useful context but not the answer, mention what insights it provides.
    6. If the document is irrelevant to the question, state "This document is irrelevant to the question."

    Output Format:
    {{"summary": "<reading note>", "relevance": "<high | medium | low | irrelevant>", "answer": "<short answer or 'unknown'>"}}

    Question: {question}
    Document: {page_text}
    """

        for i, page_text in enumerate(texts):
            page_prompt = text_con_prompt_template.format(question=question, page_text=page_text)
            page_text_device = move_to_device([page_text], device_text)
            try:
                summary, _ = text_agent.predict(page_prompt, texts=page_text_device, images=None, with_sys_prompt=False)
                text_reading_notes[str(i)] = summary.strip()
            except Exception as e:
                print(f"[text_agent] Error on page {i}: {e}")
                text_reading_notes[str(i)] = "Error during summarization."

        image_agent = self.agents[0]
        device_image = get_model_device(image_agent)
        image_reading_notes = {}

        image_con_prompt_template = """Task Description:
    1. Read the given question and the document content (either text or image).
    2. Write a short reading note summarizing the key points from the document.
    3. Assess how relevant the document is to the question and explain briefly.
    4. If the document answers the question, say so clearly and extract the answer.
    5. If it gives useful context but not the answer, mention what insights it provides.
    6. If the document is irrelevant to the question, state "This document is irrelevant to the question."

    Output Format:
    {{"summary": "<reading note>", "relevance": "<high | medium | low | irrelevant>", "answer": "<short answer or 'unknown'>"}}

    Question: {question}
    Document: {image_context}
    """

        for i, image in enumerate(images):
            image_context = "[Image content available]"  # You can replace with OCR output if available
            page_prompt = image_con_prompt_template.format(question=question, image_context=image_context)
            image_tensor_device = move_to_device([image], device_image)
            try:
                summary, _ = image_agent.predict(page_prompt, texts=None, images=image_tensor_device, with_sys_prompt=False)
                image_reading_notes[str(i)] = summary.strip()
            except Exception as e:
                print(f"[image_agent] Error on page {i}: {e}")
                image_reading_notes[str(i)] = "Error during summarization."

        # Compose messages for final summarizer
        # all_messages = "General Agent:\n" + general_response + "\n"
        all_messages = "Text Reading Notes:\n" + json.dumps(text_reading_notes, indent=2) + "\n"
        all_messages += "Image Reading Notes:\n" + json.dumps(image_reading_notes, indent=2) + "\n"

        final_ans, final_messages = self.sum(all_messages)
        return final_ans, final_messages, text_reading_notes, image_reading_notes


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
