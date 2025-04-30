import json
import re
from dataclasses import dataclass
from PIL import Image
import os
import pymupdf
from tqdm import tqdm
from datetime import datetime
import glob

@dataclass
class Content:
    image: Image
    image_path: str
    txt: str
    
class BaseDataset():
    def __init__(self, config):
        self.config = config
        self.IM_FILE = lambda doc_name,index: f"{self.config.extract_path}/{doc_name}_{index}.png"
        self.TEXT_FILE = lambda doc_name,index: f"{self.config.extract_path}/{doc_name}_{index}.txt"
        self.EXTRACT_DOCUMENT_ID = lambda sample: re.sub("\\.pdf$", "", sample["doc_id"]).split("/")[-1] 

        current_time = datetime.now()
        self.time = current_time.strftime("%Y-%m-%d-%H-%M")
    
    def load_data(self, use_retreival=True):
        # print("→ load_data") 
        path = self.config.sample_path
        print(path)
        if use_retreival:
            try:
                assert(os.path.exists(self.config.sample_with_retrieval_path))
                path = self.config.sample_with_retrieval_path
                print(path)
            except:
                print("Use original sample path!")
                
        assert(os.path.exists(path))
        print("  loading from", path)
        with open(path, 'r') as f:
            samples = json.load(f)
            
        return samples
    
    def dump_data(self, samples, use_retreival=True):
        print("→ dump_data") 
        if use_retreival:
            path = self.config.sample_with_retrieval_path
        else:
            path = self.config.sample_path
        print("  dumping to", path)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        
        return path
    
    def load_latest_results(self):
        print("→ load_latest_results") 
        print(self.config.result_dir)
        path = find_latest_json(self.config.result_dir)
        with open(path, 'r') as f:
            samples = json.load(f)
        return samples, path
    
    def dump_reults(self, samples):
        os.makedirs(self.config.result_dir, exist_ok=True)
        path = os.path.join(self.config.result_dir, self.time + ".json")
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        return path
    
    def load_retrieval_data(self):
        # print("→ load_retrieval_data") 
        assert(os.path.exists(self.config.sample_with_retrieval_path))
        with open(self.config.sample_with_retrieval_path, 'r') as f:
            samples = json.load(f)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_retrieval_data(sample)
        return samples
    

    def load_subquery_retrieval_data(self, sample, subquery_id: str):
        print("Loading the data for subquery")
        content_list = self.load_processed_content(sample, disable_load_image=False)

        texts = []
        images = []

        retrieval_info = sample.get("retrieval_info", {})
        subquery_info = retrieval_info.get(subquery_id, {})

        if "text" in subquery_info:
            for idx in subquery_info["text"][:self.config.top_k]:
                texts.append(content_list[idx].txt.replace("\n", ""))

        if "image" in subquery_info:
            for idx in subquery_info["image"][:self.config.top_k]:
                images.append(content_list[idx].image_path)

        subquery_question = None
        if "reasoning_dag" in sample:
            for parent, child in sample["reasoning_dag"]:
                if child.strip().startswith(subquery_id):
                    subquery_question = child.strip()
                    break

        if subquery_question is None:
            raise ValueError(f"Subquery ID {subquery_id} not found in reasoning DAG!")

        return subquery_question, texts, images
    
    # DAG method
    def load_sample_retrieval_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)

        question: str = sample[self.config.question_key]
        texts = []
        images = []

        print(self.config.r_text_key)
        if self.config.use_mix:
            if self.config.r_mix_key in sample:
                for page in sample[self.config.r_mix_key][:self.config.top_k]:
                    if page in sample[self.config.r_image_key]:
                        origin_image_path = content_list[page].image_path
                        images.append(origin_image_path)
                    if page in sample[self.config.r_text_key]:
                        texts.append(content_list[page].txt.replace("\n", ""))
        else:
            if self.config.r_text_key in sample:
                for page in sample[self.config.r_text_key][:self.config.top_k]:
                    texts.append(content_list[page].txt.replace("\n", ""))
            if self.config.r_image_key in sample:
                for page in sample[self.config.r_image_key][:self.config.top_k]:
                    origin_image_path = content_list[page].image_path
                    images.append(origin_image_path)

        return question, texts, images

    # # MDocAgent Method
    # def load_sample_retrieval_data(self, sample):
    #     content_list = self.load_processed_content(sample, disable_load_image=True)
    #     question:str = sample[self.config.question_key]
    #     texts = []
    #     images = []
    #     if self.config.use_mix:
    #         if self.config.r_mix_key in sample:
    #             for page in sample[self.config.r_mix_key][:self.config.top_k]:
    #                 if page in sample[self.config.r_image_key]:
    #                     origin_image_path = ""
    #                     origin_image_path = content_list[page].image_path
    #                     images.append(origin_image_path)
    #                 if page in sample[self.config.r_text_key]:
    #                     texts.append(content_list[page].txt.replace("\n", ""))
    #     else:
    #         if self.config.r_text_key in sample:
    #             for page in sample[self.config.r_text_key][:self.config.top_k]:
    #                 texts.append(content_list[page].txt.replace("\n", ""))
    #         if self.config.r_image_key in sample:
    #             for page in sample[self.config.r_image_key][:self.config.top_k]:
    #                 origin_image_path = ""
    #                 origin_image_path = content_list[page].image_path
    #                 images.append(origin_image_path)
                        
    #     return question, texts, images
    
    # # Graph method
    # def load_sample_retrieval_data(self, sample):
    #     # print("→ load_sample_retrieval_data")
    #     content_list = self.load_processed_content(sample, disable_load_image=False)
    #     question:str = sample[self.config.question_key]
    #     texts = []
    #     images = []
    #     if self.config.use_mix:
    #         # print(self.config.r_mix_key)
    #         if self.config.r_mix_key in sample:
    #             for i, node in enumerate(sample[self.config.r_mix_key][:self.config.top_k]):
    #                 try:
    #                     _, page_str = node.rsplit("_page", 1)
    #                     page_no = int(page_str)
    #                     # print(page_no)
    #                 except ValueError:
    #                     continue

    #                 # if this node was selected for image retrieval
    #                 if node in sample[self.config.r_mix_key]:
    #                     images.append(content_list[i].image_path)

    #                 # if this node was selected for text retrieval
    #                 if node in sample[self.config.r_mix_key]:
    #                     # strip newlines and append
    #                     texts.append(content_list[i].txt.replace("\n", ""))
    #     else:
    #         if self.config.r_text_key in sample:
    #             for page in sample[self.config.r_text_key][:self.config.top_k]:
    #                 texts.append(content_list[page].txt.replace("\n", ""))
    #         if self.config.r_image_key in sample:
    #             for page in sample[self.config.r_image_key][:self.config.top_k]:
    #                 origin_image_path = ""
    #                 origin_image_path = content_list[page].image_path
    #                 images.append(origin_image_path)
                        
    #     return question, texts, images
    
    def load_full_data(self):
        print("→ load_full_data") 
        samples = self.load_data(use_retreival=False)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_full_data(sample)
        return samples
    
    def load_sample_full_data(self, sample):
        print("→ load_sample_full_data") 
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question:str = sample[self.config.question_key]
        texts = []
        images = []
        
        if self.config.page_id_key in sample:
            sample_no_list = sample[self.config.page_id_key]
        else:
            sample_no_list = [i for i in range(0,min(len(content_list),self.config.vlm_max_page))]
        for page in sample_no_list:
            texts.append(content_list[page].txt.replace("\n", ""))
            origin_image_path = ""
            origin_image_path = content_list[page].image_path
            images.append(origin_image_path)
                        
        return question, texts, images
    

    # # DAG method
    def load_processed_content(self, sample: dict, disable_load_image=True) -> list[Content]:
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        content_list = []
        for page_idx in range(self.config.max_page):
            im_file = self.IM_FILE(doc_name, page_idx)
            text_file = self.TEXT_FILE(doc_name, page_idx)
            if not os.path.exists(im_file):
                break
            img = None
            if not disable_load_image:
                img = self.load_image(im_file)
            txt = self.load_txt(text_file)
            content_list.append(Content(image=img, image_path=im_file, txt=txt)) 
        return content_list

    
    # # MDocAgent Method
    # def load_processed_content(self, sample: dict, disable_load_image=True)->list[Content]:
    #     doc_name = self.EXTRACT_DOCUMENT_ID(sample)
    #     content_list = []
    #     for page_idx in range(self.config.max_page):
    #         im_file = self.IM_FILE(doc_name, page_idx)
    #         text_file = self.TEXT_FILE(doc_name, page_idx)
    #         if not os.path.exists(im_file):
    #             break
    #         img = None
    #         if not disable_load_image:
    #             img = self.load_image(im_file)
    #         txt = self.load_txt(text_file)
    #         content_list.append(Content(image=img, image_path=im_file, txt=txt)) 
    #     return content_list

    # # Graph Method
    # def load_processed_content(self, sample: dict, disable_load_image=False) -> list[Content]:
    #     # print("→ load_processed_content") 
    #     content_list: list[Content] = []
    #     # the JSON field holding all doc_page nodes:
    #     nodes = sample.get("graph-top-10-question_node", [])
    #     for node in nodes:
    #         # skip anything not of form "..._page<k>"
    #         if "_page" not in node:
    #             continue
    #         # split into doc_name and page index
    #         doc_name, page_str = node.rsplit("_page", 1)
    #         try:
    #             page_idx = int(page_str)
    #         except ValueError:
    #             continue

    #         # build filepaths
    #         im_file  = self.IM_FILE(doc_name, page_idx)
    #         txt_file = self.TEXT_FILE(doc_name, page_idx)
    #         # if the image for that page doesn’t exist, skip
    #         if not os.path.exists(im_file):
    #             print("Yes")
    #             continue

    #         # load image if requested
    #         img = None
    #         if not disable_load_image:
    #             img = self.load_image(im_file)
    #             # print(im_file)

    #         # load & truncate text
    #         txt = self.load_txt(txt_file)
    #         # print(txt_file)
    #         content_list.append(Content(image=img, image_path=im_file, txt=txt))

    #     return content_list
    
    def load_image(self, file):
        # print("→ load_image") 
        pil_im = Image.open(file)
        return pil_im

    def load_txt(self, file):
        # print("→ load_txt") 
        max_length = self.config.max_character_per_page
        with open(file, 'r') as file:
            content = file.read()
        content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return content[:max_length]
    
    def extract_content(self, resolution=144):
        print("→ extract_content") 
        samples = self.load_data()
        for sample in tqdm(samples):
            self._extract_content(sample, resolution=resolution)
            
    def _extract_content(self, sample, resolution=144):
        print("→ _extract_content") 
        max_pages=self.config.max_page
        os.makedirs(self.config.extract_path, exist_ok=True)
        image_list = list()
        text_list = list()
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        with pymupdf.open(os.path.join(self.config.document_path, sample["doc_id"])) as pdf:
            for index, page in enumerate(pdf[:max_pages]):
                # save page as an image
                im_file = self.IM_FILE(doc_name,index)
                if not os.path.exists(im_file):
                    im = page.get_pixmap(dpi=resolution)
                    im.save(im_file)
                image_list.append(im_file)
                # save page text
                txt_file = self.TEXT_FILE(doc_name,index)
                if not os.path.exists(txt_file):
                    text = page.get_text("text")
                    with open(txt_file, 'w') as f:
                        f.write(text)
                text_list.append(txt_file)
                
        return image_list, text_list
    
def extract_time(file_path):
    file_name = os.path.basename(file_path)
    time_str = file_name.split(".json")[0]
    return datetime.strptime(time_str, "%Y-%m-%d-%H-%M")

def find_latest_json(result_dir):
    pattern = os.path.join(result_dir, "*-*-*-*-*.json")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith('_results.json')]
    if not files:
        print(f"Json file not found at {result_dir}")
        return None
    latest_file = max(files, key=extract_time)
    return latest_file