import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from retrieval.base_retrieval import BaseRetrieval
import hydra
import importlib
import torch

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.retrieval.cuda_visible_devices
    print(torch.cuda.device_count())
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))

    retrieval_class_path = cfg.retrieval.class_path
    module_name, class_name = retrieval_class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    retrieval_class = getattr(module, class_name)
    
    print("-------------Loading data-------------")
    dataset = BaseDataset(cfg.dataset)
    print("-------------Loading retrieval-------------")
    retrieval_model:BaseRetrieval = retrieval_class(cfg.retrieval)

    print("-------------Finding TopK-------------")
    retrieval_model.find_top_k(dataset)

if __name__ == "__main__":
    main()