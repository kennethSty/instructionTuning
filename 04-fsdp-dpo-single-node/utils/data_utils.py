from datasets import load_dataset
import logging 
import torch.distributed as dist 

from utils.distributed_utils import rank0_first

def load_preference_dataset( 
    dataset_name: str="trl-lib/ultrafeedback_binarized", 
    split="train", 
    ):
    with rank0_first():
        dataset = load_dataset(
            dataset_name, 
            split=split
        )
    dist.get_rank()
    logging.info("Dataset loaded")
    return dataset
