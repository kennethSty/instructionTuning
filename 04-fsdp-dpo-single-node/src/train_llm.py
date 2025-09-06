from threading import local
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import wandb
from datasets import load_dataset
from contextlib import contextmanager
from trl import DPOConfig, DPOTrainer
from torch.distributed import  barrier
import torch.distributed as dist
from typing import Optional
from torch.distributed.elastic.multiprocessing.errors import record
import logging 

from utils.logger import setup_logger
from src.regul_dpo_trainer import RegulDPOTrainer
from utils.distributed_utils import rank0_first
from utils.model_utils import load_model_rank0
from utils.data_utils import load_preference_dataset

def get_dpo_config(
    output_dir="/data/Qwen2-4B-DPO-Regul", 
    run_name: str="dpo_regul_exp", 
    ld_alpha: Optional[float]=None
    ):

    args_dpo = DPOConfig(
        output_dir=output_dir,
        report_to="wandb",
        logging_steps=50,
        run_name=run_name,
        ld_alpha=ld_alpha,
    )

    return args_dpo


def test_regul_dpo():
    rank = dist.get_rank()
    model, tokenizer = load_model_rank0(rank=rank)
    dataset = load_preference_dataset()
        
    # --- W&B ---
    if rank == 0:
        logging.info("initializing wandb")
        wandb.init(
            project="dpo_experiments", 
            name="dpo-Qwen3-reg-length",
            reinit="finish_previous"
        )
        logging.info("done initializing wandb")
    else:
        wandb.init(mode="disabled")

    args_dpo = get_dpo_config(
        output_dir="/data/Qwen2-0.5B-DPO-Regul", 
        run_name="dpo_regul_exp", 
        )

    # --- Trainer: Standard DPO ---
    trainer_regul_dpo = RegulDPOTrainer(
        model=model,
        length_alpha=0.4,
        args=args_dpo,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer_regul_dpo.train()
    del model, tokenizer, dataset

def test_ld_dpo():
    rank = dist.get_rank()
    # Reload to start from same model state
    model, tokenizer = load_model_rank0(rank=rank)
    dataset = load_preference_dataset()
        
    # --- W&B ---
    if rank == 0:
        logging.info("initializing wandb")
        wandb.init(
            project="dpo_experiments", 
            name="dpo-Qwen3-length-desens", 
            reinit="finish_previous"
        )
        logging.info("done initializing wandb")

    args_dpo = get_dpo_config(
        output_dir="data/Qwen2-0.5B-DPO-ld",   
        run_name="run_ld_exp", 
        ld_alpha=0.4)

    # --- Trainer: Standard DPO ---
    logging.info("Constucting DPOTrainer")
    trainer_dpo = DPOTrainer(
        model=model,
        args=args_dpo,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if rank == 0:
        logging.info("starting training")
    trainer_dpo.train()
    if rank == 0:
        logging.info("finished training")
    del model, tokenizer, dataset

def get_hf_token():
    load_dotenv()
    return os.environ["HUGGINGFACE_TOKEN"]   

@record
def main():
    local_rank = int(os.environ.get("LOCAL_RANK"))
    device = torch.device(f"cuda:{local_rank}")
    logging.info("Local Rank:", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', device_id=device)
    logging.info("Testing Length Desensitization")
    test_ld_dpo()
    dist.barrier()
    logging.info("Test LD Done")
    test_regul_dpo()
    dist.destroy_process_group() 


if __name__ == "__main__":
    main()
