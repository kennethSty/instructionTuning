from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import logging 
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from helper.generate_utils import generate_response
from helper.logger import setup_logger
from helper.utils import get_parser
from data_preparation.DistributedDataProvider import DistributedDataProvider

def test_instructions(fsdp_model, distributed_data_provider, device):
    header = (
        f"\n{'='*80}\n"
        f"*****************TEST INSTRUCTION FOLLOWING ABILITY:********************\n"
        f"{'='*80}"
    )
    
    rank = dist.get_rank()
    if rank == 0:
        logging.info(header)

    _, distributed_test_loader = distributed_data_provider.get_loaders()
    instructions_from_training_set = [
        "Research and summarize the common practices for caring of rabbits.",
        "Generate a list of 5 books that discuss the theme of resilience",
        "Identify the key characters of the novel 'The Hobbit'",
        "Write a function which reverses the order of words in a sentence.",
        "Generate a list of five advantages of using a certain product."
    ]

    for i, instruction in enumerate(instructions_from_training_set):
        dist.barrier() # Enforces pretty test output
        if rank == 0: logging.info(f"\n{'='*80}\nTEST {i+1}:\n{'='*80}")
        response = generate_response(
                fsdp_model=fsdp_model,
                raw_instruction_text=instruction,
                device=device,
                preprocessor=distributed_data_provider.preprocessor
                )
        logging.info(f"###Generated Text: {response}")
        dist.barrier()

    footer = (
        f"\n{'='*80}\n"
        f"*****************FINISH TEST:********************\n"
        f"{'='*80}"
    )

    if rank == 0: logging.info(footer)
    
