from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import logging 
import os
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from helper.generate_utils import generate_response
from helper.model_utils import load_and_shard_pretrained, load_sharded_from_local
from helper.logger import setup_logger
from helper.utils import get_parser, setup_directories, rank0_first
from data_preparation.PreProcessor import PreProcessor 

def test_instructions(fsdp_model, model_name, preprocessor, device):

    instructions_from_training_set = [
        "Research and summarize the common practices for caring of rabbits.",
        "Generate a list of 5 books that discuss the theme of resilience",
        "Identify the key characters of the novel 'The Hobbit'",
        "Write a function which reverses the order of words in a sentence.",
        "Generate a list of five advantages of using a certain product."
    ]

    rank = dist.get_rank()

    for i, instruction in enumerate(instructions_from_training_set):
        dist.barrier() # Enforces pretty test output
        if rank == 0: logging.info(f"\n{'='*80}\nTEST {i+1}:\n{'='*80}")
        response = generate_response(
                fsdp_model=fsdp_model,
                raw_instruction_text=instruction,
                device=device,
                preprocessor=preprocessor
                )
        logging.info(f"###Response by {model_name}: {response}")
        dist.barrier()

    footer = (
        f"\n{'='*80}\n"
        f"*****************FINISH TEST:********************\n"
        f"{'='*80}"
    )

    if rank == 0: logging.info(footer)

@record
def main():
    parser = get_parser()
    args = parser.parse_args()
    seed = torch.manual_seed(args.seed)

    dist.init_process_group()
    local_rank = int(os.environ["LOCAL_RANK"]) #passed in by torchrun
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device) # use local_rank in all implicit .cuda() calls

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    setup_logger(rank)
    logging.info(f"Setting up rank {rank} out of {world_size} ranks")

    exp_dir = Path(f"{args.save_dir}/{args.experiment_name}")
    setup_directories(exp_dir, rank)

    # Load data
    with rank0_first():
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        preprocessor = PreProcessor(args=args, config=config)

    fsdp_base = load_and_shard_pretrained(
            args=args,
            config=config,
            rank=rank,
            local_rank=local_rank,
            pad_is_added=preprocessor.pad_is_added(),
            vocab_size=preprocessor.get_vocab_size(),
            device=device)
    
    header_base = (
        f"\n{'='*80}\n"
        f"*****************TEST INSTRUCTION FOLLOWING ABILITY BASE MODEL:********************\n"
        f"{'='*80}"
    )
    
    if rank == 0:
        logging.info(f"\n{'='*80}\n COMPARISON BASE VS FINETUNED \n{'='*80}")
        logging.info(header_base)

    test_instructions(
            fsdp_model=fsdp_base, 
            model_name="Base Model", 
            preprocessor=preprocessor, 
            device=device)

    fsdp_finetuned = load_sharded_from_local(fsdp_base, exp_dir)
    
    header_finetuned = (
        f"\n{'='*80}\n"
        f"*****************TEST INSTRUCTION FOLLOWING ABILITY FINETUNED MODEL:********************\n"
        f"{'='*80}"
    )
    
    if rank == 0:
        logging.info(header_finetuned)
    
    test_instructions(
            fsdp_model=fsdp_finetuned, 
            model_name="Finetuned Model", 
            preprocessor=preprocessor, 
            device=device)
    
    if rank == 0:
        logging.info(f"\n{'='*80}\nCOMPARISON COMPLETE!\n{'='*80}")
    
    dist.destroy_process_group()
     

if __name__ == "__main__":
    main()
