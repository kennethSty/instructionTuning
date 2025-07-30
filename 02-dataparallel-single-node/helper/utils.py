import argparse
import torch
from contextlib import contextmanager
import torch.distributed as dist

def get_parser() -> argparse.ArgumentParser: 
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", default=None, required=True)
    parser.add_argument("-d", "--dataset_name", default=None, required=True)
    parser.add_argument("-m", "--model_name", default=None, required=True)
    parser.add_argument("--instruction_col_name", default=None, required=True)
    parser.add_argument("--response_col_name", default=None, required=True)
    parser.add_argument("--test_split", default=None, required=True, type=float)

    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--weight_decay", default=0.1)
    parser.add_argument("--dtype", default=torch.float32)
    parser.add_argument("--save_dir", default="../outputs")
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkp_freq", default=500)
    parser.add_argument("--log_freq", default=100)
    
    return parser

@contextmanager
def rank0_first():
    """
    Context executing whatever is within the context on rank 0 first.
    IMPORTANT: assumes dist.init_process_group() is already called. 
    Otherwise no rank and dist information is available
    """
    rank = dist.get_rank()
    if rank == 0:
        yield #execute code within context here
    dist.barrier()
    if rank > 0:
        yield #execute code within context here
    dist.barrier()
