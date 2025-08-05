import argparse
import torch
from contextlib import contextmanager
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from functools import partial
import wandb
import json
import logging

def get_parser() -> argparse.ArgumentParser: 
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", default=None, required=True)
    parser.add_argument("-d", "--dataset_name", default=None, required=True)
    parser.add_argument("-m", "--model_name", default=None, required=True)
    parser.add_argument("--instruction_col_name", default=None, required=True)
    parser.add_argument("--response_col_name", default=None, required=True)
    parser.add_argument("--test_split", default=None, required=True, type=float)
    
    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--dtype", default=torch.float32)
    parser.add_argument("--save_dir", default="../outputs")
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--num_epochs", default=12, type=int)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--ignore_token_id", default=-100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkp_freq", default=500)
    parser.add_argument("--log_freq", default=100)
    parser.add_argument("--mask_instruction", default=False, type=bool, 
                        help="If true instruction part of will be masked in labels")
    parser.add_argument("--shift_labels", default=False, 
                        help="If True, label token positions are shifted by 1")
    parser.add_argument("--numel_to_wrap", default=100000000, 
                        type=int, help="Only apply FSDP to modules with num params > this value")
    parser.add_argument("--offload_params_to_cpu", default=False, type=bool)
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


def load_model_lr_state(model, lr_scheduler, state, exp_dir, device):
    """Loads model, state and learning rate scheduler states from a prev. run inplace"""
    model.load_state_dict(
        torch.load(exp_dir/"model.pt", map_location=device, weights_only=True)
    )
    lr_scheduler.load_state_dict(
        torch.load(exp_dir/"lr_scheduler.pt", map_location=device, weights_only=True)
    )

    with open(exp_dir/"state.json") as f:
        state = json.load(f)

def log_and_update_state(state, timers, step_i, lr_scheduler, train_loader, args, rank, world_size, device):
    tokens_per_step = world_size * args.batch_size * args.max_length
    ms_per_train_step = sum(t.get_avg_elapsed_ms() for t in timers.values())
    timing_dict = {f"Avg. time (ms) per {name}": timer.get_avg_elapsed_ms() for name, timer in timers.items()}
    log_dict = {
        "global_step": state["global_step"],
        "lr": lr_scheduler.get_last_lr()[0],
        "running_loss": state["running_loss"] / args.log_freq, # Avg loss over log_freq steps
        "epoch": state["epoch"],
        "epoch_progress": state["epoch_step"]/len(train_loader),
        "num_batches_remaining": len(train_loader) - step_i,
        **get_memory_stats(device),
        **timing_dict,
        "tokens/s": 1000 * (tokens_per_step / ms_per_train_step),
        "ms per training step": ms_per_train_step,
        "tokens per training step": tokens_per_step
    }
    logging.info(log_dict)

    if rank == 0: #only log to wandb for rank 0
        wandb.log(log_dict, step=state["global_step"])

    # Reset loss after each log, to accumulte loss for next training period
    state["running_loss"] = 0 
    for t in timers.values():
        t.reset()
   
def get_memory_stats(device):
    memory_stats = torch.cuda.memory_stats(device)
    properties = torch.cuda.get_device_properties(device)
    memory_info_dict = {
        "total_gb": 1e-9 * properties.total_memory,
        "current_alloc_gb": 1e-9 * memory_stats["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * memory_stats["allocated_bytes.all.peak"],
        "current_reserved_gb": 1e-9 * memory_stats["reserved_bytes.all.current"],
        "peak_reserved_gb": 1e-9 * memory_stats["reserved_bytes.all.peak"]
    }
    
    # Reset memory history usage to start accumulating for next period
    torch.cuda.reset_peak_memory_stats(device)
    return memory_info_dict

def save_checkpoint(model, lr_scheduler, optimizer, exp_dir, state):
    logging.info("Saving checkpoints for model and lr_scheduler")
    if not isinstance(optimizer, ZeroRedundancyOptimizer):
        logging.info("Saving optmizer too as it is not sharded")
        torch.save(optimizer.state_dict(), exp_dir/"optimizer.pt")
    else: 
        logging.info("Warning: not saving optimizer as it is sharded")
    torch.save(model.state_dict(), exp_dir/"model.pt")
    torch.save(lr_scheduler, exp_dir/"lr_scheduler.pt")
    with open(exp_dir/"state.json", "w") as f:
        json.dump(state, f)
