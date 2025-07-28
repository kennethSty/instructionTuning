import torch
import argparse
import json
import wandb
import subprocess

from helper.logger import LOGGER

def get_free_gpu():
    try:
        # Query memory from nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )

        print("=========Selecting Free GPU============\n")
        print(result)
        # Parse memory values
        memory_free = [int(x) for x in result.stdout.strip().split('\n')]
        best_gpu = int(torch.tensor(memory_free).argmax())
        print(f"CHOSEN GPU ID: {best_gpu}")
        # Set device
        return torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")

    except Exception as e:
        print(f"Failed to auto-select GPU: {e}")
        return torch.device("cpu")


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "current_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "current_reserved_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_reserved_gb": 1e-9 * mem["reserved_bytes.all.peak"]
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    #Required
    parser.add_argument("-e", "--experiment_name", default=None, required=True)
    parser.add_argument("-d", "--dataset_name", default=None, required=True)
    parser.add_argument("-m", "--model_name", default=None, required=True)
    parser.add_argument("--instruction_col_name", default=None, required=True)
    parser.add_argument("--response_col_name", default=None, required=True)
    parser.add_argument("--test_split", default=None, required=True, type=float)
    #Optional
    parser.add_argument("--dtype", default=torch.float32)
    parser.add_argument("--ignore_token_id", default=-100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="../outputs")
    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--max_length", default=1024)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--checkp_freq", default=500)
    parser.add_argument("--log_freq", default=100)

    return parser

def _load_to_device(path, device):
    return torch.load(path, map_location=device, weights_only=True)

def resume_run(model, optimizer, lr_scheduler, exp_dir, device):
    # Load model state from last state via checkpoint
    model.load_state_dict(_load_to_device(exp_dir/"model.pt", device))
    optimizer.load_state_dict(_load_to_device(exp_dir/"optimizer.pt", device))
    lr_scheduler.load_state_dict(_load_to_device(exp_dir/"lr_scheduler.pt", device))
    
    # Update state 
    with open(exp_dir/"state.json") as f:
        state = json.load(f)


def log_and_update_state(state, timers, step_i, lr_scheduler, train_loader, args, device): 
    tokens_per_step = args.batch_size * args.max_length 
    ms_per_step = sum(t.get_avg_elapsed_ms() for t in timers.values()) 
    info ={ 
        "global_step": state["global_step"], 
        "lr": lr_scheduler.get_last_lr()[0], 
        "running_loss": state["running_loss"] / args.log_freq, 
        "epoch": state["epoch"], 
        "epoch_progress": state["epoch_step"] / len(train_loader), 
        "num_batches_remaining": len(train_loader) - step_i, 
        **get_mem_stats(device), 
        "tokens/s": 1000 * (tokens_per_step / ms_per_step), 
        "ms per step": ms_per_step, 
        **{ 
            f"Avg. time (ms) per {timed_element}": timer.get_avg_elapsed_ms() 
            for timed_element, timer in timers.items() 
        } 
    } 
 
    LOGGER.info(info) 
    wandb.log(info, step=state["global_step"]) 
    torch.cuda.reset_peak_memory_stats(device) #clears memory usage history 
    state["running_loss"] = 0 
    for t in timers.values(): 
        t.reset() 
 
def save_checkpoint(model, optimizer, lr_scheduler, exp_dir, state): 
    LOGGER.info("Saving checkpoint.") 
    torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt") 
    torch.save(model.state_dict(), exp_dir / "model.pt") 
    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt") 
    with open(exp_dir / "state.json", "w") as f: 
        json.dump(state, f) 
         
 
