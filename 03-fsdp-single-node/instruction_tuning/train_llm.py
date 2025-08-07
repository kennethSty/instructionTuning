import wandb
import os
import logging
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoConfig
from torch.optim import AdamW

from evaluation.compare_models import test_instructions
from helper.model_utils import load_and_shard_pretrained, resume_states, save_checkpoint
from helper.logger import setup_logger
from helper.utils import get_parser, setup_directories, rank0_first, log_and_update_state, get_memory_stats
from helper.LocalTimer import LocalTimer
from data_preparation.DistributedDataProvider import DistributedDataProvider

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
        distributed_data_provider = DistributedDataProvider(args=args, config=config)
        distributed_train_loader, distributed_test_loader = distributed_data_provider.get_loaders()

    # Load on cpu via rank0, shard it and dispatch the sharded weights to other ranks
    # Internally params are changed from Tensor to DTensor (D = distributed)
    fsdp_model = load_and_shard_pretrained(
            args=args,
            config=config, 
            rank=rank, 
            local_rank=local_rank, 
            pad_is_added=distributed_data_provider.pad_is_added(),
            vocab_size=distributed_data_provider.get_vocab_size(),
            device=device)
    logging.info(f"Memory stats after loading model:{get_memory_stats(device)}")

    # Optimizer is init for only the shard of model weights local to the current device
    # fsdp_model params are DTensor => optimizer states are DTensor
    optimizer = AdamW(
            fsdp_model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            fused=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=len(distributed_train_loader) * args.num_epochs, 
            eta_min=args.lr*0.1
            )
    
    # Check if we resume a previous experiment run
    state = {
            "epoch": 0,
            "global_step": 0,
            "epoch_step": 0,
            "running_loss": 0
            }
    run_is_resumed = False
    state_dir_exists = (exp_dir / "states.json").exists()
    if state_dir_exists:
        resume_states(fsdp_model, optimizer, lr_scheduler)
        with open(exp_dir/"states.json") as f:
            state = json.load(f)
            run_is_resumed = True
    logging.info(f"Resumed = {run_is_resumed} | State = {state}")
    dist.barrier()

    # Setup tracking in weights and biases
    if rank == 0:
        train_size, _ = distributed_data_provider.get_train_test_size()
        wandb.init(
                project="distributed_train_ground",
                dir=exp_dir,
                name=args.experiment_name,
                id=args.experiment_name,
                resume="must" if run_is_resumed else None,
                save_code=True,
                config={
                    "args": vars(args),
                    "training_data_size": train_size,
                    "num_batches": len(distributed_train_loader) * world_size,
                    "world_size": world_size
                    }
                )
   
    steps_to_time = ["forward", "backward", "build_batch", "optimizer_step"]
    timers = {k: LocalTimer(device) for k in steps_to_time}
    
    train_fsdp_model(
            fsdp_model=fsdp_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            distributed_data_provider=distributed_data_provider,
            timers=timers,
            state=state,
            args=args,
            exp_dir=exp_dir,
            rank=rank,
            world_size=world_size,
            device=device
            )

    dist.destroy_process_group()


def train_fsdp_model(
        fsdp_model, optimizer, lr_scheduler, distributed_data_provider,
        timers, state, args, exp_dir, rank, world_size, device):
    
    for state["epoch"] in range(state["epoch"], args.num_epochs):
        logging.info(f"Start epoch {state['epoch']} at step {state['epoch_step']}")
        num_train_batches = distributed_data_provider.get_num_train_batches()
        progress_bar = tqdm(range(num_train_batches), disable=(rank>0))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        # Create iterable so that we can time batch construction
        distributed_train_loader, _ = distributed_data_provider.get_loaders()
        distributed_train_loader.sampler.set_epoch(state["epoch"]) #Make sampler shuffle differently in every epoch
        distributed_train_batches = iter(distributed_train_loader)

        for step_i in range(len(distributed_train_loader)):
            if step_i < state["epoch_step"]:
                continue # For resuming

            with timers["build_batch"], torch.no_grad():
                batch = next(distributed_train_batches)
                input_ids = batch[0].to(device)
                label_ids = batch[1].to(device)

            with timers["forward"]:
                outputs = fsdp_model(input_ids=input_ids, labels=label_ids)

            with timers["backward"]:
                optimizer.zero_grad(set_to_none=True)
                outputs.loss.backward()

            with timers["optimizer_step"]:
                optimizer.step()
                lr_scheduler.step()

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                log_and_update_state(
                    state=state,
                    timers=timers,
                    step_i=step_i,
                    lr_scheduler=lr_scheduler,
                    train_loader=distributed_train_loader,
                    args=args,
                    device=device,
                    rank=rank,
                    world_size=world_size
                    )

            if state["global_step"] % args.checkp_freq == 0:
                dist.barrier()
                logging.info(f"=========== RANK {rank} SAVING MODEL SHARD ============")
                save_checkpoint(
                    fsdp_model=fsdp_model,
                    lr_scheduler=lr_scheduler,
                    optimizer=optimizer,
                    exp_dir=exp_dir,
                    state=state
                    )
                dist.barrier()
                test_instructions(
                    fsdp_model=fsdp_model,
                    model_name="Current Training State",
                    preprocessor=distributed_data_provider.preprocessor,
                    device=device
                    )
                dist.barrier()

        state["epoch_step"] = 0

        logging.info("=============SAVING & TESTING FINAL STATE==============")
        dist.barrier()
        save_checkpoint(
            fsdp_model=fsdp_model,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            exp_dir=exp_dir,
            state=state
            )
        dist.barrier()
        test_instructions(
            fsdp_model=fsdp_model,
            model_name="Final Training State",
            preprocessor=distributed_data_provider.preprocessor,
            device=device
            )
        dist.barrier()

        logging.info("=================TRAINING FINISHED===================")

    # Finish script when all ranks are finished
    dist.barrier()

if __name__ == "__main__":
    main()


