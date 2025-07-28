import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import datasets
import multiprocessing
from typing import List, Dict
from itertools import chain
import json
import torch 
import wandb
import tqdm

from data_preparation.DataProvider import DataProvider
from helper.LocalTimer import LocalTimer
from helper.utils import (
        get_mem_stats, get_parser, resume_run,
        save_checkpoint, log_and_update_state, get_free_gpu
        )
from helper.logger import LOGGER
from evaluation.compare_models import test_instructions

def main():
    #Parse CLI commands
    parser = get_parser()
    args = parser.parse_args()
    exp_dir = Path(f"{args.save_dir}/{args.experiment_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    device = get_free_gpu()
    dtype = torch.float32
    LOGGER.info(f"Environment: {os.environ}\n\n")
    LOGGER.info(f"CL Arguments: {args} \n\n")
    
    #Prepare Data
    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    data_provider = DataProvider(args=args, config=config)
    LOGGER.info(f"Instances in trainloader: {len(data_provider.train_loader)}\n\n")
    LOGGER.info(f"Instances in testloader:L {len(data_provider.test_loader)}\n\n")
    
    #Prepare Model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    if data_provider.is_pad_added_manually():
        model.resize_token_embeddings(data_provider.get_vocab_size())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(data_provider.train_loader) * args.num_epochs, eta_min=args.lr * 0.1
    )
    LOGGER.info(f"Model: {model}\n\n")
    LOGGER.info(f"Optimizer: {optimizer}\n\n")

    #Load model if it is a resumed run
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0
    }
    resumed = False
    if (exp_dir / "state.json").exists():
        resume_run(model, optimizer, lr_scheduler, exp_dir, device)
        resumed = True
    LOGGER.info(f"Resumed: {resumed} \n State: {state}")

    #Initiate tracking in weights and biases (wandb)
    num_train_batches = data_provider.get_num_train_batches()
    wandb.init(
        project="distributed_training_trainground",
        dir=exp_dir,
        name=args.experiment_name,
        id=args.experiment_name,
        resume="must" if resumed else None,
        save_code=True,
        config={
            "args": vars(args), #turns args into pretty string
            "training_data_size": num_train_batches * args.batch_size,
            "num_batches": num_train_batches
        }
    )

    # Setup LocalTimer objects to gather the times needed in each training step
    steps_to_time = ["forward", "backward", "build_batch", "optimizer_step"]
    timers = {step: LocalTimer(device) for step in steps_to_time}

    # Train the model 
    train_model(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        data_provider=data_provider,
        timers=timers,
        state=state,
        args=args,
        exp_dir=exp_dir,
        device=device
    )


def train_model(
        model, optimizer, lr_scheduler, data_provider, 
        timers, state, args, exp_dir, device):

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Start epoch {state['epoch']} at step {state['epoch_step']}")
        num_train_batches = data_provider.get_num_train_batches()
        progress_bar = tqdm.tqdm(range(num_train_batches))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])


        training_batches = iter(data_provider.train_loader)
        for step_i in range(num_train_batches):
            if step_i < state["epoch_step"]:
                continue

            with timers["build_batch"], torch.no_grad():
                batch = next(training_batches)
                input_ids = batch[0].to(device)
                label_ids = batch[1].to(device)
           
            with timers["forward"]:
                outputs = model(
                    input_ids=input_ids, 
                    labels = label_ids
                )

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
                    train_loader=data_provider.train_loader, 
                    args=args, 
                    device=device)
            if state["global_step"] % args.checkp_freq == 0:
                save_checkpoint(
                    model=model, 
                    optimizer=optimizer, 
                    lr_scheduler=lr_scheduler, 
                    exp_dir=exp_dir,
                    state=state
                    )
        
        # Use model to get some first answers
        test_instructions(
                finetuned_model=model,
                data_provider=data_provider,
                device=device)
        state["epoch_step"] = 0



if __name__ == "__main__":
    main()
