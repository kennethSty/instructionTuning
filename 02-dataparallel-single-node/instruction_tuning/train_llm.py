import logging
from tqdm import tqdm
import torch
from pathlib import Path
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoConfig
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer
import wandb
import os

from evaluation.compare_models import test_instructions
from helper.logger import setup_logger
from helper.utils import get_parser, rank0_first, log_and_update_state, save_checkpoint
from helper.LocalTimer import LocalTimer
from data_preparation.DistributedDataProvider import DistributedDataProvider


@record
def main():
    #get arguments
    parser = get_parser()
    args = parser.parse_args()
    dtype = args.dtype

    # Do torch.cuda.set_device before init process 
    # Tells torch which gpu this ranks is to use for comm. and comp.
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device) 
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    setup_logger(rank=rank)
    logging.info(f"Setting up rank {rank} out of {world_size} ranks")
    print(f"=============== RankyRanky: {rank}======================")
    
    #setup directory
    exp_dir = Path(f"{args.save_dir}/{args.experiment_name}")
    if rank == 0:
        logging.info("Setting up experiment directory: {exp_dir}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        logging.info("FINISHED Setting up experiment directory")
    dist.barrier()

    seed = torch.manual_seed(args.seed)

    # Load ddp_model and data on rank 0 first
    with rank0_first():
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
        config = AutoConfig.from_pretrained(args.model_name)
        distributed_data_provider = DistributedDataProvider(args=args, config=config)
        distributed_train_loader, distributed_test_loader = distributed_data_provider.get_loaders()
        if distributed_data_provider.pad_is_added():
            model.resize_token_embeddings(distributed_data_provider.get_vocab_size())

    # Wrap ddp_model in DDP for synching gradients via mean allreduce before optim. steps
    # Note: DDP ensures ddp_models are initiated with same weights. As ddp_model is pretrained, this is given by default.
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    logging.info(
       f"Using {args.model_name} ddp_model with {sum(p.numel() for p in ddp_model.parameters())} params"
    ) 

    # Shard optimizer states across ranks
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(distributed_train_loader) * args.num_epochs, eta_min=args.lr * 1e-1
    )
    logging.info(
        f"Optimizer: {optimizer}"
    )

    # Load ddp_model if it is a resumed run
    # Note: no with_rank0 first needed, because ddp_model files are downloaded already as it is a resumed run
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0
    }
    runIsResumed = False
    
    with rank0_first():
        if (exp_dir / "state.json").exists():
            resume_run(ddp_model=ddp_model, 
                       lr_scheduler=lr_scheduler, 
                       state=state, 
                       exp_dir=exp_dir, 
                       device=device)
            runIsResumed = True
        logging.info(f"Resumed: {runIsResumed} | State: {state}")
        dist.barrier()

    # Weights and Biases tracks only loss state on rank0 
    if rank == 0:
        print(f"exp name {args.experiment_name}")
        train_size, test_size = distributed_data_provider.get_train_test_size()
        wandb.init(
            project="distributed_training_trainground",
            dir=exp_dir,
            name=args.experiment_name,
            id=args.experiment_name,
            resume="must" if runIsResumed else None,
            save_code=True,
            config={
                "args": vars(args),  #turns args Namespace into string
                "training_data_size": train_size, 
                "num_batches": len(distributed_train_loader) * world_size,
                "world_size": world_size
            }

        )

    steps_to_time = ["forward", "backward", "build_batch", "optimizer_step"]
    timers = {k: LocalTimer(device) for k in steps_to_time}

    train_ddp_model(
        ddp_model=ddp_model,
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
    

def train_ddp_model(
        ddp_model, optimizer, lr_scheduler, distributed_data_provider,
        timers, state, args, exp_dir, rank, world_size, device):
    
    for state["epoch"] in range(state["epoch"], args.num_epochs):
        logging.info(f"Start epoch {state['epoch']} at step {state['epoch_step']}")
        num_train_batches = distributed_data_provider.get_num_train_batches()
        progress_bar = tqdm(range(num_train_batches), disable=(rank>0))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])
        
        # Create iterable so that we can time batch construction
        distributed_train_loader, _ = distributed_data_provider.get_loaders()
        distributed_train_batches = iter(distributed_train_loader)

        for step_i in range(len(distributed_train_loader)):
            if step_i < state["epoch_step"]:
                continue

            with timers["build_batch"], torch.no_grad():
                batch = next(distributed_train_batches)
                input_ids = batch[0].to(device)
                label_ids = batch[1].to(device)

            with timers["forward"]:
                outputs = ddp_model(input_ids=input_ids, labels=label_ids)

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
                if rank == 0:
                    logging.info("=========== RANK 0 SAVING MODEL ============")
                    save_checkpoint(
                        model=ddp_model,
                        lr_scheduler=lr_scheduler,
                        optimizer=optimizer,
                        exp_dir=exp_dir,
                        state=state
                        )

                    test_instructions(
                        finetuned_model=ddp_model.module, #unwrap to use .generate()
                        distributed_data_provider=distributed_data_provider,
                        device=device
                        )
                # Make other ranks wait until model is saved and evaluated
                dist.barrier()

        state["epoch_step"] = 0

    if rank == 0:
        logging.info("=============SAVING & TESTING FINAL STATE==============")
        save_checkpoint(
            model=ddp_model,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            exp_dir=exp_dir,
            state=state
            )

        test_instructions(
            finetuned_model=ddp_model.module, #unwrap to use .generate()
            distributed_data_provider=distributed_data_provider,
            device=device
            )
        
        logging.info("=================TRAINING FINISHED===================")
    
    # Finish script when all ranks are finished
    dist.barrier()


if __name__ == "__main__":
    main()
