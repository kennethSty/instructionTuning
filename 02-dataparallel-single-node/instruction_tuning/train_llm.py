import logging
import torch
from pathlib import Path
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoConfig
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer

from helper.logger import setup_logger
from helper.utils import get_parser, rank0_first
from data_preparation.DistributedDataProvider import DistributedDataProvider


@record
def main():
   
    #get arguments
    parser = get_parser()
    args = parser.parse_args()

    #setup distributed communicator
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()

    #setup directory
    exp_dir = Path(f"{args.save_dir}/{args.experiment_name}")
    if rank == 0:
        exp_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Setting up experiment directory: {exp_dir}")

    #setting up device
    dtype = args.dtype
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    setup_logger(rank=rank)
    logging.info(f"Set device to {local_rank}")

    seed = torch.manual_seed(args.seed)

    # Load model and data on rank 0 first
    with rank0_first():
        with device:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)

    with rank0_first():
        config = AutoConfig.from_pretrained(args.model_name)
        distributed_data_provider = DistributedDataProvider(args=args, config=config)
        distributed_train_loader, distributed_test_loader = distributed_data_provider.get_loaders()
    
    # Wrap model in DDP for synching gradients via mean allreduce before optim. steps
    # Note: DDP ensures models are initiated with same weights. As model is pretrained, this is given by default.
    model = DistributedDataParallel(model, device_ids=[local_rank])
    logging.info(
       f"Using {args.model_name} model with {sum(p.numel() for p in model.parameters())} params"
    ) 

    # Shard optimizer states across ranks
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(distributed_train_loader) * args.num_epochs, eta_min=args.lr * 1e-1
    )
    logging.info(
        f"Optimizer: {optimizer}"
    )

    # Load model if it is a resumed run
    # Note: no with_rank0 first needed, because model files are downloaded already as it is a resumed run
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0
    }
    runIsResumed = False
    
    with rank0_first():
        if (exp_dir / "state.json").exists():
            resume_run(model=model, 
                       lr_scheduler=lr_scheduler, 
                       state=state, 
                       exp_dir=exp_dir, 
                       device=device)
            runIsResumed = True
        logging.info(f"Resumed: {runIsResumed} | State: {state}")
        dist.barrier()



    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
