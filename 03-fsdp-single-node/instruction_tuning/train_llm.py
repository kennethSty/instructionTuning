import wandb
import os
import logging
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
import torch
from torch.distributed.checkpoint.state_dict import set_state_dict
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import load_checkpoint_and_dispatch
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict,
        StateDictOptions
        )
from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel,
        CPUOffload,
        ShardingStrategy
        )
from functools import partial

from evaluation.compare_models import test_instructions
from helper.model_prep import reset_params_for_fsdp
from helper.logger import setup_logger
from helper.utils import get_parser, rank0_first, log_and_update_state, get_memory_stats
from helper.LocalTimer import LocalTimer
from data_preparation.DistributedDataProvider import DistributedDataProvider

@record
def main():
    parser = get_parser()
    args = parser.parse_args()
    seed = torch.manual_seed(args.seed)

    # Setup device variable and device default usage
    dist.init_process_group()
    local_rank = int(os.environ["LOCAL_RANK"]) #passed in by torchrun
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device) # use local_rank in all implicit .cuda() calls

    # Log process rank id
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    setup_logger(rank)
    logging.info(f"Setting up rank {rank} out of {world_size} ranks")


    # Setup Directory with Rank 0
    exp_dir = Path(f"{args.save_dir}/{args.experiment_name}")
    if rank == 0:
        logging.info(f"Setting up experiment directory: {exp_dir}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Finished setting up directory")
    dist.barrier()
    
    # Load data
    with rank0_first():
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        distributed_data_provider = DistributedDataProvider(args=args, config=config)
        distributed_train_loader, distributed_test_loader = distributed_data_provider.get_loaders()

    # Load model to CPU using rank 0 to avoid double loading and memory issues
    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=args.dtype, device_map="cpu")
        # Adapt model embedding size by increased vocab size
        if distributed_data_provider.pad_is_added():
            model.resize_token_embeddings(distributed_data_provider.get_vocab_size())
        full_state_dict = model.state_dict()
    else:
        full_state_dict = None

    # Load empty model shell on each GPU with only shape specs of params
    with rank0_first():
        with torch.device("meta"):
            # Adapt embedding layer according to tokenizer change
            # Change config as in "meta" mode no other change possible
            if distributed_data_provider.pad_is_added():
                config.vocab_size = distributed_data_provider.get_vocab_size()
            model = AutoModelForCausalLM.from_config(config, torch_dtype=args.dtype) 
    
    # Prepare Sharding Threshold
    # Wrap policy function which defines based on which criteria modules will be wrapped 
    # Uses size-based policy prefilled with threshold value from args
    size_based_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(args.numel_to_wrap)
            )

    # Pseudo-implement reset_parameters() function for each module as expected by FSDP
    reset_params_for_fsdp()
    # Use empty parameter model to specify how modules are to be sharded
    model = FullyShardedDataParallel(
            module=model,
            device_id=local_rank,
            sync_module_states=False, #sync not needed as pretrained weights are same across ranks
            auto_wrap_policy=size_based_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD, #equivalent to ZeRO3
            cpu_offload=CPUOffload(offload_params=args.offload_params_to_cpu)
            )

    # Now load the into the model shards
    # Automatically dispatches the weights according to the swrapped sharding
    dcp_options = StateDictOptions(
            full_state_dict=True, #rank 0 hs the full
            broadcast_from_rank0=True
            )

    set_model_state_dict(
            model=model,
            model_state_dict=full_state_dict,
            options=dcp_options
            )

    logging.info(f"Memory stats: {get_memory_stats(device)}")
    
    dist.destroy_process_group()



if __name__ == "__main__":
    main()


