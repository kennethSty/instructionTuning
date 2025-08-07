import torch
import json
import logging
import torch.distributed as dist
from functools import partial
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, Conv1D
from transformers import AutoModelForCausalLM, AutoConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict,
        get_model_state_dict,
        get_state_dict,
        StateDictOptions
        )
from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel,
        CPUOffload,
        ShardingStrategy
        )

from helper.utils import rank0_first, get_memory_stats


def save_checkpoint(fsdp_model, lr_scheduler, optimizer, exp_dir, state):
    """
    Saves a fsdp-sharded model and optimizer checkpoint using rank-specific 
    and therefore shard specific .pt files.
    Also saves learning rate scheduler states, which are assumed to not be sharded.
    """

    dist.barrier() # All ranks are ready to save
    logging.info("Saving checkpoints for sharded model, optimizer and full lr_scheduler")
    
    checkpoint_options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    
    # Get state dicts of the shard of the current rank
    sharded_model_state, sharded_optimizer_state = get_state_dict(
        fsdp_model, optimizer, options = checkpoint_options
        )
    # Save shard state dicts into checkpoint directory
    save(
        dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
        checkpoint_id=exp_dir / "checkpoint"
        )
    # Only rank 0 save lr_scheduler and state as these are not sharded
    if dist.get_rank() == 0:
        torch.save(lr_scheduler, exp_dir/"lr_scheduler.pt")
        with open(exp_dir/"state.json", "w") as f:
            json.dump(state, f)
    dist.barrier()



def resume_states(fsdp_model, optimizer, lr_scheduler):
    """
    Loads sharded state dicts from a previous run into model and optimizer inplace 
    """

    checkpoint_options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    
    # Load current state of sharded model params and optimizer states
    sharded_model_state, sharded_optimizer_state = get_state_dict(
        model, optimizer, options=checkpoint_options
        )


    # Load checkpointed sharded statedicts into sharded dicts of model and optimizer
    # Each rank loads its own shard .pt file from checkpoint dir automatically
    load(
        dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
        checkpoint_id=exp_dir/"checkpoint"
        )

    # Set sharded state dicts into sharded model and optimizer instance
    set_state_dict(
        model,
        optimizer,
        model_state_dict=sharded_model_state,
        optim_state_dict=sharded_optimizer_state,
        options=checkpoint
        )
    
    # LR scheduler is not sharded and can be loaded normally
    lr_scheduler.load_state_dict(
        torch.load(
            exp_dir / "lr_scheduler.pt", map_location=device, weights_only=True
            )
        )


def load_sharded_from_local(fsdp_model, exp_dir):
    """
    Loads the weights of a sharded model stored locally via "save_checkpoint" 
    into a sharded model state created from a huggingface instance created using model_name.
    """
    
    checkpoint_options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    sharded_state_dict = get_model_state_dict(fsdp_model, options=checkpoint_options)
    
    logging.info(f"Loading finetuned shards from {exp_dir/'checkpoint'}")
    if not (exp_dir/"checkpoint").exists():
        raise FileNotFoundError("No checkpoint exists at {exp_dir/'checkpoint'}")
    
    dist.barrier()

    # Assumption: Each rank automatically loads the right shard
    load(
        dict(model=sharded_state_dict),
        checkpoint_id = exp_dir / "checkpoint"
        )
    dist.barrier()
    set_model_state_dict(
        model=fsdp_model,
        model_state_dict=sharded_state_dict,
        options=checkpoint_options
        )
    dist.barrier()
    return fsdp_model


def load_and_shard_pretrained(args, config, rank, local_rank, pad_is_added, vocab_size, device):   
    """
    Utility function to load a pretrained model from huggingface and shard it 
    using FSDP.
    """

    full_state_dict = get_full_state_dict(
            args=args, 
            rank=rank, 
            pad_is_added=pad_is_added,
            vocab_size=vocab_size
            )
    
    # Load empty model shell on each GPU with only shape specs of params
    with rank0_first():
        with torch.device("meta"):
            # Adapt embedding layer according to tokenizer change
            # Change config as in "meta" mode no other change possible
            if pad_is_added:
                config.vocab_size = vocab_size
            model = AutoModelForCausalLM.from_config(config, torch_dtype=args.dtype) 
    
    # Prepare Sharding Threshold
    # Wrap policy function which defines based on which criteria modules will be wrapped 
    # Uses size-based policy prefilled with threshold value from args
    size_based_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(args.numel_to_wrap)
            )

    # Pseudo-implement reset_parameters() function for each module as expected by FSDP
    dummy_impl_reset()
    # Use empty parameter model to specify how modules are to be sharded
    fsdp_model = FullyShardedDataParallel(
            module=model,
            device_id=local_rank,
            sync_module_states=False, #sync not needed as pretrained weights are same across ranks
            auto_wrap_policy=size_based_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD, #equivalent to ZeRO3
            cpu_offload=CPUOffload(offload_params=args.offload_params_to_cpu)
            )

    # Now load the into the model shards
    # Automatically dispatches the shards of the fsdp model
    dcp_options = StateDictOptions(
            full_state_dict=True, #rank 0 has the full state dict and 
            broadcast_from_rank0=True
            )

    set_model_state_dict(
            model=fsdp_model,
            model_state_dict=full_state_dict,
            options=dcp_options
            )

    dist.barrier()
    
    logging.info(f"Loaded {args.model_name} and received shards")
    return fsdp_model


def dummy_impl_reset():
    """
    Implements the reset_parameters function for 
    modules that miss the implementation to be compatible with FSDP
    """
    # Expects weights are loaded from pretrained and therefore does nothing
    GPT2Attention.reset_parameters = lambda _: None
    Conv1D.reset_parameters = lambda _: None


def get_full_state_dict(args, rank, pad_is_added, vocab_size):
    """
    Loads the entire model on the CPU, to obtain and return the full model state dict.
    """

    # Load model to CPU using rank 0 to avoid double loading and memory issues
    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=args.dtype, 
            device_map="cpu"
            )
        # Adapt model embedding size by increased vocab size
        if pad_is_added:
            model.resize_token_embeddings(vocab_size)
        full_state_dict = model.state_dict()
        del model 
    # Other ranks will get parameters via broadcasting 
    else: 
        full_state_dict = None
    return full_state_dict
