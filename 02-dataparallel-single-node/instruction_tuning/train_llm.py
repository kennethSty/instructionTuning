import logging
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM

from helper.logger import setup_logger
from helper.utils import get_parser, rank0_first


@record
def main():
   
    parser = get_parser()
    args = parser.parse_args()

    dist.init_process_group()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()

    dtype = args.dtype
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    setup_logger(rank=rank)
    logging.info(f"Set device to {local_rank}")

    seed = torch.manual_seed(args.seed)

    with rank0_first(), device:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    logging.info(
       f"Using {args.model_name} model with {sum(p.numel() for p in model.parameters())} params"
    ) 

    #more logic follows
    
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
