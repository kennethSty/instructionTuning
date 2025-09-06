import torch.distributed as dist
from contextlib import contextmanager

@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
       yield
    dist.barrier()
