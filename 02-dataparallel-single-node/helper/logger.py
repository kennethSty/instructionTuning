import logging

def setup_logger(rank: int):
    """
    Configures rank aware global logging for distributed training.
    """
    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO
   )
