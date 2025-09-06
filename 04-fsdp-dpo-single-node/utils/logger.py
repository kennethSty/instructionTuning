import logging

def setup_logger(rank: int):
    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO
    )
