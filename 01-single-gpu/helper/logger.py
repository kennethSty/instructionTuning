import logging

LOGGER = logging.getLogger("train_llm")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s -%(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
