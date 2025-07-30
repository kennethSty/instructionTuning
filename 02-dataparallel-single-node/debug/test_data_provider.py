from argparse import Namespace
from transformers import AutoConfig
import torch.distributed as dist
from data_preparation.DataProvider import DataProvider
from helper.utils import rank0_first
from helper.logger import setup_logger
def main():
    args = Namespace(
        experiment_name="data test",
        model_name="openai-community/gpt2",
        dataset_name="tatsu-lab/alpaca",
        test_split=0.2,
        max_length=1024,
        ignore_token_id=-100,
        response_col_name="output",
        instruction_col_name="instruction",
        batch_size=8
    )

    dist.init_process_group()
    rank = dist.get_rank()
    setup_logger(rank)
    config = AutoConfig.from_pretrained(args.model_name)
    
    with rank0_first():
        data_provider = DataProvider(args, config)


if __name__ == "__main__":
    main()
