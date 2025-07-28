from argparse import Namespace
from data_preparation.DataProvider import DataProvider
from transformers import AutoConfig

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

    config = AutoConfig.from_pretrained(args.model_name)
    data_provider = DataProvider(args, config)


if __name__ == "__main__":
    main()
