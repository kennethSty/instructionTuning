from argparse import Namespace
from transformers import AutoConfig
from data_preparation.PreProcessor import PreProcessor

def test_preprocessor():
    args = Namespace(
        experiment_name="preproc test",
        model_name="openai-community/gpt2",
        dataset_name="tatsu-lab/alpaca",
        teest_split=0.2,
        max_length=1024,
        ignore_token_id=-100,
        response_col_name="output",
        instruction_col_name="instruction",
        batch_size=8
    )

    config = AutoConfig.from_pretrained(args.model_name)
    preprocessor = PreProcessor(args, config)
    data = preprocessor.get_preprocessed_data()
    print(f"Loaded dataset of size: {len(data)}")

if __name__ == "__main__":
    test_preprocessor() 

