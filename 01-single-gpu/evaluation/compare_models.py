from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
from helper.logger import LOGGER
from helper.utils import get_parser, _load_to_device
from data_preparation.DataProvider import DataProvider

def generate_response(model, data_provider, instruction, device, model_name="Model"):
    """Generate response for a given instruction"""
    system_text = "Below is an instruction that describes a task. Write a response that completes the request.\n\n"
    instruction_text = f"###Instruction:\n{instruction}\n\n###Response:\n"
    input_text_with_format = f"{system_text}{instruction_text}"
    input_ids_formatted = data_provider.encode(
        input_text_with_format, return_tensors="pt"
    ).to(device)

    prompt_log = (
        f"\n{'-'*20} {model_name} {'-'*20}\n"
        f"Input: {instruction}\n"
    )
    LOGGER.info(prompt_log)

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **input_ids_formatted,
            max_length=input_ids_formatted.input_ids.shape[1] + 100,
            do_sample=False,
            pad_token_id=data_provider.get_pad_token_id(),
            eos_token_id=data_provider.get_eos_token_id(),
            repetition_penalty=1.1
        )

    generated_tokens = output[0][input_ids_formatted.input_ids.shape[1]:]
    generated_text = data_provider.decode(generated_tokens, skip_special_tokens=True)
    
    LOGGER.info(f"Response: {generated_text}")
    return generated_text

def test_instructions(finetuned_model, data_provider, device, base_model=None):
    header = (
        f"\n{'='*80}\n"
        f"*****************TEST INSTRUCTION FOLLOWING ABILITY:********************\n"
        f"{'='*80}"
    )
    LOGGER.info(header)

    instructions_from_training_set = [
        "Research and summarize the common practices for caring of rabbits.",
        "Generate a list of 5 books that discuss the theme of resilience",
        "Identify the key characters of the novel 'The Hobbit'",
        "Write a function which reverses the order of words in a sentence.",
        "Generate a list of five advantages of using a certain product."
    ]

    for i, instruction in enumerate(instructions_from_training_set):
        LOGGER.info(f"\n{'='*80}\nTEST {i+1}:\n{'='*80}")

        if base_model is not None:
            generate_response(
                base_model, data_provider, instruction, device, 
                "PRETRAINED MODEL"
            )

        generate_response(
            finetuned_model, data_provider, instruction, device,
            "INSTRUCTION-TUNED MODEL"
        )

        LOGGER.info(f"\n{'='*50}")

def main():
    parser = get_parser()
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model_name)
    exp_dir = Path(f"{args.save_dir}/{args.experiment_name}")
    device = torch.device(args.device)
    dtype = torch.float32

    intro_log = (
        f"{'='*80}\n"
        f"MODEL COMPARISON: Pretrained vs Instruction-Tuned\n"
        f"{'='*80}"
    )
    LOGGER.info(intro_log)

    data_provider = DataProvider(args, config)

    LOGGER.info("\nLoading original pretrained model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=dtype
    ).to(device)
    LOGGER.info(f"Loaded pretrained {args.model_name}")

    LOGGER.info(f"\nLoading instruction-tuned model from {exp_dir}...")
    if not (exp_dir / "model.pt").exists():
        LOGGER.error(f"ERROR: No trained model found at {exp_dir}/model.pt")
        LOGGER.error("Please make sure you've trained a model first.")
        return

    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=dtype
    )
    if data_provider.is_pad_added_manually():
        finetuned_model.resize_token_embeddings(data_provider.get_vocab_size())
    finetuned_model.load_state_dict(_load_to_device(exp_dir/"model.pt", device))
    finetuned_model = finetuned_model.to(device)
    LOGGER.info(f"Loaded instruction-tuned model")

    test_instructions(
        finetuned_model=finetuned_model, 
        data_provider=data_provider,
        device=device,
        base_model=base_model
    )

    LOGGER.info(f"\n{'='*80}\nCOMPARISON COMPLETE!\n{'='*80}")

if __name__ == "__main__":
    main()

