import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_hf_token():
    load_dotenv()
    return os.environ["HUGGINGFACE_TOKEN"]

def load_model_and_tokenizer(model_name: str="Qwen/Qwen2-0.5B-Instruct", local_only=False):
    hf_token = get_hf_token()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_only,
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_only,
        token=hf_token
    )

    return model, tokenizer

def load_model_rank0(rank: int, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
    if rank == 0:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name, 
            local_only=False
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name, 
            local_only=True
        )
    return model, tokenizer
