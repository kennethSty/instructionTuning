import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, Conv1D

def reset_params_for_fsdp():
    # Expects weights are loaded from pretrained and therefore does nothing
    GPT2Attention.reset_parameters = lambda _: None
    Conv1D.reset_parameters = lambda _: None


