# Instruction Tuning with Increasing Scale

This repository (work in progress) explores **instruction tuning techniques** for large language models. It starts with a minimal single-GPU setup and gradually introduces more advanced parallelization strategies to enable efficient large-scale training on an NVIDIA DGX system.

## Project Structure

Each stage of the project is organized in its own directory: 
```
├── 01-single-gpu/ # Instruction tuning on a single GPU
└── 02-dataparallel-single-node/ # Instruction tuning with Data Parallelism on up to 8 GPUs
└── 03-fsdp-single-node/ # Instruction tuning with Fully Sharded Data Parallelism on up to 8 GPUs 

```
To run a specific stage, navigate to the corresponding folder (e.g., `01-single-gpu`) and follow the usage instructions in that folder’s `README.md`.

## Requirements

Dependencies and environment setup are described in the subdirectory READMEs. Typically, you'll need:
- Python 3.10+
- PyTorch
- CUDA-compatible GPU drivers
- [Hugging Face Transformers](https://github.com/huggingface/transformers) (for most examples)

## Acknowledgements
The code builds upon:
- [*LLMs from Scratch*](https://github.com/rasbt/llms-from-scratch) by **Sebastian Raschka**
- [*Distributed training guide*](https://github.com/LambdaLabsML/distributed-training-guide) from **Lambda Labs**


**Note: 
All training hyperparameters such as learning rate, batch size, number of epochs were chosen for illustration purposes and not further optimized. 
