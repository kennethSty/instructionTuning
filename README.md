# Instruction Tuning with Increasing Scale

This repository explores **instruction tuning techniques** for large language models. It starts with a minimal single-GPU setup and gradually introduces more advanced parallelization strategies to enable efficient large-scale training on an NVIDIA DGX system.

## Project Structure

Each stage of the project is organized in its own directory: 
```
├── 01-single-gpu/ 
└── ... work in progress .. 
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
