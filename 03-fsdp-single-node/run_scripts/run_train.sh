#!/bin/bash

echo "Run training script"

export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=8

torchrun --standalone \
	--nproc-per-node=4 \
	-m "instruction_tuning.train_llm" \
	-e "fsdp-test-$(date +%Y-%m-%d-%H-%M-%S)" \
	--model_name "openai-community/gpt2" \
	--dataset_name "tatsu-lab/alpaca" \
	--instruction_col_name instruction \
	--response_col_name output \
	--test_split 0.1 \
	--checkp_freq 1

echo "Training completed at: $(date)"
