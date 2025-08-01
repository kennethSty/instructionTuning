#!/bin/bash

# Usage: ./run_scripts/run_compare.sh experiment_name
# Example: ./run_scripts/run_compare.sh ddp-gpt2-2025-08-01-13-22-40

if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 ddp-gpt2-2025-08-01-13-22-40"
    exit 1
fi

EXPERIMENT_NAME=$1

echo " Comparing pretrained vs instruction-tuned model: $EXPERIMENT_NAME"
echo ""

torchrun --nproc_per_node 1  -m evaluation.compare_models \
    -e "$EXPERIMENT_NAME" \
    -d "tatsu-lab/alpaca" \
    --model_name "openai-community/gpt2" \
    --instruction_col_name "instruction" \
    --response_col_name "output" \
    --test_split 0.2
