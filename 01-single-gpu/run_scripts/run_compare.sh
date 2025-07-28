#!/bin/bash

# Usage: ./run_scripts/run_compare.sh experiment_name
# Example: ./run_scripts/run_compare.sh gpt2-alpaca-single-gpu-2025-07-26T11-11-34

if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 gpt2-alpaca-single-gpu-2025-07-26T11-11-34"
    exit 1
fi

EXPERIMENT_NAME=$1

echo " Comparing pretrained vs instruction-tuned model: $EXPERIMENT_NAME"
echo ""

python -m evaluation.compare_models \
    -e "$EXPERIMENT_NAME" \
    -d "tatsu-lab/alpaca" \
    -m "openai-community/gpt2" \
    --instruction_col_name "instruction" \
    --response_col_name "output" \
    --test_split 0.2
