#!/bin/bash

python -m instruction_tuning.train_llm \
    -e "gpt2-124M-single-gpu-$(date +%Y-%m-%dT%H-%M-%S)" \
    -d "tatsu-lab/alpaca" \
    -m "openai-community/gpt2" \
    --num_epochs 16 \
    --instruction_col_name "instruction" \
    --response_col_name "output"\
    --test_split 0.2 \
    --batch_size 16
   
        
