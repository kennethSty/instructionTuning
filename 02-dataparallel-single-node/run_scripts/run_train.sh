#!/bin/bash

# Usage: from src.
# 1. set executable rights: chmod +x ./run_scripts/run_train.sh
# 2. execute: ./run_scripts/run_train.sh

SESSION_NAME="training-$(date +%Y-%m-%D-%H-%M-%S)"
echo "Starting training with $SESSION_NAME"

tmux new-session -s "$SESSION_NAME" -c "$(pwd)" bash -c "
    export TORCHELASTIC_ERROR_FILE=../error.json
    export OMP_NUM_THREADS=4

    torchrun --standalone \
	    --nproc-per-node 4 \
	    -m instruction_tuning.train_llm \
	        -e \"ddp-instructionTune-\$(date +%Y-%m-%D-%H-%M-%s)\" \
                -m openai-community/gpt2 \
		-d tatsu-lab/alpaca \
                --instruction_col_name instruction \
		--response_col_name output \
		--test_split 0.1 
    echo ''
    echo \"Training completed at: \$(date)\"
    echo Press any key to exit the temux session
    read -n 1
"
