#!/bin/bash

SESSION_NAME="training-$(date +%Y-%m-%D-%H-%M-%S)"
echo "Run training script with $SESSION_NAME"

tmux new-session -s "$SESSION_NAME" -c "$(pwd)" bash -c "

	export TORCHELASTIC_ERROR_FILE=../error.json
	export OMP_NUM_THREADS=8
	
	torchrun --standalone \
		--nproc-per-node=4 \
		-m instruction_tuning.train_llm \
                -e ddp-gpt2-$(date +%Y-%m-%d-%H-%M-%S) \
		--model_name openai-community/gpt2 \
		--dataset_name tatsu-lab/alpaca \
		--instruction_col_name instruction \
		--response_col_name output \
		--test_split 0.1		
	echo ''
    	echo \"Training completed at: \$(date)\"
    	echo Press any key to exit the temux session
   	read -n 1
"

