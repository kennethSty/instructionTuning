#!/bin/bash

# Script to run training in a new attached tmux session
# Usage: ./run_train_tmux.sh

# Generate a unique session name with timestamp (e.g., training-20250728-123456)
SESSION_NAME="training-$(date +%Y%m%d-%H%M%S)"

# Inform the user about the session details
echo "Starting training in tmux session: $SESSION_NAME"
echo "You can detach from the session with Ctrl+B, then D"
echo "To reattach later, use: tmux attach-session -t $SESSION_NAME"
echo "To list all tmux sessions, use: tmux list-sessions"
echo ""

# Start a new tmux session (attached by default)
# -s "$SESSION_NAME": sets the session name
# -c "$(pwd)": starts the tmux session in the current working directory
# bash -c "...": runs the entire block of training commands inside a bash shell
tmux new-session -s "$SESSION_NAME" -c "$(pwd)" bash -c "
    # Run your training script as a Python module
    # Make sure 'train_llm.py' is importable as a module (i.e., no '.py' in -m)
    python -m instruction_tuning.train_llm \
        -e \"gpt2-specialPAD-alpaca-single-gpu-\$(date +%Y-%m-%dT%H-%M-%S)\" \
        -d \"tatsu-lab/alpaca\" \
        -m \"openai-community/gpt2\" \
        --num_epochs 20 \
        --instruction_col_name \"instruction\" \
        --response_col_name \"output\" \
        --test_split 0.05 \
        --batch_size 16

    # Print a confirmation message with timestamp after training finishes
    echo ''
    echo 'Training completed at: \$(date)'

    # Pause and wait for any key press before ending the session
    # This prevents the session from closing immediately so you can review output
    echo 'Press any key to exit this tmux session...'
    read -n 1
"

