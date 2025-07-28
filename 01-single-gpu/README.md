# Instruction Tuning of GPT2 124M on a single GPU

## Folder Structure
This subdirectory trains GPT2 with 124M parameters on the 52k Alpaca instruction tuning dataset.
The structure of the directory is the following:
1. `data_preparation`: includes classes that encapsulate preparing training and validation data loaders for instruction finetuning.
2. `debug`: includes scripts to see a training batch instance for debugging.
3. `helper`: contains various helpers for logging, elapsed time measurements and other utils.
4. `instruction_tuning`: contains the file that executes instruction training.
5. `evaluation`: contains a script for qualitatively evaluating the effect of instruction tuning.
6. `run_scripts`: contains shell scripts that execute evaluation or training scripts with CLI arguments.

## Usage
Remain in the source directory and execute:
```
./run_scripts/run_train_tmux.sh
```
This will execute a run on you available NVIDIA GPU in a tmux session from which you can detach.
**Note:**
It is assumed that you logged into you weights and biases account before executing the scripts. 
If not done, do so via `wandb login`

## Example Result
To compare the performance of the base model with your instruction tuned version say `gpt2-specialPAD-alpaca-single-gpu-2025-07-28T10-57-47` run:
```
./run_scripts/run_compare.sh gpt2-specialPAD-alpaca-single-gpu-2025-07-28T10-57-47
```
You can see an excerpt of a comparison here:
```
================================================================================
TEST 3:
================================================================================
2025-07-28 17:23:23,050 - INFO -
-------------------- PRETRAINED MODEL --------------------
Input: Identify the key characters of the novel 'The Hobbit'

2025-07-28 17:23:23,113 - INFO -Response: , "Hello world!" , "" ; print (response); }
2025-07-28 17:23:23,113 - INFO -
-------------------- INSTRUCTION-TUNED MODEL --------------------
Input: Identify the key characters of the novel 'The Hobbit'

2025-07-28 17:23:23,263 - INFO -Response: The key characters in The Hobbits are Bilbo Baggins, Gandalf and Thorin Oakenshield; they all have their own unique personalities and motivations.
```
As can be seen, the instruction tuned model significantly extends the instruction following capability of the basemodel.
