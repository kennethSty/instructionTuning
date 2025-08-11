# Instruction Tuning of GPT2 124M on a several GPUs of one node

## Folder Structure
This subdirectory uses the FSDP strategy to train GPT2 with 124M parameters on the 52k Alpaca instruction tuning dataset across the GPUs of a given node. 
The structure of the directory is the following:
1. `data_preparation`: includes classes that encapsulate preparing training and validation data loaders for instruction finetuning. The key difference to the `data_preparation` in `01-single-gpu` is the refactoring of the `DataProvider` class into the `DistributedDataProvider` class, which uses torch's `DistributedSampler` to implement the data parallel approach. In contrast
2. `debug`: includes scripts to see a training batch instance for debugging. 
3. `helper`: contains various helpers for logging, elapsed time measurements and other utils. Key difference to `02-dataparallel-single-node` is the inclusion of a custom generate function as no such function exists for distributed FSDP models. Additionally, it includes scripts that facilitate the loading, sharding and checkpointing of the used model.
4. `instruction_tuning`: contains the file that executes FSDP instruction training. 
5. `evaluation`: contains a script for qualitatively evaluating the effect of instruction tuning.
6. `run_scripts`: contains shell scripts that execute evaluation or training scripts with CLI arguments.

## Usage
Before executing the scripts, modify their executable permissions. 
```
chmod +x ./run_scripts/run_train_tmux.sh
chmod +x ./run_scripts/run_compare.sh
```

Remain in the source directory and execute:
```
./run_scripts/run_train_tmux.sh
```
This will execute a run on your available NVIDIA GPU in a tmux session from which you can detach.
**Note:**
It is assumed that you logged into you weights and biases account before executing the scripts. 
If not done, do so via `wandb log

## Example
Below is an example showing the response of the baseline model to its instruction fine-tuned version. 
```
TEST 4:
================================================================================
-------------------- BASE MODEL --------------------
[rank=2] [2025-08-11 17:07:50,653] INFO: ###Response by Base Model: 
Write a function which reverses the order of words in a sentence.

###Response:

Write a function which reverses the order of words in a sentence.

###Response:

Write a function which reverses the order of words in a sentence.

###Response:

Write a function which reverses the order of words in a sentence.

###Response:

Write a function which reverses the order of words in a sentence.

-------------------- INSTRUCTION-TUNED MODEL --------------------
[rank=2] [2025-08-11 17:08:00,379] INFO: ###Response by Finetuned Model: def reverse_words(sentence):

```
