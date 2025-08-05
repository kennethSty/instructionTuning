from transformers import AutoTokenizer, AutoConfig
import datasets
import multiprocessing
from typing import List, Dict
from itertools import chain

class PreProcessor:
    def __init__(self, args: Dict, config: AutoConfig):
        self.args = args
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.set_pad_token()
        self.max_length = self.__get_max_length()
        self.full_text_col_name = "full_text"
        self.label_col_name = "labels"
        self.ignore_token_id = args.ignore_token_id
        self.preprocessed_dataset = self.preprocess(
            datasets.load_dataset(args.dataset_name, trust_remote_code=True)["train"]
        )
    
    def preprocess(self, dataset):
        return dataset\
            .map(
                self._format_full_text,
                batched=True,
                load_from_cache_file=True,
                num_proc=multiprocessing.cpu_count(),
                desc=f"Format Instruction and Response together")\
            .map(
                self._tokenize_batch,
                batched=True,
                load_from_cache_file=True,
                num_proc=multiprocessing.cpu_count(),
                desc=f"Tokenize dataset")\
            .filter(
                lambda x: len(x["input_ids"]) < self.max_length,
                load_from_cache_file=True,
                num_proc=multiprocessing.cpu_count(),
                desc="Filter out instructions that do not fit into model"
            )

    def set_pad_token(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.is_pad_added_manually = True
        print(f"\n Using padtoken id: {self.tokenizer.pad_token}") 

    def get_preprocessed_data(self):
        return self.preprocessed_dataset


    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id

    def get_ignore_token_id(self):
        return self.ignore_token_id
    
    def _format_full_text(self, batch: Dict[str, List]):
        # Format with proper instruction-response structure that matches evaluation
        formatted_texts = []
        for instruction, response in zip(batch[self.args.instruction_col_name],\
                                         batch[self.args.response_col_name]):
            # This format matches what we use during evaluation. 
            # Add EOS token to indicate end of sequence.
            system_text = "Below is an instruction that describes a task. Write a response that completes the request.\n\n"
            instruction_text = f"###Instruction:\n{instruction}\n\n"
            response_text = f"###Response:\n{response}" 
            formatted_text = f"{system_text}{instruction_text}{response_text}{self.tokenizer.eos_token}"
            formatted_texts.append(formatted_text)
        
        return {self.full_text_col_name: formatted_texts}

    def _tokenize_batch(self, batch: Dict[str, List]):
        return self.tokenizer(batch[self.full_text_col_name])
    
    def __get_max_length(self) -> int:
        max_length = self.args.max_length or self.tokenizer.model_max_length
        if max_length > self.config.max_position_embeddings:
            max_length = self.config.max_position_embeddings
        return max_length
