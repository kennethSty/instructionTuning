from typing import List, Dict
from itertools import chain
from transformers import AutoTokenizer, AutoConfig
import datasets
import logging
import multiprocessing

class PreProcessor:
    """
    A utility class for preprocessing instruction-response datasets for language model fine-tuning.

    This class handles tokenizer setup, special token configuration, dataset formatting, tokenization, 
    and filtering based on the model's context window. Special tokens like [PAD] and EOS are automatically 
    managed and made accessible through getter methods.
    """
    def __init__(self, args: Dict, config: AutoConfig):
        self.__args = args
        self.__config = config
        self.__tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.__ignore_token_id = args.ignore_token_id
        self.__full_text_col_name = "full_text"
        self.__label_col_name = "labels"
        self.__set_pad_token()
        self.__max_length = self.__get_max_length()
        self.preprocessed_dataset = self.__preprocess(
            datasets.load_dataset(args.dataset_name, trust_remote_code=True)["train"]
        )
        self.is_pad_added = False


    def get_preprocessed_data(self):
        return self.preprocessed_dataset

    def encode(self, text, **kwargs):
        return self.__tokenizer(text, **kwargs)

    def batch_encode(self, batch, **kwargs):
        return self.__tokenizer.batch_encode(batch, **kwargs)

    def batch_decode(self, batch, **kwargs):
        return self.__tokenizer.batch_decode(batch, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.__tokenizer.decode(token_ids, **kwargs)

    def get_pad_token_id(self):
        return self.__tokenizer.pad_token_id

    def get_eos_token_id(self):
        return self.__tokenizer.eos_token_id

    def get_ignore_token_id(self):
        return self.__ignore_token_id
    
    def __set_pad_token(self):
        if self.__tokenizer.pad_token_id is None:
            self.__tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.__pad_is_added = True

        logging.info(f"\n Using padtoken: {self.__tokenizer.pad_token}")

    def pad_is_added(self):
        return self.__pad_is_added

    def get_vocab_size(self):
        return len(self.__tokenizer)

    def __preprocess(self, dataset):
        return dataset\
            .map(
                self.__format_full_text,
                batched=True,
                load_from_cache_file=True, #Assumes rank0 loads first so no file corruption
                desc="Format instruction and response together"
            )\
            .map(
                self.__tokenize_batch,
                batched=True,
                load_from_cache_file=True,
                desc="Tokenize formatted text"
            )\
            .filter(
                lambda x: len(x["input_ids"]) < self.__max_length, #keeps tokenized sequ. within context window
                load_from_cache_file=True,
                num_proc=multiprocessing.cpu_count(),
                desc="Filter out instructions that exceed context window of model"
            )

    def __format_full_text(self, batch: Dict[str, List]) -> Dict[str, List]:
        formatted_texts = []
        
        for instruction, response in zip(
                batch[self.__args.instruction_col_name],
                batch[self.__args.response_col_name]
                ):
            system_text = "Below is an instruction that describes a task. Write a response that completes the request. \n\n"
            instruction_text = f"###Instruction:\n{instruction}\n\n"
            response_text = f"###Response:\n{response}"
            full_text = f"{system_text}{instruction_text}{response_text}{self.__tokenizer.eos_token}" #IMPORTANT add EOS 
            formatted_texts.append(full_text)

        return {self.__full_text_col_name: formatted_texts}

    def __tokenize_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        return self.__tokenizer(batch[self.__full_text_col_name])

    def __get_max_length(self) -> int:
        return min(
            self.__args.max_length,
            self.__tokenizer.model_max_length,
            self.__config.max_position_embeddings
            )

