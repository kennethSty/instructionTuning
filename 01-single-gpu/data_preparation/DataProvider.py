from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from transformers import AutoConfig, default_data_collator
import torch

from data_preparation.PreProcessor import PreProcessor
from helper.logger import LOGGER


class DataProvider:
    """
    Encapsulates the logic that provides the trainind and validation data
    in the form of torch.utils.data.DataLoader. 
    This particularly includes a custom collator which formats input and label tensors like this:
        INPUT TOKENS:

        ----------------

        tensor([21106,   318,   281, 12064,   326,  8477,   257,  4876,    13, 19430,
                  257,  2882,   326, 32543,   262,  2581,    13,   198,   198, 21017,
                 6310,  2762,    25,   198, 33234,  1958,   262, 10825,  6241,   287,
                  262, 21247,    13,   198,   198, 21017, 31077,    25,   198, 41338,
                   11,  2695,   434,    11, 10974, ...,  50257, 50257])

         LABEL TOKENS (Unshifted - as expected by huggingface AutoModelForCausalLM):

        ---------------

        tensor([21106,   318,   281, 12064,   326,  8477,   257,  4876,    13, 19430,
                  257,  2882,   326, 32543,   262,  2581,    13,   198,   198, 21017,
                 6310,  2762,    25,   198, 33234,  1958,   262, 10825,  6241,   287,
                  262, 21247,    13,   198,   198, 21017, 31077,    25,   198, 41338,
                   11,  2695,   434,    11, 10974, ..., -100,  -100])

    In case training does not work as expected, consider setting shift_labels=True in collation.
    This is needed when the trained model does not internally shift the labels. 
    """

    def __init__(self, args: Dict, config: AutoConfig):
        self.preprocessor = PreProcessor(args=args, config=config)
        self.train_loader, self.test_loader = self.build_loaders(args)
        self.log_data_sanity_check()

    def get_loaders(self):
        return self.train_loader, self.test_loader

    def build_loaders(self, args: Dict):
        assert args.test_split <= 1.0 and args.test_split > 0
        data = self.preprocessor.get_preprocessed_data()
        split = data.train_test_split(
            test_size=args.test_split
        )
        
        train_loader = DataLoader(
            split["train"],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True, #otherwise last batch too small
            collate_fn=self._custom_collate_function
        ) 

        test_loader = DataLoader(
            split["test"],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False, #in test batch size does not matter
            collate_fn=self._custom_collate_function
        )

        return train_loader, test_loader

    def get_num_train_batches(self):
        return len(self.train_loader)

    def is_pad_added_manually(self):
        return self.preprocessor.is_pad_added_manually

    def get_pad_token_id(self):
        return self.preprocessor.get_pad_token_id()

    def get_eos_token_id(self):
        return self.preprocessor.get_eos_token_id()

    def get_vocab_size(self):
        return len(self.preprocessor.tokenizer)

    def encode(self, text, **kwargs):
        return self.preprocessor.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.preprocessor.decode(token_ids, **kwargs)

    def _custom_collate_function(self, batch: List[Dict], mask_instruction=False, shift_labels=False):
        """
        Collator function that takes in a dictionary of input_id and label_id tokens and formats them for model training.
        Operations on input_ids: 
            - Add pad_token_id to the right, until all input sequences are of equal length
        Operation on labels: 
            - Mask padding tokens with -100 (will be ignored by loss computation).
            - Optionally also mask instruction part.

        Returns: Tuple[torch.Tensor(input_ids), torch.Tensor(label_ids)]
        """

        batch_max_length = max(len(item["input_ids"]) for item in batch)
        pad_token_id = self.preprocessor.get_pad_token_id()
        input_ids_list, label_ids_list = [], []

        for item in batch:
            input_ids = item["input_ids"]
            padded_input = input_ids + [pad_token_id] * (batch_max_length - len(input_ids))
            
            # Note AutoModelForCausalLM mostly shift labels internally -> no shift needed
            # Usual Models loaded from other parts may expect shifted labels
            inputs = torch.tensor(padded_input[:-1]) if shift_labels else torch.tensor(padded_input)
            labels = torch.tensor(padded_input[1:]) if shift_labels else torch.tensor(padded_input) 
            
            if mask_instruction:
                response_start_idx = 0
                full_text = self.preprocessor.tokenizer.decode(input_ids)
                response_start_text = "###Response:\n"
                instruction_end = full_text.find(response_start_text) #returns start index
                if instruction_end != -1:
                    # Mask Instruction +  "###Response"-Tag 
                    instructtion_end_index = instruction_end + len(response_start_text) - 1 #-1 because id starts at 0 
                    instruction_part = full_text[:instruction_end_index]                    
                    instruction_tokens = self.preprocessor.tokenizer.encode(instruction_part)
                    response_start_idx = len(instruction_tokens)
            
                # Mask instruction part - only train on response
                if response_start_idx > 0:
                    labels[:response_start_idx] = self.preprocessor.ignore_token_id
            
            # Mask padding tokens in labels
            mask = labels == pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1: 
                labels[indices] = self.preprocessor.ignore_token_id 

            input_ids_list.append(inputs)
            label_ids_list.append(labels)
            
        return torch.stack(input_ids_list), torch.stack(label_ids_list)
    
    def log_data_sanity_check(self):
        log = "\n============FINISHED PREPROCESSING DATA===============\n"
        log += "\n============PRINTING SAMPLES===============\n"
        for batch in self.train_loader:
            log += self.get_instance_text(batch)
            log += "\n============FINISHED PRINTING SAMPLES===============\n"
            LOGGER.info(log)
            return

    def get_instance_text(self, batch: Tuple[torch.Tensor]):
        log = "\n============Printing 1 Decoded Training Instances=============\n"
        input_batch_text = self._decode_inputs(batch)
        label_text_batch = self._decode_labels(batch)
        for input_text, label_text in zip(input_batch_text, label_text_batch):
            log += "\n********new instance*********\n"
            log += "INPUT TEXT:"
            log += "\n----------------\n"
            log += input_text
            log += "\n LABEL TEXT:"
            log += "\n---------------\n"
            log += label_text
            log += "\n********end instance********\n"
            break #only print first instance for clarity
        
        log += "\n============Printing Tokenized Training Instances=============\n"
        for input_tokens, label_tokens in zip(batch[0], batch[1]):
            log += "\n********new instance*********\n"
            "INPUT TOKENS:"
            log += "\n----------------\n"
            log += str(input_tokens)
            log += "\n LABEL TOKENS:"
            log += "\n---------------\n"
            log += str(label_tokens)
            log += "\n********end instance********\n"
            break
        return log

    def get_first_train_batch_text(self):
        for batch in self.train_loader:
            return self._decode_inputs(batch) 

    def _decode_inputs(self, batch: Tuple[torch.Tensor]):
        return self.preprocessor.tokenizer.batch_decode(batch[0])

    def _decode_labels(self, batch: Tuple[torch.Tensor]):
        # Ignore token id cannot be decoded. Needs to be replaced for decoding and printing.
        target = batch[1].clone()
        target[target == self.preprocessor.ignore_token_id] = self.preprocessor.get_pad_token_id()
        return self.preprocessor.tokenizer.batch_decode(target)
