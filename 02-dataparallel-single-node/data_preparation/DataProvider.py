from typing import Dict, List, Tuple
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_preparation.PreProcessor import PreProcessor

class DataProvider:
    def __init__(self, args, config):
        self.__preprocessor = PreProcessor(args, config)
        self.__build_loaders(args)
        self.log_data_sanity_check()

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.__train_loader, self.__test_loader
    
    def log_data_sanity_check(self):
        log = "\n============FINISHED PREPROCESSING DATA===============\n"
        log += "\n============PRINTING SAMPLES===============\n"
        for batch in self.__train_loader:
            log += self.__get_instance_text(batch)
            log += "\n============FINISHED PRINTING SAMPLES===============\n"
            logging.info(log)
            return

    def __build_loaders(self, args: Dict):
        assert args.test_split <= 1.0 and args.test_split >= 0
        data = self.__preprocessor.get_preprocessed_data()
        split = data.train_test_split(test_size=args.test_split)

        self.__train_loader = DataLoader(
            split["train"],
            batch_size=args.batch_size,
            sampler=DistributedSampler(split["train"], shuffle=True, drop_last=True),
            collate_fn=self.__custom_collate_function
        )

        self.__test_loader = DataLoader(
            split["test"],
            batch_size=args.batch_size,
            sampler=DistributedSampler(split["test"], shuffle=True, drop_last=True),
            collate_fn=self.__custom_collate_function
        )

    def __custom_collate_function(self, batch: List[Dict], 
            shift_labels=False, 
            mask_instruction=False) -> Tuple[torch.Tensor, torch.Tensor]:
        max_batch_size = max(len(i["input_ids"]) for i in batch)
        pad_token_id = self.__preprocessor.get_pad_token_id()
        input_ids_list, label_ids_list = [], []

        for item in batch:
            input_ids = item["input_ids"]
            padded_input = input_ids + [pad_token_id] * (max_batch_size - len(input_ids))
            input_ids = torch.tensor(padded_input[:-1]) if shift_labels else torch.tensor(padded_input) #truncate last
            label_ids = torch.tensor(padded_input[1:]) if shift_labels else torch.tensor(padded_input)  #shift right

            if mask_instruction:
                response_start_idx = 0
                full_text = self.__preprocessor.decode(input_ids)
                response_text = "###Response:\n"
                instruction_end = full_text.find(response_text)
                if instruction_end != -1:
                    instruction_end_index = instruction_end + len(response_text)
                    instruction_part = full_text[:instruction_end_index]
                    instruction_tokens = self.__preprocessor.encode(instruction_part)
                    response_start_idx = len(instruction_tokens)

                if response_start_idx > 0:
                    label_ids[:response_start_idx] = self.__preprocessor.get_ignore_token_id()

            #mask padding
            mask = (label_ids == pad_token_id)
            if torch.nonzero(mask).numel() > 1:
                label_ids[mask] = self.__preprocessor.get_ignore_token_id()
            
            input_ids_list.append(input_ids)
            label_ids_list.append(label_ids)

        return torch.stack(input_ids_list), torch.stack(label_ids_list) #stacks along first dim.


    def __get_instance_text(self, batch: Tuple[torch.Tensor]):
        log = "\n============Printing 1 Decoded Training Instance=============\n"
        input_batch_text = self.__decode_inputs(batch)
        label_text_batch = self.__decode_labels(batch)
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
        
        log += "\n============Printing 1 Tokenized Training Instance=============\n"
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

    def __decode_inputs(self, batch: Tuple[torch.Tensor]):
        return self.__preprocessor.batch_decode(batch[0])

    def __decode_labels(self, batch: Tuple[torch.Tensor]):
        # Ignore token id cannot be decoded. Needs to be replaced for decoding and printing.
        target = batch[1].clone()
        target[target == self.__preprocessor.get_ignore_token_id()] = self.__preprocessor.get_pad_token_id()
        return self.__preprocessor.batch_decode(target)
        

