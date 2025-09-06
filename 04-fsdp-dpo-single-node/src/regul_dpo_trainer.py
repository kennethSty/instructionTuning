# --- imports & env ---
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Any, Union, Literal, Callable

from transformers import (
    DataCollator,
    PreTrainedModel,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from datasets import IterableDataset, Dataset, load_dataset
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import compute_accuracy, flush_left, flush_right, selective_log_softmax

import wandb

# --- Custom Trainer with length regularization term ---
class RegulDPOTrainer(DPOTrainer):
    def __init__(
        self,
        length_alpha: float,  # length regularization strength
        model: Union[str, nn.Module, PreTrainedModel],
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        self.length_alpha = length_alpha  # keep a clear name; self.beta comes from args.beta
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

    def compute_loss(  # correct HF hook name
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return loss, metrics
        return loss

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:

        model_output = self.concatenated_forward(model, batch)

        # reference policy logps
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rej_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rej_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss_regul(
            chosen_logps=model_output["chosen_logps"],
            rejected_logps=model_output["rejected_logps"],
            ref_chosen_logps=ref_chosen_logps,
            ref_rej_logps=ref_rej_logps,
            model_output=model_output,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics

    def dpo_loss_regul(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rej_logps: torch.FloatTensor,
        model_output: dict[str, torch.FloatTensor] | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        device = self.accelerator.device

        log_ratios = (chosen_logps - rejected_logps).to(device)
        ref_log_ratios = (ref_chosen_logps - ref_rej_logps).to(device)
        log_ratio_diff = log_ratios - ref_log_ratios

        # length regularization on completion lengths (chosen vs rejected)
        if model_output["chosen_lengths"] is None or  model_output["rejected_logps"] is None:
            raise ValueError("chosen_lengths and rejected_lengths missing. Set alpha to get them")

        chosen_lengths = model_output["chosen_lengths"].to(device)
        rejected_lengths = model_output["rejected_lengths"].to(device)

        log_ratio_diff_regul = self.beta * log_ratio_diff + self.length_alpha * (chosen_lengths - rejected_lengths)

        losses = -F.logsigmoid(log_ratio_diff_regul)
        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rej_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        
        # Get number of different promopts in batch before expanding 
        num_examples = batch["prompt_input_ids"].shape[0]
        
        # Expands batch by factor of 2. Having one instance for prompt + rejected answer
        # and one for prompt + chosen answer
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {"use_cache": False, "output_hidden_states": True}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            # Encoder Decoder Architecture takes in only prompt at Encoder
            # Then compute loss based on output sequence wrt to labels 
            labels = completion_input_ids.clone()
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Decoder-only architecture takes in prompt + completion as input and 
            # Internally shifts tokens by 1 for labels 
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )
            # Handles too long input by flushing left or right 
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush-left gets rid of left padding s.t. all seqs start at 0
                    # Then truncate end -> ensures prompt tokens all persist  
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush-right gets rid of right padding until the longest seq.
                    # Then truncate beginning -> ensures completion tokens all persist
                    # Preferred variant for DPO as we are interested in probabs of completion
                    attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError("Unknown truncation mode")
            else:
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            if self.use_logits_to_keep:
                # compute how many of output logits are completion logits
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep

            if self.padding_free:
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute log softmax ouput for each position that is not masked
        labels = labels.clone()
        # Set label = 0 for all tokens that should not be in loss -> i.e. prmpts
        labels[~loss_mask] = 0 
        # Computing log probab of label tokens via log softmax 
        # Instead of computing them for each token in vocab do so only for labels
        per_token_logps = selective_log_softmax(logits, labels)  
        # set log probabs for all non completion tokens to 0
        per_token_logps[~loss_mask] = 0
        # Shifts logits by one as labels are not yet shifted when cloned
        # Ensures logp at position i corresponds to prob of token i+1
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        # Total log probability of completion sequence 
        # per_token_logps has shape [batch_size, seq_len + 1] due to torch_roll.
        # Cut first token out for this reason 
        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)

        # Compute token lengths of rejected and chosen answers for regularizer
        # By concatenation last num_examples are instances with chosen answers
        completion_lengths = loss_mask.sum(dim=1) # completion_length shape: (batch_size)
        output["chosen_lengths"] = completion_lengths[:num_examples] 
        output["rejected_lengths"] = completion_lengths[num_examples:]

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if self.padding_free:
            split_idx = (model_kwargs["position_ids"] == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output


