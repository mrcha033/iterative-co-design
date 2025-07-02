"""
Evaluation utilities for measuring model performance.

This module provides functions for calculating task-specific metrics including
perplexity for language modeling tasks and accuracy for sequence classification tasks.
Properly handles variable sequence lengths and padding tokens.
"""

import torch
from tqdm import tqdm
import math
import logging
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools

logger = logging.getLogger(__name__)


def calculate_task_metric(model, data_loader, metric: str, max_steps: Optional[int] = None):
    """
    Calculates the specified evaluation metric in a safe, no-gradient context.

    Args:
        model: The model to evaluate
        data_loader: DataLoader with the evaluation data
        metric: The name of the metric to calculate ("perplexity" or "accuracy")
        max_steps: The maximum number of batches to evaluate.
        
    Returns:
        Dictionary with metric values and evaluation metadata
    """
    model.eval()
    with torch.no_grad():
        if metric == "perplexity":
            value, actual_steps = calculate_perplexity(model, data_loader, max_steps=max_steps)
            logger.info(f"Evaluation completed: {metric}={value:.4f} (using {actual_steps} steps)")
            return {"perplexity": value, "eval_steps": actual_steps, "eval_max_steps": max_steps}
        elif metric == "accuracy":
            value, actual_steps = calculate_accuracy(model, data_loader, max_steps=max_steps)
            logger.info(f"Evaluation completed: {metric}={value:.4f} (using {actual_steps} steps)")
            return {"accuracy": value, "eval_steps": actual_steps, "eval_max_steps": max_steps}
        else:
            raise ValueError(f"Unknown metric type: {metric}")


def mask_labels(input_ids, attention_mask, pad_token_id, ignore_index=-100):
    """
    Creates labels for language modeling, ignoring padded tokens.
    """
    labels = input_ids.clone()
    # In transformers, padded tokens are masked with 0.
    labels[attention_mask == 0] = ignore_index
    return labels


def calculate_perplexity(model, data_loader, fp16: bool = True, max_steps: Optional[int] = None):
    """
    Calculates the perplexity of a language model on a given dataset with
    correct handling for padding and device placement.
    
    Returns:
        Tuple of (perplexity, actual_steps) where actual_steps is the number of batches processed
    """
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    actual_steps = 0

    # Use the model's pad_token_id, or eos_token_id as a fallback
    pad_token_id = getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", None)
    
    # Limit the number of steps for evaluation if specified
    if max_steps is not None:
        data_loader = itertools.islice(data_loader, max_steps)

    with torch.amp.autocast(device_type=device.type, enabled=(fp16 and device.type == 'cuda')), torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Perplexity", total=max_steps):
            batch = {k: v.to(device) for k, v in batch.items()}
            actual_steps += 1
            
            model_inputs = {"input_ids": batch["input_ids"]}
            if "attention_mask" in batch:
                model_inputs["attention_mask"] = batch["attention_mask"]

            # Create labels and ensure padding is properly masked
            if 'labels' in batch:
                labels = batch['labels']
            else:
                labels = batch["input_ids"].clone()
                # Explicitly mask padding tokens in labels to be ignored in loss calculation
                if pad_token_id is not None:
                    labels[labels == pad_token_id] = -100  # HF default ignore_index
                
                # Create an attention mask if not present and we have padding tokens
                if "attention_mask" not in model_inputs and pad_token_id is not None:
                    # 1 for real tokens, 0 for padding
                    attention_mask = (batch["input_ids"] != pad_token_id).long()
                    model_inputs["attention_mask"] = attention_mask
            
            model_inputs["labels"] = labels
            outputs = model(**model_inputs)
            
            # Loss is already averaged, so we multiply by the number of non-padded tokens
            nll = outputs.loss * labels.ne(-100).sum()
            total_nll += nll.item()
            total_tokens += labels.ne(-100).sum().item()

    if total_tokens == 0:
        logger.warning("No tokens were processed, cannot calculate perplexity.")
        return float('inf'), actual_steps

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    
    # Log information about the evaluation
    logger.info(f"Perplexity calculated over {actual_steps} steps ({total_tokens} tokens)")
    
    return perplexity, actual_steps


def calculate_accuracy(model, data_loader, max_steps: Optional[int] = None):
    """
    Calculates the accuracy of a classification model.
    
    Returns:
        Tuple of (accuracy, actual_steps) where actual_steps is the number of batches processed
    """
    model.eval()
    device = next(model.parameters()).device

    correct_predictions = 0
    total_predictions = 0
    actual_steps = 0

    # Limit the number of steps for evaluation if specified
    if max_steps is not None:
        data_loader = itertools.islice(data_loader, max_steps)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Accuracy", total=max_steps):
            # Move all tensors to the model's device
            inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            actual_steps += 1
            
            labels = inputs.pop("labels", None) # Pop labels to avoid passing them as model inputs
            
            if labels is None:
                continue

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    if total_predictions == 0:
        return 0.0, actual_steps

    accuracy = correct_predictions / total_predictions
    
    # Log information about the evaluation
    logger.info(f"Accuracy calculated over {actual_steps} steps ({total_predictions} examples)")
    
    return accuracy, actual_steps
