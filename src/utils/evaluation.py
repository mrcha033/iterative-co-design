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

logger = logging.getLogger(__name__)


def calculate_task_metric(model, data_loader, metric: str):
    """
    Calculates the specified evaluation metric.

    Args:
        model: The model to evaluate
        data_loader: DataLoader with the evaluation data
        metric: The name of the metric to calculate ("perplexity" or "accuracy")

    Returns:
        dict: A dictionary with the metric name and value
    """
    if metric == "perplexity":
        value = calculate_perplexity(model, data_loader)
        return {"perplexity": value}
    elif metric == "accuracy":
        value = calculate_accuracy(model, data_loader)
        return {"accuracy": value}
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


def calculate_perplexity(model, data_loader, fp16: bool = True):
    """
    Calculates the perplexity of a language model on a given dataset with
    correct handling for padding and device placement.
    """
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    # Use the model's pad_token_id, or eos_token_id as a fallback
    pad_token_id = getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", None)

    with torch.cuda.amp.autocast(enabled=(fp16 and device.type == 'cuda')), torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # If the dataloader provides labels (like our SlidingWindowDataset), use them.
            # Otherwise, create them from input_ids for standard Causal LM evaluation.
            if 'labels' in batch:
                outputs = model(**batch)
                labels = batch['labels']
            else:
                labels = batch["input_ids"].clone()
                if pad_token_id is not None:
                    labels[labels == pad_token_id] = -100 # HF default ignore_index
                outputs = model(**batch, labels=labels)
            
            # Loss is already averaged, so we multiply by the number of non-padded tokens
            nll = outputs.loss * labels.ne(-100).sum()
            total_nll += nll.item()
            total_tokens += labels.ne(-100).sum().item()

    if total_tokens == 0:
        logger.warning("No tokens were processed, cannot calculate perplexity.")
        return float('inf')

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity


def calculate_accuracy(model, data_loader):
    """
    Calculates the accuracy of a classification model.
    """
    model.eval()
    device = next(model.parameters()).device

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Accuracy"):
            # Move all tensors to the model's device
            inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            labels = inputs.pop("labels", None) # Pop labels to avoid passing them as model inputs
            
            if labels is None:
                continue

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    if total_predictions == 0:
        return 0.0

    return correct_predictions / total_predictions
