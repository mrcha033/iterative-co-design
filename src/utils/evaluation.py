"""
Evaluation utilities for measuring model performance.

This module provides functions for calculating task-specific metrics including
perplexity for language modeling tasks and accuracy for sequence classification tasks.
Properly handles variable sequence lengths and padding tokens.
"""

import torch
from tqdm import tqdm
import math


def calculate_task_metric(model, tokenizer, data_loader, task_type):
    """
    Calculates the appropriate evaluation metric based on the task type.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        data_loader: DataLoader with the evaluation data
        task_type: Either "language_modeling" or "sequence_classification"

    Returns:
        dict: A dictionary with the metric name and value
    """
    if task_type == "language_modeling":
        perplexity = calculate_perplexity(model, tokenizer, data_loader)
        return {"perplexity": perplexity}
    elif task_type == "sequence_classification":
        accuracy = calculate_accuracy(model, data_loader)
        return {"accuracy": accuracy}
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def calculate_perplexity(model, tokenizer, data_loader):
    """
    Calculates the perplexity of a language model on a given dataset.

    Properly accounts for variable sequence lengths by counting actual tokens
    using the attention mask, rather than assuming fixed-length sequences.
    """
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Perplexity"):
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Count actual tokens using attention mask (excludes padding tokens)
            # The attention mask has 1s for real tokens and 0s for padding
            batch_tokens = inputs["attention_mask"].sum().item()

            # Loss is already averaged over the sequence length and batch size by transformers
            # We need to denormalize it to get the total loss for this batch
            batch_loss = loss.item() * batch_tokens

            total_loss += batch_loss
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def calculate_accuracy(model, data_loader):
    """
    Calculates the accuracy of a classification model.
    """
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Accuracy"):
            # Assuming batch is a dict with 'input_ids', 'attention_mask', 'labels'
            inputs = (
                {k: v.cuda() for k, v in batch.items() if k != "labels"}
                if torch.cuda.is_available()
                else batch
            )
            labels = (
                batch["labels"].cuda() if torch.cuda.is_available() else batch["labels"]
            )

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    return correct_predictions / total_predictions
