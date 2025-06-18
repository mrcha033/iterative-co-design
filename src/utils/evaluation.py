import torch
from tqdm import tqdm
import math


def calculate_perplexity(model, tokenizer, data_loader):
    """
    Calculates the perplexity of a language model on a given dataset.

    Note: This is a simplified implementation. A real-world scenario might
    require more sophisticated handling of sliding windows or tokenization.
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

            total_loss += loss.item() * inputs["input_ids"].size(0)
            total_tokens += inputs["input_ids"].size(0)

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
