#!/usr/bin/env python3
"""Debug permutation to identify why perplexity explodes"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.models.wrapper import ModelWrapper
from src.utils.evaluation import calculate_task_metric

def test_identity_permutation():
    """Test if identity permutation preserves model functionality"""
    print("Loading Mamba-370M model...")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model hidden_size: {model.config.hidden_size}")
    print(f"Model vocab_size: {model.config.vocab_size}")
    
    # Load small dataset for testing
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    val_dataset = dataset["validation"].select(range(10))  # Very small sample
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text"])
    data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1)
    
    if torch.cuda.is_available():
        model.cuda()
    
    # Test 1: Original model perplexity
    print("\n=== Testing original model ===")
    orig_metric = calculate_task_metric(model, tokenizer, data_loader, "language_modeling")
    print(f"Original perplexity: {orig_metric['perplexity']:.2f}")
    
    # Test 2: Wrapped model (no permutation)
    print("\n=== Testing wrapped model (no permutation) ===")
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()
    
    wrapped_metric = calculate_task_metric(wrapped_model, tokenizer, data_loader, "language_modeling")
    print(f"Wrapped perplexity: {wrapped_metric['perplexity']:.2f}")
    
    # Test 3: Identity permutation
    print("\n=== Testing identity permutation ===")
    identity_perm = list(range(model.config.hidden_size))
    print(f"Applying identity permutation of length {len(identity_perm)}")
    
    try:
        wrapped_model.permute_model_weights(identity_perm)
        identity_metric = calculate_task_metric(wrapped_model, tokenizer, data_loader, "language_modeling")
        print(f"Identity permutation perplexity: {identity_metric['perplexity']:.2f}")
    except Exception as e:
        print(f"Error during identity permutation: {e}")
        return
    
    # Test 4: Check layer shapes
    print("\n=== Layer shape analysis ===")
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            if isinstance(module, torch.nn.Linear):
                print(f"Linear {name}: {module.weight.shape}")
            elif isinstance(module, torch.nn.Embedding):
                print(f"Embedding {name}: {module.weight.shape}")
            elif isinstance(module, torch.nn.LayerNorm):
                print(f"LayerNorm {name}: {module.weight.shape}")

if __name__ == "__main__":
    test_identity_permutation() 