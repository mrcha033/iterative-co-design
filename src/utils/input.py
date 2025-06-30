"""
Utilities for creating safe and robust model inputs.
"""
import torch

def make_dummy_input(model, tokenizer, device, seq_len=512):
    """
    Creates a safe dummy input tensor for profiling or testing.

    - Prioritizes using the tokenizer's pad_token_id for valid, non-disruptive input.
    - Falls back to a random integer tensor if a pad token is not available.
    - Safely gets vocab_size from the model's config, not Hydra's.
    """
    # Respect the model's maximum sequence length
    seq_len = min(seq_len, getattr(model.config, "max_position_embeddings", seq_len))

    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        dummy_ids = torch.full((1, seq_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    else:
        # Fallback: randint but bounded by the model's actual vocab_size
        vocab_size = getattr(model.config, "vocab_size", 50257) # 50257 is a common default
        dummy_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long, device=device)
    
    return {"input_ids": dummy_ids, "attention_mask": torch.ones_like(dummy_ids)} 