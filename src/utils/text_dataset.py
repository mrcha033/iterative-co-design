"""
Text dataset utilities for language modeling tasks.
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, List
import mmap


class TextDataset(Dataset):
    """
    Dataset for text data with configurable sequence length.
    Optimized for memory efficiency with memory mapping.
    """
    
    def __init__(
        self,
        file_path: str,
        sequence_length: int = 1024,
        num_samples: Optional[int] = None,
        tokenizer=None
    ):
        """
        Initialize the text dataset.
        
        Args:
            file_path: Path to the text file
            sequence_length: Length of each text sequence
            num_samples: Maximum number of samples to load
            tokenizer: Tokenizer to use (if None, uses character-level)
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        
        # Read and process the text
        self._load_text()
        
        # Create vocabulary if using character-level tokenization
        if self.tokenizer is None:
            self._create_vocab()
    
    def _load_text(self):
        """Load text from file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Remove empty lines and extra whitespace
        lines = [line.strip() for line in self.text.split('\n') if line.strip()]
        self.text = ' '.join(lines)
        
        print(f"Loaded {len(self.text)} characters from {self.file_path}")
    
    def _create_vocab(self):
        """Create character-level vocabulary."""
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)
        
        # Create character to index mapping
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Created vocabulary with {self.vocab_size} unique characters")
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        else:
            # Character-level tokenization
            return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def __len__(self):
        """Get dataset length."""
        if self.num_samples is not None:
            return min(self.num_samples, len(self.text) // self.sequence_length)
        return len(self.text) // self.sequence_length
    
    def __getitem__(self, idx):
        """Get a text sequence."""
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        
        # Extract text sequence
        text_seq = self.text[start_idx:end_idx]
        
        # Tokenize
        tokens = self._tokenize(text_seq)
        
        # Pad if necessary
        if len(tokens) < self.sequence_length:
            tokens.extend([0] * (self.sequence_length - len(tokens)))
        
        # Convert to tensor
        input_ids = torch.tensor(tokens[:self.sequence_length], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
            'text': text_seq
        }