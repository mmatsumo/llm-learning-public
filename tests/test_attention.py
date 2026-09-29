import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from attention import Transformer

def test_transformer():
    """Test full seq2seq transformer."""
    batch_size, seq_len, vocab_size, d_model = 4, 64, 1000, 512
    
    # Source (encoder input)
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Target (decoder input)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model, num_layers=2, num_heads=8)
    
    # Forward pass
    logits = model(src, tgt)
    
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: {(batch_size, seq_len, vocab_size)}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size), "Shape mismatch!"
    print("All tests passed!")

if __name__ == "__main__":
    test_transformer()