import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from attention import Transformer, CharTokenizer, TransformerGenerator

def test_transformer():
    """Test full seq2seq transformer."""
    print("\n=== Test: Transformer Forward Pass ===")
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
    print("Transformer test passed!")


def test_generation_dummy():
    """Test text generation (dummy tokenizer)."""
    print("\n=== Test: Generation (Dummy) ===")
    batch_size, seq_len, vocab_size, d_model = 2, 32, 100, 256
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model, num_layers=2, num_heads=8)
    
    # Source
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    
    # Generator (no tokenizer)
    generator = TransformerGenerator(model, tokenizer=None, device='cpu')
    
    # Generate
    generated = generator.generate_greedy(src, max_len=20, start_token=1, end_token=2)
    
    print(f"Source shape: {src.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= 21
    print("Generation test passed!")


def test_generation_with_tokenizer():
    """Test text generation with CharTokenizer."""
    print("\n=== Test: Generation with CharTokenizer ===")
    
    # Create tokenizer
    vocab_text = "abcdefghijklmnopqrstuvwxyz "
    tokenizer = CharTokenizer(vocab_text)
    vocab_size = tokenizer.vocab_size
    
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"START_TOKEN: {tokenizer.START_TOKEN}")
    print(f"END_TOKEN: {tokenizer.END_TOKEN}")
    
    # Create model with correct vocab size
    d_model = 128
    seq_len = 10
    batch_size = 1
    
    model = Transformer(vocab_size, vocab_size, d_model, num_layers=2, num_heads=8)
    
    # Create generator
    generator = TransformerGenerator(model, tokenizer=tokenizer, device='cpu')
    
    # Create dummy source
    src = torch.randint(0, len(vocab_text), (batch_size, seq_len))
    print("source: ", src)
    
    # Generate
    generated = generator.generate_greedy(
        src, 
        max_len=20, 
        start_token=tokenizer.START_TOKEN, 
        end_token=tokenizer.END_TOKEN
    )
    
    print(f"Source shape: {src.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    # Decode
    decoded_text = tokenizer.decode(generated[0])
    print(f"Decoded text: '{decoded_text}'")
    
    assert generated.shape[0] == batch_size
    print("Generation with tokenizer test passed!")
    print("decoded text: ", decoded_text)


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running Transformer Tests")
    print("=" * 50)
    
    test_transformer()
    test_generation_dummy()
    test_generation_with_tokenizer()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()