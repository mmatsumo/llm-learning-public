# Transformer Implementation from Scratch

A complete implementation of the **Transformer architecture** (Vaswani et al., 2017) 
built from first principles in PyTorch.

## Architecture

- **Encoder**: 6 layers of self-attention + feed-forward
- **Decoder**: 6 layers of masked self-attention + cross-attention + feed-forward
- **Multi-head attention**: 8 heads, scaled dot-product
- **Positional encoding**: Sinusoidal position embeddings

## Components

- `ScaledDotProductAttention`: Core attention mechanism
- `MultiHeadAttention`: Multiple attention heads in parallel
- `TransformerBlockUnit`: Single encoder block
- `TransformerDecoderUnit`: Single decoder block with cross-attention
- `PositionalEncoding`: Sinusoidal position embeddings

## Usage

```python
from attention import Transformer

# Create model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_layers=6,
    num_heads=8
)

# Forward pass
src = torch.randint(0, 10000, (4, 64))  # [batch, seq_len]
tgt = torch.randint(0, 10000, (4, 64))
logits = model(src, tgt)  # [batch, seq_len, vocab_size]