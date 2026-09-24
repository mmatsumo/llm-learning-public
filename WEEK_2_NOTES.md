# Week 2: Multi-Head Attention Implementation

## Day 1

**Accomplished:**
- Debugged git merge conflicts
- Understood dimension reshaping for multi-head attention
- Implemented MultiHeadAttention class with:
  - 4 linear projection layers (W_q, W_k, W_v, W_o)
  - Proper head splitting and concatenation
  - Causal masking support
  - Tested with (batch=4, seq_len=64, d_model=512, num_heads=8)

**Key Insight:**
Multi-head attention = 8 parallel 64-dim attention heads, not one 512-dim head.
seq_len, any input tokesn lenght
Reshaping: (batch, seq_len, 512) → (batch, 8, seq_len, 64) → compute → concatenate

**Code Location:** `src/attention.py` - MultiHeadAttention class

## Day 2 Summary:

**Accomplished:**

Fixed git merge conflicts (README.md sync issues)
Debugged MultiHeadAttention implementation:
Fixed K projection bug (was using self.W_q instead of self.W_k)
Fixed attention call (was passing Q twice instead of Q, K, V)
Fixed mask broadcasting (reshaped to (1, 1, seq_len, seq_len))
Successfully implemented and tested MultiHeadAttention with:
8 parallel attention heads
Proper dimension reshaping: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
Causal masking support
Output projection
Tested with shape: (batch_size=4, seq_len=64, d_model=512, num_heads=8)
Created WEEK_2_NOTES.md file

**Key Insights:**

Q, K, V projections must use different weight matrices (W_q, W_k, W_v)
Each head operates on reduced dimension (d_k = d_model / num_heads = 64)
Heads run in parallel, then concatenate back to d_model
Mask needs to broadcast across batch and head dimensions


## Day 3 (Today)

**Accomplished:**
- Implemented PositionalEncoding (sinusoidal)
- Implemented FeedForward (2-layer MLP)
- Implemented TransformerBlockUnit (attention + FF + norms + residuals)
- Implemented TransformerEncoder (stack of blocks with PE)
- Tested with (batch=4, seq_len=64, d_model=512, num_layers=2)
**Key Insights:**
- Residual connections: x + attention(x), x + ff(x)
- Layer normalization after each sub-layer
- Position encoding added once at input
- All 8 blocks share the same weights (parameterized once in __init__)

