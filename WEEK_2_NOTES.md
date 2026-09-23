# Week 2: Multi-Head Attention Implementation

## Day 1 (Today)

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