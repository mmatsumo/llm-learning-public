## attention function ... softmax Q K^T V b
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
torch.manual_seed(404)

# e.g.
seq_len = 8 #number of tokens
d_model = 100 #model dimension
batch_size, seq_len, d_model = 4, 8, 100
Q = torch.randn((batch_size, seq_len,d_model)) #shape
K = torch.randn(batch_size, seq_len, d_model) #shape
V = torch.randn(batch_size, seq_len, d_model) #shape

print(Q.shape)

#print(K.shape, 'K = ', K[:,0])
#print(V.shape, 'V = ', V[:,0])

# ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, Q,K,V, mask=None):
        """
        Args:
            Q: [batch_size, seq_len, d_model]
            K: [batch_size, seq_len, d_model]
            V: [batch_size, seq_len, d_model]
            mask: Optional causal mask
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # attention = softmax(Q@(K^T)/sqrt(K.shape[1]))@V
        scores = torch.matmul(Q,K.transpose(-2, -1))
        d_k = Q.shape[-1]
        seq_len = Q.shape[-2]
        scores = scores/math.sqrt(d_k)

        # Causal mask optional, past tokens combinations
        if mask is not None:
            scores=scores.masked_fill((mask== 0), float('-inf'))
            #print(scores.sum(dim=-1))  # Should be all 1.0

        att_weights = F.softmax(scores, dim=-1)
       
        
        # Apply dropout
        att_weights = self.dropout(att_weights)

        # V: values matrix
        output= torch.matmul(att_weights,V)
        return output, att_weights
    # MultiHeadAttention

attention= ScaledDotProductAttention(d_model)

mask = torch.tril(torch.ones(seq_len, seq_len))

att, weights= attention(Q,K,V,mask)
print("Attention ", att.shape)
print("Attention ", att)

print("Weights ", weights.shape)