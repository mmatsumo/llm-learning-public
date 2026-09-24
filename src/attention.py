## attention function ... softmax Q K^T V b
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
torch.manual_seed(404)

# e.g.
seq_len = 64 #number of input tokens
d_model = 512 #model dimension
batch_size, seq_len, d_model = 4, 64, 512
Q = torch.randn(batch_size, seq_len,d_model) #shape
K = torch.randn(batch_size, seq_len, d_model) #shape
V = torch.randn(batch_size, seq_len, d_model) #shape


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
        d_k = Q.shape[-1] #seq_len
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "check, d_model should be divisible by num_heads, mod == 0"

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads  # dimension per head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_model,dropout=dropout)
        
                
    def forward(self, Q,K,V, mask=None):
        """
        Args:
            Q: [batch_size, seq_len, d_model]
            K: [batch_size, seq_len, d_model]
            V: [batch_size, seq_len, d_model]
            mask: Optional causal mask
        
        Internal:
            linear projections
            Q_w: [batch_size, seq_len, d_model]
            K_w:
            V_w:
            
        Returns MultiHeadAttention:
            output: [batch_size, seq_len, d_model]
        """
        
        batch_size = Q.shape[0]
        seq_len = Q.shape[-2]

        # Linear projection before scaled dot product
       
        K_w = (self.W_k(K)) # shape (batch_size, seq_len, d_model)
        Q_w = (self.W_q(Q))
        V_w = (self.W_v(V))
        print("Q_w projection", Q_w.shape)

        # Reshape for the num_heads
        K_head = K_w.reshape(batch_size,-1,self.num_heads,self.d_k)
        K_head = K_head.transpose(1,2) # transpose to [batch_size, num_heads, seq_len, d_k] 

        Q_head = Q_w.reshape(batch_size,-1,self.num_heads,self.d_k)
        Q_head = Q_head.transpose(1,2) # transpose to [batch_size, num_heads, seq_len, d_k]
        print("Q_head shape after transpose:", Q_head.shape)

        V_head = V_w.reshape(batch_size,-1,self.num_heads,self.d_k)
        V_head = V_head.transpose(1,2) # transpose to [batch_size, num_heads, seq_len, d_k]
        
        # scaled dot product with each of the num_heads
        att_output, att_wei = self.attention(Q_head,K_head,V_head,mask) # [batch_size, num_heads, seq_len, d_k]
        print("att_output ", att_output.shape)
        att_output = att_output.transpose(1, 2).contiguous() # [batch_size, seq_len, num_heads, d_k]
        print("att_output after transpose ", att_output.shape)

        concat_head = att_output.view(batch_size, seq_len, self.d_model) # [batch_size, seq_len, d_model]
        print("concat_head ", concat_head.shape)
        output = self.W_o(concat_head)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model=d_model

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            x + positional_encoding: (batch_size, seq_len, d_model)
        """
        seq_len = Q.shape[-2]
        pos_encoding = torch.zeros(seq_len,self.d_model)
        

        for pos in range(seq_len):
            for i in range(self.d_model //2):
                pos_encoding[pos,2*i] = math.sin(pos/(10000**(2*i/self.d_model)) )
                pos_encoding[pos,2*i+1] = math.cos(pos/(10000**(2*i/self.d_model)) )

        pos_encoding = pos_encoding.unsqueeze(0)  # (1, seq_len, d_model)
        # adding pos_encoding to input x

        return pos_encoding + x

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        # If d_ff not provided, use d_model * 4
        if d_ff is None:
            d_ff = d_model * 4

        self.fc1 = nn.Linear(d_model, d_ff)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # TODO: Apply linear → ReLU → linear → dropout
        x = self.fc1(x) # x expanded
        x= self.ReLU( x) # x hidden
        x = self.fc2(x)
        output = self.dropout(x)

        return output

"""
attention= ScaledDotProductAttention(d_model)

mask = torch.tril(torch.ones(seq_len, seq_len))

att, weights= attention(Q,K,V,mask)
print("Attention ", att.shape)
#print("Attention ", att)

print("Weights ", weights.shape) 

multi_attention = MultiHeadAttention(d_model) 
mask = torch.tril(torch.ones(seq_len, seq_len))

#[1, 1, seq_len, seq_len] for broadcasting
mask = mask.unsqueeze(0).unsqueeze(0)  
output = multi_attention(Q,K,V, mask)



ff_test = FeedForward(d_model)
Q_o = ff_test(Q)
print(Q_o.shape)

"""