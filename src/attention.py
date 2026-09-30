## attention function ... softmax Q K^T V b
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

torch.manual_seed(404)

"""
# e.g.
seq_len = 64 #number of input tokens
d_model = 512 #model dimension
batch_size, seq_len, d_model = 4, 64, 512
Q = torch.randn(batch_size, seq_len,d_model) #shape
K = torch.randn(batch_size, seq_len, d_model) #shape
V = torch.randn(batch_size, seq_len, d_model) #shape
"""
### Functional elements of the transformer: 
# Scaled Dot Product, Multihead Attention, Positional Encoding, Feed Forward NN

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
        
        # Reshape for the num_heads
        K_head = K_w.reshape(batch_size,-1,self.num_heads,self.d_k)
        K_head = K_head.transpose(1,2) # transpose to [batch_size, num_heads, seq_len, d_k] 

        Q_head = Q_w.reshape(batch_size,-1,self.num_heads,self.d_k)
        Q_head = Q_head.transpose(1,2) # transpose to [batch_size, num_heads, seq_len, d_k]
        #print("Q_head shape after transpose:", Q_head.shape)

        V_head = V_w.reshape(batch_size,-1,self.num_heads,self.d_k)
        V_head = V_head.transpose(1,2) # transpose to [batch_size, num_heads, seq_len, d_k]
        
        # scaled dot product with each of the num_heads
        att_output, att_wei = self.attention(Q_head,K_head,V_head,mask) # [batch_size, num_heads, seq_len, d_k]
        #print("att_output ", att_output.shape)
        att_output = att_output.transpose(1, 2).contiguous() # [batch_size, seq_len, num_heads, d_k]
        #print("att_output after transpose ", att_output.shape)

        concat_head = att_output.view(batch_size, seq_len, self.d_model) # [batch_size, seq_len, d_model]
        #print("concat_head ", concat_head.shape)
        output = self.W_o(concat_head)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model=d_model
        """ Alternative with buffer
                # Pre-compute positional encodings
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                -(math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term)[:-1]
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]
            ## in forward
            seq_len = x.shape[1]
            return x + self.pe[:, :seq_len, :]
        """

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            x + positional_encoding: (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        device = x.device  # Get device from input

        pos_encoding = torch.zeros(seq_len,self.d_model, device= device)
        

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

### Encoder

class TransformerBlockUnit(nn.Module):
    def __init__(self, d_model, d_ff: int = None, dropout = 0.1, num_heads = 8):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_heads = num_heads
        self.mh = MultiHeadAttention(self.d_model, self.num_heads, self.dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, self.d_ff, self.dropout)
    
    def forward(self, x, mask = None):
        """ Args:
        x
        - Q query
        - K keys
        - V values
        Output:
        - block unit output x
        """
        
        Q=x
        K=x
        V=x
        
        # Multi Head Attention
        attention_output = self.mh(Q,K,V, mask=None)

        # Add and Norm
        x = self.norm1(attention_output + x)

        # Feedforward
        
        ff_output = self.ff(x)
        x = self.norm2(ff_output + x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers = 6, d_ff=None, num_heads = 8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlockUnit(d_model, d_ff,  dropout, num_heads)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
    
    def forward(self, x, mask = None):
        x=self.embedding(x)
        x = x * math.sqrt(self.d_model)

        x= self.pe(x)
        for layer in self.layers:  
            x = layer(x, mask)
        return x

### Decoder

class TransformerDecoderUnit(nn.Module):
    def __init__(self, d_model, d_ff: int = None, dropout = 0.1, num_heads = 8):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_heads = num_heads

        #Attention layers, add and norm
        self.mh = MultiHeadAttention(self.d_model, self.num_heads, self.dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, self.d_ff, self.dropout)
        
    
    def forward(self, x, encoder_output, mask = None):
        """
        Args:
            x: decoder input [batch_size, seq_len, d_model]
            encoder_output: encoder output [batch_size, seq_len, d_model]
            tgt_mask: causal mask for self-attention [seq_len, seq_len]
        
        Returns:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.shape[1]

        if mask==None:
            # Causal maskl for first self attention step
            mask = torch.tril(torch.ones(seq_len, seq_len))
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len] for broadcasting

        Q=x
        K=x
        V=x


        attention_output = self.mh(Q,K,V, mask)
        x=self.norm1(x+attention_output)
    
        Q= x
        K= encoder_output
        V= encoder_output

        cross_attention_output = self.mh(Q,K,V, mask=None)
        x = self.norm2(x+ cross_attention_output)

        ff_output = self.ff(x)
        x = self.norm3(ff_output+x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers = 6, d_ff=None, num_heads = 8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderUnit(d_model, d_ff,  dropout, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    
    def forward(self, x, encoder_output, mask = None):
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x= self.pe(x) #x is shifted right
        for layer in self.layers:  
            x = layer(x, encoder_output, mask=None)

        x=self.norm(x)

        logits = self.output_proj(x)
        
        return logits
    
class Transformer(nn.Module):
    """Complete sequence-to-sequence transformer, seq2seq"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers=6, 
                 num_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_layers, d_ff, num_heads, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_layers, d_ff, num_heads, dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, mask=src_mask)
        decoder_output = self.decoder(tgt, encoder_output, mask=tgt_mask)
        return decoder_output

class TransformerGenerator(nn.Module):
    #Autoregressive text generation with transformer
    
    def __init__(self, model, tokenizer= None, device='cpu'):
        """
        Args:
            model: Trained Transformer model
            tokenizer: Tokenizer with encode/decode methods
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(device)  # Move model to device
    
    def generate(self, src, max_len=50, start_token=1, end_token=2, temperature=1.0, top_k=None):
        """
        Autoregressive generation: feed one token at a time
        
        Args:
            src: Source sequence [batch_size, seq_len] (encoder_input)
            max_len: Maximum length of generated sequence
            start_token: Token ID to start generation (usually <START>)
            end_token: Token ID that stops generation (usually <END>)
            temperature: Controls randomness (1.0 = normal, >1 = more random, <1 = less random)
            top_k: If set, only sample from top-k tokens (e.g. top_k=50)
        
        Returns:
            generated: Generated token IDs [batch_size, generated_seq_len]
        """
        self.model.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad(): # context-manager and decorator
            # Encode source
            encoder_output = self.model.encoder(src)
            
            # Initialize target with start token
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=self.device)
            
            for step in range(max_len):
                # Create causal mask for current target length
                tgt_len = tgt.shape[1]
                tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.device))
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                
                # Forward pass
                logits = self.model.decoder(tgt, encoder_output, mask=tgt_mask)
                
                # Get next token logits (last position)
                next_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature
                next_logits = next_logits / temperature
                
                # Top-k sampling (optional)
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                    next_logits_filtered = torch.full_like(next_logits, float('-inf'))
                    next_logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
                    next_logits = next_logits_filtered
                
                # Sample from distribution
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                
                # Append to sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if all sequences generated end token
                if (next_token == end_token).all():
                    break
        
        return tgt
    
    def generate_greedy(self, src, max_len=50, start_token=1, end_token=2):
        """
        Greedy decoding: always pick highest probability token.
        (Faster, deterministic, but may not be optimal)
        
        Args:
            src: Source sequence [batch_size, seq_len]
            max_len: Maximum length
            start_token: Start token ID
            end_token: End token ID
        
        Returns:
            generated: Token IDs [batch_size, generated_seq_len]
        """
        self.model.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad():
            encoder_output = self.model.encoder(src)
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=self.device)
            
            for step in range(max_len):
                tgt_len = tgt.shape[1]
                tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.device))
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                
                logits = self.model.decoder(tgt, encoder_output, mask=tgt_mask)
                next_logits = logits[:, -1, :]
                
                # Greedy: argmax
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [batch_size, 1]
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if (next_token == end_token).all():
                    break
        
        return tgt

class CharTokenizer:
    ## Simple character-level tokenizer ##
    
    def __init__(self, text):
        """
        Args:
            text: Raw text to build vocabulary from
        """
        self.chars = sorted(set(text)) #no repetition
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
        # Special tokens
        self.START_TOKEN = self.vocab_size
        self.END_TOKEN = self.vocab_size + 1
        

        # Add special tokens to idx_to_char
        self.idx_to_char[self.START_TOKEN] = '<START>'
        self.idx_to_char[self.END_TOKEN] = '<END>'
        self.vocab_size += 2  # Account for special tokens

        print(f"Tokenizer created: {len(self.chars)} chars + 2 special tokens = {self.vocab_size} vocab")
    
    def encode(self, text):
        """Convert text to token IDs.
        
        Args:
            text: String to encode
        
        Returns:
            List of token IDs
        """
        return [self.char_to_idx[c] for c in text]
    
    def decode(self, token_ids):
        """Convert token IDs back to text.
        
        Args:
            token_ids: List or tensor of token IDs
        
        Returns:
            Decoded string
        """
        # Handle tensor input
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        result = []
        for i in token_ids:
            if i < len(self.chars):  # Regular character
                result.append(self.idx_to_char[i])
        return ''.join(result)
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



x = torch.randn(batch_size, seq_len, d_model)
x_en = torch.randn(batch_size, seq_len, d_model)
decoder = TransformerDecoderUnit(d_model, d_ff=2)

output = decoder(x, x_en)
print("Output shape:", output.shape)  # (4, 64, 512)

"""

"""" # Decoder testing
batch_size, seq_len, d_model, vocab_size = 4, 64, 512, 1000

# Encoder
src = torch.randint(0, vocab_size, (batch_size, seq_len))
encoder = TransformerEncoder(vocab_size, d_model, num_layers=2)
encoder_output = encoder(src)

# Decoder
tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
decoder = TransformerDecoder(vocab_size, d_model, num_layers=2)
logits = decoder(tgt, encoder_output)

print("Encoder output shape:", encoder_output.shape)  # [4, 64, 512]
print("Decoder logits shape:", logits.shape)  # [4, 64, 1000] 
"""