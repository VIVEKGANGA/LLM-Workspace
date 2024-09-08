import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        
        assert d_out % num_heads == 0, 'num_heads should be a divisor of d_out'
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.drop_out = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        query = self.W_query(x) #Shape is b, num_tokens, d_out
        key = self.W_key(x)
        value = self.W_value(x)
        
        query = query.view(b, num_tokens, self.num_heads, self.head_dim) #Shape is b, num_tokens, num_heads, head_dim)
        key = key.view(b, num_tokens, self.num_heads, self.head_dim) #Shape is b, num_tokens, num_heads, head_dim)
        value = value.view(b, num_tokens, self.num_heads, self.head_dim) #Shape is b, num_tokens, num_heads, head_dim)
        
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)
        
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attention_score = query @ key.transpose(2,3)
        
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        #use the maske to fill attn_score
        attention_score.masked_fill_(mask_bool, -torch.inf)
        
        attention_weight = torch.softmax(attention_score/ key.shape[-1]**0.5, dim=-1)
        attention_weight = self.drop_out(attention_weight) #shape (b, num_heads, num_tokens, head_dim)
        
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attention_weight @ value).transpose(1,2)
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
        
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/(torch.sqrt(var + self.eps))
        return self.scale * norm_x + self.shift
        
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x,3))
            ))
        
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
            )
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
