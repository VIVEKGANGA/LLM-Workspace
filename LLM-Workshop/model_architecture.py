import torch.nn as nn
from model_utils import TransformerBlock, LayerNorm
import torch
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 8,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    
#Testing the working of GPT architecture
tokenizer = tiktoken.get_encoding("gpt2")
"""
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
"""

# Generate Text

def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:,-1,:] 
        probas = torch.softmax(logits, dim=-1) #(batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1) #(batch, n_tokens+1)
    return idx

"""
input_text = "Hello, How are"
encoded = tokenizer.encode(input_text)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

out = generate_text(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=3, 
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

"""