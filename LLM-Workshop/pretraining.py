import tiktoken
import torch
from model_architecture import GPTModel, generate_text
from model_utils import calc_loss_loader, calc_loss_batch, plot_losses, create_dataloader_v1

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(124)

def text_to_token_ids(text, tokenizer):
    token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    return token_ids

def token_ids_to_text(token_ids, tokenizer):
    text = tokenizer.decode(token_ids.squeeze(0).tolist())
    return text

#Testing
input_text = "Atheletes of India are"
tokenizer  = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)
token_ids = generate_text(model=model,
                          idx=text_to_token_ids(input_text, tokenizer),
                          max_new_tokens=10,
                          context_size=GPT_CONFIG_124M["context_length"])

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Data Preparation
with open('boy_knight.txt','r') as file:
    text = file.read()
    
print(text[:100])
print("\n")
print(text[-100:])


total_characters = len(text)
total_tokens = len(tokenizer.encode(text))

print("Characters:", total_characters)
print("Tokens:", total_tokens)


train_ratio = 0.9
split_idx = int(train_ratio * len(text))
train_data = text[:split_idx]
val_data = text[split_idx:]

torch.manual_seed(1234)

train_loader = create_dataloader_v1(train_data, batch_size=4, max_length=256)

first_batch = next(iter(train_loader))  
print(first_batch[0].shape, first_batch[1].shape)

val_loader = create_dataloader_v1(train_data, batch_size=4, max_length=256)
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##Check loss functions with out training
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    with torch.no_grad():
        sample = generate_text(model, 
                               text_to_token_ids(start_context, tokenizer), 
                               50, 
                               GPT_CONFIG_124M["context_length"])
        print(token_ids_to_text(sample, tokenizer))
    model.train()

def train_model(model, train_loader, val_loader, device, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    #Main training loop
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
                if eval_iter is not None and global_step >= eval_iter:
                    break
        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_token_seen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, track_token_seen = train_model(model, train_loader, val_loader, device, optimizer, num_epochs, eval_freq=1000, eval_iter=None, start_context="Atheletes of India are", tokenizer=tokenizer)

torch.save(model.state_dict(), 'gpt_model.pth')


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, track_token_seen, train_losses, val_losses)