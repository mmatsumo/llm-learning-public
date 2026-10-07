import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from attention import Transformer
from prepare_shakespeare import run_preparation
from attention import TransformerGenerator

# data prep
vocab_size, tokenizer, train_loader, val_loader = run_preparation()

# Model
#model = Transformer(vocab_size, vocab_size, d_model=256, num_layers=4, num_heads=8)
model = Transformer(vocab_size, vocab_size, d_model=32, num_layers=2, num_heads=8)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Training loop
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        logits = model(x, y)  # [batch, seq_len, vocab_size]
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(loader)}: Loss {loss.item():.4f}")
    
    return total_loss / len(loader)

# Validation loop
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x, y)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(loader)

# Train for 5 epochs
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pt')

