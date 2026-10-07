import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from attention import CharTokenizer

class ShakespeareDataset(Dataset):
    def __init__(self, data, seq_len=256):
        """
        Args:
            data: tensor of token IDs
            seq_len: sequence length for training
        """
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

def run_preparation():
    print("Preparing Shakespeare data...")
    
    # Load text
    current_dir = os.getcwd()
    input_file_path = os.path.join(current_dir, 'data', 'input_shakespeare.txt')
    
    with open(input_file_path, 'r') as f:
        data = f.read()
    
    print(f"Length of dataset: {len(data):,} characters")
    
    # Create tokenizer from full text
    tokenizer = CharTokenizer(data)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Encode all text
    encoded = torch.tensor(tokenizer.encode(data), dtype=torch.long)
    
    # Split into train/val
    train_size = int(0.9 * len(encoded))
    train_data = encoded[:train_size]
    val_data = encoded[train_size:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")

    # testing different settings
    #seq_length= 256
    #batch_size_s = 32

    seq_length= 64
    batch_size_s = 32
        
    # Create datasets for processing?
    train_dataset = ShakespeareDataset(train_data, seq_len=seq_length)
    val_dataset = ShakespeareDataset(val_data, seq_len=seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_s, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_s, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Return vars needed for training
    return vocab_size, tokenizer, train_loader, val_loader

if __name__ == "__main__":
    vocab_size, tokenizer, train_loader, val_loader = run_preparation()