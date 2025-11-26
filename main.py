import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import GreenhouseDataset
from src.model import GreenhouseGRU
from src.train import train_model
import torch.nn as nn
import torch.optim as optim

def main():
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LENGTH = 24
    INPUT_SIZE = 11 # u1, u2, u3, yd, yc, yt, yh, d_i, d_c, d_t, d_h
    CNN_FILTERS = 64
    CNN_KERNEL = 3
    GRU_HIDDEN = 32
    OUTPUT_SIZE = 3 # u1, u2, u3
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # 1. Prepare Data
    print("Preparing Data...")
    # Use files 1-15 for training, 16-20 for validation
    train_indices = list(range(1, 16))
    val_indices = list(range(16, 21))
    path_template = "data/mpc_log_r1_mpc_24_{}.csv"

    train_dataset = GreenhouseDataset(indices=train_indices, path_template=path_template, seq_length=SEQ_LENGTH)
    val_dataset = GreenhouseDataset(indices=val_indices, path_template=path_template, seq_length=SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    print("Initializing Model...")
    model = GreenhouseGRU()
    
    # 3. Setup Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Train
    print("Starting Training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)
    
    print("Training Complete!")

    # 5. Evaluate
    print("Starting Evaluation...")
    from src.evaluate import evaluate_model
    evaluate_model(model, val_loader, criterion, device=DEVICE)
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()
