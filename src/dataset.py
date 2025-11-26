import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils import load_data, data_preprocessing

class GreenhouseDataset(Dataset):
    def __init__(self, indices, path_template, seq_length=24):
        """
        Args:
            indices (list): List of file indices to load.
            path_template (str): Template for file paths.
            seq_length (int): Length of the input sequence.
        """
        self.seq_length = seq_length
        
        # Load and preprocess data
        raw_data = load_data(indices, path_template)
        self.X, self.y = data_preprocessing(raw_data, n_past=seq_length, n_future=1)
        
        # Convert to torch tensors
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
