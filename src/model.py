import torch
import torch.nn as nn
import torch.nn.functional as F

class GreenhouseGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=11, hidden_size=16, num_layers=2,
            dropout=0, batch_first=True, bidirectional=True
        )
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
