import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # CNN backbone
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # remove classifier
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        
        B, T, C, H, W = x.size()
        
        x = x.view(B * T, C, H, W)
        
        features = self.cnn(x)  # (B*T, 512)
        features = features.view(B, T, 512)
        
        lstm_out, _ = self.lstm(features)
        
        out = lstm_out[:, -1, :]  # last timestep
        
        return self.fc(out)