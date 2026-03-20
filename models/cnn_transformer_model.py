import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ---------------- POSITIONAL ENCODING ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# ---------------- MAIN MODEL ----------------
class CNN_Transformer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=2):
        super().__init__()

        # CNN Backbone
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # remove final layer

        # Positional Encoding
        self.pos_enc = PositionalEncoding(hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        features = self.cnn(x)  # (B*T, 512)

        features = features.view(B, T, -1)  # (B, T, 512)

        # add positional encoding
        features = self.pos_enc(features)

        # transformer
        out = self.transformer(features)  # (B, T, 512)

        # take last token
        out = out[:, -1, :]

        return self.fc(out)