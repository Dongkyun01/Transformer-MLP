import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)].to(x.device))

class TransformerWithMLP(nn.Module):
    def __init__(self, num_tokens, d_model=64, nhead=4, num_encoder_layers=2, d_ff=256, 
                 additional_feature_dim=8, output_dim=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.mlp = nn.Sequential(
            nn.Linear(additional_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.final = nn.Sequential(
            nn.Linear(d_model + 64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, seq_input, feature_input):
        x = self.embedding(seq_input)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # 마지막 timestep의 임베딩

        features = self.mlp(feature_input)
        combined = torch.cat([x, features], dim=-1)
        return self.final(combined).squeeze(-1)
