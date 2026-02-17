import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

class SLTModel(nn.Module):
    def __init__(self, num_classes, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        
        # GCN Feature Extractor
        self.gcn_spatial = nn.Linear(3, 128) # Increased feature width
        self.ln1 = nn.LayerNorm(128)
        
        # Transformer (d_model = 21 * 128 = 2688)
        self.pos_encoder = PositionalEncoding(d_model=2688)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2688, nhead=8, dim_feedforward=2048, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.classifier = nn.Sequential(
            nn.Linear(2688, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, f, j, c = x.shape
        # Spatial Graph Step
        x = torch.matmul(self.adj, x) 
        x = self.gcn_spatial(x)
        x = self.ln1(x)
        
        # Temporal Step
        x = x.view(b, f, -1) 
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Feature Pooling
        x = x.mean(dim=1) 
        return self.classifier(x)