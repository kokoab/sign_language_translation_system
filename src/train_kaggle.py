import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os

# --- 1. ENHANCED GEOMETRIC EXTRACTOR ---
def extract_distances(data):
    def dist(p1, p2):
        return torch.sqrt(torch.sum((p1 - p2)**2, dim=-1) + 1e-8)

    # These 10 features are specifically chosen for R/U and S/T/M/N discrimination
    f1 = dist(data[:, :, 4], data[:, :, 8])   # Thumb-Index
    f2 = dist(data[:, :, 8], data[:, :, 12])  # Index-Middle (Crucial for R vs U)
    f3 = dist(data[:, :, 12], data[:, :, 16]) # Middle-Ring (Crucial for M vs N)
    f4 = dist(data[:, :, 16], data[:, :, 20]) # Ring-Pinky
    f5 = dist(data[:, :, 4], data[:, :, 12])  # Thumb-Middle (Crucial for M)
    f6 = dist(data[:, :, 4], data[:, :, 16])  # Thumb-Ring (Crucial for N)
    f7 = dist(data[:, :, 4], data[:, :, 20])  # Thumb-Pinky (Crucial for S)
    f8 = dist(data[:, :, 0], data[:, :, 4])   # Thumb Extension
    f9 = dist(data[:, :, 0], data[:, :, 8])   # Index Extension
    f10 = dist(data[:, :, 0], data[:, :, 17]) # Hand Breadth
    
    return torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dim=-1)

# --- 2. MODEL DEFINITION ---
class SLTModel(nn.Module):
    def __init__(self, num_classes, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        self.gcn_spatial = nn.Linear(3, 128)
        self.ln1 = nn.LayerNorm(128)
        self.projection = nn.Linear(21 * 128, 256)
        self.geom_fc = nn.Linear(10, 64)
        self.d_model = 320
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, self.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, f, j, c = x.shape
        geom_emb = F.relu(self.geom_fc(extract_distances(x)))
        x = torch.matmul(self.adj, x) 
        x = self.gcn_spatial(x)
        x = self.ln1(x).view(b, f, -1)
        x = F.relu(self.projection(x))
        x = torch.cat([x, geom_emb], dim=-1) + self.pos_encoder
        return self.classifier(self.transformer(x).mean(dim=1))

# --- 3. DATASET (30-FRAME DOWNSAMPLING) ---
class SignDataset(Dataset):
    def __init__(self, data_path, augment=True):
        self.data_path = data_path
        self.file_list = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        self.labels = sorted(list(set([f.split('_')[0] for f in self.file_list])))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.augment = augment
        
    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_path, self.file_list[idx])).astype(np.float32)
        data = data.reshape(-1, 21, 3)
        
        # Downsample to exactly 30 frames
        if data.shape[0] >= 60: data = data[::2, :, :]
        data = data[:30, :, :]
        
        if self.augment:
            if np.random.rand() > 0.5: data[:, :, 0] = 1.0 - data[:, :, 0]
            data *= np.random.uniform(0.8, 1.2)
            data += np.random.normal(0, 0.003, data.shape)

        # Centering
        data -= data[:, 0, np.newaxis, :]
        norm = np.max(np.linalg.norm(data, axis=2))
        if norm > 0: data /= norm
        
        label = self.label_to_idx[self.file_list[idx].split('_')[0]]
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)

# --- 4. TRAINING ENGINE ---
def train():
    device = torch.device("cuda")
    DATA_PATH = '/kaggle/input/datasets/kokoab/alphabets-01/landmarks' 
    dataset = SignDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    adj = torch.eye(21)
    conns = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20)]
    for i, j in conns: adj[i,j]=1; adj[j,i]=1

    model = nn.DataParallel(SLTModel(num_classes=len(dataset.labels), adj=adj)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')

    for epoch in range(80): # 80 epochs is usually enough for 30-frame models
        model.train()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
        print(f"Epoch {epoch+1} | Acc: {100.*correct/total:.2f}%")

    torch.save({'model_state_dict': model.module.state_dict(), 'labels': dataset.labels}, 'slt_model.pth')

if __name__ == "__main__":
    train()