import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import SLTModel
from dataset import SignDataset
from utils import get_hand_adj
import psutil

def train_model():
    device = torch.device("mps")
    print(f"🚀 Optimized M4 Training Initialized")

    dataset = SignDataset('data/landmarks')
    
    # Set pin_memory=False as MPS doesn't support it yet
    train_loader = DataLoader(
        dataset, 
        batch_size=512, # Start at 128, move to 256 if stable
        shuffle=True, 
        num_workers=8, 
        pin_memory=False, 
        persistent_workers=True,
        prefetch_factor=2
    )
    
    num_classes = len(dataset.labels)
    adj = get_hand_adj().to(device)
    model = SLTModel(num_classes=num_classes, adj=adj).to(device)
    
    # REMOVED torch.compile as it causes the "Not enough SMs" hang on M4
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # AMP Scaler for MPS
    scaler = torch.amp.GradScaler('mps')

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # set_to_none=True is more efficient on Unified Memory
            optimizer.zero_grad(set_to_none=True) 
            
            with torch.amp.autocast(device_type='mps'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping is slow on MPS; only use if model is unstable
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 512**2:.2f} MB")

    if not os.path.exists('weights'): os.makedirs('weights')
    torch.save({'model_state_dict': model.state_dict(), 'labels': dataset.labels}, 'weights/slt_model.pth')
    print("✅ Training Complete.")

if __name__ == "__main__":
    train_model()
