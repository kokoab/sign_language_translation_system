import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SignDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        self.labels = sorted(list(set([f.split('_')[0] for f in self.file_list])))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        data = np.load(os.path.join(self.data_path, file_name)).astype(np.float32)
        data = data.reshape(60, 21, 3)
        
        # 1. Wrist Centering
        wrist = data[:, 0, :]
        data = data - wrist[:, np.newaxis, :]
        
        # 2. Max-Distance Scaling (Normalization)
        # This scales the hand size to a unit sphere
        max_dist = np.max(np.linalg.norm(data, axis=2))
        if max_dist > 0:
            data = data / max_dist
        
        label_str = file_name.split('_')[0]
        label = self.label_to_idx[label_str]
        
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)