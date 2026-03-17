"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT Stage 1 — Isolated Sign Classification (Dual Hand Edition)  ║
║  DS-GCN Spatial Encoder + Transformer Encoder + Classifier Head  ║
║  Input : [B, 32, 42, 10] (Both Hands: xyz + vel + acc + mask)    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
# 🔥 FORCE KAGGLE TO ONLY USE 1 GPU (Prevents duplicate processes & RAM OOM)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import numpy as np
import json
import math
import logging
import shutil
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split 

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-S1")

torch.backends.cudnn.benchmark = True

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — DUAL-HAND GRAPH & DS-GCN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

# Base edges for a single hand
_EDGES_SINGLE = [
    (0,1),(1,2),(2,3),(3,4),        
    (0,5),(5,6),(6,7),(7,8),        
    (0,9),(9,10),(10,11),(11,12),   
    (0,13),(13,14),(14,15),(15,16), 
    (0,17),(17,18),(18,19),(19,20), 
    (5,9),(9,13),(13,17),            
]
# Replicate the exact same connections for the second hand (offset by +21)
_EDGES = _EDGES_SINGLE + [(u+21, v+21) for u, v in _EDGES_SINGLE]

# 42 Nodes total (21 Left + 21 Right)
def build_adjacency_matrices(num_nodes: int = 42) -> torch.Tensor:
    def _norm(M):
        deg = M.sum(axis=1, keepdims=True).clip(min=1)
        return M / deg
    A_self = np.eye(num_nodes, dtype=np.float32)
    A_out  = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A_in   = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in _EDGES:
        A_out[src, dst] = 1.0
        A_in[dst, src]  = 1.0
    return torch.from_numpy(np.stack([_norm(A_self), _norm(A_out), _norm(A_in)]))

class DSGCNBlock(nn.Module):
    def __init__(self, C_in, C_out, temporal_kernel=3, dropout=0.1, num_groups=8):
        super().__init__()
        K = 3
        self.dw_weights    = nn.Parameter(torch.ones(K, C_in))
        nn.init.uniform_(self.dw_weights, 0.8, 1.2)
        self.pointwise     = nn.Linear(K * C_in, C_out, bias=False)
        self.temporal_conv = nn.Conv1d(C_out, C_out, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=C_out, bias=False)
        self.temporal_norm = nn.GroupNorm(num_groups, C_out)
        self.norm          = nn.LayerNorm(C_out)
        self.act           = nn.GELU()
        self.drop          = nn.Dropout(dropout)
        self.residual      = nn.Linear(C_in, C_out, bias=False) if C_in != C_out else nn.Identity()

    def forward(self, x, A):
        B, T, N, C = x.shape
        residual = self.residual(x)
        agg = torch.einsum('knm,btnc->kbtnc', A, x)                
        agg = agg * self.dw_weights.view(3, 1, 1, 1, C)
        h   = agg.permute(1, 2, 3, 0, 4).reshape(B, T, N, 3 * C)  
        h     = self.drop(self.pointwise(h))                        
        C_out = h.shape[-1]
        h_t   = h.permute(0, 2, 3, 1).reshape(B * N, C_out, T)      
        h_t   = self.temporal_norm(self.temporal_conv(h_t))
        h     = h_t.reshape(B, N, C_out, T).permute(0, 3, 1, 2)    
        return self.act(self.norm(h + residual))

_THUMB_MCP  = 2;  _THUMB_IP   = 3;  _THUMB_TIP  = 4
_INDEX_MCP  = 5;  _INDEX_PIP  = 6;  _INDEX_TIP  = 8
_MIDDLE_MCP = 9;  _MIDDLE_PIP = 10; _MIDDLE_TIP = 12
_RING_MCP   = 13; _RING_PIP   = 14; _RING_TIP   = 16
_PINKY_MCP  = 17; _PINKY_PIP  = 18; _PINKY_TIP  = 20

# 12 Features per hand x 2 hands = 24 features
N_GEO_FEATURES = 24 

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0: return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * (torch.rand(shape, device=x.device) < keep) / keep

class DSGCNEncoder(nn.Module):
    def __init__(self, in_channels=10, d_model=256, nhead=8, num_transformer_layers=4, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.register_buffer('A', build_adjacency_matrices(42)) # 42 Node Adjacency
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Sequential(nn.Linear(in_channels, 64), nn.LayerNorm(64), nn.GELU())
        self.gcn1 = DSGCNBlock(64,  128, temporal_kernel=3, dropout=dropout)
        self.gcn2 = DSGCNBlock(128, 128, temporal_kernel=3, dropout=dropout)
        self.gcn3 = DSGCNBlock(128, d_model, temporal_kernel=5, dropout=dropout)
        self.node_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        dp_rates = [drop_path_rate * i / max(num_transformer_layers - 1, 1) for i in range(num_transformer_layers)]
        self.transformer_layers, self.drop_paths = nn.ModuleList(), nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))
        self.transformer_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'transformer' in name: continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    @staticmethod
    def _geo_dist(a, b): return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz):
        d = self._geo_dist
        
        # Helper to compute features for a specific hand (base offset 0 or 21)
        def get_hand_features(base):
            tips = [d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_TIP]), d(xyz[:,:,base+_INDEX_TIP], xyz[:,:,base+_MIDDLE_TIP]), d(xyz[:,:,base+_MIDDLE_TIP], xyz[:,:,base+_RING_TIP]), d(xyz[:,:,base+_RING_TIP], xyz[:,:,base+_PINKY_TIP]), d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_PINKY_TIP])]
            curls = [d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_TIP]) / (d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_IP]) + 1e-4), d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_TIP]) / (d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_PIP]) + 1e-4), d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_TIP]) / (d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_PIP]) + 1e-4), d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_TIP]) / (d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_PIP]) + 1e-4), d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_TIP]) / (d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_PIP]) + 1e-4)]
            cross_idx_mid  = xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0]
            d_thumb_idxmcp = d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_MCP])
            return tips + curls + [cross_idx_mid, d_thumb_idxmcp]

        feats_hand1 = get_hand_features(0)  # Left Hand
        feats_hand2 = get_hand_features(21) # Right Hand
        
        # Concatenate 12 features per hand = 24 features total
        return torch.stack(feats_hand1 + feats_hand2, dim=-1)

    def forward(self, x):
        xyz = x[:, :, :, :3]
        h = self.input_proj(self.input_norm(x))
        h = self.gcn3(self.gcn2(self.gcn1(h, self.A), self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)
        h = self.geo_proj(torch.cat([h, self.geo_norm(self._compute_geo_features(xyz))], dim=-1)) + self.pos_enc
        for layer, dp in zip(self.transformer_layers, self.drop_paths): h = h + dp(layer(h) - h)
        return self.transformer_norm(h)

class ClassifierHead(nn.Module):
    def __init__(self, d_model=256, num_classes=29, dropout=0.4):
        super().__init__()
        self.frame_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, num_classes))
    def forward(self, x):
        attn = F.softmax(self.frame_attn(x).squeeze(-1), dim=1)
        return self.net((x * attn.unsqueeze(-1)).sum(dim=1))

class SLTStage1(nn.Module):
    def __init__(self, num_classes, in_channels=10, d_model=256, nhead=8, num_transformer_layers=4, dropout=0.1, head_dropout=0.4, drop_path_rate=0.1):
        super().__init__()
        self.encoder = DSGCNEncoder(in_channels=in_channels, d_model=d_model, nhead=nhead, num_transformer_layers=num_transformer_layers, dropout=dropout, drop_path_rate=drop_path_rate)
        self.head = ClassifierHead(d_model=d_model, num_classes=num_classes, dropout=head_dropout)
    def forward(self, x): return self.head(self.encoder(x))

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — DATASET, LOADER, & EMA
# ══════════════════════════════════════════════════════════════════

class SignDataset(Dataset):
    def __init__(self, data_path: str, label_to_idx: dict, manifest: dict, cache_path: str = None):
        self.label_to_idx = label_to_idx
        self.num_classes  = len(label_to_idx)
        data_path = Path(data_path)

        if cache_path and Path(cache_path).exists():
            log.info(f"Loading cache: {cache_path}")
            cache = torch.load(cache_path, weights_only=True)
            if cache.get('label_to_idx') == label_to_idx and cache.get('num_files') == len(manifest):
                self.data, self.targets = cache['data'], cache['targets']
                self.filenames = cache.get('filenames', []) 
                log.info(f"  {len(self.targets)} samples | RAM: ~{self.data.nbytes / 1e6:.0f} MB")
                return
            log.warning("Dataset/Manifest changed — rebuilding cache from .npy files.")

        log.info("Building dataset from .npy files using strict manifest...")
        data_list, target_list, filename_list = [], [], []
        skipped_manifest = 0
        first_shape = None
        
        for fname in sorted(f for f in os.listdir(data_path) if f.endswith('.npy')):
            if fname not in manifest:
                skipped_manifest += 1
                continue
                
            label = manifest[fname]
            if label not in label_to_idx:
                skipped_manifest += 1
                continue
                
            arr = np.load(data_path / fname).astype(np.float32)

            # Accept ONLY (32, 42, 10)
            if arr.shape != (32, 42, 10):
                continue
                
            data_list.append(arr)
            target_list.append(label_to_idx[label])
            filename_list.append(fname)

        if len(data_list) == 0:
            raise ValueError("🛑 CRITICAL: Every file was skipped! Ensure your .npy files are actually (32, 42, 10).")

        self.data      = torch.from_numpy(np.stack(data_list))
        self.targets   = torch.tensor(target_list, dtype=torch.long)
        self.filenames = filename_list
        log.info(f"  {len(self.targets)} samples loaded | {skipped_manifest} skipped missing from manifest")

        if cache_path:
            torch.save({'data': self.data, 'targets': self.targets,
                        'label_to_idx': label_to_idx, 'filenames': self.filenames,
                        'num_files': len(manifest)}, cache_path)

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

    def class_weights(self, temperature: float = 0.5) -> torch.Tensor:
        counts = Counter(self.targets.tolist())
        raw = torch.tensor([len(self.targets) / counts[t.item()] for t in self.targets], dtype=torch.float32)
        return raw ** temperature

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.decay = decay
        self.backup = {}

    def to(self, device):
        self.shadow = {n: p.to(device) for n, p in self.shadow.items()}
        return self

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])

# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — TRAINING, EVAL, SCHEDULER
# ══════════════════════════════════════════════════════════════════

def _batch_rotation_matrices(batch_size, max_deg, device):
    angles = torch.randn(batch_size, 3, device=device) * max_deg
    rad = angles * (math.pi / 180.0)
    cx, sx = torch.cos(rad[:, 0]), torch.sin(rad[:, 0])
    cy, sy = torch.cos(rad[:, 1]), torch.sin(rad[:, 1])
    cz, sz = torch.cos(rad[:, 2]), torch.sin(rad[:, 2])
    zero, one = torch.zeros_like(cx), torch.ones_like(cx)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=1).view(-1, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=1).view(-1, 3, 3)
    return Rx @ Ry @ Rz

def online_augment(x, rotation_deg=10.0, scale_lo=0.85, scale_hi=1.15, noise_std=0.003):
    B, T, N, C = x.shape
    device = x.device
    R = _batch_rotation_matrices(B, rotation_deg, device)         
    
    # ⚠️ Keep this logic aligned with your 10 channels. We rotate the first 9 (xyz, v, a)
    # and safely stitch the 10th (mask) back on without rotating it.
    spatial_features = x[..., :9]
    mask_features    = x[..., 9:]
    
    xr = spatial_features.view(B, T, N, 3, 3)                                
    xr = torch.einsum('btngi,bij->btngj', xr, R)                  
    xr = xr.reshape(B, T, N, 9)
    
    # Stitch mask back
    x_rotated = torch.cat([xr, mask_features], dim=-1)
    
    scale = scale_lo + torch.rand(B, 1, 1, 1, device=device) * (scale_hi - scale_lo)
    return x_rotated * scale + torch.randn_like(x_rotated) * noise_std

def apply_mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs: scale = (e + 1) / self.warmup_epochs
        else:
            progress = ((e - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]

@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss, correct_1, correct_5, sample_total = 0.0, 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            total_loss += loss.item()
            
        preds = logits.argmax(dim=1)
        correct_1 += (preds == y).sum().item()
        
        _, top5_preds = logits.topk(5, dim=1)
        correct_5 += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()
        
        sample_total += y.size(0)
        
    return {
        "val_loss": total_loss / len(loader), 
        "acc": (correct_1 / max(sample_total, 1)) * 100,
        "top5_acc": (correct_5 / max(sample_total, 1)) * 100
    }

def make_checkpoint(model, optimizer, scheduler, ema, epoch, val_acc, best_acc, trigger_times, label_to_idx, idx_to_label, num_classes, in_channels, d_model, nhead, num_transformer_layers):
    unwrapped = model.module if hasattr(model, 'module') else model
    return {
        'encoder_state_dict':     unwrapped.encoder.state_dict(),
        'head_state_dict':        unwrapped.head.state_dict(),
        'model_state_dict':       unwrapped.state_dict(),
        'optimizer_state_dict':   optimizer.state_dict(),
        'scheduler_state_dict':   scheduler.state_dict(),
        'ema_shadow':             ema.shadow if ema else None,
        'epoch': epoch, 'best_acc': best_acc, 'trigger_times': trigger_times,
        'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label, 'num_classes': num_classes,
        'in_channels': in_channels, 'd_model': d_model, 'nhead': nhead, 'num_transformer_layers': num_transformer_layers,
        'val_acc': val_acc, 'stage': 1,
    }

def train(
    # UPDATE THIS to your Kaggle dataset path containing BOTH .npy files and manifest.json
    data_path = '/kaggle/input/datasets/kokoab/batch-1/ASL_landmarks_float16',
    save_dir  = '/kaggle/working/',
    
    smoke_test = False,
    
    epochs = 200, batch_size = 256, accum_steps = 4, lr = 1e-3, weight_decay = 0.01,
    warmup_epochs = 5, label_smoothing = 0.05, grad_clip = 5.0, patience = 40,
    sampler_temperature = 0.5, in_channels = 10, d_model = 256, nhead = 8,
    num_transformer_layers = 4, dropout = 0.10, head_dropout = 0.15,
    mixup_alpha = 0.2
):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_CKPT, LAST_CKPT = save_dir / 'best_model.pth', save_dir / 'last_checkpoint.pth'

    log.info(f"Device: {device} | AMP: {use_amp} | Devices Found: {torch.cuda.device_count() if use_amp else 0}")
    
    if smoke_test:
        log.warning("🚬 SMOKE TEST MODE ACTIVATED! Running on subset for 3 epochs.")
        epochs, patience = 3, 3

    manifest_path = Path(data_path) / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"CRITICAL: manifest.json not found in {data_path}. Upload it first!")
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    unique_labels = sorted(list(set(manifest.values())))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    
    log.info(f"Loaded strict manifest mapping {len(manifest)} files to {len(unique_labels)} classes.")
    
    cache_path = str(save_dir / 'ds_cache.pt')
    full_ds    = SignDataset(data_path, label_to_idx, manifest=manifest, cache_path=cache_path)
    
    indices = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=full_ds.targets.numpy()
    )
    
    if smoke_test:
        train_idx = train_idx[:400]
        val_idx = val_idx[:100]

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    log.info(f"Dataset Stratified Split: 70% Train ({len(train_ds)}) | 30% Val ({len(val_ds)})")

    sample_weights = full_ds.class_weights(temperature=sampler_temperature)[train_ds.indices]

    on_gpu = False
    if device.type == 'cuda':
        log.info("Preloading dataset to GPU...")
        full_ds.data    = full_ds.data.to(device, non_blocking=True)
        full_ds.targets = full_ds.targets.to(device, non_blocking=True)
        torch.cuda.empty_cache() 
        on_gpu = True

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    nw = 0 if on_gpu else (2 if device.type == 'cuda' else 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=nw, pin_memory=(not on_gpu and use_amp), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=(not on_gpu and use_amp))

    model = SLTStage1(num_classes=full_ds.num_classes, in_channels=in_channels, d_model=d_model, nhead=nhead, num_transformer_layers=num_transformer_layers, dropout=dropout, head_dropout=head_dropout)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = ModelEMA(model)

    start_epoch, best_acc, trigger_times = 1, 0.0, 0
    
    if LAST_CKPT.exists():
        try:
            ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'ema_shadow' in ckpt and ckpt['ema_shadow'] is not None:
                ema.shadow = ckpt['ema_shadow']
            start_epoch, best_acc, trigger_times = ckpt['epoch'] + 1, ckpt.get('best_acc', 0.0), ckpt.get('trigger_times', 0)
            log.info(f"Resumed from epoch {ckpt['epoch']} | Best so far: {best_acc:.2f}%")
        except Exception as e: 
            log.warning(f"Resume failed ({e}). Starting fresh.")

    model.to(device)
    ema.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(device)

    history = []
    converged_epoch = None

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = online_augment(x)
            
            x, y_a, y_b, lam = apply_mixup(x, y, alpha=mixup_alpha)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                loss = (lam * loss_a + (1 - lam) * loss_b) / accum_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                ema.update(model.module if hasattr(model, 'module') else model)

            epoch_loss += loss.item() * accum_steps
            
            if epoch == 1 and i == 0 and use_amp:
                log.info(f"GPU 0 Mem Usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        ema.apply(model.module if hasattr(model, 'module') else model)
        val_metrics = evaluate(model, val_loader, device, use_amp)
        ema.restore(model.module if hasattr(model, 'module') else model)

        train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc, val_top5 = val_metrics["val_loss"], val_metrics["acc"], val_metrics["top5_acc"]

        log.info(f"Ep {epoch:03d} | LR: {cur_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_acc:.2f}% | Val Top-5: {val_top5:.2f}%")

        history.append({"epoch": epoch, "val_acc": round(val_acc, 3), "val_top5": round(val_top5, 3), "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5), "lr": round(cur_lr, 8)})

        should_save_last = (epoch % 5 == 0 or epoch == epochs)
        if val_acc > best_acc or should_save_last:
            ckpt = make_checkpoint(model, optimizer, scheduler, ema, epoch, val_acc, max(val_acc, best_acc), trigger_times, label_to_idx, idx_to_label, full_ds.num_classes, in_channels, d_model, nhead, num_transformer_layers)
            
            if should_save_last:
                torch.save(ckpt, LAST_CKPT)

            if val_acc > best_acc:
                best_acc, trigger_times = val_acc, 0
                converged_epoch = epoch
                torch.save(ckpt, BEST_CKPT)
                log.info(f"  ✨ Best Checkpoint Saved! (Top-1 Acc: {best_acc:.2f}%)")
            else:
                trigger_times += 1
        else:
            trigger_times += 1

        if trigger_times >= patience:
            log.info(f"🛑 CONVERGENCE REACHED: Early stopping triggered at Epoch {epoch}.")
            break

    log.info(f"🏆 Best model peaked at Epoch {converged_epoch} with Validation Accuracy: {best_acc:.2f}%")
    with open(save_dir / 'history.json', 'w') as f: json.dump(history, f, indent=2)

    try:
        shutil.make_archive('/kaggle/working/SLT_Stage1_Results', 'zip', '/kaggle/working/')
        log.info("✅ Successfully zipped! Download SLT_Stage1_Results.zip from the Kaggle Output panel.")
    except Exception as e:
        log.error(f"Failed to zip files: {e}")

if __name__ == "__main__":
    train()