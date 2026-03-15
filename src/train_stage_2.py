"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT Stage 2 — Continuous Sign Language Recognition (CTC)        ║
║  Frozen DS-GCN Encoder (Stage 1) + BiLSTM + CTC Decoder          ║
║  Input : Variable-length sequences of 32-frame landmark clips    ║
║  Output: Gloss sequences decoded via CTC                         ║
║  Target: Kaggle T4 (16GB VRAM)                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  UPDATES: S2-5 Alignment Fix, S2-8 Frozen Params Fix, AMP added, ║
║  Cosine Scheduler, EMA, and proper DropPath restored.            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Lock to 1 GPU for memory safety

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import json
import math
import logging
import shutil
from pathlib import Path
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-S2")
torch.backends.cudnn.benchmark = True

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — DS-GCN ENCODER (Copied verbatim from Stage 1 v5.1)
# ══════════════════════════════════════════════════════════════════

_EDGES_SINGLE = [
    (0,1),(1,2),(2,3),(3,4),        
    (0,5),(5,6),(6,7),(7,8),        
    (0,9),(9,10),(10,11),(11,12),   
    (0,13),(13,14),(14,15),(15,16), 
    (0,17),(17,18),(18,19),(19,20), 
    (5,9),(9,13),(13,17),            
]
_EDGES = _EDGES_SINGLE + [(u+21, v+21) for u, v in _EDGES_SINGLE]

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
        self.dw_weights = nn.Parameter(torch.ones(3, C_in))
        nn.init.uniform_(self.dw_weights, 0.8, 1.2)
        self.pointwise = nn.Linear(3 * C_in, C_out, bias=False)
        self.temporal_conv = nn.Conv1d(C_out, C_out, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=C_out, bias=False)
        self.temporal_norm = nn.GroupNorm(num_groups, C_out)
        self.norm = nn.LayerNorm(C_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Linear(C_in, C_out, bias=False) if C_in != C_out else nn.Identity()

    def forward(self, x, A):
        B, T, N, C = x.shape
        residual = self.residual(x)
        agg = torch.einsum('knm,btnc->kbtnc', A, x)                
        agg = agg * self.dw_weights.view(3, 1, 1, 1, C)
        h = agg.permute(1, 2, 3, 0, 4).reshape(B, T, N, 3 * C)  
        h = self.drop(self.pointwise(h))                        
        C_out = h.shape[-1]
        h_t = h.permute(0, 2, 3, 1).reshape(B * N, C_out, T)      
        h_t = self.temporal_norm(self.temporal_conv(h_t))
        h = h_t.reshape(B, N, C_out, T).permute(0, 3, 1, 2)    
        return self.act(self.norm(h + residual))

_THUMB_MCP = 2; _THUMB_IP = 3; _THUMB_TIP = 4; _INDEX_MCP = 5; _INDEX_PIP = 6; _INDEX_TIP = 8
_MIDDLE_MCP = 9; _MIDDLE_PIP = 10; _MIDDLE_TIP = 12; _RING_MCP = 13; _RING_PIP = 14; _RING_TIP = 16
_PINKY_MCP = 17; _PINKY_PIP = 18; _PINKY_TIP = 20
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
        self.register_buffer('A', build_adjacency_matrices(42))
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
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))
            
        self.transformer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _geo_dist(a, b): return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz):
        d = self._geo_dist
        def get_hand_features(base):
            tips = [d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_TIP]), d(xyz[:,:,base+_INDEX_TIP], xyz[:,:,base+_MIDDLE_TIP]), d(xyz[:,:,base+_MIDDLE_TIP], xyz[:,:,base+_RING_TIP]), d(xyz[:,:,base+_RING_TIP], xyz[:,:,base+_PINKY_TIP]), d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_PINKY_TIP])]
            curls = [d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_TIP]) / (d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_IP]) + 1e-4), d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_TIP]) / (d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_PIP]) + 1e-4), d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_TIP]) / (d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_PIP]) + 1e-4), d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_TIP]) / (d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_PIP]) + 1e-4), d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_TIP]) / (d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_PIP]) + 1e-4)]
            cross_idx_mid = xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0]
            d_thumb_idxmcp = d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_MCP])
            return tips + curls + [cross_idx_mid, d_thumb_idxmcp]
        return torch.stack(get_hand_features(0) + get_hand_features(21), dim=-1)

    def forward(self, x):
        xyz = x[:, :, :, :3]
        h = self.input_proj(self.input_norm(x))
        h = self.gcn3(self.gcn2(self.gcn1(h, self.A), self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)
        h = self.geo_proj(torch.cat([h, self.geo_norm(self._compute_geo_features(xyz))], dim=-1)) + self.pos_enc
        for layer, dp in zip(self.transformer_layers, self.drop_paths): h = h + dp(layer(h) - h)
        return self.transformer_norm(h)

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — STAGE 2 CTC ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class SLTStage2CTC(nn.Module):
    def __init__(self, vocab_size, stage1_ckpt=None, d_model=256, lstm_hidden=512, lstm_layers=2, dropout=0.3):
        super().__init__()
        
        self.encoder = DSGCNEncoder(in_channels=10, d_model=d_model)
        
        if stage1_ckpt and Path(stage1_ckpt).exists():
            log.info(f"Loading pre-trained Stage 1 weights from {stage1_ckpt}")
            ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
            enc_state = {k.replace('encoder.', ''): v for k, v in ckpt['model_state_dict'].items() if 'encoder.' in k}
            self.encoder.load_state_dict(enc_state)
        
        # Freeze encoder to save VRAM and preserve spatial-temporal features
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Compress each 32-frame sign into exactly 4 sequential tokens
        self.temporal_pool = nn.AdaptiveAvgPool1d(4)

        # BiLSTM Sequence Decoder
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x, x_lens):
        B = x.size(0)
        out_seqs, out_lens = [], []
        
        # Process sequences individually to avoid massive padding VRAM spikes
        for b in range(B):
            valid_x = x[b, :x_lens[b]] # [T, 42, 10]
            num_clips = valid_x.size(0) // 32
            clips = valid_x.view(num_clips, 32, 42, 10)
            
            with torch.no_grad():
                enc_out = self.encoder(clips) # [num_clips, 32, 256]
                
            enc_out = enc_out.permute(0, 2, 1) # [num_clips, 256, 32]
            pooled = self.temporal_pool(enc_out) # [num_clips, 256, 4]
            pooled = pooled.permute(0, 2, 1) # [num_clips, 4, 256]
            
            seq_features = pooled.reshape(num_clips * 4, -1) # [num_clips * 4, 256]
            out_seqs.append(seq_features)
            out_lens.append(num_clips * 4)

        padded_seqs = pad_sequence(out_seqs, batch_first=True) # [B, max_tokens, 256]
        packed = pack_padded_sequence(padded_seqs, out_lens, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True) # [B, max_tokens, 1024]
        
        logits = self.classifier(unpacked) # [B, max_tokens, Vocab_Size]
        return logits, torch.tensor(out_lens, dtype=torch.long, device=x.device)

# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — DATASET & UTILS
# ══════════════════════════════════════════════════════════════════

class SyntheticCTCDataset(Dataset):
    def __init__(self, data_path, manifest, gloss_to_idx, num_samples=5000, min_len=2, max_len=6):
        self.data_path = Path(data_path)
        self.manifest = manifest
        self.gloss_to_idx = gloss_to_idx
        
        self.gloss_files = defaultdict(list)
        for f, gloss in manifest.items():
            if gloss in gloss_to_idx:
                self.gloss_files[gloss].append(f)
                
        self.vocab_keys = list(self.gloss_files.keys())
        self.samples = []
        
        log.info(f"Generating {num_samples} synthetic continuous sequences...")
        for _ in range(num_samples):
            seq_len = random.randint(min_len, max_len)
            seq_glosses = random.choices(self.vocab_keys, k=seq_len)
            seq_files = [random.choice(self.gloss_files[g]) for g in seq_glosses]
            self.samples.append((seq_files, [gloss_to_idx[g] for g in seq_glosses]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        files, target_glosses = self.samples[idx]
        arrays = []
        valid_targets = []
        
        # S2-5 Fix: Strict alignment between loaded arrays and targets
        for f, tgt in zip(files, target_glosses):
            arr = np.load(self.data_path / f).astype(np.float32)
            if arr.shape == (32, 42, 10):
                arrays.append(arr)
                valid_targets.append(tgt)
        
        # Fallback if somehow all files are skipped
        if len(arrays) == 0:
            return np.zeros((32, 42, 10), dtype=np.float32), []
            
        x = np.concatenate(arrays, axis=0) 
        return x, valid_targets

def collate_ctc(batch):
    xs = [torch.from_numpy(b[0]) for b in batch if len(b[1]) > 0]
    ys = [torch.tensor(b[1], dtype=torch.long) for b in batch if len(b[1]) > 0]
    
    if len(xs) == 0: return None, None, None, None
    
    x_lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
    
    x_pad = pad_sequence(xs, batch_first=True) # [B, max_T, 42, 10]
    y_flat = torch.cat(ys) # CTC expects flattened targets
    
    return x_pad, y_flat, x_lens, y_lens

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
    
    spatial_features = x[..., :9]
    mask_features    = x[..., 9:]
    xr = spatial_features.view(B, T, N, 3, 3)                                
    xr = torch.einsum('btngi,bij->btngj', xr, R)                  
    xr = xr.reshape(B, T, N, 9)
    
    x_rotated = torch.cat([xr, mask_features], dim=-1)
    scale = scale_lo + torch.rand(B, 1, 1, 1, device=device) * (scale_hi - scale_lo)
    return x_rotated * scale + torch.randn_like(x_rotated) * noise_std

def calculate_wer(reference, hypothesis):
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.uint8)
    for i in range(len(reference) + 1): d[i][0] = i
    for j in range(len(hypothesis) + 1): d[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[len(reference)][len(hypothesis)] / max(len(reference), 1)

def decode_ctc(log_probs, out_lens, blank=0):
    preds = log_probs.argmax(dim=-1).cpu().numpy()
    decoded_batch = []
    for b in range(preds.shape[0]):
        seq = preds[b, :out_lens[b]]
        decoded = []
        last_tok = blank
        for tok in seq:
            if tok != blank and tok != last_tok:
                decoded.append(tok)
            last_tok = tok
        decoded_batch.append(decoded)
    return decoded_batch

# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — TRAINING COMPONENTS (SCHEDULER, EMA, CKPT)
# ══════════════════════════════════════════════════════════════════

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

class ModelEMA:
    def __init__(self, model, decay=0.999):
        # Only track trainable parameters
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

def make_checkpoint(model, optimizer, scheduler, ema, epoch, val_wer, best_wer, trigger_times, gloss_to_idx, idx_to_gloss, vocab_size):
    unwrapped = model.module if hasattr(model, 'module') else model
    return {
        'model_state_dict':       unwrapped.state_dict(),
        'optimizer_state_dict':   optimizer.state_dict(),
        'scheduler_state_dict':   scheduler.state_dict(),
        'ema_shadow':             ema.shadow if ema else None,
        'epoch': epoch, 'best_wer': best_wer, 'trigger_times': trigger_times,
        'gloss_to_idx': gloss_to_idx, 'idx_to_gloss': idx_to_gloss, 'vocab_size': vocab_size,
        'val_wer': val_wer, 'stage': 2,
    }

# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train_stage2(
    data_path = '/kaggle/input/datasets/kokoab/batch-1/ASL_landmarks_float16',
    stage1_ckpt = '/kaggle/input/datasets/kokoab/model-dataset/best_model.pth', # ⚠️ UPDATE THIS!
    save_dir = '/kaggle/working/',
    smoke_test = False,
    epochs = 50, batch_size = 32, lr = 1e-3, warmup_epochs = 5, patience = 15
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_CKPT, LAST_CKPT = save_dir / 'stage2_best_model.pth', save_dir / 'stage2_last_checkpoint.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    
    if smoke_test:
        log.warning("🚬 SMOKE TEST MODE ACTIVATED! Running on subset for 3 epochs.")
        epochs, patience = 3, 3

    manifest_path = Path(data_path) / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    unique_labels = sorted(list(set(manifest.values())))
    unique_labels = ['<BLANK>'] + unique_labels # CTC Blank token is Index 0
    gloss_to_idx = {gloss: i for i, gloss in enumerate(unique_labels)}
    idx_to_gloss = {i: gloss for gloss, i in gloss_to_idx.items()}
    vocab_size = len(gloss_to_idx)
    
    log.info(f"CTC Vocab Size: {vocab_size} (including BLANK at index 0)")

    train_ds = SyntheticCTCDataset(data_path, manifest, gloss_to_idx, num_samples=100 if smoke_test else 8000)
    val_ds = SyntheticCTCDataset(data_path, manifest, gloss_to_idx, num_samples=20 if smoke_test else 1000)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ctc, num_workers=2, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ctc, num_workers=2, pin_memory=use_amp)

    model = SLTStage2CTC(vocab_size=vocab_size, stage1_ckpt=stage1_ckpt)
    
    # S2-8 Fix: Only pass trainable params to optimizer (saves huge VRAM)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = ModelEMA(model)

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch, best_wer, trigger_times = 1, float('inf'), 0
    
    # S2-9 Fix: Checkpoint Resume Logic
    if LAST_CKPT.exists():
        try:
            ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'ema_shadow' in ckpt and ckpt['ema_shadow'] is not None:
                ema.shadow = ckpt['ema_shadow']
            start_epoch, best_wer, trigger_times = ckpt['epoch'] + 1, ckpt.get('best_wer', float('inf')), ckpt.get('trigger_times', 0)
            log.info(f"Resumed from epoch {ckpt['epoch']} | Best WER so far: {best_wer:.2f}%")
        except Exception as e: 
            log.warning(f"Resume failed ({e}). Starting fresh.")

    model.to(device)
    ema.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(device)

    history = []
    converged_epoch = None
    log.info("Starting Stage 2 CTC Training...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for i, (x_pad, y_flat, x_lens, y_lens) in enumerate(train_loader):
            if x_pad is None: continue # Skip if batch was fully invalid
            
            x_pad = x_pad.to(device, non_blocking=True)
            y_flat = y_flat.to(device, non_blocking=True)
            
            # S2-3 Fix: Apply Augmentation
            x_pad = online_augment(x_pad)
            optimizer.zero_grad(set_to_none=True)
            
            # S2-7 Fix: Applied AMP
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, out_lens = model(x_pad, x_lens)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1) # [T, B, C]
                loss = ctc_loss_fn(log_probs, y_flat, out_lens, y_lens)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            ema.update(model)
            epoch_loss += loss.item()
            
            if epoch == 1 and i == 0 and use_amp:
                log.info(f"GPU 0 Mem Usage (Batch 1): {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
            
        # Validation Loop
        ema.apply(model)
        model.eval()
        total_wer = 0
        total_samples = 0
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for x_pad, y_flat, x_lens, y_lens in val_loader:
                if x_pad is None: continue
                x_pad = x_pad.to(device, non_blocking=True)
                logits, out_lens = model(x_pad, x_lens)
                log_probs = F.log_softmax(logits, dim=-1)
                
                decoded_preds = decode_ctc(log_probs, out_lens, blank=0)
                
                targets = []
                idx = 0
                for length in y_lens:
                    targets.append(y_flat[idx:idx+length].cpu().tolist())
                    idx += length
                    
                for ref, hyp in zip(targets, decoded_preds):
                    total_wer += calculate_wer(ref, hyp)
                    total_samples += 1
                    
        ema.restore(model)
        
        train_loss = epoch_loss / len(train_loader)
        val_wer = (total_wer / max(total_samples, 1)) * 100
        
        log.info(f"Ep {epoch:02d} | LR: {cur_lr:.2e} | Train Loss: {train_loss:.4f} | Val WER: {val_wer:.2f}%")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_wer": round(val_wer, 2), "lr": round(cur_lr, 8)})

        # Checkpoint (WER: lower is better!)
        should_save_last = (epoch % 5 == 0 or epoch == epochs)
        if val_wer < best_wer or should_save_last:
            ckpt = make_checkpoint(model, optimizer, scheduler, ema, epoch, val_wer, min(val_wer, best_wer), trigger_times, gloss_to_idx, idx_to_gloss, vocab_size)
            if should_save_last:
                torch.save(ckpt, LAST_CKPT)
            if val_wer < best_wer:
                best_wer, trigger_times = val_wer, 0
                converged_epoch = epoch
                torch.save(ckpt, BEST_CKPT)
                log.info(f"  ✨ Best Checkpoint Saved! (WER: {best_wer:.2f}%)")
            else: trigger_times += 1
        else: trigger_times += 1

        if trigger_times >= patience:
            log.info(f"🛑 CONVERGENCE REACHED: Early stopping triggered at Epoch {epoch}.")
            break

    log.info(f"🏆 Best model peaked at Epoch {converged_epoch} with Validation WER: {best_wer:.2f}%")
    with open(save_dir / 'stage2_history.json', 'w') as f: json.dump(history, f, indent=2)

if __name__ == '__main__':
    train_stage2()