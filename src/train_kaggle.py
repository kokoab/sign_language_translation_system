"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT Stage 1 — Isolated Sign Classification                      ║
║  DS-GCN Spatial Encoder + Transformer Encoder + Classifier Head  ║
║  Input : [B, 32, 21, 9]  (xyz + velocity + acceleration)         ║
║  Output: [B, num_classes]                                        ║
║  Target: Kaggle Dual T4 (DataParallel) + local RTX 3080/4050     ║
╠══════════════════════════════════════════════════════════════════╣
║  PIPELINE ROADMAP                                                ║
║  Stage 1 (this file) — isolated sign → single gloss label        ║
║    Train: cross-entropy loss, classifier head on mean-pooled seq ║
║    Output: best_model.pth  (encoder weights transfer to Stage 2) ║
║                                                                  ║
║  Stage 2 (future: train_stage2.py)                               ║
║    Load DSGCNEncoder weights from Stage 1 best_model.pth         ║
║    Replace classifier with CTC head                              ║
║    Train on continuous signing videos → gloss sequences          ║
║                                                                  ║
║  Stage 3 (future: train_stage3.py)                               ║
║    Fine-tune mT5-small on (gloss sequence → English sentence)    ║
║    Pure text-to-text, no video model involved                    ║
╠══════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE DESIGN FOR STAGE COMPATIBILITY                     ║
║  The model is split into two cleanly separable parts:            ║
║    DSGCNEncoder  — DS-GCN + Transformer Encoder                  ║
║                    outputs [B, T, d_model] (sequence preserved)  ║
║                    THIS is what Stage 2 loads and reuses         ║
║    ClassifierHead — mean pool + linear layers                    ║
║                    outputs [B, num_classes]                      ║
║                    Stage 2 discards this and replaces with CTC   ║
║                                                                  ║
║  Stage 1 checkpoint saves both encoder and head weights,         ║
║  plus all hyperparameters needed to reconstruct for inference.   ║
╠══════════════════════════════════════════════════════════════════╣
║  KNOWN HISTORY & FIXES                                           ║
║  Run 1 (v2.0): lr=3e-4, collapsed epoch 5 — LR too high          ║
║  Run 2 (v2.2): lr=5e-5, frequency bias (J/P/Q only)              ║
║  Run 3 (v2.3): WeightedRandomSampler temperature=0.5, working    ║
║  Run 4 (v2.4): Reached 95.66%, collapsed epoch 33                ║
║    → CosineAnnealingWarmRestarts spiked LR at epoch 30           ║
║    → Fixed: replaced with CosineWarmupScheduler (no restarts)    ║
║  Stage 1 v1.0 (this): encoder/head split, Stage 2 ready          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import numpy as np
import os
import json
import math
import logging
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-S1")


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — HAND GRAPH
#  Fixed adjacency matrices for MediaPipe 21-node hand topology.
#  Three partitions: self-loop, outward (wrist→tip), inward (tip→wrist).
#  These never change — registered as buffers, not learned parameters.
# ══════════════════════════════════════════════════════════════════

_EDGES = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17),           # Palm cross-connectors
]

def build_adjacency_matrices(num_nodes: int = 21) -> torch.Tensor:
    def _norm(M):
        deg = M.sum(axis=1, keepdims=True).clip(min=1)
        return M / deg
    A_self = np.eye(num_nodes, dtype=np.float32)
    A_out  = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A_in   = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in _EDGES:
        A_out[src, dst] = 1.0
        A_in[dst, src]  = 1.0
    return torch.from_numpy(
        np.stack([_norm(A_self), _norm(A_out), _norm(A_in)])
    )  # [3, 21, 21]


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — DS-GCN BLOCK
#  Depthwise Separable Graph Convolutional Block.
#  [B, T, N, C_in] -> [B, T, N, C_out]
#
#  GroupNorm (not BatchNorm): identical behavior at B=256 training
#  and B=1 webcam inference. BatchNorm uses running statistics that
#  create train/eval mismatch — unacceptable for real-time use.
#
#  Vectorized einsum over all 3 adjacency partitions simultaneously.
#  'knm, btnc -> kbtnc' contracts over m (neighbor axis, not self axis).
#  Verified: agg[k,b,t,n,c] = sum_m A[k,n,m] * x[b,t,m,c]
# ══════════════════════════════════════════════════════════════════

class DSGCNBlock(nn.Module):
    def __init__(self, C_in, C_out, temporal_kernel=3, dropout=0.1, num_groups=8):
        super().__init__()
        K = 3
        assert C_out % num_groups == 0, \
            f"C_out ({C_out}) must be divisible by num_groups ({num_groups})"

        self.dw_weights    = nn.Parameter(torch.ones(K, C_in))
        nn.init.uniform_(self.dw_weights, 0.8, 1.2)

        self.pointwise     = nn.Linear(K * C_in, C_out, bias=False)
        self.temporal_conv = nn.Conv1d(
            C_out, C_out,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            groups=C_out,
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(num_groups, C_out)
        self.norm          = nn.LayerNorm(C_out)
        self.act           = nn.GELU()
        self.drop          = nn.Dropout(dropout)
        self.residual      = (nn.Linear(C_in, C_out, bias=False)
                              if C_in != C_out else nn.Identity())

    def forward(self, x, A):
        # x: [B, T, N, C_in]   A: [3, N, N]
        B, T, N, C = x.shape
        residual = self.residual(x)

        agg = torch.einsum('knm,btnc->kbtnc', A, x)               # [3, B, T, N, C]
        agg = agg * self.dw_weights.view(3, 1, 1, 1, C)
        h   = agg.permute(1, 2, 3, 0, 4).reshape(B, T, N, 3 * C)  # [B, T, N, 3C]

        h     = self.drop(self.pointwise(h))                        # [B, T, N, C_out]
        C_out = h.shape[-1]
        h_t   = h.permute(0, 2, 3, 1).reshape(B * N, C_out, T)     # [B*N, C_out, T]
        h_t   = self.temporal_norm(self.temporal_conv(h_t))
        h     = h_t.reshape(B, N, C_out, T).permute(0, 3, 1, 2)    # [B, T, N, C_out]

        return self.act(self.norm(h + residual))


# ══════════════════════════════════════════════════════════════════
#  SECTION 2b — JOINT INDICES, GEOMETRIC FEATURES, DROPPATH
# ══════════════════════════════════════════════════════════════════

_THUMB_MCP  = 2;  _THUMB_IP   = 3;  _THUMB_TIP  = 4
_INDEX_MCP  = 5;  _INDEX_PIP  = 6;  _INDEX_TIP  = 8
_MIDDLE_MCP = 9;  _MIDDLE_PIP = 10; _MIDDLE_TIP = 12
_RING_MCP   = 13; _RING_PIP   = 14; _RING_TIP   = 16
_PINKY_MCP  = 17; _PINKY_PIP  = 18; _PINKY_TIP  = 20

N_GEO_FEATURES = 12


class DropPath(nn.Module):
    """Stochastic depth: randomly skip transformer residuals during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * (torch.rand(shape, device=x.device) < keep) / keep


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — ENCODER
#  Produces a temporal sequence of sign features.
#  Output: [B, T, d_model]  — sequence is PRESERVED, not pooled.
#
#  Stage 1 pools this and classifies.
#  Stage 2 feeds this directly into a CTC head — no structural change
#  to the encoder itself, just load these weights and swap the head.
# ══════════════════════════════════════════════════════════════════

class DSGCNEncoder(nn.Module):
    """
    Shared encoder used by all three stages.

    Input : [B, 32, 21, 9]
    Output: [B, 32, d_model]   — full temporal sequence, no pooling

    Stage 1: downstream head attention-pools this and classifies
    Stage 2: downstream CTC head takes this directly
    Stage 3: not involved (pure text stage)
    """
    def __init__(self, in_channels=9, d_model=256,
                 nhead=8, num_transformer_layers=4, dropout=0.1,
                 drop_path_rate=0.1):
        super().__init__()

        A = build_adjacency_matrices(21)
        self.register_buffer('A', A)   # [3, 21, 21] fixed, not learned

        self.input_norm = nn.LayerNorm(in_channels)

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # GCN stack: 64 → 128 → 128 → d_model
        self.gcn1 = DSGCNBlock(64,       128,     temporal_kernel=3, dropout=dropout)
        self.gcn2 = DSGCNBlock(128,      128,     temporal_kernel=3, dropout=dropout)
        self.gcn3 = DSGCNBlock(128,      d_model, temporal_kernel=5, dropout=dropout)

        # Attention pooling over 21 nodes (replaces lossy mean pool)
        self.node_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # Geometric feature branch: 12 hand-shape descriptors → fused with GCN output
        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)

        # Learned positional encoding for the 32-frame temporal axis
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Transformer with stochastic depth (DropPath)
        dp_rates = [drop_path_rate * i / max(num_transformer_layers - 1, 1)
                    for i in range(num_transformer_layers)]
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ))
            self.drop_paths.append(DropPath(dp))
        self.transformer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'transformer' in name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _geo_dist(a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz):
        """12 hand-shape descriptors from xyz positions [B, T, 21, 3]."""
        d = self._geo_dist

        tips = [
            d(xyz[:,:,_THUMB_TIP],  xyz[:,:,_INDEX_TIP]),
            d(xyz[:,:,_INDEX_TIP],  xyz[:,:,_MIDDLE_TIP]),
            d(xyz[:,:,_MIDDLE_TIP], xyz[:,:,_RING_TIP]),
            d(xyz[:,:,_RING_TIP],   xyz[:,:,_PINKY_TIP]),
            d(xyz[:,:,_THUMB_TIP],  xyz[:,:,_PINKY_TIP]),
        ]

        curls = [
            d(xyz[:,:,_THUMB_MCP],  xyz[:,:,_THUMB_TIP])
                / (d(xyz[:,:,_THUMB_MCP],  xyz[:,:,_THUMB_IP])   + 1e-4),
            d(xyz[:,:,_INDEX_MCP],  xyz[:,:,_INDEX_TIP])
                / (d(xyz[:,:,_INDEX_MCP],  xyz[:,:,_INDEX_PIP])  + 1e-4),
            d(xyz[:,:,_MIDDLE_MCP], xyz[:,:,_MIDDLE_TIP])
                / (d(xyz[:,:,_MIDDLE_MCP], xyz[:,:,_MIDDLE_PIP]) + 1e-4),
            d(xyz[:,:,_RING_MCP],   xyz[:,:,_RING_TIP])
                / (d(xyz[:,:,_RING_MCP],   xyz[:,:,_RING_PIP])  + 1e-4),
            d(xyz[:,:,_PINKY_MCP],  xyz[:,:,_PINKY_TIP])
                / (d(xyz[:,:,_PINKY_MCP],  xyz[:,:,_PINKY_PIP]) + 1e-4),
        ]

        cross_idx_mid  = xyz[:,:,_INDEX_TIP,0] - xyz[:,:,_MIDDLE_TIP,0]
        d_thumb_idxmcp = d(xyz[:,:,_THUMB_TIP], xyz[:,:,_INDEX_MCP])

        return torch.stack(tips + curls + [cross_idx_mid, d_thumb_idxmcp], dim=-1)

    def forward(self, x):
        # x: [B, 32, 21, 9]
        xyz = x[:, :, :, :3]

        h = self.input_norm(x)           # [B, 32, 21, 9]
        h = self.input_proj(h)           # [B, 32, 21, 64]
        h = self.gcn1(h, self.A)         # [B, 32, 21, 128]
        h = self.gcn2(h, self.A)         # [B, 32, 21, 128]
        h = self.gcn3(h, self.A)         # [B, 32, 21, d_model]

        # Attention pooling over 21 nodes
        attn = self.node_attn(h).squeeze(-1)          # [B, T, 21]
        attn = F.softmax(attn, dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)       # [B, T, d_model]

        # Fuse geometric features
        geo = self._compute_geo_features(xyz)         # [B, T, 12]
        geo = self.geo_norm(geo)
        h = self.geo_proj(torch.cat([h, geo], dim=-1))  # [B, T, d_model]

        h = h + self.pos_enc

        # Transformer with stochastic depth
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            h = h + dp(layer(h) - h)
        h = self.transformer_norm(h)

        return h                         # Stage 2 loads up to here


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — CLASSIFIER HEAD  (Stage 1 only)
#  Takes the encoder sequence [B, T, d_model] and produces
#  a single class prediction per sample.
#  Stage 2 discards this entirely and replaces it with a CTC head.
# ══════════════════════════════════════════════════════════════════

class ClassifierHead(nn.Module):
    """
    Stage 1 classification head.
    [B, T, d_model] → attention pool → [B, d_model] → [B, num_classes]

    Discarded entirely in Stage 2 — the encoder weights are what matter.
    """
    def __init__(self, d_model=256, num_classes=29, dropout=0.4):
        super().__init__()
        self.frame_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, T, d_model]
        attn = self.frame_attn(x).squeeze(-1)        # [B, T]
        attn = F.softmax(attn, dim=1)
        h = (x * attn.unsqueeze(-1)).sum(dim=1)      # [B, d_model]
        return self.net(h)                            # [B, num_classes]


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — FULL STAGE 1 MODEL
#  Encoder + ClassifierHead wired together.
#  Checkpoint saves encoder and head state dicts separately
#  so Stage 2 can load encoder only without awkward surgery.
# ══════════════════════════════════════════════════════════════════

class SLTStage1(nn.Module):
    def __init__(self, num_classes, in_channels=9, d_model=256,
                 nhead=8, num_transformer_layers=4,
                 dropout=0.1, head_dropout=0.4, drop_path_rate=0.1):
        super().__init__()
        self.encoder = DSGCNEncoder(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.head = ClassifierHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=head_dropout,
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — DATASET
# ══════════════════════════════════════════════════════════════════

class SignDataset(Dataset):
    """
    Loads .npy landmark files of shape [32, 21, 9].

    Cache: first run loads all .npy files and saves a single .pt tensor.
    Subsequent runs load the .pt file directly (~3 sec vs ~2 min).
    Cache is invalidated automatically if label_to_idx changes
    (i.e. when you add new sign classes to the dataset).
    """

    def __init__(self, data_path: str, label_to_idx: dict, cache_path: str = None):
        self.label_to_idx = label_to_idx
        self.num_classes  = len(label_to_idx)
        data_path = Path(data_path)

        if cache_path and Path(cache_path).exists():
            log.info(f"Loading cache: {cache_path}")
            cache = torch.load(cache_path, weights_only=True)
            if cache.get('label_to_idx') == label_to_idx:
                self.data, self.targets = cache['data'], cache['targets']
                log.info(f"  {len(self.targets)} samples | "
                         f"RAM: ~{self.data.nbytes / 1e6:.0f} MB")
                return
            log.warning("Label map changed — rebuilding cache from .npy files.")

        log.info("Building dataset from .npy files (first run, cached after)...")
        data_list, target_list, skipped = [], [], 0
        for fname in sorted(f for f in os.listdir(data_path) if f.endswith('.npy')):
            label = fname.split('_')[0]
            if label not in label_to_idx:
                skipped += 1
                continue
            arr = np.load(data_path / fname).astype(np.float32)
            if arr.shape != (32, 21, 9):
                log.warning(f"  Skipping {fname}: shape {arr.shape}")
                skipped += 1
                continue
            data_list.append(arr)
            target_list.append(label_to_idx[label])

        self.data    = torch.from_numpy(np.stack(data_list))
        self.targets = torch.tensor(target_list, dtype=torch.long)
        log.info(f"  {len(self.targets)} samples loaded | {skipped} skipped")

        if cache_path:
            torch.save({'data': self.data, 'targets': self.targets,
                        'label_to_idx': label_to_idx}, cache_path)
            log.info(f"  Cache saved → {cache_path}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def class_weights(self, temperature: float = 0.5) -> torch.Tensor:
        """
        Temperature-softened inverse-frequency weights for WeightedRandomSampler.

        temperature=1.0  full inverse-frequency (aggressive, caused collapse in run 1)
        temperature=0.5  square root — gentler balance (what we use)
        temperature=0.0  uniform, no rebalancing

        For this dataset: J(832) vs GOODBYE(104) → raw 8x ratio → sqrt = 2.8x.
        Prevents the minority class flood that destabilizes early training.
        """
        counts = Counter(self.targets.tolist())
        raw = torch.tensor(
            [len(self.targets) / counts[t.item()] for t in self.targets],
            dtype=torch.float32,
        )
        return raw ** temperature


def stratified_split(dataset: SignDataset, val_ratio: float = 0.15,
                     seed: int = 42):
    """
    Stratified train/val split with fixed seed for reproducibility.
    Every class appears in validation proportionally — no class gets
    accidentally excluded (unlike random_split on small classes).
    """
    indices_per_class = defaultdict(list)
    for idx, label in enumerate(dataset.targets.tolist()):
        indices_per_class[label].append(idx)

    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for label, idxs in indices_per_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ══════════════════════════════════════════════════════════════════
#  SECTION 6b — ONLINE AUGMENTATION
#  Applied per-batch during training so the model sees different
#  perturbations every epoch, even for the same underlying sample.
#  All three channel groups (xyz, velocity, acceleration) are 3D
#  vectors in the same coordinate frame, so rotation applies to all.
# ══════════════════════════════════════════════════════════════════

def _batch_rotation_matrices(batch_size, max_deg, device):
    """Random rotation matrix per sample. Returns [B, 3, 3]."""
    angles = torch.randn(batch_size, 3, device=device) * max_deg
    rad = angles * (math.pi / 180.0)
    cx, sx = torch.cos(rad[:, 0]), torch.sin(rad[:, 0])
    cy, sy = torch.cos(rad[:, 1]), torch.sin(rad[:, 1])
    cz, sz = torch.cos(rad[:, 2]), torch.sin(rad[:, 2])

    zero = torch.zeros_like(cx)
    one  = torch.ones_like(cx)

    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=1).view(-1, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=1).view(-1, 3, 3)
    return Rx @ Ry @ Rz


def online_augment(x, rotation_deg=10.0, scale_lo=0.85, scale_hi=1.15,
                   noise_std=0.003):
    """
    Fast batched augmentation on GPU.

    x : [B, 32, 21, 9]  (xyz | velocity | acceleration)
    Returns augmented tensor (same shape, new storage).
    """
    B, T, N, C = x.shape
    device = x.device

    R = _batch_rotation_matrices(B, rotation_deg, device)         # [B, 3, 3]
    xr = x.view(B, T, N, C // 3, 3)                              # [B, 32, 21, 3, 3]
    xr = torch.einsum('btngi,bij->btngj', xr, R)                 # single fused op
    x  = xr.reshape(B, T, N, C)

    scale = scale_lo + torch.rand(B, 1, 1, 1, device=device) * (scale_hi - scale_lo)
    return x * scale + torch.randn_like(x) * noise_std


def mixup_batch(x, y, alpha=0.2):
    """
    MixUp: linearly interpolate between random pairs of samples.

    Forces the model to learn smooth decision boundaries instead of
    memorizing individual samples. Returns mixed inputs and both
    sets of labels so the loss can weight them by lambda.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 — SCHEDULER
#  Linear warmup → cosine decay → min_lr. No restarts, ever.
#
#  History: CosineAnnealingWarmRestarts caused the epoch-33 collapse.
#  With T_0=10, T_mult=2, LR spiked back to max at epoch 30,
#  immediately after the model had reached 95.66% accuracy.
#  This scheduler is strictly monotonically decaying after warmup.
# ══════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs,
                 min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / self.warmup_epochs
        else:
            progress = ((e - self.warmup_epochs) /
                        (self.max_epochs - self.warmup_epochs))
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 — EVALUATION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, num_classes, use_amp):
    model.eval()
    total_loss = 0.0
    correct    = torch.zeros(num_classes, dtype=torch.long)
    total      = torch.zeros(num_classes, dtype=torch.long)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits     = model(x)
            total_loss += F.cross_entropy(logits, y).item()
        preds = logits.argmax(dim=1)
        for c in range(num_classes):
            mask       = (y == c)
            correct[c] += (preds[mask] == c).sum().cpu()
            total[c]   += mask.sum().cpu()

    per_class = (correct.float() / total.float().clamp(min=1)).numpy()
    overall   = correct.sum().item() / max(total.sum().item(), 1)
    return {
        "loss":          total_loss / len(loader),
        "acc":           overall * 100,
        "per_class_acc": per_class,
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 9 — CHECKPOINT HELPERS
#  Saves encoder and head state dicts separately.
#  Stage 2 loads only 'encoder_state_dict' — no surgery required.
# ══════════════════════════════════════════════════════════════════

def make_checkpoint(model, optimizer, scheduler, epoch,
                    val_acc, best_acc, trigger_times,
                    label_to_idx, idx_to_label,
                    num_classes, in_channels, d_model,
                    nhead, num_transformer_layers):
    """
    Builds the checkpoint dict.

    Saves encoder and head state dicts separately so Stage 2 can do:
        ckpt = torch.load('best_model.pth')
        encoder.load_state_dict(ckpt['encoder_state_dict'])
    without needing to strip 'encoder.' prefixes from a monolithic dict.
    """
    unwrapped = model.module if hasattr(model, 'module') else model
    return {
        # ── Weights (split for Stage 2 compatibility) ─────────────
        'encoder_state_dict':     unwrapped.encoder.state_dict(),
        'head_state_dict':        unwrapped.head.state_dict(),

        # ── Full model (for Stage 1 resume / inference) ───────────
        'model_state_dict':       unwrapped.state_dict(),

        # ── Training state (for resume) ───────────────────────────
        'optimizer_state_dict':   optimizer.state_dict(),
        'scheduler_state_dict':   scheduler.state_dict(),
        'epoch':                  epoch,
        'best_acc':               best_acc,
        'trigger_times':          trigger_times,

        # ── Architecture (everything Stage 2 needs to reconstruct) ─
        'label_to_idx':           label_to_idx,
        'idx_to_label':           idx_to_label,
        'num_classes':            num_classes,
        'in_channels':            in_channels,
        'd_model':                d_model,
        'nhead':                  nhead,
        'num_transformer_layers': num_transformer_layers,

        # ── Accuracy record ───────────────────────────────────────
        'val_acc':                val_acc,
        'stage':                  1,   # so Stage 2 can assert it loaded Stage 1
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 10 — TRAINING
# ══════════════════════════════════════════════════════════════════

def train(
    # ── Paths ──────────────────────────────────────────────────────
    data_path = '/kaggle/input/datasets/kokoab/005-claude/landmarks/landmarks',
    meta_path = '/kaggle/input/datasets/kokoab/005-claude/dataset_meta.json',
    save_dir  = '/kaggle/working/',

    # ── Training ───────────────────────────────────────────────────
    epochs              = 200,
    batch_size          = 512,     # GPU-preloaded data eliminates CPU bottleneck
    lr                  = 5e-4,    # scaled up with batch_size (128→512)
    weight_decay        = 0.01,
    warmup_epochs       = 5,      # was 10 — reach peak LR faster
    val_ratio           = 0.15,
    label_smoothing     = 0.05,
    grad_clip           = 5.0,
    patience            = 30,
    sampler_temperature = 0.5,

    # ── Model ──────────────────────────────────────────────────────
    in_channels            = 9,   # xyz + velocity + acceleration
    d_model                = 256,
    nhead                  = 8,
    num_transformer_layers = 4,
    dropout                = 0.10,
    head_dropout           = 0.15,
):
    # ── Device ─────────────────────────────────────────────────────
    device  = torch.device('cuda' if torch.cuda.is_available() else
                           'mps'  if torch.backends.mps.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    BEST_CKPT = save_dir / 'best_model.pth'
    LAST_CKPT = save_dir / 'last_checkpoint.pth'

    log.info(f"Device: {device} | AMP: {use_amp}")
    log.info(f"Stage 1 — Isolated Sign Classification")

    # ── Label map ──────────────────────────────────────────────────
    with open(meta_path) as f:
        meta = json.load(f)

    if 'label_to_idx' in meta:
        label_to_idx = meta['label_to_idx']
    else:
        # Alphabetical sort = deterministic mapping across machines
        label_to_idx = {l: i for i, l in
                        enumerate(sorted(meta['label_counts'].keys()))}
        log.info("Built label_to_idx from label_counts (sorted alphabetically)")

    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    log.info(f"Classes ({len(label_to_idx)}): {sorted(label_to_idx.keys())}")

    if 'label_counts' in meta:
        log.info("Class distribution:")
        for label, count in sorted(meta['label_counts'].items()):
            bar = '█' * (count // 20)
            log.info(f"   {label:>10}: {count:5d}  {bar}")

    # ── Dataset ────────────────────────────────────────────────────
    cache_path = str(save_dir / 'ds_cache.pt')
    full_ds    = SignDataset(data_path, label_to_idx, cache_path=cache_path)
    train_ds, val_ds = stratified_split(full_ds, val_ratio, seed=42)
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Compute sample weights while data is still on CPU (avoids GPU sync overhead)
    sample_weights = full_ds.class_weights(temperature=sampler_temperature)[
        train_ds.indices]

    # Preload entire dataset to GPU.
    # 16K × [32, 21, 9] × 4 bytes ≈ 400 MB — fits easily in 15 GB T4 VRAM.
    # Eliminates all CPU→GPU transfers and DataLoader worker overhead.
    on_gpu = False
    if device.type == 'cuda':
        ds_mb = full_ds.data.nelement() * full_ds.data.element_size() / 1e6
        log.info(f"Preloading dataset to GPU ({ds_mb:.0f} MB)...")
        full_ds.data    = full_ds.data.to(device, non_blocking=True)
        full_ds.targets = full_ds.targets.to(device, non_blocking=True)
        on_gpu = True
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)
    log.info(f"WeightedRandomSampler | temperature={sampler_temperature}")

    # num_workers=0 when data is GPU-resident (CUDA tensors can't cross
    # process boundaries). pin_memory is irrelevant for GPU tensors.
    nw = 0 if on_gpu else (2 if device.type == 'cuda' else 0)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=nw, pin_memory=(not on_gpu and use_amp),
        persistent_workers=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=(not on_gpu and use_amp),
        persistent_workers=False,
    )

    # ── Model ──────────────────────────────────────────────────────
    model = SLTStage1(
        num_classes=full_ds.num_classes,
        in_channels=in_channels,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dropout=dropout,
        head_dropout=head_dropout,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch   = 1
    best_acc      = 0.0
    trigger_times = 0

    # ── Auto-resume ────────────────────────────────────────────────
    # Kaggle sessions time out. last_checkpoint.pth saves every epoch
    # so you never lose more than one epoch of training.
    if LAST_CKPT.exists():
        try:
            log.info(f"Found checkpoint — attempting resume from {LAST_CKPT}")
            ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch   = ckpt['epoch'] + 1
            best_acc      = ckpt.get('best_acc', 0.0)
            trigger_times = ckpt.get('trigger_times', 0)
            log.info(f"Resumed from epoch {ckpt['epoch']} | "
                     f"Best so far: {best_acc:.2f}%")
        except Exception as e:
            log.warning(f"Resume failed ({e}). Starting fresh.")

    # DataParallel disabled: model is only 3.5M params (~14 MB).
    # The per-batch cost to replicate weights + gather outputs across
    # GPUs exceeds the compute time at this model size — single GPU
    # with a larger batch is faster. Re-enable for models > 50M params.
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_params = sum(p.numel() for p in
                     (model.module.encoder if hasattr(model, 'module')
                      else model.encoder).parameters())
    log.info(f"Total parameters: {n_params:,} "
             f"(encoder: {enc_params:,} | head: {n_params - enc_params:,})")
    log.info(f"\n{'='*62}")
    log.info(f"  Stage 1 | epochs={epochs} | batch={batch_size} | lr={lr}")
    log.info(f"  Warmup={warmup_epochs} | clip={grad_clip} | patience={patience}")
    log.info(f"  Scheduler: CosineWarmup (NO restarts — they caused epoch-33 collapse)")
    log.info(f"{'='*62}\n")

    history = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss  = 0.0
        train_corr  = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = online_augment(x)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss   = F.cross_entropy(logits, y,
                                         label_smoothing=label_smoothing)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            train_corr  += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ── Validation ─────────────────────────────────────────────
        val_metrics = evaluate(
            model, val_loader, device, full_ds.num_classes, use_amp)
        train_acc = 100 * train_corr / train_total
        val_acc   = val_metrics["acc"]

        log.info(
            f"Epoch {epoch:03d}/{epochs} | LR: {cur_lr:.2e} | "
            f"Train: {train_acc:.1f}% | Val: {val_acc:.2f}% | "
            f"Loss: {epoch_loss / len(train_loader):.4f}"
        )

        # Per-class breakdown every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            log.info("  Per-class validation accuracy:")
            per_class = val_metrics["per_class_acc"]
            struggling = []
            for idx in range(full_ds.num_classes):
                label = idx_to_label[str(idx)]
                acc   = per_class[idx] * 100
                bar   = '█' * int(per_class[idx] * 20)
                flag  = '  <-- needs attention' if acc < 70 else ''
                if acc < 70:
                    struggling.append(label)
                log.info(f"    {label:>10}: {acc:5.1f}%  {bar}{flag}")
            if struggling:
                log.info(f"  Struggling classes: {struggling}")

        history.append({
            "epoch":     epoch,
            "train_acc": round(train_acc, 3),
            "val_acc":   round(val_acc, 3),
            "val_loss":  round(val_metrics["loss"], 5),
            "lr":        round(cur_lr, 8),
        })

        # ── Checkpoints ────────────────────────────────────────────
        ckpt = make_checkpoint(
            model, optimizer, scheduler, epoch,
            val_acc, max(val_acc, best_acc), trigger_times,
            label_to_idx, idx_to_label,
            full_ds.num_classes, in_channels, d_model,
            nhead, num_transformer_layers,
        )

        # Always save last — enables resume after Kaggle timeout
        torch.save(ckpt, LAST_CKPT)

        if val_acc > best_acc:
            best_acc      = val_acc
            trigger_times = 0
            torch.save(ckpt, BEST_CKPT)
            log.info(f"  ✨ New best: {val_acc:.2f}% → best_model.pth")
            log.info(f"     (encoder_state_dict saved separately for Stage 2)")
        else:
            trigger_times += 1
            log.info(f"  No improvement ({trigger_times}/{patience})")
            if trigger_times >= patience:
                log.info(f"  Early stopping at epoch {epoch} | "
                         f"Best: {best_acc:.2f}%")
                break

    # ── Save history ───────────────────────────────────────────────
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    log.info(f"\n{'='*62}")
    log.info(f"  Stage 1 complete.")
    log.info(f"  Best val acc : {best_acc:.2f}%")
    log.info(f"  Best model   : {BEST_CKPT}")
    log.info(f"  Last ckpt    : {LAST_CKPT}")
    log.info(f"")
    log.info(f"  To load encoder in Stage 2:")
    log.info(f"    ckpt = torch.load('best_model.pth')")
    log.info(f"    encoder.load_state_dict(ckpt['encoder_state_dict'])")
    log.info(f"{'='*62}")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train(
        data_path = '/kaggle/input/datasets/kokoab/005-claude/landmarks/landmarks',
        meta_path = '/kaggle/input/datasets/kokoab/005-claude/dataset_meta.json',
        save_dir  = '/kaggle/working',
    )