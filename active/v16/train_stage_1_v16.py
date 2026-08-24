"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v16 — Stage 1 Training (Isolated Sign Classification)      ║
║  Squeezeformer encoder, d_model=256, 5-channel input             ║
║                                                                  ║
║  Usage:                                                          ║
║    python src_v16/train_stage_1_v16.py \                         ║
║      --data_path ASL_landmarks_v16 \                             ║
║      --save_dir models/output_v16 \                              ║
║      --epochs 150 --lr 3e-4 --batch_size 256                    ║
║                                                                  ║
║  Expected training time:                                         ║
║    RTX 5090: ~2 hours | Kaggle P100: ~4 hours                   ║
║    Kaggle Dual T4: ~3.5 hours                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import logging
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add parent directory for model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v16 import SLTStage1V16, make_checkpoint_v16, NUM_NODES, DEFAULT_IN_CHANNELS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class SignDatasetV16(Dataset):
    """Dataset for v16 extracted .npy files.

    Expected file format: [32, 61, 5] float16
    Files are in a flat directory: CLASS_hash.npy

    Split: 70% train / 15% val / 15% test (stratified by sorted file order).
    """
    def __init__(self, data_path, split='train', manifest_path=None):
        self.data_path = data_path
        self.split = split

        # Discover classes from filenames using file manifest if available
        files_by_class = {}
        file_manifest_path = manifest_path.replace('.json', '_files.json') if manifest_path else None

        if file_manifest_path and os.path.exists(file_manifest_path):
            # Use pre-built filename→class mapping (handles multi-underscore classes)
            with open(file_manifest_path) as fm:
                file_manifest = json.load(fm)
            for fname, cls in file_manifest.items():
                fpath = os.path.join(data_path, fname)
                if os.path.exists(fpath):
                    if cls not in files_by_class:
                        files_by_class[cls] = []
                    files_by_class[cls].append(fpath)
        else:
            # Fallback: parse class from filename
            for f in os.listdir(data_path):
                if not f.endswith('.npy'):
                    continue
                cls = f.rsplit('_', 2)[0]
                if cls not in files_by_class:
                    files_by_class[cls] = []
                files_by_class[cls].append(os.path.join(data_path, f))

        # Build label mapping
        classes = sorted(files_by_class.keys())
        self.label_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
        self.num_classes = len(classes)

        # Load manifest if available (for consistent class ordering)
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Detect format: v16 manifest = {class_name: index}, old = {filename: class_name}
            first_val = next(iter(manifest.values()))
            if isinstance(first_val, int):
                # v16 format: class_name → index
                self.label_to_idx = {str(k): int(v) for k, v in manifest.items()}
            else:
                # Old format: filename → class_name
                manifest_classes = sorted(set(manifest.values()))
                self.label_to_idx = {c: i for i, c in enumerate(manifest_classes)}
            self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
            self.num_classes = len(self.label_to_idx)

        # Split files: 70/15/15 with RANDOM SHUFFLE (seed=42 for reproducibility)
        # This ensures all video sources/signers are mixed across splits
        import random as _rng
        self.files = []
        self.targets = []

        for cls in classes:
            cls_files = sorted(files_by_class[cls])
            # Deterministic shuffle so splits are consistent across runs
            _rng.seed(42)
            _rng.shuffle(cls_files)

            n = len(cls_files)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            if split == 'train':
                selected = cls_files[:train_end]
            elif split == 'val':
                selected = cls_files[train_end:val_end]
            elif split == 'test':
                selected = cls_files[val_end:]
            else:
                selected = cls_files  # all

            label = self.label_to_idx.get(cls, -1)
            if label < 0:
                continue

            for fp in selected:
                self.files.append(fp)
                self.targets.append(label)

        self.targets = torch.tensor(self.targets, dtype=torch.long)
        log.info(f"[{split}] {len(self.files)} samples, {self.num_classes} classes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)

        # Handle different extraction formats
        T, N, C = arr.shape

        # Pad 47 nodes → 61 nodes if needed (older extraction)
        if N == 47:
            pad = np.zeros((T, 14, C), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
            N = 61

        # Convert old 10-channel format to v16 5-channel format
        if C == 10:
            # Old format: X(0), Y(1), Z(2)=zero, vel_x(3), vel_y(4), vel_z(5)=zero,
            #             acc_x(6), acc_y(7), acc_z(8)=zero, mask(9)
            # v16 format: X(0), Y(1), Z(2)=perspective, mask(3), palm_scale(4)
            new_arr = np.zeros((T, N, 5), dtype=np.float32)
            new_arr[:, :, 0] = arr[:, :, 0]  # X
            new_arr[:, :, 1] = arr[:, :, 1]  # Y
            # Z: compute perspective from palm scale
            for hs in [0, 21]:
                wrist = arr[:, hs, :2]
                mcp = arr[:, hs + 9, :2]
                palm_len = np.sqrt(((mcp - wrist) ** 2).sum(axis=-1))
                valid = palm_len > 0.01
                if valid.sum() >= 3:
                    ref = np.median(palm_len[valid])
                    if ref > 0.01:
                        for t in range(T):
                            if palm_len[t] > 0.01:
                                new_arr[t, hs:hs+21, 2] = np.log(ref / palm_len[t])
            new_arr[:, :, 3] = arr[:, :, 9]  # mask
            # Palm scale
            for hs in [0, 21]:
                wrist = arr[:, hs, :2]
                mcp = arr[:, hs + 9, :2]
                palm_len = np.sqrt(((mcp - wrist) ** 2).sum(axis=-1))
                valid = palm_len > 0.01
                if valid.sum() >= 3:
                    ref = np.median(palm_len[valid])
                    if ref > 0.01:
                        for t in range(T):
                            ps = palm_len[t] / ref if palm_len[t] > 0.01 else 1.0
                            new_arr[t, hs:hs+21, 4] = ps
                    else:
                        new_arr[:, hs:hs+21, 4] = 1.0
                else:
                    new_arr[:, hs:hs+21, 4] = 1.0
            # Face/body nodes: palm_scale = 1.0 (default)
            new_arr[:, 42:, 4] = 1.0
            arr = new_arr
        elif C == 5:
            pass  # already v16 format
        elif C == 16:
            new_arr = np.zeros((T, N, 5), dtype=np.float32)
            new_arr[:, :, 0] = arr[:, :, 0]
            new_arr[:, :, 1] = arr[:, :, 1]
            new_arr[:, :, 2] = arr[:, :, 2]
            new_arr[:, :, 3] = arr[:, :, 9]
            new_arr[:, :, 4] = 1.0
            arr = new_arr

        # Add velocity and acceleration channels (computed from XY)
        # vel = central difference of XY, acc = central difference of vel
        T2, N2, C2 = arr.shape
        xy = arr[:, :, :2]  # [T, N, 2]
        vel = np.zeros_like(xy)
        vel[1:-1] = (xy[2:] - xy[:-2]) / 2.0
        vel[0] = vel[1]; vel[-1] = vel[-2]
        acc = np.zeros_like(xy)
        acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
        acc[0] = acc[1]; acc[-1] = acc[-2]
        # Append: [T, N, 5] + [T, N, 2] vel + [T, N, 2] acc = [T, N, 9]
        arr = np.concatenate([arr, vel, acc], axis=-1)

        x = torch.from_numpy(arr)

        # Ensure correct temporal length
        if x.shape[0] != 32:
            x = F.interpolate(
                x.permute(2, 1, 0).unsqueeze(0),
                size=32, mode='nearest'
            ).squeeze(0).permute(2, 1, 0)

        # Validate shape (9ch after vel/acc appended)
        assert x.shape == (32, 61, 9), \
            f"Bad shape {x.shape} from {self.files[idx]} (raw was {np.load(self.files[idx]).shape})"

        return x, self.targets[idx]


# ═══════════════════════════════════════════════════════════
# Augmentation
# ═══════════════════════════════════════════════════════════

def online_augment(x):
    """Apply augmentation to a batch. x: [B, T, V, C]"""
    B, T, V, C = x.shape

    # Random horizontal flip with hand swap (p=0.5)
    if torch.rand(1, device=x.device) < 0.5:
        x[..., 0] = -x[..., 0]  # flip X
        # Swap left hand (0-20) and right hand (21-41)
        x_swapped = x.clone()
        x_swapped[:, :, 0:21] = x[:, :, 21:42]
        x_swapped[:, :, 21:42] = x[:, :, 0:21]
        x = x_swapped

    # Independent X/Y scale (simulates aspect-ratio variation: portrait vs landscape)
    # Base scale 0.85-1.15 per axis, plus 30% chance of portrait-like distortion
    scale_x = (0.85 + torch.rand(B, 1, 1, 1, device=x.device) * 0.3)
    scale_y = (0.85 + torch.rand(B, 1, 1, 1, device=x.device) * 0.3)
    # 30% chance to simulate portrait recording (compress X, stretch Y)
    portrait_mask = (torch.rand(B, 1, 1, 1, device=x.device) < 0.3).float()
    scale_x = scale_x * (1.0 - portrait_mask * 0.25)   # compress X up to 25%
    scale_y = scale_y * (1.0 + portrait_mask * 0.25)   # stretch Y up to 25%
    x[..., 0:1] = x[..., 0:1] * scale_x
    x[..., 1:2] = x[..., 1:2] * scale_y
    # Scale velocity (ch5,6) and acceleration (ch7,8) if present — they're derivatives of XY
    if C > 5:
        x[..., 5:6] = x[..., 5:6] * scale_x
        x[..., 6:7] = x[..., 6:7] * scale_y
    if C > 7:
        x[..., 7:8] = x[..., 7:8] * scale_x
        x[..., 8:9] = x[..., 8:9] * scale_y

    # Wider-framing simulation (p=0.2): shrink hands relative to body
    # Mimics recordings where signer is farther from camera
    if torch.rand(1, device=x.device) < 0.2:
        zoom = 0.6 + torch.rand(1, device=x.device).item() * 0.3  # 0.6-0.9
        x[..., :42, :2] = x[..., :42, :2] * zoom
        if C > 5:
            x[..., :42, 5:7] = x[..., :42, 5:7] * zoom
        if C > 7:
            x[..., :42, 7:9] = x[..., :42, 7:9] * zoom

    # Random rotation (±15 degrees)
    angle = (torch.rand(B, device=x.device) - 0.5) * 30 * np.pi / 180
    cos_a = torch.cos(angle).view(B, 1, 1)
    sin_a = torch.sin(angle).view(B, 1, 1)
    x_rot = x[..., 0] * cos_a - x[..., 1] * sin_a
    y_rot = x[..., 0] * sin_a + x[..., 1] * cos_a
    x[..., 0] = x_rot
    x[..., 1] = y_rot

    # Random temporal speed perturbation (0.8x-1.2x)
    if torch.rand(1, device=x.device) < 0.5:
        rate = 0.8 + torch.rand(1).item() * 0.4
        new_T = max(4, int(T * rate))
        vc = x.reshape(B, T, V * C).permute(0, 2, 1)  # [B, V*C, T]
        vc = F.interpolate(vc, size=new_T, mode='nearest')
        vc = F.interpolate(vc, size=T, mode='nearest')
        x = vc.permute(0, 2, 1).reshape(B, T, V, C)

    # Gaussian noise on spatial channels (X, Y, Z)
    if torch.rand(1, device=x.device) < 0.5:
        noise = torch.randn_like(x[..., :3]) * 0.01
        x[..., :3] = x[..., :3] + noise

    # Joint dropout (ST-GCN++ inspired): zero 10-20% of joints per sample
    # Forces distributed representation, improves robustness to missing landmarks
    if torch.rand(1, device=x.device) < 0.3:
        # Random number of joints to drop per sample (10-20% of 61 = 6-12 joints)
        n_drop = torch.randint(6, 13, (B,), device=x.device)
        for b in range(B):
            drop_idx = torch.randperm(V, device=x.device)[:n_drop[b]]
            x[b, :, drop_idx, :] = 0.0

    # Skeleton retargeting: vary hand proportions to simulate different signers
    if torch.rand(1, device=x.device) < 0.3:
        for hs in [0, 21]:  # each hand
            finger_scales = 0.85 + torch.rand(5, device=x.device) * 0.3
            finger_ranges = [(1, 5), (5, 9), (9, 13), (13, 17), (17, 20)]
            wrist_xy = x[:, :, hs, :2].clone()
            for fi, (start, end) in enumerate(finger_ranges):
                for node in range(start, end):
                    disp = x[:, :, hs + node, :2] - wrist_xy
                    x[:, :, hs + node, :2] = wrist_xy + disp * finger_scales[fi]

    return x


# ═══════════════════════════════════════════════════════════
# ASAM (Adaptive Sharpness-Aware Minimization)
# ═══════════════════════════════════════════════════════════

class ASAM:
    """Adaptive SAM — seeks flat minima for better generalization.
    Wraps an existing optimizer. Call ascent_step() then descent_step() per batch."""

    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = {}

    @torch.no_grad()
    def ascent_step(self):
        grads = []
        params = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            # Adaptive: scale by parameter magnitude
            scale = torch.abs(p.data).clamp(min=1e-7)
            grads.append((p.grad * scale).reshape(-1))
            params.append(p)
        grad_norm = torch.cat(grads).norm()
        for p in params:
            scale = torch.abs(p.data).clamp(min=1e-7)
            e_w = p.grad * scale * scale * self.rho / (grad_norm + 1e-12)
            p.data.add_(e_w)
            self.state[p] = e_w

    @torch.no_grad()
    def descent_step(self):
        for p in self.model.parameters():
            if p in self.state:
                p.data.sub_(self.state[p])
        self.optimizer.step()
        self.state = {}


# ═══════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    if v.dtype in (torch.long, torch.int, torch.int64):
                        self.shadow[k] = v.clone()  # skip EMA for integer params
                    else:
                        self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, use_amp=True, in_channels=9):
    model.eval()
    correct_1 = correct_5 = total = 0
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if x.shape[-1] > in_channels:
            x = x[..., :in_channels]
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct_1 += (preds == y).sum().item()
        _, top5 = logits.topk(5, dim=1)
        correct_5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)

    return {
        'loss': total_loss / len(loader),
        'acc': correct_1 / max(total, 1) * 100,
        'top5': correct_5 / max(total, 1) * 100,
    }


# ═══════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════

def train_stage1_v16(
    data_path,
    save_dir='models/output_v16',
    manifest_path='models/manifest.json',
    epochs=100,
    batch_size=256,
    accum_steps=2,
    lr=3e-4,
    weight_decay=0.01,
    label_smoothing=0.15,
    patience=30,
    in_channels=9,
    dim=256,
    depth=4,
    dropout=0.1,
    drop_path=0.0,
    head_dropout=0.3,
    use_pairwise=False,
    use_angles=False,
    teacher_ckpt=None,
    distill_temp=3.0,
    distill_alpha=0.7,
    clean_labels=False,
    clean_threshold=0.95,
    smoke_test=False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    log.info(f"Device: {device}, AMP: {use_amp}")

    os.makedirs(save_dir, exist_ok=True)

    # ── Data ──
    train_ds = SignDatasetV16(data_path, split='train', manifest_path=manifest_path)

    # ── Label Cleaning ──
    if clean_labels:
        log.info("Label cleaning: training quick model to find mislabeled samples...")
        from model_v16 import SLTStage1V16 as _CleanModel
        _cm = _CleanModel(num_classes=train_ds.num_classes, in_channels=DEFAULT_IN_CHANNELS,
                          dim=128, depth=2).to(device)  # small fast model
        _opt = torch.optim.AdamW(_cm.parameters(), lr=1e-3)
        _loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        # Quick 5-epoch train
        for _ep in range(5):
            _cm.train()
            for _x, _y in _loader:
                _x, _y = _x.to(device), _y.to(device)
                _loss = F.cross_entropy(_cm(_x), _y)
                _loss.backward(); _opt.step(); _opt.zero_grad()
        # Find mislabeled: samples where model is confident the label is WRONG
        _cm.eval()
        _clean_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
        bad_indices = []
        idx_offset = 0
        with torch.no_grad():
            for _x, _y in _clean_loader:
                _x, _y = _x.to(device), _y.to(device)
                probs = F.softmax(_cm(_x), dim=-1)
                pred_conf, pred_cls = probs.max(dim=-1)
                for i in range(len(_y)):
                    # Model is confident (>threshold) AND predicts a DIFFERENT class
                    if pred_conf[i] > clean_threshold and pred_cls[i] != _y[i]:
                        bad_indices.append(idx_offset + i)
                idx_offset += len(_y)
        del _cm, _opt
        if bad_indices:
            log.info(f"  Found {len(bad_indices)} likely mislabeled samples ({len(bad_indices)/len(train_ds)*100:.1f}%)")
            # Remove bad samples
            keep = set(range(len(train_ds.files))) - set(bad_indices)
            train_ds.files = [train_ds.files[i] for i in sorted(keep)]
            train_ds.targets = torch.tensor([train_ds.targets[i].item() for i in sorted(keep)])
            log.info(f"  Cleaned: {len(train_ds)} samples remaining")
        else:
            log.info(f"  No mislabeled samples found (threshold={clean_threshold})")
    val_ds = SignDatasetV16(data_path, split='val', manifest_path=manifest_path)

    if smoke_test:
        # Use small subset for testing
        train_ds.files = train_ds.files[:500]
        train_ds.targets = train_ds.targets[:500]
        val_ds.files = val_ds.files[:100]
        val_ds.targets = val_ds.targets[:100]
        epochs = 3
        batch_size = min(batch_size, 32)
        log.info(f"SMOKE TEST: {len(train_ds.files)} train, {len(val_ds.files)} val, batch={batch_size}")

    num_classes = train_ds.num_classes

    # Weighted sampler for class imbalance
    class_counts = Counter(train_ds.targets.numpy().tolist())
    weights = [1.0 / max(class_counts[t], 1) for t in train_ds.targets.numpy().tolist()]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    num_workers = 0 if smoke_test else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=use_amp,
                              persistent_workers=(num_workers > 0),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_amp,
                            persistent_workers=(num_workers > 0))

    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {num_classes}")
    log.info(f"Architecture: Squeezeformer d={dim} depth={depth} in_ch={in_channels}")

    # ── Model ──
    model = SLTStage1V16(
        num_classes=num_classes,
        in_channels=in_channels,
        dim=dim,
        depth=depth,
        dropout=dropout,
        head_dropout=head_dropout,
        drop_path=drop_path,
        use_pairwise=use_pairwise,
        use_angles=use_angles,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    log.info(f"Features: pairwise={use_pairwise}, angles={use_angles}")
    log.info(f"DropPath: {drop_path}")
    log.info(f"Parameters: {params:,}")

    # ── Teacher model for self-distillation ──
    teacher = None
    if teacher_ckpt:
        log.info(f"Loading teacher from {teacher_ckpt}")
        t_ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        teacher = SLTStage1V16(
            num_classes=t_ckpt['num_classes'],
            in_channels=t_ckpt.get('in_channels', in_channels),
            dim=t_ckpt.get('d_model', dim),
            depth=t_ckpt.get('depth', depth),
            use_pairwise=t_ckpt.get('use_pairwise', use_pairwise),
            use_angles=t_ckpt.get('use_angles', use_angles),
        ).to(device)
        t_sd = t_ckpt.get('ema_shadow', t_ckpt['model_state_dict'])
        t_sd = {k.replace('_orig_mod.', ''): v for k, v in t_sd.items()}
        teacher.load_state_dict(t_sd, strict=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        log.info(f"  Teacher: d={t_ckpt.get('d_model', dim)}, acc={t_ckpt.get('best_acc', '?')}%")
        log.info(f"  Distillation: T={distill_temp}, alpha={distill_alpha}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=0.999)

    # ── Training ──
    best_acc = 0.0
    trigger_times = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        t0 = time.time()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Truncate channels if needed (e.g., 9ch dataset → 5ch model)
            if x.shape[-1] > in_channels:
                x = x[..., :in_channels]

            # Augment
            x = online_augment(x)

            if x.shape[2] != 61:
                log.error(f"SHAPE BUG: x.shape={x.shape} after augment — expected V=61")
                continue

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)

                if teacher is not None:
                    # Self-distillation: soft labels from teacher
                    with torch.no_grad():
                        t_logits = teacher(x)
                    # KL divergence on softened probabilities
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / distill_temp, dim=-1),
                        F.softmax(t_logits / distill_temp, dim=-1),
                        reduction='batchmean',
                    ) * (distill_temp ** 2)
                    ce_loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                    loss = distill_alpha * kd_loss + (1 - distill_alpha) * ce_loss
                else:
                    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            epoch_loss += loss.item() * accum_steps

        scheduler.step()

        # ── Evaluate with EMA weights ──
        ema.apply(model)
        val_metrics = evaluate(model, val_loader, device, use_amp, in_channels)
        ema.restore(model)

        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={epoch_loss/len(train_loader):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.2f}% | "
            f"top5={val_metrics['top5']:.2f}% | "
            f"lr={current_lr:.6f} | "
            f"{epoch_time:.0f}s"
        )

        # History
        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_top5': val_metrics['top5'],
            'lr': current_lr,
        })

        # ── Checkpoint ──
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            trigger_times = 0

            ckpt = make_checkpoint_v16(
                model, optimizer, scheduler, ema,
                epoch=epoch,
                metric_value=val_metrics['acc'],
                best_metric=best_acc,
                trigger_times=trigger_times,
                label_maps={
                    'label_to_idx': train_ds.label_to_idx,
                    'idx_to_label': train_ds.idx_to_label,
                },
                vocab_size=num_classes,
                dim=dim,
                depth=depth,
                in_channels=in_channels,
                stage=1,
                use_angles=use_angles,
                use_pairwise=use_pairwise,
            )

            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(ckpt, save_path)
            log.info(f"  ★ Best model saved: {val_metrics['acc']:.2f}%")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                log.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        # Save history
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    log.info(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")

    # ── Final Test Evaluation ──
    log.info("\n" + "="*60)
    log.info("  FINAL TEST SET EVALUATION")
    log.info("="*60)

    # Load best checkpoint
    best_ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device, weights_only=False)
    best_sd = best_ckpt.get('ema_shadow') or best_ckpt.get('model_state_dict')
    model.load_state_dict(best_sd, strict=False)

    test_ds = SignDatasetV16(data_path, split='test', manifest_path=manifest_path)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_amp,
                             persistent_workers=(num_workers > 0))

    from sklearn.metrics import precision_score, recall_score, f1_score

    eval_results = {}
    for split_name, loader in [('val', val_loader), ('test', test_loader)]:
        model.eval()
        all_preds, all_labels = [], []
        correct_1 = correct_5 = total = 0
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if x.shape[-1] > in_channels:
                    x = x[..., :in_channels]
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                total_loss += loss.item()
                preds = logits.argmax(1)
                correct_1 += (preds == y).sum().item()
                _, t5 = logits.topk(5, dim=1)
                correct_5 += (t5 == y.unsqueeze(1)).any(1).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())

        acc = correct_1 / max(total, 1) * 100
        top5 = correct_5 / max(total, 1) * 100
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

        eval_results[split_name] = {
            'accuracy': round(acc, 2), 'top5_accuracy': round(top5, 2),
            'precision_weighted': round(prec, 2), 'recall_weighted': round(rec, 2),
            'f1_weighted': round(f1, 2), 'loss': round(total_loss / len(loader), 4),
            'total_samples': total,
        }

        log.info(f"  [{split_name.upper():5s}] {total:5d} samples | "
                 f"Acc={acc:.2f}% | Top5={top5:.2f}% | "
                 f"P={prec:.2f}% R={rec:.2f}% F1={f1:.2f}%")

    # Save eval results
    eval_path = os.path.join(save_dir, 'eval_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    log.info(f"  Saved: {eval_path}")

    return best_acc


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLT v16 Stage 1 Training')
    parser.add_argument('--data_path', default='ASL_landmarks_v16',
                        help='Path to v16 extracted .npy directory')
    parser.add_argument('--save_dir', default='models/output_v16',
                        help='Output directory for checkpoints')
    parser.add_argument('--manifest', default='models/manifest.json',
                        help='Class manifest JSON')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--accum_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--in_channels', type=int, default=9,
                        help='Input channels (5=raw XYZ+mask+palm, 9=+vel/acc)')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--drop_path', type=float, default=0.0,
                        help='DropPath rate (stochastic depth, linearly increasing per block)')
    parser.add_argument('--use_pairwise', action='store_true',
                        help='Add curated pairwise hand distances (68 features)')
    parser.add_argument('--use_angles', action='store_true',
                        help='Add signer-invariant angle features (110 features)')
    parser.add_argument('--teacher_ckpt', default=None,
                        help='Path to teacher checkpoint for self-distillation')
    parser.add_argument('--distill_temp', type=float, default=3.0,
                        help='Distillation temperature (default 3.0)')
    parser.add_argument('--distill_alpha', type=float, default=0.7,
                        help='Weight for distillation loss vs CE (default 0.7)')
    parser.add_argument('--clean_labels', action='store_true',
                        help='Run label cleaning before training (removes likely mislabeled samples)')
    parser.add_argument('--clean_threshold', type=float, default=0.95,
                        help='Confidence threshold for label cleaning (default 0.95)')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test (3 epochs, small data)')
    args = parser.parse_args()

    train_stage1_v16(
        data_path=args.data_path,
        save_dir=args.save_dir,
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        in_channels=args.in_channels,
        dim=args.dim,
        depth=args.depth,
        drop_path=args.drop_path,
        use_pairwise=args.use_pairwise,
        use_angles=args.use_angles,
        teacher_ckpt=args.teacher_ckpt,
        distill_temp=args.distill_temp,
        distill_alpha=args.distill_alpha,
        clean_labels=args.clean_labels,
        clean_threshold=args.clean_threshold,
        smoke_test=args.smoke,
    )
