"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT Stage 2 — Continuous Sign Recognition (CTC)                ║
║  DS-GCN Encoder (transferred from Stage 1) + CTC Head          ║
║  Input : [B, T, 21, 9]  variable-length landmark sequences      ║
║  Output: gloss sequences via CTC decoding                       ║
╠══════════════════════════════════════════════════════════════════╣
║  PREREQUISITES                                                   ║
║  1. Stage 1 training complete → best_model.pth                  ║
║  2. Isolated sign data (synthetic mode) OR                      ║
║     Continuous signing videos with gloss annotations            ║
╠══════════════════════════════════════════════════════════════════╣
║  DATA MODES                                                      ║
║  Synthetic: concatenates isolated 32-frame clips into sequences  ║
║    of 2-8 glosses. Good for bootstrapping, no new data needed.  ║
║  Real: loads variable-length [T, 21, 9] sequences with gloss    ║
║    annotations from JSON.  Use when you have continuous data.   ║
╠══════════════════════════════════════════════════════════════════╣
║  CTC VOCABULARY                                                  ║
║  Index 0 = CTC blank token                                      ║
║  Index 1..num_classes = gloss labels (Stage 1 index + 1)        ║
║  The model outputs [B, T, num_classes+1] log-probabilities.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import json
import math
import logging
from pathlib import Path
from collections import defaultdict

from train_kaggle import (
    DSGCNBlock,
    build_adjacency_matrices,
    DropPath,
    CosineWarmupScheduler,
    online_augment,
    N_GEO_FEATURES,
    _THUMB_MCP, _THUMB_IP, _THUMB_TIP,
    _INDEX_MCP, _INDEX_PIP, _INDEX_TIP,
    _MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP,
    _RING_MCP, _RING_PIP, _RING_TIP,
    _PINKY_MCP, _PINKY_PIP, _PINKY_TIP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-S2")


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — SINUSOIDAL POSITIONAL ENCODING
#  Replaces the learned 32-frame PE from Stage 1.
#  Supports arbitrary sequence lengths without retraining.
# ══════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — STAGE 2 ENCODER
#  Architecturally identical to Stage 1's DSGCNEncoder except:
#   - Sinusoidal PE (variable-length sequences)
#   - Accepts src_key_padding_mask for the transformer
#  Loads GCN, attention, and transformer weights from Stage 1.
# ══════════════════════════════════════════════════════════════════

class Stage2Encoder(nn.Module):
    def __init__(self, in_channels=9, d_model=256,
                 nhead=8, num_transformer_layers=4, dropout=0.1,
                 drop_path_rate=0.1):
        super().__init__()

        A = build_adjacency_matrices(21)
        self.register_buffer('A', A)

        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        self.gcn1 = DSGCNBlock(64,  128,     temporal_kernel=3, dropout=dropout)
        self.gcn2 = DSGCNBlock(128, 128,     temporal_kernel=3, dropout=dropout)
        self.gcn3 = DSGCNBlock(128, d_model, temporal_kernel=5, dropout=dropout)

        self.node_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)

        self.pos_enc = SinusoidalPE(d_model)

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

    @staticmethod
    def _geo_dist(a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz):
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

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, T, 21, 9] — T is variable
        xyz = x[:, :, :, :3]

        h = self.input_norm(x)
        h = self.input_proj(h)
        h = self.gcn1(h, self.A)
        h = self.gcn2(h, self.A)
        h = self.gcn3(h, self.A)

        attn = self.node_attn(h).squeeze(-1)
        attn = F.softmax(attn, dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)

        geo = self._compute_geo_features(xyz)
        geo = self.geo_norm(geo)
        h = self.geo_proj(torch.cat([h, geo], dim=-1))

        h = self.pos_enc(h)

        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            h = h + dp(layer(h, src_key_padding_mask=src_key_padding_mask) - h)
        h = self.transformer_norm(h)

        return h    # [B, T, d_model]

    def load_stage1_weights(self, ckpt_path):
        """Load compatible weights from Stage 1 encoder checkpoint."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        s1_state = ckpt['encoder_state_dict']
        own_state = self.state_dict()

        loaded, skipped = [], []
        for name, param in s1_state.items():
            if name == 'pos_enc':
                skipped.append(name)
                continue
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name] = param
                loaded.append(name)
            else:
                skipped.append(name)

        self.load_state_dict(own_state)
        log.info(f"Stage 1 weights: {len(loaded)} loaded, "
                 f"{len(skipped)} skipped ({skipped})")
        return loaded, skipped


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — CTC HEAD + FULL MODEL
# ══════════════════════════════════════════════════════════════════

class SLTStage2(nn.Module):
    """
    Full Stage 2 model: encoder + CTC projection.

    vocab_size = number of gloss classes (29).
    Output has vocab_size + 1 channels (index 0 = CTC blank).
    """
    def __init__(self, vocab_size, in_channels=9, d_model=256,
                 nhead=8, num_transformer_layers=4,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.encoder = Stage2Encoder(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.ctc_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size + 1),   # +1 for CTC blank
        )
        nn.init.trunc_normal_(self.ctc_proj[1].weight, std=0.02)
        nn.init.zeros_(self.ctc_proj[1].bias)

    def forward(self, x, src_key_padding_mask=None):
        h = self.encoder(x, src_key_padding_mask)   # [B, T, d_model]
        return self.ctc_proj(h)                      # [B, T, vocab_size+1]


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — SYNTHETIC CONTINUOUS DATASET
#  Concatenates isolated 32-frame sign clips into sequences.
#  Each synthetic sequence contains 2-8 randomly chosen glosses.
#  No new data collection needed — uses Stage 1 isolated data.
# ══════════════════════════════════════════════════════════════════

class SyntheticContinuousDataset(Dataset):
    """
    Generates synthetic continuous sequences by concatenating isolated clips.

    Each sample:
      landmarks: [T, 21, 9]  where T = num_signs × 32
      targets:   [num_signs]  gloss indices (1-indexed; 0 = CTC blank)

    GPU preloading: call .to_device(device) after construction to move
    all clips to GPU. Eliminates CPU→GPU transfer every batch.
    """
    def __init__(self, data_path, label_to_idx, cache_path=None,
                 min_signs=2, max_signs=5, samples_per_epoch=4000,
                 shared_clips=None):
        self.min_signs = min_signs
        self.max_signs = max_signs
        self.samples_per_epoch = samples_per_epoch
        self.label_to_idx = label_to_idx
        self.num_classes = len(label_to_idx)

        if shared_clips is not None:
            self.clips_by_class = shared_clips
            return

        data_path = Path(data_path)
        self.clips_by_class = defaultdict(list)

        if cache_path and Path(cache_path).exists():
            log.info(f"Loading isolated sign cache: {cache_path}")
            cache = torch.load(cache_path, weights_only=True)
            if cache.get('label_to_idx') == label_to_idx:
                data, targets = cache['data'], cache['targets']
                for i in range(len(targets)):
                    self.clips_by_class[targets[i].item()].append(data[i])
                log.info(f"  {len(targets)} clips across {len(self.clips_by_class)} classes")
                return

        log.info("Loading isolated sign clips from .npy files...")
        for fname in sorted(f for f in os.listdir(data_path) if f.endswith('.npy')):
            label = fname.split('_')[0]
            if label not in label_to_idx:
                continue
            arr = np.load(data_path / fname).astype(np.float32)
            if arr.shape != (32, 21, 9):
                continue
            cls_idx = label_to_idx[label]
            self.clips_by_class[cls_idx].append(torch.from_numpy(arr))
        total = sum(len(v) for v in self.clips_by_class.values())
        log.info(f"  {total} clips across {len(self.clips_by_class)} classes")

    def to_device(self, device):
        """Move all clips to GPU. Call once after construction."""
        for cls_idx in self.clips_by_class:
            self.clips_by_class[cls_idx] = [
                c.to(device) for c in self.clips_by_class[cls_idx]]
        return self

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        n_signs = torch.randint(self.min_signs, self.max_signs + 1, (1,)).item()
        classes = torch.randint(0, self.num_classes, (n_signs,))

        clips = []
        targets = []
        for cls_idx in classes.tolist():
            pool = self.clips_by_class[cls_idx]
            clip = pool[torch.randint(0, len(pool), (1,)).item()]
            clips.append(clip)
            targets.append(cls_idx + 1)   # +1 because CTC blank = 0

        landmarks = torch.cat(clips, dim=0)               # [T, 21, 9]
        targets = torch.tensor(targets, dtype=torch.long,
                               device=landmarks.device)    # [num_signs]
        return landmarks, targets


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — REAL CONTINUOUS DATASET (for actual continuous data)
#  Expected format:
#    sequences/seq_001.npy  →  [T, 21, 9]  variable-length
#    annotations.json       →  {"seq_001": ["HELLO", "M", "Y", ...]}
# ══════════════════════════════════════════════════════════════════

class ContinuousSignDataset(Dataset):
    """Loads real continuous signing sequences with gloss annotations."""

    def __init__(self, seq_dir, annotation_path, label_to_idx):
        self.label_to_idx = label_to_idx
        seq_dir = Path(seq_dir)

        with open(annotation_path) as f:
            annotations = json.load(f)

        self.samples = []
        skipped = 0
        for seq_name, glosses in annotations.items():
            npy_path = seq_dir / f"{seq_name}.npy"
            if not npy_path.exists():
                skipped += 1
                continue
            valid = all(g in label_to_idx for g in glosses)
            if not valid:
                skipped += 1
                continue
            self.samples.append((npy_path, glosses))

        log.info(f"Continuous dataset: {len(self.samples)} sequences, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, glosses = self.samples[idx]
        landmarks = torch.from_numpy(
            np.load(npy_path).astype(np.float32))        # [T, 21, 9]
        targets = torch.tensor(
            [self.label_to_idx[g] + 1 for g in glosses],  # +1 for CTC blank
            dtype=torch.long)
        return landmarks, targets


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — COLLATE, DECODING, METRICS
# ══════════════════════════════════════════════════════════════════

def ctc_collate(batch):
    """Pad variable-length sequences and build CTC inputs."""
    landmarks_list, targets_list = zip(*batch)

    input_lengths = torch.tensor([lm.size(0) for lm in landmarks_list],
                                 dtype=torch.long)
    landmarks = pad_sequence(landmarks_list, batch_first=True)  # [B, T_max, 21, 9]

    target_lengths = torch.tensor([t.size(0) for t in targets_list],
                                  dtype=torch.long)
    targets = torch.cat(targets_list)  # [sum(target_lengths)]

    # True where padded — used by transformer to ignore padding
    T_max = landmarks.size(1)
    padding_mask = torch.arange(T_max).unsqueeze(0) >= input_lengths.unsqueeze(1)

    return landmarks, targets, input_lengths, target_lengths, padding_mask


def ctc_greedy_decode(log_probs, input_lengths):
    """
    Greedy CTC decoding: argmax per frame, collapse repeats, remove blanks.

    log_probs:     [B, T, C]
    input_lengths: [B]
    Returns: list of decoded sequences (list of int lists)
    """
    preds = log_probs.argmax(dim=-1)  # [B, T]
    decoded = []
    for i in range(preds.size(0)):
        seq = preds[i, :input_lengths[i]].tolist()
        collapsed = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:
                collapsed.append(idx)
            prev = idx
        decoded.append(collapsed)
    return decoded


def edit_distance(hyp, ref):
    """Levenshtein distance between two integer sequences."""
    m, n = len(hyp), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(hypotheses, references):
    """Word Error Rate: total edit distance / total reference length."""
    total_edits = sum(edit_distance(h, r) for h, r in zip(hypotheses, references))
    total_words = sum(len(r) for r in references)
    return total_edits / max(total_words, 1)


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 — CHECKPOINT
# ══════════════════════════════════════════════════════════════════

def make_checkpoint(model, optimizer, scheduler, epoch,
                    wer_val, best_wer, trigger_times,
                    label_to_idx, idx_to_label,
                    vocab_size, in_channels, d_model,
                    nhead, num_transformer_layers):
    unwrapped = model.module if hasattr(model, 'module') else model
    return {
        'encoder_state_dict':   unwrapped.encoder.state_dict(),
        'ctc_proj_state_dict':  unwrapped.ctc_proj.state_dict(),
        'model_state_dict':     unwrapped.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch':                epoch,
        'best_wer':             best_wer,
        'trigger_times':        trigger_times,
        'label_to_idx':         label_to_idx,
        'idx_to_label':         idx_to_label,
        'vocab_size':           vocab_size,
        'in_channels':          in_channels,
        'd_model':              d_model,
        'nhead':                nhead,
        'num_transformer_layers': num_transformer_layers,
        'stage':                2,
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 — TRAINING
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, use_amp, idx_to_label):
    model.eval()
    all_hyps, all_refs = [], []
    total_loss = 0.0
    n_batches = 0

    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    for landmarks, targets, input_lengths, target_lengths, padding_mask in loader:
        landmarks    = landmarks.to(device)
        targets      = targets.to(device)
        input_lengths  = input_lengths.to(device)
        target_lengths = target_lengths.to(device)
        padding_mask   = padding_mask.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(landmarks, src_key_padding_mask=padding_mask)
            log_probs = F.log_softmax(logits, dim=-1)
            # CTC loss expects [T, B, C]
            loss = ctc_loss_fn(
                log_probs.permute(1, 0, 2),
                targets, input_lengths, target_lengths,
            )
        total_loss += loss.item()
        n_batches += 1

        decoded = ctc_greedy_decode(log_probs.float(), input_lengths.cpu())
        all_hyps.extend(decoded)

        offset = 0
        for tl in target_lengths:
            all_refs.append(targets[offset:offset + tl].cpu().tolist())
            offset += tl

    avg_loss = total_loss / max(n_batches, 1)
    wer = compute_wer(all_hyps, all_refs)
    seq_acc = sum(1 for h, r in zip(all_hyps, all_refs) if h == r) / max(len(all_hyps), 1)

    return avg_loss, wer, seq_acc, all_hyps, all_refs


def train(
    # ── Paths ──────────────────────────────────────────────────────
    data_path   = '/kaggle/input/datasets/kokoab/005-claude/landmarks/landmarks',
    meta_path   = '/kaggle/input/datasets/kokoab/005-claude/dataset_meta.json',
    stage1_ckpt = '/kaggle/input/stage1-model/best_model.pth',
    save_dir    = '/kaggle/working/',

    # ── Synthetic data ────────────────────────────────────────────
    min_signs        = 2,
    max_signs        = 5,
    train_samples    = 4000,
    val_samples      = 600,

    # ── Training ──────────────────────────────────────────────────
    epochs              = 100,
    batch_size          = 32,
    lr                  = 1e-4,
    weight_decay        = 0.01,
    warmup_epochs       = 5,
    grad_clip           = 5.0,
    patience            = 25,

    # ── Model ─────────────────────────────────────────────────────
    in_channels            = 9,
    d_model                = 256,
    nhead                  = 8,
    num_transformer_layers = 4,
    dropout                = 0.10,
    drop_path_rate         = 0.1,

    # ── Data mode ─────────────────────────────────────────────────
    use_real_data    = False,
    real_seq_dir     = None,
    real_annot_path  = None,
):
    device  = torch.device('cuda' if torch.cuda.is_available() else
                           'mps'  if torch.backends.mps.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    BEST_CKPT = save_dir / 'ctc_model.pth'
    LAST_CKPT = save_dir / 'ctc_last_checkpoint.pth'

    log.info(f"Device: {device} | AMP: {use_amp}")
    log.info(f"Stage 2 — Continuous Sign Recognition (CTC)")

    # ── Label map (from Stage 1 meta) ─────────────────────────────
    with open(meta_path) as f:
        meta = json.load(f)

    if 'label_to_idx' in meta:
        label_to_idx = meta['label_to_idx']
    else:
        label_to_idx = {l: i for i, l in
                        enumerate(sorted(meta['label_counts'].keys()))}

    idx_to_label = {str(v + 1): k for k, v in label_to_idx.items()}
    idx_to_label['0'] = '<blank>'
    vocab_size = len(label_to_idx)

    log.info(f"Vocabulary: {vocab_size} glosses + 1 blank = {vocab_size + 1} CTC tokens")
    log.info(f"Glosses: {sorted(label_to_idx.keys())}")

    # ── Dataset ───────────────────────────────────────────────────
    cache_path = str(save_dir / 'ds_cache.pt')

    if use_real_data and real_seq_dir and real_annot_path:
        log.info("Using REAL continuous data")
        full_ds = ContinuousSignDataset(real_seq_dir, real_annot_path, label_to_idx)
        n_val = max(1, int(len(full_ds) * 0.15))
        n_train = len(full_ds) - n_val
        indices = list(range(len(full_ds)))
        np.random.RandomState(42).shuffle(indices)
        train_ds = Subset(full_ds, indices[:n_train])
        val_ds   = Subset(full_ds, indices[n_train:])
    else:
        log.info("Using SYNTHETIC continuous data (concatenated isolated clips)")
        train_ds = SyntheticContinuousDataset(
            data_path, label_to_idx, cache_path=cache_path,
            min_signs=min_signs, max_signs=max_signs,
            samples_per_epoch=train_samples,
        )
        val_ds = SyntheticContinuousDataset(
            data_path, label_to_idx,
            min_signs=min_signs, max_signs=max_signs,
            samples_per_epoch=val_samples,
            shared_clips=train_ds.clips_by_class,
        )

    # Preload clips to GPU — eliminates all CPU→GPU transfers
    on_gpu = False
    if device.type == 'cuda' and not use_real_data:
        log.info("Preloading clips to GPU...")
        train_ds.to_device(device)
        on_gpu = True

    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # num_workers=0 when clips are GPU-resident (CUDA tensors can't cross
    # process boundaries). Sequences are generated on GPU directly.
    nw = 0 if on_gpu else 2
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=(not on_gpu and use_amp),
        collate_fn=ctc_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(not on_gpu and use_amp),
        collate_fn=ctc_collate,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = SLTStage2(
        vocab_size=vocab_size,
        in_channels=in_channels,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )

    if Path(stage1_ckpt).exists():
        log.info(f"Loading Stage 1 encoder weights from {stage1_ckpt}")
        model.encoder.load_stage1_weights(stage1_ckpt)
    else:
        log.warning(f"Stage 1 checkpoint not found: {stage1_ckpt}")
        log.warning("Training encoder from scratch (not recommended)")

    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    log.info(f"Total parameters: {n_params:,} "
             f"(encoder: {enc_params:,} | CTC head: {n_params - enc_params:,})")

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    start_epoch   = 1
    best_wer      = float('inf')
    trigger_times = 0

    # ── Auto-resume ───────────────────────────────────────────────
    if LAST_CKPT.exists():
        try:
            log.info(f"Found checkpoint — attempting resume from {LAST_CKPT}")
            ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch   = ckpt['epoch'] + 1
            best_wer      = ckpt.get('best_wer', float('inf'))
            trigger_times = ckpt.get('trigger_times', 0)
            model.to(device)
            log.info(f"Resumed from epoch {ckpt['epoch']} | "
                     f"Best WER: {best_wer:.4f}")
        except Exception as e:
            log.warning(f"Resume failed ({e}). Starting fresh.")

    log.info(f"\n{'='*62}")
    log.info(f"  Stage 2 | epochs={epochs} | batch={batch_size} | lr={lr}")
    log.info(f"  Warmup={warmup_epochs} | clip={grad_clip} | patience={patience}")
    log.info(f"  Mode: {'real' if use_real_data else 'synthetic'}")
    log.info(f"  Metric: Word Error Rate (lower is better)")
    log.info(f"{'='*62}\n")

    history = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss  = 0.0
        n_batches   = 0

        for landmarks, targets, input_lengths, target_lengths, padding_mask in train_loader:
            landmarks      = landmarks.to(device, non_blocking=True)
            targets        = targets.to(device, non_blocking=True)
            input_lengths  = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            padding_mask   = padding_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(landmarks, src_key_padding_mask=padding_mask)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = ctc_loss_fn(
                    log_probs.permute(1, 0, 2),   # [T, B, C] for CTC
                    targets, input_lengths, target_lengths,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────
        val_loss, wer_val, seq_acc, hyps, refs = evaluate(
            model, val_loader, device, use_amp, idx_to_label)

        log.info(f"Epoch {epoch:03d}/{epochs} | LR: {cur_lr:.2e} | "
                 f"Train loss: {avg_train_loss:.4f} | "
                 f"Val loss: {val_loss:.4f} | "
                 f"WER: {wer_val:.4f} | SeqAcc: {seq_acc*100:.1f}%")

        # Show sample predictions every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            n_show = min(3, len(hyps))
            for i in range(n_show):
                hyp_str = ' '.join(idx_to_label.get(str(t), '?') for t in hyps[i])
                ref_str = ' '.join(idx_to_label.get(str(t), '?') for t in refs[i])
                log.info(f"  Pred: [{hyp_str}]")
                log.info(f"  True: [{ref_str}]")
                log.info(f"  ---")

        history.append({
            'epoch': epoch,
            'lr': cur_lr,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'wer': wer_val,
            'seq_acc': seq_acc,
        })

        # ── Checkpoints ──────────────────────────────────────────
        ckpt = make_checkpoint(
            model, optimizer, scheduler, epoch,
            wer_val, min(wer_val, best_wer), trigger_times,
            label_to_idx, idx_to_label,
            vocab_size, in_channels, d_model,
            nhead, num_transformer_layers,
        )

        torch.save(ckpt, LAST_CKPT)

        if wer_val < best_wer:
            best_wer      = wer_val
            trigger_times = 0
            torch.save(ckpt, BEST_CKPT)
            log.info(f"  ✨ New best WER: {wer_val:.4f} → ctc_model.pth")
        else:
            trigger_times += 1
            log.info(f"  No improvement ({trigger_times}/{patience})")
            if trigger_times >= patience:
                log.info(f"  Early stopping at epoch {epoch} | "
                         f"Best WER: {best_wer:.4f}")
                break

    with open(save_dir / 'history_stage2.json', 'w') as f:
        json.dump(history, f, indent=2)

    log.info(f"\n{'='*62}")
    log.info(f"  Stage 2 complete.")
    log.info(f"  Best WER     : {best_wer:.4f}")
    log.info(f"  Best model   : {BEST_CKPT}")
    log.info(f"  Last ckpt    : {LAST_CKPT}")
    log.info(f"")
    log.info(f"  To load encoder in Stage 3:")
    log.info(f"    ckpt = torch.load('ctc_model.pth')")
    log.info(f"    encoder.load_state_dict(ckpt['encoder_state_dict'])")
    log.info(f"{'='*62}")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train(
        data_path   = '/kaggle/input/datasets/kokoab/005-claude/landmarks/landmarks',
        meta_path   = '/kaggle/input/datasets/kokoab/005-claude/dataset_meta.json',
        stage1_ckpt = '/kaggle/input/stage1-model/best_model.pth',
        save_dir    = '/kaggle/working',
    )
