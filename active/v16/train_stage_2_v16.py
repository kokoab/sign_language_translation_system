"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v16 — Stage 2 Training (Continuous Sign Recognition CTC)   ║
║  Squeezeformer encoder (from Stage 1) + MultiScaleTCN           ║
║  + Sequence Squeezeformer + CTC head                            ║
║                                                                  ║
║  Usage:                                                          ║
║    python src_v16/train_stage_2_v16.py \                         ║
║      --data_path ASL_landmarks_v16 \                             ║
║      --stage1_ckpt src_v16/output_v16_d384/best_model.pth \     ║
║      --save_dir models/output_stage2_v16 \                      ║
║      --epochs 60 --lr 5e-4 --batch_size 16                      ║
║                                                                  ║
║  Training strategy:                                              ║
║    Epochs 1-30: encoder frozen, train TCN + SeqTransformer + CTC ║
║    Epochs 31-60: encoder unfrozen at 0.1x LR for fine-tuning    ║
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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v16 import (
    SLTStage2V16CTC, make_checkpoint_v16,
    NUM_NODES, DEFAULT_IN_CHANNELS, DEFAULT_DIM
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Dataset: Synthetic CTC sequences
# ═══════════════════════════════════════════════════════════

class SyntheticCTCDataset(Dataset):
    """Creates synthetic continuous sequences by concatenating isolated clips.

    Each sample: 1-8 random signs concatenated with short transitions.
    Target: sequence of gloss indices (for CTC loss).
    """
    def __init__(self, data_path, split='train', manifest_path=None,
                 num_sequences=10000, max_signs=8):
        self.max_signs = max_signs
        self.num_sequences = num_sequences
        self.clip_len = 32

        # Load all isolated clips grouped by class
        files_by_class = {}
        file_manifest_path = manifest_path.replace('.json', '_files.json') if manifest_path else None

        if file_manifest_path and os.path.exists(file_manifest_path):
            with open(file_manifest_path) as fm:
                file_manifest = json.load(fm)
            for fname, cls in file_manifest.items():
                fpath = os.path.join(data_path, fname)
                if os.path.exists(fpath):
                    if cls not in files_by_class:
                        files_by_class[cls] = []
                    files_by_class[cls].append(fpath)
        else:
            for f in os.listdir(data_path):
                if not f.endswith('.npy'):
                    continue
                cls = f.rsplit('_', 2)[0]
                if cls not in files_by_class:
                    files_by_class[cls] = []
                files_by_class[cls].append(os.path.join(data_path, f))

        classes = sorted(files_by_class.keys())

        # Load manifest for consistent ordering
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Detect format: v16 = {class: index}, old = {filename: class}
            first_val = next(iter(manifest.values()))
            if isinstance(first_val, int):
                # v16 format: use indices directly, shift by +1 for blank
                self.gloss_to_idx = {str(k): int(v) + 1 for k, v in manifest.items()}
            else:
                # Old format
                manifest_classes = sorted(set(manifest.values()))
                self.gloss_to_idx = {c: i + 1 for i, c in enumerate(manifest_classes)}
        else:
            self.gloss_to_idx = {c: i + 1 for i, c in enumerate(classes)}

        self.idx_to_gloss = {i: c for c, i in self.gloss_to_idx.items()}
        self.vocab_size = len(self.gloss_to_idx) + 1  # +1 for blank token
        self.blank_idx = 0

        # Split files 70/15/15 with RANDOM SHUFFLE (seed=42)
        import random as _rng
        self.clips_by_class = {}
        for cls in classes:
            cls_files = sorted(files_by_class[cls])
            _rng.seed(42)
            _rng.shuffle(cls_files)
            n = len(cls_files)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            if split == 'train':
                selected = cls_files[:train_end]
            elif split == 'val':
                selected = cls_files[train_end:val_end]
            else:
                selected = cls_files[val_end:]

            if selected:
                self.clips_by_class[cls] = selected

        # Only keep classes that exist in the gloss vocabulary
        self.clips_by_class = {c: v for c, v in self.clips_by_class.items()
                               if c in self.gloss_to_idx}
        self.class_names = list(self.clips_by_class.keys())
        log.info(f"[{split}] {len(self.class_names)} classes, "
                 f"{sum(len(v) for v in self.clips_by_class.values())} clips, "
                 f"{num_sequences} synthetic sequences, vocab={self.vocab_size}")

    def _load_clip(self, filepath):
        """Load and convert a single clip to v16 format [32, 61, 5]."""
        arr = np.load(filepath).astype(np.float32)
        T, N, C = arr.shape

        # Pad 47 → 61 nodes if needed
        if N == 47:
            arr = np.concatenate([arr, np.zeros((T, 14, C), dtype=np.float32)], axis=1)
            N = 61

        # Convert old format to v16
        if C == 10:
            new_arr = np.zeros((T, N, 5), dtype=np.float32)
            new_arr[:, :, 0] = arr[:, :, 0]  # X
            new_arr[:, :, 1] = arr[:, :, 1]  # Y
            # Perspective Z from palm scale
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
            new_arr[:, 42:, 4] = 1.0
            arr = new_arr
        elif C == 5:
            pass  # already v16
        elif C == 16:
            new_arr = np.zeros((T, N, 5), dtype=np.float32)
            new_arr[:, :, 0] = arr[:, :, 0]
            new_arr[:, :, 1] = arr[:, :, 1]
            new_arr[:, :, 2] = arr[:, :, 2]
            new_arr[:, :, 3] = arr[:, :, 9]
            new_arr[:, :, 4] = 1.0
            arr = new_arr

        # Ensure 32 frames
        if arr.shape[0] != self.clip_len:
            from scipy.interpolate import interp1d
            src_t = np.linspace(0, 1, arr.shape[0])
            dst_t = np.linspace(0, 1, self.clip_len)
            flat = arr.reshape(arr.shape[0], -1)
            resampled = np.zeros((self.clip_len, flat.shape[1]), dtype=np.float32)
            for c in range(flat.shape[1]):
                resampled[:, c] = np.interp(dst_t, src_t, flat[:, c])
            arr = resampled.reshape(self.clip_len, N, 5)

        return arr

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """Generate one synthetic continuous sequence."""
        rng = np.random.RandomState(idx)

        # Random number of signs (1-max_signs)
        n_signs = rng.randint(1, self.max_signs + 1)

        # Random sign sequence
        chosen_classes = rng.choice(self.class_names, size=n_signs, replace=True)

        clips = []
        targets = []

        for cls in chosen_classes:
            # Random clip from this class
            clip_files = self.clips_by_class[cls]
            clip_path = clip_files[rng.randint(len(clip_files))]
            clip = self._load_clip(clip_path)

            # Trim 2 frames from start/end (remove static holds)
            clip = clip[2:-2]  # 32 → 28 frames

            clips.append(clip)
            targets.append(self.gloss_to_idx[cls])

        # Concatenate all clips
        sequence = np.concatenate(clips, axis=0)  # [N*28, 61, 5]

        # Pad to multiple of 32
        T = sequence.shape[0]
        C = sequence.shape[-1]
        pad_to = ((T + 31) // 32) * 32
        if pad_to > T:
            pad = np.zeros((pad_to - T, NUM_NODES, C), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=0)

        x = torch.from_numpy(sequence)
        y = torch.tensor(targets, dtype=torch.long)

        return x, y


class RealPhraseCTCDataset(Dataset):
    """Real continuous phrase videos (pre-extracted).

    Expected directory layout (v16):
        <phrase_path>/
            PHRASE_NAME_hash.npy       (one .npy per video)
            phrase_annotations.json    {filename: [gloss1, gloss2, ...]}

    Falls back to legacy v15 filename parsing (PHRASE_NAME_NNNN_hash.npy)
    when phrase_annotations.json is not present.
    """
    def __init__(self, phrase_path, gloss_to_idx):
        self.gloss_to_idx = gloss_to_idx
        self.files = []
        self.targets = []

        if not os.path.exists(phrase_path):
            log.warning(f"Phrase path not found: {phrase_path}")
            return

        # Preferred: load gloss sequences from phrase_annotations.json
        ann_path = os.path.join(phrase_path, 'phrase_annotations.json')
        annotations = None
        if os.path.exists(ann_path):
            with open(ann_path) as fh:
                annotations = json.load(fh)
            log.info(f"[phrases] loaded annotations from {ann_path} ({len(annotations)} entries)")

        # Known multi-word glosses (compound labels in manifest). Checked in order.
        COMPOUND = ['HARD_DIFFICULT', 'EAT_FOOD', 'DRIVE_CAR', 'MAKE_CREATE',
                    'HIS_HER', 'EXCUSE ME', 'RIGHT (ADJECTIVE)', 'RIGHT (PREPOSITION)']

        def parse_filename_fallback(fname):
            """Parse PHRASE_NAME_hash.npy or PHRASE_NAME_NNNN_hash.npy → gloss list.
            Splits by underscore but re-joins known compound glosses.
            """
            stem = fname.replace('.npy', '')
            # Strip trailing _hash (and optional _NNNN_hash)
            parts = stem.split('_')
            # Drop hash (always last token with hex-like content or length>=6)
            if parts and (all(c in '0123456789abcdef' for c in parts[-1].lower()) or len(parts[-1]) >= 6):
                parts = parts[:-1]
            # Drop NNNN index (legacy v15)
            if parts and parts[-1].isdigit():
                parts = parts[:-1]
            phrase_name = '_'.join(parts)
            glosses = []
            i = 0
            tokens = phrase_name.split('_')
            while i < len(tokens):
                # Try matching compound glosses first (check 2 or 3 tokens joined)
                matched = False
                for w in (3, 2):
                    cand = '_'.join(tokens[i:i+w])
                    if cand in COMPOUND or cand in gloss_to_idx:
                        glosses.append(cand)
                        i += w
                        matched = True
                        break
                if not matched:
                    glosses.append(tokens[i])
                    i += 1
            return glosses

        for f in sorted(os.listdir(phrase_path)):
            if not f.endswith('.npy'):
                continue

            # Get gloss sequence
            if annotations and f in annotations:
                glosses = annotations[f]
            else:
                glosses = parse_filename_fallback(f)

            # Convert glosses to indices (only keep if all glosses found in vocab)
            target = []
            for g in glosses:
                if g in gloss_to_idx:
                    target.append(gloss_to_idx[g])

            if target and len(target) == len(glosses):
                self.files.append(os.path.join(phrase_path, f))
                self.targets.append(target)

        log.info(f"[phrases] {len(self.files)} phrase videos loaded")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)

        # Handle old formats (same logic as SyntheticCTCDataset._load_clip)
        T, N, C = arr.shape
        if N == 47:
            arr = np.concatenate([arr, np.zeros((T, 14, C), dtype=np.float32)], axis=1)
            N = 61
        if C == 10:
            new_arr = np.zeros((T, N, 5), dtype=np.float32)
            new_arr[:, :, 0] = arr[:, :, 0]
            new_arr[:, :, 1] = arr[:, :, 1]
            new_arr[:, :, 3] = arr[:, :, 9]
            new_arr[:, :, 4] = 1.0
            arr = new_arr
        elif C == 16:
            new_arr = np.zeros((T, N, 5), dtype=np.float32)
            new_arr[:, :, 0] = arr[:, :, 0]
            new_arr[:, :, 1] = arr[:, :, 1]
            new_arr[:, :, 2] = arr[:, :, 2]
            new_arr[:, :, 3] = arr[:, :, 9]
            new_arr[:, :, 4] = 1.0
            arr = new_arr

        # Pad to multiple of 32
        pad_to = ((T + 31) // 32) * 32
        if pad_to > T:
            pad = np.zeros((pad_to - T, N, 5), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        x = torch.from_numpy(arr)
        y = torch.tensor(self.targets[idx], dtype=torch.long)

        return x, y


# ═══════════════════════════════════════════════════════════
# Collate function for variable-length CTC
# ═══════════════════════════════════════════════════════════

def collate_ctc(batch):
    """Collate variable-length sequences for CTC training.

    Pads all sequences to the longest in the batch.
    Returns:
        x: [B, T_max, 61, 5]
        targets: [sum(target_lengths)]
        input_lengths: [B] — number of output tokens per sample
        target_lengths: [B] — number of target glosses per sample
    """
    xs, ys = zip(*batch)

    # Find max length
    max_T = max(x.shape[0] for x in xs)
    # Pad to multiple of 32
    max_T = ((max_T + 31) // 32) * 32

    B = len(xs)
    C = xs[0].shape[-1]  # get actual channel count from data
    x_padded = torch.zeros(B, max_T, NUM_NODES, C)
    targets_flat = []
    input_lengths = []
    target_lengths = []

    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.shape[0]
        x_padded[i, :T] = x
        targets_flat.append(y)
        # Output tokens = num_clips × out_tokens_per_clip (4)
        num_clips = max_T // 32
        input_lengths.append(num_clips * 4)
        target_lengths.append(len(y))

    targets = torch.cat(targets_flat)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return x_padded, targets, input_lengths, target_lengths


# ═══════════════════════════════════════════════════════════
# WER computation
# ═══════════════════════════════════════════════════════════

def compute_wer(predictions, targets):
    """Compute Word Error Rate using edit distance."""
    total_words = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        total_words += len(target)

        # Edit distance (Levenshtein)
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        total_errors += dp[m][n]

    return total_errors / max(total_words, 1) * 100


def ctc_greedy_decode(logits, blank=0):
    """Greedy CTC decoding: argmax → remove blanks → merge duplicates.

    Args:
        logits: [B, S, V]
    Returns:
        list of lists of token indices
    """
    preds = logits.argmax(dim=-1)  # [B, S]
    decoded = []
    for b in range(preds.shape[0]):
        tokens = preds[b].tolist()
        # Remove blanks
        tokens = [t for t in tokens if t != blank]
        # Merge consecutive duplicates
        merged = []
        for t in tokens:
            if not merged or merged[-1] != t:
                merged.append(t)
        decoded.append(merged)
    return decoded


# ═══════════════════════════════════════════════════════════
# EMA (reuse from Stage 1)
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
                        self.shadow[k] = v.clone()
                    else:
                        self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


# ═══════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════

def train_stage2_v16(
    data_path,
    stage1_ckpt,
    save_dir='models/output_stage2_v16',
    phrase_data=None,
    manifest_path='models/manifest.json',
    epochs=60,
    batch_size=16,
    lr=5e-4,
    weight_decay=0.01,
    patience=35,
    unfreeze_epoch=30,
    encoder_lr_scale=0.1,
    dim=DEFAULT_DIM,
    inter_ctc_weight=0.1,
    num_sequences=10000,
    smoke_test=False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    log.info(f"Device: {device}, AMP: {use_amp}")

    os.makedirs(save_dir, exist_ok=True)

    # ── Data ──
    synth_train_ds = SyntheticCTCDataset(
        data_path, split='train', manifest_path=manifest_path,
        num_sequences=num_sequences)
    synth_val_ds = SyntheticCTCDataset(
        data_path, split='val', manifest_path=manifest_path,
        num_sequences=max(1000, num_sequences // 10))

    vocab_size = synth_train_ds.vocab_size
    gloss_to_idx = synth_train_ds.gloss_to_idx
    idx_to_gloss = synth_train_ds.idx_to_gloss

    if smoke_test:
        synth_train_ds.num_sequences = 100
        synth_val_ds.num_sequences = 20
        epochs = 3
        batch_size = min(batch_size, 4)

    # ── Real phrase data (optional, combined with synthetic) ──
    train_ds = synth_train_ds
    val_ds = synth_val_ds
    if phrase_data:
        from torch.utils.data import ConcatDataset
        phrase_ds = RealPhraseCTCDataset(phrase_data, gloss_to_idx)
        if len(phrase_ds) > 0:
            # Random 85/15 train/val split on phrase files (deterministic seed)
            import random as _random
            n_ph = len(phrase_ds)
            idx = list(range(n_ph))
            _random.Random(42).shuffle(idx)
            n_val = max(1, int(n_ph * 0.15))
            val_idx = set(idx[:n_val])
            train_idx = [i for i in idx if i not in val_idx]

            phrase_train = torch.utils.data.Subset(phrase_ds, train_idx)
            phrase_val = torch.utils.data.Subset(phrase_ds, list(val_idx))

            train_ds = ConcatDataset([synth_train_ds, phrase_train])
            val_ds = ConcatDataset([synth_val_ds, phrase_val])
            log.info(f"[phrases] combined training: {len(synth_train_ds)} synthetic "
                     f"+ {len(phrase_train)} real phrases = {len(train_ds)}")
            log.info(f"[phrases] combined validation: {len(synth_val_ds)} synthetic "
                     f"+ {len(phrase_val)} real phrases = {len(val_ds)}")
        else:
            log.warning(f"[phrases] no usable phrase files found in {phrase_data}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_ctc, num_workers=2,
                              pin_memory=use_amp, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_ctc, num_workers=2,
                            pin_memory=use_amp)

    log.info(f"Train: {len(train_ds)} sequences | Val: {len(val_ds)} | Vocab: {vocab_size}")

    # ── Detect architecture from Stage 1 checkpoint ──
    s1_ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    in_channels = s1_ckpt.get('in_channels', DEFAULT_IN_CHANNELS)
    dim = s1_ckpt.get('d_model', dim)
    encoder_depth = s1_ckpt.get('depth', 4)
    use_angles = s1_ckpt.get('use_angles', False)
    use_pairwise = s1_ckpt.get('use_pairwise', False)
    log.info(f"Stage 1 checkpoint: in_channels={in_channels}, d_model={dim}, "
             f"depth={encoder_depth}, use_angles={use_angles}, use_pairwise={use_pairwise}")

    # ── Model ──
    model = SLTStage2V16CTC(
        vocab_size=vocab_size,
        stage1_ckpt=stage1_ckpt,
        in_channels=in_channels,
        dim=dim,
        encoder_depth=encoder_depth,
        use_angles=use_angles,
        use_pairwise=use_pairwise,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,} | Trainable (encoder frozen): {trainable_params:,}")

    # ── Optimizer (only trainable params initially) ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=0.999)

    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # ── Training ──
    best_wer = 100.0
    trigger_times = 0
    history = []

    for epoch in range(1, epochs + 1):
        # ── Unfreeze encoder at epoch N ──
        if epoch == unfreeze_epoch + 1:
            log.info(f"*** Unfreezing encoder at epoch {epoch} with {encoder_lr_scale}x LR ***")
            model.unfreeze_encoder(lr_scale=encoder_lr_scale)

            # Rebuild optimizer with encoder params at lower LR
            encoder_params = list(model.encoder.parameters())
            other_params = [p for n, p in model.named_parameters()
                          if not n.startswith('encoder.') and p.requires_grad]
            optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': lr * encoder_lr_scale},
                {'params': other_params, 'lr': lr},
            ], weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=lr * 0.01)

            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"  Trainable params: {trainable_now:,}")

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, targets, input_lengths, target_lengths) in enumerate(train_loader):
            x = x.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, inter_logits = model(x)

                # CTC loss
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # [S, B, V]
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

                # InterCTC loss
                if inter_logits is not None and inter_ctc_weight > 0:
                    inter_log_probs = F.log_softmax(inter_logits, dim=-1).permute(1, 0, 2)
                    inter_loss = ctc_loss_fn(inter_log_probs, targets,
                                            input_lengths, target_lengths)
                    loss = loss + inter_ctc_weight * inter_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

            epoch_loss += loss.item()

        scheduler.step()

        # ── Evaluate ──
        ema.apply(model)
        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for x, targets, input_lengths, target_lengths in val_loader:
                x = x.to(device)
                targets_dev = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _ = model(x)
                    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                    loss = ctc_loss_fn(log_probs, targets_dev, input_lengths, target_lengths)
                    val_loss += loss.item()

                # Decode predictions
                decoded = ctc_greedy_decode(logits)
                all_preds.extend(decoded)

                # Split flat targets back into per-sample lists
                offset = 0
                for tl in target_lengths:
                    tl = tl.item()
                    all_targets.append(targets[offset:offset+tl].tolist())
                    offset += tl

        ema.restore(model)

        val_wer = compute_wer(all_preds, all_targets)
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        log.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={epoch_loss/len(train_loader):.4f} | "
            f"val_loss={val_loss/len(val_loader):.4f} | "
            f"WER={val_wer:.2f}% | "
            f"lr={current_lr:.6f} | "
            f"{epoch_time:.0f}s"
        )

        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'wer': val_wer,
            'lr': current_lr,
        })

        # ── Checkpoint ──
        if val_wer < best_wer:
            best_wer = val_wer
            trigger_times = 0

            ckpt = make_checkpoint_v16(
                model, optimizer, scheduler, ema,
                epoch=epoch,
                metric_value=val_wer,
                best_metric=best_wer,
                trigger_times=trigger_times,
                label_maps={
                    'gloss_to_idx': gloss_to_idx,
                    'idx_to_gloss': idx_to_gloss,
                },
                vocab_size=vocab_size,
                dim=dim,
                depth=encoder_depth,
                in_channels=in_channels,
                stage=2,
                use_angles=use_angles,
                use_pairwise=use_pairwise,
            )

            save_path = os.path.join(save_dir, 'stage2_best_model.pth')
            torch.save(ckpt, save_path)
            log.info(f"  ★ Best model saved: WER={val_wer:.2f}%")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                log.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        # Save history
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    log.info(f"\nTraining complete. Best WER: {best_wer:.2f}%")
    return best_wer


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLT v16 Stage 2 CTC Training')
    parser.add_argument('--data_path', default='ASL_landmarks_v16',
                        help='Path to v16 extracted .npy directory')
    parser.add_argument('--stage1_ckpt', default='src_v16/output_v16_d384/best_model.pth',
                        help='Stage 1 checkpoint for encoder weights')
    parser.add_argument('--save_dir', default='models/output_stage2_v16',
                        help='Output directory')
    parser.add_argument('--phrase_data', default=None,
                        help='Path to real phrase .npy directory (optional)')
    parser.add_argument('--manifest', default='models/manifest.json',
                        help='Class manifest JSON')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--unfreeze_epoch', type=int, default=30)
    parser.add_argument('--dim', type=int, default=DEFAULT_DIM)
    parser.add_argument('--num_sequences', type=int, default=10000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    train_stage2_v16(
        data_path=args.data_path,
        stage1_ckpt=args.stage1_ckpt,
        save_dir=args.save_dir,
        phrase_data=args.phrase_data,
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        unfreeze_epoch=args.unfreeze_epoch,
        dim=args.dim,
        num_sequences=args.num_sequences,
        smoke_test=args.smoke,
    )
