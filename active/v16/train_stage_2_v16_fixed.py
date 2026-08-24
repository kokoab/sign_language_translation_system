"""Stage 2 training with 3 fixes to prevent template memorization:

FIX A: Partial-sequence augmentation on real phrase data.
       Each phrase gets N subsequence training samples (e.g., first-sign-only, last-two-signs).
       Breaks "after HELLO always HOW" pattern.

FIX B: Heavier synthetic mixing (20K synthetic vs 663 real phrases).
       Dilutes the 9 fixed templates with diverse random concatenations.

FIX C: Isolated-sign sequences added (N=3000).
       Single isolated Stage 1 clips used as 1-sign Stage 2 training samples.
       Teaches model that short videos can legitimately produce 1-sign outputs.

All three fixes combined in one training script; toggle via CLI flags.
"""
import os, sys, json, time, logging, argparse, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v16 import SLTStage2V16CTC, make_checkpoint_v16, NUM_NODES, DEFAULT_IN_CHANNELS, DEFAULT_DIM
from train_stage_2_v16 import (
    SyntheticCTCDataset, RealPhraseCTCDataset, collate_ctc,
    compute_wer, ctc_greedy_decode, ModelEMA,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# FIX A: Partial-sequence augmentation
# ─────────────────────────────────────────────────

class PartialPhraseDataset(Dataset):
    """Takes real phrase videos and creates subsequence training samples.
    For a 3-sign phrase with N clips, generates:
      - Full phrase (1 sample)
      - First sign only (first ~N/3 clips) (1 sample)
      - Last sign only (last ~N/3 clips) (1 sample)
      - First two signs (first ~2N/3 clips) (1 sample)
      - Last two signs (last ~2N/3 clips) (1 sample)

    Each source phrase video becomes ~5 training samples.
    This breaks template memorization by showing that phrase prefixes/suffixes can appear alone.
    """
    def __init__(self, phrase_path, gloss_to_idx, indices=None):
        self.gloss_to_idx = gloss_to_idx
        self.items = []

        base = RealPhraseCTCDataset(phrase_path, gloss_to_idx)
        file_indices = indices if indices is not None else range(len(base.files))

        for i in file_indices:
            fpath = base.files[i]
            target = base.targets[i]
            n_signs = len(target)

            # Load once to check length
            arr = np.load(fpath).astype(np.float32)
            T = arr.shape[0]
            clips_total = T // 32 if T >= 32 else 1
            if clips_total == 0:
                continue

            # Full phrase (always)
            self.items.append((fpath, list(target), 0, T))

            # Only generate partials if we have enough clips
            if n_signs >= 2 and clips_total >= 2:
                clips_per_sign = clips_total / n_signs

                # First sign only
                end_clip = max(1, round(clips_per_sign))
                self.items.append((fpath, [target[0]], 0, end_clip * 32))

                # Last sign only
                start_clip = max(0, clips_total - max(1, round(clips_per_sign)))
                self.items.append((fpath, [target[-1]], start_clip * 32, T))

                if n_signs >= 3:
                    # First two signs
                    end_clip2 = max(2, round(clips_per_sign * 2))
                    self.items.append((fpath, list(target[:2]), 0, min(end_clip2 * 32, T)))

                    # Last two signs
                    start_clip2 = max(0, clips_total - max(2, round(clips_per_sign * 2)))
                    self.items.append((fpath, list(target[-2:]), start_clip2 * 32, T))

        log.info(f"[partial-phrase] generated {len(self.items)} samples from {len(list(file_indices))} source videos")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, target, start, end = self.items[idx]
        arr = np.load(fpath).astype(np.float32)
        arr = arr[start:end]
        if arr.shape[0] < 32:
            # Pad to minimum single clip
            pad = np.zeros((32 - arr.shape[0], 61, 5), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        T = arr.shape[0]
        if T % 32 != 0:
            pad = np.zeros((((T + 31) // 32) * 32 - T, 61, 5), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return torch.from_numpy(arr), torch.tensor(target, dtype=torch.long)


# ─────────────────────────────────────────────────
# FIX C: Isolated single-sign sequences
# ─────────────────────────────────────────────────

class IsolatedSignDataset(Dataset):
    """Uses Stage 1 isolated samples as 1-sign Stage 2 training sequences.
    Teaches model that a single 32-frame input can produce a single-gloss output
    (not always a multi-sign phrase).
    """
    def __init__(self, data_path, manifest_path, gloss_to_idx, max_samples=3000, split='train'):
        self.gloss_to_idx = gloss_to_idx
        self.items = []

        # Use same manifest loading as SyntheticCTCDataset
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Check format
        first_val = next(iter(manifest.values()))
        if isinstance(first_val, int):
            # class_name → index format; need filename manifest
            files_manifest_path = manifest_path.replace('.json', '_files.json')
            if os.path.exists(files_manifest_path):
                with open(files_manifest_path) as f:
                    file_manifest = json.load(f)
            else:
                file_manifest = {}
                for f in os.listdir(data_path):
                    if f.endswith('.npy'):
                        cls = f.rsplit('_', 2)[0]
                        file_manifest[f] = cls
        else:
            file_manifest = manifest

        # Split per class
        files_by_class = defaultdict(list)
        for fname, cls in file_manifest.items():
            if cls in gloss_to_idx:
                files_by_class[cls].append(fname)

        # Replicate Stage 1's per-class split (seed=42, 70/15/15)
        all_files = []
        for cls in sorted(files_by_class.keys()):
            cls_files = sorted(files_by_class[cls])
            rng = random.Random(); rng.seed(42); rng.shuffle(cls_files)
            n = len(cls_files)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            if split == 'train':
                selected = cls_files[:train_end]
            elif split == 'val':
                selected = cls_files[train_end:val_end]
            else:
                selected = cls_files[val_end:]
            for fname in selected:
                all_files.append((fname, cls))

        # Random sample
        rng = random.Random(42); rng.shuffle(all_files)
        all_files = all_files[:max_samples]

        for fname, cls in all_files:
            fpath = os.path.join(data_path, fname)
            if os.path.exists(fpath):
                self.items.append((fpath, gloss_to_idx[cls]))

        log.info(f"[isolated] {len(self.items)} single-sign samples from {split} split")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, cls_idx = self.items[idx]
        arr = np.load(fpath).astype(np.float32)
        if arr.shape != (32, 61, 5):
            # Handle old 10ch format
            if arr.shape[-1] == 10:
                new_arr = np.zeros((32, 61, 5), dtype=np.float32)
                new_arr[:, :, 0] = arr[:, :, 0]
                new_arr[:, :, 1] = arr[:, :, 1]
                new_arr[:, :, 3] = arr[:, :, 9]
                new_arr[:, :, 4] = 1.0
                arr = new_arr
        return torch.from_numpy(arr), torch.tensor([cls_idx], dtype=torch.long)


# ─────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────

def train_fixed(
    data_path, stage1_ckpt, phrase_data,
    save_dir='models/output_stage2_v16_fixed',
    manifest_path='models/manifest_v16.json',
    epochs=60, batch_size=16, lr=5e-4, weight_decay=0.01,
    patience=35, unfreeze_epoch=30, encoder_lr_scale=0.1, dim=DEFAULT_DIM,
    num_sequences=20000,  # FIX B: more synthetic (was 10000)
    use_partial=True,     # FIX A: partial-phrase augmentation
    use_isolated=True,    # FIX C: isolated single-sign samples
    isolated_samples=3000,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    log.info(f"Device: {device}, AMP: {use_amp}")
    log.info(f"Fixes: partial={use_partial}, isolated={use_isolated}, synth_sequences={num_sequences}")

    os.makedirs(save_dir, exist_ok=True)

    # Synthetic datasets (FIX B: larger)
    synth_train = SyntheticCTCDataset(data_path, split='train', manifest_path=manifest_path,
                                       num_sequences=num_sequences)
    synth_val = SyntheticCTCDataset(data_path, split='val', manifest_path=manifest_path,
                                     num_sequences=max(1500, num_sequences // 10))

    vocab_size = synth_train.vocab_size
    gloss_to_idx = synth_train.gloss_to_idx
    idx_to_gloss = synth_train.idx_to_gloss

    train_parts = [synth_train]
    val_parts = [synth_val]

    # Phrase split (same seed as baseline)
    phrase_train_idx = phrase_val_idx = None
    if phrase_data and os.path.isdir(phrase_data):
        base = RealPhraseCTCDataset(phrase_data, gloss_to_idx)
        n_ph = len(base.files)
        idx = list(range(n_ph)); random.Random(42).shuffle(idx)
        n_val = max(1, int(n_ph * 0.15))
        phrase_val_idx = idx[:n_val]
        phrase_train_idx = idx[n_val:]

        if use_partial:
            # FIX A: partial-phrase augmentation
            phrase_train_ds = PartialPhraseDataset(phrase_data, gloss_to_idx, indices=phrase_train_idx)
            phrase_val_ds = PartialPhraseDataset(phrase_data, gloss_to_idx, indices=phrase_val_idx)
        else:
            phrase_train_ds = torch.utils.data.Subset(base, phrase_train_idx)
            phrase_val_ds = torch.utils.data.Subset(base, phrase_val_idx)

        train_parts.append(phrase_train_ds)
        val_parts.append(phrase_val_ds)
        log.info(f"[phrases] {len(phrase_train_ds)} train + {len(phrase_val_ds)} val phrase samples")

    # FIX C: isolated single-sign sequences
    if use_isolated:
        iso_train = IsolatedSignDataset(data_path, manifest_path, gloss_to_idx,
                                         max_samples=isolated_samples, split='train')
        iso_val = IsolatedSignDataset(data_path, manifest_path, gloss_to_idx,
                                       max_samples=max(200, isolated_samples // 10), split='val')
        train_parts.append(iso_train)
        val_parts.append(iso_val)
        log.info(f"[isolated] {len(iso_train)} train + {len(iso_val)} val isolated samples")

    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    val_ds = ConcatDataset(val_parts) if len(val_parts) > 1 else val_parts[0]

    log.info(f"TOTAL Train: {len(train_ds)} | Val: {len(val_ds)} | Vocab: {vocab_size}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_ctc, num_workers=2,
                              pin_memory=use_amp, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_ctc, num_workers=2, pin_memory=use_amp)

    # ── Detect architecture from Stage 1 checkpoint ──
    s1_ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    in_channels = s1_ckpt.get('in_channels', DEFAULT_IN_CHANNELS)
    dim = s1_ckpt.get('d_model', dim)
    encoder_depth = s1_ckpt.get('depth', 4)
    use_angles = s1_ckpt.get('use_angles', False)
    use_pairwise = s1_ckpt.get('use_pairwise', False)
    log.info(f"Stage 1 checkpoint: in_channels={in_channels}, d_model={dim}, "
             f"depth={encoder_depth}, use_angles={use_angles}, use_pairwise={use_pairwise}")

    model = SLTStage2V16CTC(
        vocab_size=vocab_size, stage1_ckpt=stage1_ckpt,
        in_channels=in_channels, dim=dim,
        encoder_depth=encoder_depth,
        use_angles=use_angles, use_pairwise=use_pairwise,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,} | Trainable (encoder frozen): {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    best_wer = 100.0
    trigger_times = 0

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch + 1:
            log.info(f"*** Unfreezing encoder at epoch {epoch} with {encoder_lr_scale}x LR ***")
            for p in model.encoder.parameters():
                p.requires_grad = True
            encoder_params = list(model.encoder.parameters())
            other_params = [p for n, p in model.named_parameters()
                            if not n.startswith('encoder.') and p.requires_grad]
            optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': lr * encoder_lr_scale},
                {'params': other_params, 'lr': lr},
            ], weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=lr * 0.01)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, targets, input_lengths, target_lengths) in enumerate(train_loader):
            x = x.to(device); targets = targets.to(device)
            input_lengths = input_lengths.to(device); target_lengths = target_lengths.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, inter_logits = model(x)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                if inter_logits is not None:
                    inter_lp = F.log_softmax(inter_logits, dim=-1).permute(1, 0, 2)
                    loss = loss + 0.1 * ctc_loss_fn(inter_lp, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0; wer_total = 0.0; wer_count = 0
        with torch.no_grad():
            for x, targets, input_lengths, target_lengths in val_loader:
                x = x.to(device); targets = targets.to(device)
                input_lengths = input_lengths.to(device); target_lengths = target_lengths.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _ = model(x)
                    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                # WER — build batched predictions + references, call compute_wer once
                probs = F.softmax(logits, dim=-1)  # [B, S, V]
                decoded_all = ctc_greedy_decode(probs, blank=0)  # list of lists
                refs = []
                target_idx = 0
                for b in range(x.shape[0]):
                    L = int(target_lengths[b])
                    refs.append(targets[target_idx:target_idx+L].cpu().tolist())
                    target_idx += L
                # compute_wer returns percentage over whole batch
                batch_wer = compute_wer(decoded_all, refs)
                total_ref_words = sum(len(r) for r in refs)
                wer_total += batch_wer * total_ref_words / 100.0  # accumulate raw edit errors
                wer_count += total_ref_words

        val_loss /= max(1, len(val_loader))
        val_wer = 100.0 * wer_total / max(1, wer_count)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t0

        log.info(f"Epoch {epoch:3d}/{epochs} | train_loss={epoch_loss/len(train_loader):.4f} | "
                 f"val_loss={val_loss:.4f} | WER={val_wer:.2f}% | lr={current_lr:.6f} | {epoch_time:.0f}s")

        if val_wer < best_wer:
            best_wer = val_wer; trigger_times = 0
            ckpt = make_checkpoint_v16(
                model, optimizer, scheduler, None,
                epoch=epoch, metric_value=val_wer, best_metric=best_wer,
                trigger_times=trigger_times,
                label_maps={'gloss_to_idx': gloss_to_idx, 'idx_to_gloss': idx_to_gloss},
                vocab_size=vocab_size, dim=dim, depth=encoder_depth,
                in_channels=in_channels, stage=2,
                use_angles=use_angles, use_pairwise=use_pairwise,
            )
            torch.save(ckpt, os.path.join(save_dir, 'stage2_best_model.pth'))
            log.info(f"  ★ Best: WER={val_wer:.2f}%")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                log.info(f"Early stop at epoch {epoch}")
                break

    log.info(f"Done. Best WER: {best_wer:.2f}%")
    return best_wer


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='ASL_landmarks_v16')
    p.add_argument('--stage1_ckpt', required=True)
    p.add_argument('--phrase_data', required=True)
    p.add_argument('--save_dir', default='models/output_stage2_v16_fixed')
    p.add_argument('--manifest', default='models/manifest_v16.json')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=35)
    p.add_argument('--num_sequences', type=int, default=20000)
    p.add_argument('--isolated_samples', type=int, default=3000)
    p.add_argument('--no_partial', action='store_true')
    p.add_argument('--no_isolated', action='store_true')
    args = p.parse_args()

    train_fixed(
        data_path=args.data_path, stage1_ckpt=args.stage1_ckpt,
        phrase_data=args.phrase_data, save_dir=args.save_dir,
        manifest_path=args.manifest,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience, num_sequences=args.num_sequences,
        use_partial=not args.no_partial, use_isolated=not args.no_isolated,
        isolated_samples=args.isolated_samples,
    )
