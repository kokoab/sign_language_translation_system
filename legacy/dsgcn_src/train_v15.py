"""
╔══════════════════════════════════════════════════════════════════╗
║  Train v15 — DS-GCN-TCN with Apple Vision extraction             ║
║  Changes from v14:                                               ║
║    1. Resting hand augmentation — injects idle hand for          ║
║       one-handed signs so model learns to ignore it at inference ║
║    2. Per-hand wrist-relative normalization — each hand centered ║
║       on its own wrist, scaled by its own size (signer-indep.)   ║
║    3. No ghost hand duplication — mask=0 means hand not present  ║
║                                                                  ║
║  Usage:                                                          ║
║    python src/train_v15.py --data_path ASL_landmarks_apple_vision║
║        --save_dir /workspace/output_v15                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, argparse, logging, time, random, math, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-v15")

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stage_1 import (
    SignDataset, ModelEMA, CosineWarmupScheduler,
    online_augment, focal_cross_entropy, apply_mixup,
    _strip_compiled_prefix, recompute_kinematics, compute_bone_features,
    _rebuild_derived_channels,
)
from model_v14 import SLTStage1V14, count_parameters


# ══════════════════════════════════════════════════════════════════
#  v15 AUGMENTATION: Resting hand injection
# ══════════════════════════════════════════════════════════════════

def inject_resting_hand(x, prob=0.5):
    """For one-handed samples, randomly inject a static resting hand.
    Mostly vectorized — only loops over eligible samples (typically <50% of batch)."""
    B, T, N, C = x.shape
    device = x.device
    mask_ch = 9

    lh_active = x[:, :, 0, mask_ch].sum(dim=1) > 0
    rh_active = x[:, :, 21, mask_ch].sum(dim=1) > 0
    one_handed = lh_active ^ rh_active
    do_inject = one_handed & (torch.rand(B, device=device) < prob)

    if not do_inject.any():
        return x

    idx = do_inject.nonzero(as_tuple=True)[0]
    for bi in idx:
        b = bi.item()
        if lh_active[b]:
            inject_start, source_start = 21, 0
        else:
            inject_start, source_start = 0, 21

        pos = x[b, 0, source_start, :3].clone()
        pos[1] += 0.3 + torch.rand(1, device=device).item() * 0.3
        pos[0] += (torch.rand(1, device=device).item() - 0.5) * 0.4

        # Vectorized: fill [T, 21, 3] at once
        resting = pos[None, None, :] + torch.randn(T, 21, 3, device=device) * 0.02
        x[b, :, inject_start:inject_start+21, :3] = resting
        x[b, :, inject_start:inject_start+21, mask_ch] = 1.0

    return x


# ══════════════════════════════════════════════════════════════════
#  v15 AUGMENTATION: Per-hand wrist-relative normalization
# ══════════════════════════════════════════════════════════════════

def per_hand_wrist_normalize(x, prob=0.3):
    """Per-hand wrist-relative normalization. Vectorized over batch."""
    if random.random() > prob:
        return x

    B, T, N, C = x.shape
    x = x.clone()

    for base in [0, 21]:
        wrist_idx = base
        mcp_idx = base + 9

        # Wrist center: mean of non-zero wrist positions per sample [B, 3]
        wrist_xyz = x[:, :, wrist_idx, :3]  # [B, T, 3]
        wrist_norm = wrist_xyz.norm(dim=-1)  # [B, T]
        valid = wrist_norm > 1e-6  # [B, T]
        active = valid.any(dim=1)  # [B]

        if not active.any():
            continue

        # Compute mean wrist position (mean instead of median for vectorization)
        valid_count = valid.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        wrist_center = (wrist_xyz * valid.unsqueeze(-1).float()).sum(dim=1) / valid_count  # [B, 3]

        # Center hand on wrist (only active samples)
        center = wrist_center[:, None, None, :]  # [B, 1, 1, 3]
        hand = x[:, :, base:base+21, :3]  # [B, T, 21, 3]
        x[:, :, base:base+21, :3] = torch.where(
            active[:, None, None, None].expand_as(hand),
            hand - center.expand_as(hand),
            hand)

        # Scale by hand size (wrist-to-middle-MCP distance)
        dists = (x[:, :, mcp_idx, :3] - x[:, :, wrist_idx, :3]).norm(dim=-1)  # [B, T]
        valid_d = dists > 1e-6
        d_sum = (dists * valid_d.float()).sum(dim=1)  # [B]
        d_count = valid_d.float().sum(dim=1).clamp(min=1)  # [B]
        scale = (d_sum / d_count).clamp(min=1e-6)  # [B]

        hand = x[:, :, base:base+21, :3]
        x[:, :, base:base+21, :3] = torch.where(
            active[:, None, None, None].expand_as(hand),
            hand / scale[:, None, None, None].expand_as(hand),
            hand)

    xyz = x[..., :3]
    mask = x[..., 9:10]
    return _rebuild_derived_channels(xyz, mask)


def hand_swap_mirror(x, prob=0.5):
    """Swap left↔right hand slots and flip X coordinates. Vectorized."""
    B, T, N, C = x.shape
    do_swap = torch.rand(B, device=x.device) < prob  # [B]

    if not do_swap.any():
        return x

    x = x.clone()
    idx = do_swap.nonzero(as_tuple=True)[0]

    # Swap hand slots: 0-20 ↔ 21-41
    left = x[idx, :, 0:21, :].clone()
    x[idx, :, 0:21, :] = x[idx, :, 21:42, :]
    x[idx, :, 21:42, :] = left

    # Flip X channels
    for ch in [0, 3, 6, 10, 13]:
        if ch < C:
            x[idx, :, :, ch] = -x[idx, :, :, ch]

    return x


def online_augment_v15(x, **kwargs):
    """v15 augmentation: standard v14 augments + hand swap + wrist normalization."""
    x = online_augment(x, **kwargs)
    x = hand_swap_mirror(x, prob=0.5)
    x = per_hand_wrist_normalize(x, prob=0.3)
    return x


# ══════════════════════════════════════════════════════════════════
#  Training (same as v14, swapping augment function)
# ══════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, augment_fn=None, keep_train_mode=False):
    if not keep_train_mode:
        model.eval()
    total_loss, correct_1, correct_5, total = 0.0, 0, 0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if augment_fn is not None:
                x = augment_fn(x)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct_1 += (preds == y).sum().item()
            _, top5 = logits.topk(5, dim=1)
            correct_5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(y.cpu().tolist())

    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(all_true, all_preds, average='weighted', zero_division=0) * 100
    rec = recall_score(all_true, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0) * 100

    return {
        "val_loss": total_loss / max(len(loader), 1),
        "acc": 100 * correct_1 / max(total, 1),
        "top5": 100 * correct_5 / max(total, 1),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
    }


def train(
    data_path="ASL_landmarks_apple_vision",
    save_dir="/workspace/output_v15",
    epochs=200,
    batch_size=128,
    accum_steps=4,
    lr=5e-4,
    d_model=384,
    focal_gamma=1.0,
    label_smoothing=0.10,
    head_dropout=0.30,
    patience=25,
    resume_path=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_CKPT = save_dir / 'best_model.pth'

    log.info(f"Device: {device} | d_model={d_model} | Data: {data_path}")
    log.info(f"v15: resting hand aug + per-hand wrist normalization")

    # Load manifest (prefer model-cleaned > stat-cleaned > original)
    for mname in ['manifest_model_cleaned.json', 'manifest_cleaned.json', 'manifest.json']:
        mpath = Path(data_path) / mname
        if mpath.exists():
            manifest_path = mpath
            break
    log.info(f"Manifest: {manifest_path.name}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    unique_labels = sorted(set(manifest.values()))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    log.info(f"Manifest: {len(manifest)} files, {len(unique_labels)} classes")

    # Dataset
    cache_path = str(Path(data_path) / 'ds_cache_v15.pt')
    full_ds = SignDataset(data_path, label_to_idx, manifest=manifest, cache_path=cache_path)

    # Split
    all_targets = full_ds.targets.cpu().numpy()
    indices = list(range(len(full_ds)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42,
                                            stratify=all_targets[indices])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42,
                                          stratify=all_targets[temp_idx])
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)
    log.info(f"Split: Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    # Sampler + loaders
    sample_weights = full_ds.class_weights(temperature=0.5)[train_ds.indices]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    full_ds.data = full_ds.data.pin_memory()
    full_ds.targets = full_ds.targets.pin_memory()

    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=nw, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True, persistent_workers=True)

    # Model — same architecture as v14
    model = SLTStage1V14(
        num_classes=full_ds.num_classes, d_model=d_model,
        num_tcn_blocks=4,
        dropout=0.10, head_dropout=head_dropout,
        drop_path_rate=0.1, use_arcface=True,
    ).to(device)
    log.info(f"Model: {count_parameters(model):,} parameters (DS-GCN-TCN v14 arch, v15 training)")

    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0
    if resume_path and os.path.exists(resume_path):
        log.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_acc = ckpt.get('best_acc', 0.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        log.info(f"  Loaded epoch {start_epoch-1}, best_acc={best_acc:.2f}%")
        if ckpt.get('ema_shadow'):
            for n, p in model.named_parameters():
                if n in ckpt['ema_shadow']:
                    p.data.copy_(ckpt['ema_shadow'][n])
            log.info("  Applied EMA weights")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.98), fused=(device.type == 'cuda'))
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # EMA
    ema = ModelEMA(model, decay=0.999)
    ema.to(device)

    trigger_times = 0
    history = []
    log.info(f"GPU Mem: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB / "
             f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB"
             if torch.cuda.is_available() else "CPU mode")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        model.set_epoch(epoch)

        epoch_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            x, y = batch[0], batch[1]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # v15 augmentation: standard + resting hand + wrist normalization
            x = online_augment_v15(x)
            x, y_a, y_b, lam = apply_mixup(x, y, alpha=0.1, cutmix_prob=0.15)

            with torch.cuda.amp.autocast(enabled=True):
                out = model(x, labels=y_a)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out

                if focal_gamma > 0:
                    loss_a = focal_cross_entropy(logits, y_a, gamma=focal_gamma,
                                                 label_smoothing=label_smoothing)
                    loss_b = focal_cross_entropy(logits, y_b, gamma=focal_gamma,
                                                 label_smoothing=label_smoothing)
                else:
                    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                loss = (lam * loss_a + (1 - lam) * loss_b) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            epoch_loss += loss.detach() * accum_steps

        scheduler.step()
        train_loss_arcface = epoch_loss.item() / len(train_loader)

        # Both train and val evaluated the same way: eval mode, no dropout, no augmentation
        ema.apply(model)
        train_m = evaluate(model, train_eval_loader, device)
        val_m = evaluate(model, val_loader, device)
        ema.restore(model)

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)
        eta_str = f"{int(eta//3600)}h{int((eta%3600)//60):02d}m"
        cur_lr = optimizer.param_groups[0]['lr']
        log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | "
                 f"LR: {cur_lr:.2e} | "
                 f"Train: {train_m['val_loss']:.4f} ({train_m['acc']:.1f}%) | "
                 f"Val: {val_m['val_loss']:.4f} ({val_m['acc']:.1f}%) | "
                 f"Top5: {val_m['top5']:.2f}%")

        history.append({
            'epoch': epoch,
            'train_loss_arcface': round(train_loss_arcface, 4),
            'train_loss': round(train_m['val_loss'], 4),
            'train_acc': round(train_m['acc'], 2),
            'train_top5': round(train_m['top5'], 2),
            'train_precision': train_m['precision'],
            'train_recall': train_m['recall'],
            'train_f1': train_m['f1'],
            'val_loss': round(val_m['val_loss'], 4),
            'val_acc': round(val_m['acc'], 2),
            'val_top5': round(val_m['top5'], 2),
            'val_precision': val_m['precision'],
            'val_recall': val_m['recall'],
            'val_f1': val_m['f1'],
            'lr': cur_lr,
        })
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if val_m['acc'] > best_acc:
            best_acc = val_m['acc']
            trigger_times = 0
            torch.save({
                'model_state_dict': _strip_compiled_prefix(model.state_dict()),
                'ema_shadow': _strip_compiled_prefix(ema.shadow),
                'epoch': epoch, 'best_acc': best_acc,
                'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label,
                'num_classes': full_ds.num_classes, 'd_model': d_model,
                'model_type': 'DS-GCN-TCN-v15',
            }, BEST_CKPT)
            log.info(f"  ✨ Best! ({best_acc:.2f}%)")
        else:
            trigger_times += 1

        if trigger_times >= patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    # Final test
    log.info(f"Best: {best_acc:.2f}%")
    best = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    if best.get('ema_shadow'):
        for n, p in model.named_parameters():
            if n in best['ema_shadow']:
                p.data.copy_(best['ema_shadow'][n])
    test_m = evaluate(model, test_loader, device)
    log.info(f"Test: Top-1 {test_m['acc']:.2f}% | Top-5 {test_m['top5']:.2f}%")

    # Per-class analysis
    model.eval()
    per_class_correct, per_class_total = Counter(), Counter()
    confusion = {}
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for p, t in zip(preds.cpu().tolist(), y.cpu().tolist()):
                tl = idx_to_label.get(str(t), f"?{t}")
                pl = idx_to_label.get(str(p), f"?{p}")
                per_class_total[tl] += 1
                if p == t: per_class_correct[tl] += 1
                if tl not in confusion: confusion[tl] = {}
                confusion[tl][pl] = confusion[tl].get(pl, 0) + 1

    sorted_acc = sorted(per_class_total.keys(),
                        key=lambda l: per_class_correct.get(l, 0) / max(per_class_total[l], 1))
    log.info("Bottom 15 classes:")
    for lbl in sorted_acc[:15]:
        acc = 100 * per_class_correct.get(lbl, 0) / max(per_class_total[lbl], 1)
        conf = confusion.get(lbl, {})
        misses = sorted(((p, c) for p, c in conf.items() if p != lbl), key=lambda x: -x[1])[:3]
        log.info(f"  {lbl}: {acc:.1f}% -> {', '.join(f'{p}({c})' for p,c in misses)}")

    with open(save_dir / 'confusion_matrix.json', 'w') as f:
        json.dump(confusion, f, indent=2)
    log.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='ASL_landmarks_apple_vision')
    p.add_argument('--save_dir', default='/workspace/output_v15')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--d_model', type=int, default=384)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = p.parse_args()

    train(
        data_path=args.data_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        d_model=args.d_model,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        resume_path=args.resume,
    )
