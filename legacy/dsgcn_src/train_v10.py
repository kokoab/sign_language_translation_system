"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v10 Training — Multi-Stream Fusion Model                   ║
║  Uses model_v10.SLTStage1MultiStream                            ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    TORCHDYNAMO_DISABLE=1 python src/train_v10.py \
        --data_path ASL_landmarks_v2 \
        --save_dir /workspace/output_v10 \
        --epochs 200
"""

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
import json
import math
import random
import logging
import argparse
import time as _time
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-v10")
torch.backends.cudnn.benchmark = True

# Import from existing code
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stage_1 import (
    SignDataset, ModelEMA, CosineWarmupScheduler,
    online_augment, focal_cross_entropy, apply_mixup,
    compute_bone_features, _strip_compiled_prefix,
)
from model_v10 import SLTStage1MultiStream, count_parameters


def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss, correct_1, correct_5, total = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        total_loss += loss.item()
        correct_1 += (logits.argmax(1) == y).sum().item()
        _, top5 = logits.topk(5, dim=1)
        correct_5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
        total += y.size(0)
    return {
        "val_loss": total_loss / len(loader),
        "acc": 100 * correct_1 / max(total, 1),
        "top5_acc": 100 * correct_5 / max(total, 1),
    }


def train(
    data_path="ASL_landmarks_v2",
    save_dir="/workspace/output_v10",
    epochs=200,
    batch_size=256,
    accum_steps=2,
    lr=5e-4,
    weight_decay=0.01,
    warmup_epochs=5,
    label_smoothing=0.10,
    focal_gamma=1.0,
    grad_clip=5.0,
    patience=50,
    d_model=512,
    center_loss_weight=0.005,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_CKPT = save_dir / 'best_model.pth'
    LAST_CKPT = save_dir / 'last_checkpoint.pth'

    log.info(f"Device: {device} | d_model={d_model} | Data: {data_path}")

    # Load manifest
    manifest_path = Path(data_path) / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    unique_labels = sorted(set(manifest.values()))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    log.info(f"Manifest: {len(manifest)} files, {len(unique_labels)} classes")

    # Load dataset
    cache_path = str(Path(data_path) / 'ds_cache_v10.pt')
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

    # Sampler
    sample_weights = full_ds.class_weights(temperature=0.5)[train_ds.indices]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Pin memory
    full_ds.data = full_ds.data.pin_memory()
    full_ds.targets = full_ds.targets.pin_memory()

    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True, persistent_workers=True)

    # Model
    model = SLTStage1MultiStream(
        num_classes=full_ds.num_classes,
        d_model=d_model,
        nhead=8,
        num_transformer_layers=6,
        dropout=0.10,
        head_dropout=0.30,
        drop_path_rate=0.1,
        use_arcface=True,
    ).to(device)
    log.info(f"Model: {count_parameters(model):,} parameters")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.98), fused=(device.type == 'cuda'))
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Compile
    use_compile = hasattr(torch, 'compile') and not os.environ.get('TORCH_COMPILE_DISABLE')
    if use_compile:
        try:
            log.info("Compiling model...")
            model = torch.compile(model, mode="default")
        except Exception as e:
            log.warning(f"torch.compile failed: {e}")

    # EMA
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    ema = ModelEMA(raw_model, decay=0.999)
    ema.to(device)

    log.info(f"GPU Mem: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Training loop
    best_acc, trigger_times = 0.0, 0
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = _time.time()
        model.train()
        raw_model.set_epoch(epoch)

        epoch_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = online_augment(x)

            x, y_a, y_b, lam = apply_mixup(x, y, alpha=0.1, cutmix_prob=0.15)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(x, labels=y_a)
                logits = output[0] if isinstance(output, tuple) else output

                # Classification loss
                if focal_gamma > 0:
                    loss_a = focal_cross_entropy(logits, y_a, gamma=focal_gamma, label_smoothing=label_smoothing)
                    loss_b = focal_cross_entropy(logits, y_b, gamma=focal_gamma, label_smoothing=label_smoothing)
                else:
                    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)

                cls_loss = lam * loss_a + (1 - lam) * loss_b
                loss = cls_loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(raw_model)

            epoch_loss += loss.detach() * accum_steps

        scheduler.step()
        train_loss = epoch_loss.item() / len(train_loader)

        # Validation
        ema.apply(raw_model)
        val_metrics = evaluate(model, val_loader, device, use_amp)
        ema.restore(raw_model)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["acc"]
        val_top5 = val_metrics["top5_acc"]

        epoch_time = _time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)
        eta_str = f"{int(eta//3600)}h{int((eta%3600)//60):02d}m" if eta >= 3600 else f"{int(eta//60)}m{int(eta%60):02d}s"
        log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_acc:.2f}% | Val Top-5: {val_top5:.2f}%")

        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 3),
                        "val_top5": round(val_top5, 3)})

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
            ckpt = {
                'model_state_dict': _strip_compiled_prefix(raw_model.state_dict()),
                'ema_shadow': _strip_compiled_prefix(ema.shadow),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch, 'best_acc': best_acc,
                'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label,
                'num_classes': full_ds.num_classes, 'd_model': d_model,
                'val_acc': val_acc, 'stage': 1, 'model_type': 'multi_stream_v10',
            }
            torch.save(ckpt, BEST_CKPT)
            log.info(f"  ✨ Best Checkpoint Saved! (Top-1 Acc: {best_acc:.2f}%)")
        else:
            trigger_times += 1

        if epoch % 5 == 0:
            torch.save({
                'model_state_dict': _strip_compiled_prefix(raw_model.state_dict()),
                'ema_shadow': _strip_compiled_prefix(ema.shadow),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch, 'best_acc': best_acc,
                'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label,
                'num_classes': full_ds.num_classes, 'd_model': d_model,
            }, LAST_CKPT)

        if trigger_times >= patience:
            log.info(f"🛑 Early stopping at epoch {epoch}")
            break

    log.info(f"🏆 Best: Epoch {epoch - trigger_times} with {best_acc:.2f}%")

    # Final test
    if BEST_CKPT.exists():
        best = torch.load(BEST_CKPT, map_location=device, weights_only=False)
        raw_model.load_state_dict(best['model_state_dict'])
        if best.get('ema_shadow'):
            for n, p in raw_model.named_parameters():
                if n in best['ema_shadow']:
                    p.data.copy_(best['ema_shadow'][n])

        test_metrics = evaluate(model, test_loader, device, use_amp)
        log.info(f"🧪 Test: Top-1 {test_metrics['acc']:.2f}% | Top-5 {test_metrics['top5_acc']:.2f}%")

        # Per-class analysis
        per_class = Counter()
        per_class_total = Counter()
        confusion = {}
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                for p, t in zip(preds.cpu().tolist(), y.cpu().tolist()):
                    tl = idx_to_label.get(str(t), f"?{t}")
                    pl = idx_to_label.get(str(p), f"?{p}")
                    per_class_total[tl] += 1
                    if p == t: per_class[tl] += 1
                    if tl not in confusion: confusion[tl] = {}
                    confusion[tl][pl] = confusion[tl].get(pl, 0) + 1

        sorted_acc = sorted(per_class_total.keys(), key=lambda l: per_class.get(l, 0) / max(per_class_total[l], 1))
        log.info("Bottom 15 classes:")
        for lbl in sorted_acc[:15]:
            acc = 100 * per_class.get(lbl, 0) / max(per_class_total[lbl], 1)
            conf = confusion.get(lbl, {})
            misses = sorted(((p, c) for p, c in conf.items() if p != lbl), key=lambda x: -x[1])[:3]
            miss_str = ", ".join(f"{p}({c})" for p, c in misses)
            log.info(f"  {lbl}: {acc:.1f}% ({per_class.get(lbl,0)}/{per_class_total[lbl]}) -> {miss_str}")

        with open(save_dir / 'confusion_matrix.json', 'w') as f:
            json.dump(confusion, f, indent=2)

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    log.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='ASL_landmarks_v2')
    p.add_argument('--save_dir', default='/workspace/output_v10')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--accum_steps', type=int, default=2)
    args = p.parse_args()
    train(
        data_path=args.data_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
    )
