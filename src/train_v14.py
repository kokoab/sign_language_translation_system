"""
╔══════════════════════════════════════════════════════════════════╗
║  Train v12 — DS-GCN-TCN                                         ║
║  Also supports knowledge distillation (SAM teacher)              ║
║  Usage:                                                          ║
║    # Normal training:                                            ║
║    python src/train_v12.py --data_path ASL_landmarks_v2          ║
║                                                                  ║
║    # Distillation from SAM teacher:                              ║
║    python src/train_v12.py --data_path ASL_landmarks_v2          ║
║        --soft_labels /workspace/sam_soft_labels.pt               ║
║        --distill_alpha 0.7 --distill_temp 4.0                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, argparse, logging, time, random, math
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
log = logging.getLogger("SLT-v14")

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
    _strip_compiled_prefix,
)
from model_v14 import SLTStage1V14, count_parameters


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct_1, correct_5, total = 0.0, 0, 0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
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
    data_path="ASL_landmarks_v2",
    save_dir="/workspace/output_v14",
    epochs=200,
    batch_size=128,
    accum_steps=4,
    lr=5e-4,
    d_model=384,
    focal_gamma=1.0,
    label_smoothing=0.10,
    head_dropout=0.30,
    patience=50,
    # Distillation params
    soft_labels_path=None,
    distill_alpha=0.7,
    distill_temp=4.0,
    # Resume from checkpoint
    resume_path=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_CKPT = save_dir / 'best_model.pth'

    log.info(f"Device: {device} | d_model={d_model} | Data: {data_path}")

    # Load manifest
    manifest_path = Path(data_path) / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    unique_labels = sorted(set(manifest.values()))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    log.info(f"Manifest: {len(manifest)} files, {len(unique_labels)} classes")

    # Dataset
    cache_path = str(Path(data_path) / 'ds_cache_v14.pt')
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

    # Load soft labels for distillation
    per_sample_soft = None  # [num_dataset_samples, num_classes] indexed by dataset position
    class_soft_labels = None  # fallback: [num_classes, num_classes] indexed by class
    sam_sample_weights = None
    if soft_labels_path and os.path.exists(soft_labels_path):
        soft_labels_data = torch.load(soft_labels_path, map_location='cpu', weights_only=False)
        raw_soft = soft_labels_data['soft_labels']
        sam_sample_weights = soft_labels_data.get('sample_weights')
        filenames_sl = soft_labels_data.get('filenames', [])
        num_sl_classes = raw_soft.shape[1]

        # Build per-sample soft labels: map each dataset sample to its SAM soft label
        ds_fnames = full_ds.filenames if hasattr(full_ds, 'filenames') else []
        fname_to_sl_idx = {f: i for i, f in enumerate(filenames_sl)}

        if ds_fnames and len(fname_to_sl_idx) > 0:
            per_sample_soft = torch.zeros(len(full_ds), num_sl_classes)
            matched_mask = torch.zeros(len(full_ds), dtype=torch.bool)
            matched = 0
            for ds_idx, fname in enumerate(ds_fnames):
                if fname in fname_to_sl_idx:
                    per_sample_soft[ds_idx] = raw_soft[fname_to_sl_idx[fname]]
                    matched_mask[ds_idx] = True
                    matched += 1

            # Build class averages for unmatched samples
            class_avg = torch.zeros(len(unique_labels), num_sl_classes)
            class_counts = torch.zeros(len(unique_labels))
            for ds_idx, fname in enumerate(ds_fnames):
                if matched_mask[ds_idx]:
                    lbl = manifest.get(fname)
                    if lbl and lbl in label_to_idx:
                        class_avg[label_to_idx[lbl]] += raw_soft[fname_to_sl_idx[fname]]
                        class_counts[label_to_idx[lbl]] += 1
            class_counts = class_counts.clamp(min=1)
            class_avg /= class_counts.unsqueeze(1)

            # Fill unmatched with class average
            unmatched = 0
            for ds_idx in range(len(ds_fnames)):
                if not matched_mask[ds_idx]:
                    lbl = manifest.get(ds_fnames[ds_idx])
                    if lbl and lbl in label_to_idx:
                        per_sample_soft[ds_idx] = class_avg[label_to_idx[lbl]]
                    unmatched += 1

            log.info(f"Distillation enabled: alpha={distill_alpha}, T={distill_temp}")
            log.info(f"Per-sample soft labels: {matched}/{len(ds_fnames)} matched, {unmatched} class-averaged fallback")
            if matched == 0:
                log.warning("WARNING: No per-sample matches! Falling back to class-averaged.")
                per_sample_soft = None
        else:
            # Fallback to class-averaged
            class_soft_labels = torch.zeros(len(unique_labels), num_sl_classes)
            class_counts = torch.zeros(len(unique_labels))
            for fi, fname in enumerate(filenames_sl):
                lbl = manifest.get(fname, manifest.get(fname.replace('.npy', ''), None))
                if lbl and lbl in label_to_idx:
                    cidx = label_to_idx[lbl]
                    class_soft_labels[cidx] += raw_soft[fi]
                    class_counts[cidx] += 1
            class_counts = class_counts.clamp(min=1)
            class_soft_labels /= class_counts.unsqueeze(1)
            log.info(f"Distillation enabled (class-averaged fallback): alpha={distill_alpha}, T={distill_temp}")

        if sam_sample_weights is not None:
            log.info(f"Sample weighting: mean={sam_sample_weights.mean():.3f}, "
                     f"min={sam_sample_weights.min():.3f}, max={sam_sample_weights.max():.3f}")

    # Sampler + loaders
    sample_weights = full_ds.class_weights(temperature=0.5)[train_ds.indices]
    if sam_sample_weights is not None and ds_fnames:
        for ti, ds_idx in enumerate(train_ds.indices):
            fname = ds_fnames[ds_idx] if ds_idx < len(ds_fnames) else None
            if fname and fname in fname_to_sl_idx:
                sample_weights[ti] *= sam_sample_weights[fname_to_sl_idx[fname]].item()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    full_ds.data = full_ds.data.pin_memory()
    full_ds.targets = full_ds.targets.pin_memory()

    # Attach per-sample soft labels directly to dataset for DataLoader access
    if per_sample_soft is not None:
        full_ds.soft_labels = per_sample_soft
        log.info(f"Attached per-sample soft labels to dataset")

        # Override __getitem__ to return soft label too
        original_getitem = full_ds.__class__.__getitem__
        def _getitem_with_soft(self, idx):
            x, y = original_getitem(self, idx)
            sl = self.soft_labels[idx]
            return x, y, sl
        full_ds.__class__.__getitem__ = _getitem_with_soft

        def collate_with_soft(batch):
            xs = torch.stack([b[0] for b in batch])
            ys = torch.stack([b[1] for b in batch]) if isinstance(batch[0][1], torch.Tensor) else torch.tensor([b[1] for b in batch])
            sls = torch.stack([b[2] for b in batch])
            return xs, ys, sls

        nw = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=nw, pin_memory=True, drop_last=True,
                                  persistent_workers=True, collate_fn=collate_with_soft)
    else:
        nw = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True, persistent_workers=True)

    # Model — v12: DS-GCN-TCN
    model = SLTStage1V14(
        num_classes=full_ds.num_classes, d_model=d_model,
        num_tcn_blocks=4,
        dropout=0.10, head_dropout=head_dropout,
        drop_path_rate=0.1, use_arcface=True,
    ).to(device)
    log.info(f"Model: {count_parameters(model):,} parameters (DS-GCN-TCN v14 angle-primary)")

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
        # Apply EMA weights if available
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
        epoch_ce_loss = torch.tensor(0.0, device=device)
        epoch_kd_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            if len(batch) == 3:
                x, y, batch_soft = batch
                batch_soft = batch_soft.to(device, non_blocking=True)
            else:
                x, y = batch
                batch_soft = None
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            x = online_augment(x)
            x, y_a, y_b, lam = apply_mixup(x, y, alpha=0.1, cutmix_prob=0.15)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x, labels=y_a)

                # Classification loss (with ArcFace margin for training)
                if focal_gamma > 0:
                    loss_a = focal_cross_entropy(logits, y_a, gamma=focal_gamma,
                                                 label_smoothing=label_smoothing)
                    loss_b = focal_cross_entropy(logits, y_b, gamma=focal_gamma,
                                                 label_smoothing=label_smoothing)
                else:
                    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                cls_loss = lam * loss_a + (1 - lam) * loss_b

                # Distillation loss (per-sample soft labels)
                has_distill = (batch_soft is not None or class_soft_labels is not None) and lam > 0.99
                if has_distill:
                    if batch_soft is not None:
                        teacher_probs = batch_soft  # per-sample, already on device
                    else:
                        teacher_probs = class_soft_labels[y_a.cpu()].to(device)
                    student_log_probs = F.log_softmax(logits / distill_temp, dim=-1)
                    teacher_soft = F.softmax(teacher_probs / distill_temp, dim=-1)
                    kd_loss = F.kl_div(student_log_probs, teacher_soft,
                                      reduction='batchmean') * (distill_temp ** 2)
                    loss = ((1 - distill_alpha) * cls_loss + distill_alpha * kd_loss) / accum_steps
                    epoch_ce_loss += cls_loss.detach()
                    epoch_kd_loss += kd_loss.detach()
                else:
                    loss = cls_loss / accum_steps
                    epoch_ce_loss += cls_loss.detach()

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
        train_loss = epoch_loss.item() / len(train_loader)
        train_ce = epoch_ce_loss.item() / len(train_loader)
        train_kd = epoch_kd_loss.item() / max(len(train_loader), 1)
        # Validation with EMA
        ema.apply(model)
        val_m = evaluate(model, val_loader, device)
        ema.restore(model)

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)
        eta_str = f"{int(eta//3600)}h{int((eta%3600)//60):02d}m"
        cur_lr = optimizer.param_groups[0]['lr']
        log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | "
                 f"LR: {cur_lr:.2e} | "
                 f"CE: {train_ce:.4f} | KD: {train_kd:.4f} | Val: {val_m['val_loss']:.4f} | "
                 f"Acc: {val_m['acc']:.2f}% | Top5: {val_m['top5']:.2f}%")

        # Save per-epoch history
        history.append({
            'epoch': epoch,
            'train_ce': round(train_ce, 4),
            'train_kd': round(train_kd, 4),
            'train_total': round(train_loss, 4),
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
                'model_type': 'DS-GCN-TCN-v14',
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
    p.add_argument('--data_path', default='ASL_landmarks_v2')
    p.add_argument('--save_dir', default='/workspace/output_v14')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--d_model', type=int, default=384)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--soft_labels', default=None, help='Path to SAM soft labels for distillation')
    p.add_argument('--distill_alpha', type=float, default=0.7)
    p.add_argument('--distill_temp', type=float, default=4.0)
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
        soft_labels_path=args.soft_labels,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
        resume_path=args.resume,
    )
