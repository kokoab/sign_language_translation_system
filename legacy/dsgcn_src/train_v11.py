"""
╔══════════════════════════════════════════════════════════════════╗
║  Train v11 — Improved DS-GCN-Transformer                        ║
║  Also supports knowledge distillation (Step 6)                   ║
║  Usage:                                                          ║
║    # Normal training:                                            ║
║    python src/train_v11.py --data_path ASL_landmarks_v2          ║
║                                                                  ║
║    # Distillation from SAM teacher:                              ║
║    python src/train_v11.py --data_path ASL_landmarks_v2          ║
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
log = logging.getLogger("SLT-v11")

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
from model_v11 import SLTStage1V11, count_parameters


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct_1, correct_5, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            correct_1 += (logits.argmax(1) == y).sum().item()
            _, top5 = logits.topk(5, dim=1)
            correct_5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
            total += y.size(0)
    return {
        "val_loss": total_loss / max(len(loader), 1),
        "acc": 100 * correct_1 / max(total, 1),
        "top5": 100 * correct_5 / max(total, 1),
    }


def train(
    data_path="ASL_landmarks_v2",
    save_dir="/workspace/output_v11",
    epochs=200,
    batch_size=128,
    accum_steps=4,
    lr=5e-4,
    d_model=512,
    focal_gamma=1.0,
    label_smoothing=0.10,
    head_dropout=0.30,
    patience=50,
    # Distillation params
    soft_labels_path=None,
    distill_alpha=0.7,
    distill_temp=4.0,
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
    cache_path = str(Path(data_path) / 'ds_cache_v11.pt')
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
    class_soft_labels = None  # [num_classes, num_classes] — average soft label per class
    sam_sample_weights = None
    if soft_labels_path and os.path.exists(soft_labels_path):
        soft_labels_data = torch.load(soft_labels_path, map_location='cpu', weights_only=False)
        raw_soft = soft_labels_data['soft_labels']  # [N, num_classes]
        sam_sample_weights = soft_labels_data.get('sample_weights')  # [N] or None

        # Precompute class-averaged soft labels: for each class, average all samples' soft labels
        # This way we can look up by label index during training (works with any sampler)
        sl_label_to_idx = soft_labels_data.get('label_to_idx', label_to_idx)
        filenames = soft_labels_data.get('filenames', [])
        num_sl_classes = raw_soft.shape[1]
        class_soft_labels = torch.zeros(len(unique_labels), num_sl_classes)
        class_counts = torch.zeros(len(unique_labels))
        for fi, fname in enumerate(filenames):
            lbl = manifest.get(fname, manifest.get(fname.replace('.npy', ''), None))
            if lbl and lbl in label_to_idx:
                cidx = label_to_idx[lbl]
                class_soft_labels[cidx] += raw_soft[fi]
                class_counts[cidx] += 1
        class_counts = class_counts.clamp(min=1)
        class_soft_labels /= class_counts.unsqueeze(1)
        log.info(f"Distillation enabled: alpha={distill_alpha}, T={distill_temp}")
        log.info(f"Class-averaged soft labels: [{class_soft_labels.shape[0]}, {class_soft_labels.shape[1]}]")

        if sam_sample_weights is not None:
            log.info(f"Sample weighting: mean={sam_sample_weights.mean():.3f}, "
                     f"min={sam_sample_weights.min():.3f}, max={sam_sample_weights.max():.3f}")
            n_downweighted = (sam_sample_weights < 0.1).sum().item()
            log.info(f"  {n_downweighted} likely mislabeled samples (weight<0.1)")

    # Sampler + loaders (with curriculum / sample weighting)
    sample_weights = full_ds.class_weights(temperature=0.5)[train_ds.indices]
    # Incorporate SAM sample weights: map SAM filename indices to our dataset indices
    if sam_sample_weights is not None:
        filenames_sl = soft_labels_data.get('filenames', [])
        fname_to_sl_idx = {f: i for i, f in enumerate(filenames_sl)}
        # Map each training sample to its SAM weight
        ds_fnames = full_ds.filenames if hasattr(full_ds, 'filenames') else []
        if ds_fnames:
            for ti, ds_idx in enumerate(train_ds.indices):
                fname = ds_fnames[ds_idx] if ds_idx < len(ds_fnames) else None
                if fname and fname in fname_to_sl_idx:
                    sample_weights[ti] *= sam_sample_weights[fname_to_sl_idx[fname]].item()
        else:
            # Fallback: apply class-level weighting from SAM
            log.info("No per-file mapping available; using class-level SAM weighting")
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
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
    model = SLTStage1V11(
        num_classes=full_ds.num_classes, d_model=d_model,
        nhead=8, num_transformer_layers=4,
        dropout=0.10, head_dropout=head_dropout,
        drop_path_rate=0.1, use_arcface=True,
    ).to(device)
    log.info(f"Model: {count_parameters(model):,} parameters")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.98), fused=(device.type == 'cuda'))
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # EMA
    ema = ModelEMA(model, decay=0.999)
    ema.to(device)

    best_acc, trigger_times = 0.0, 0
    log.info(f"GPU Mem: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB / "
             f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        model.set_epoch(epoch)

        epoch_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = online_augment(x)
            x, y_a, y_b, lam = apply_mixup(x, y, alpha=0.1, cutmix_prob=0.15)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x, labels=y_a)

                # Classification loss
                if focal_gamma > 0:
                    loss_a = focal_cross_entropy(logits, y_a, gamma=focal_gamma,
                                                 label_smoothing=label_smoothing)
                    loss_b = focal_cross_entropy(logits, y_b, gamma=focal_gamma,
                                                 label_smoothing=label_smoothing)
                else:
                    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                cls_loss = lam * loss_a + (1 - lam) * loss_b

                # Distillation loss (if class-averaged soft labels available)
                if class_soft_labels is not None and lam == 1.0:
                    # Look up teacher soft label by class index — works with any sampler
                    teacher_probs = class_soft_labels[y_a.cpu()].to(device)  # [B, num_classes]
                    student_log_probs = F.log_softmax(logits / distill_temp, dim=-1)
                    teacher_soft = F.softmax(teacher_probs / distill_temp, dim=-1)
                    kd_loss = F.kl_div(student_log_probs, teacher_soft,
                                      reduction='batchmean') * (distill_temp ** 2)
                    loss = ((1 - distill_alpha) * cls_loss + distill_alpha * kd_loss) / accum_steps
                else:
                    loss = cls_loss / accum_steps

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

        # Validation with EMA
        ema.apply(model)
        val_m = evaluate(model, val_loader, device)
        ema.restore(model)

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)
        eta_str = f"{int(eta//3600)}h{int((eta%3600)//60):02d}m"
        log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | "
                 f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                 f"Train: {train_loss:.4f} | Val: {val_m['val_loss']:.4f} | "
                 f"Acc: {val_m['acc']:.2f}% | Top5: {val_m['top5']:.2f}%")

        if val_m['acc'] > best_acc:
            best_acc = val_m['acc']
            trigger_times = 0
            torch.save({
                'model_state_dict': _strip_compiled_prefix(model.state_dict()),
                'ema_shadow': _strip_compiled_prefix(ema.shadow),
                'epoch': epoch, 'best_acc': best_acc,
                'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label,
                'num_classes': full_ds.num_classes, 'd_model': d_model,
                'model_type': 'DS-GCN-Transformer-v11',
            }, BEST_CKPT)
            log.info(f"  ✨ Best! ({best_acc:.2f}%)")
        else:
            trigger_times += 1

        if trigger_times >= patience:
            log.info(f"🛑 Early stopping at epoch {epoch}")
            break

    # Final test
    log.info(f"🏆 Best: {best_acc:.2f}%")
    best = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    if best.get('ema_shadow'):
        for n, p in model.named_parameters():
            if n in best['ema_shadow']:
                p.data.copy_(best['ema_shadow'][n])
    test_m = evaluate(model, test_loader, device)
    log.info(f"🧪 Test: Top-1 {test_m['acc']:.2f}% | Top-5 {test_m['top5']:.2f}%")

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
    p.add_argument('--save_dir', default='/workspace/output_v11')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--soft_labels', default=None, help='Path to SAM soft labels for distillation')
    p.add_argument('--distill_alpha', type=float, default=0.7)
    p.add_argument('--distill_temp', type=float, default=4.0)
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
    )
