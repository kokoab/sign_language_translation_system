"""
╔══════════════════════════════════════════════════════════════════╗
║  Fine-tune SAM-SLR-v2 (SL-GCN) on 310-class ASL dataset        ║
║  Uses WLASL-2000 pretrained weights                             ║
║  4 streams: joint, bone, joint_motion, bone_motion              ║
║                                                                  ║
║  Data format matches SAM exactly:                                ║
║    - Joint: [N, 3, T, 27, 1] raw XYZ, normalization=True,      ║
║             is_vector=False, window_size=120                     ║
║    - Bone/Motion: is_vector=True, window_size=100               ║
║    - Normalization: center X,Y on nose (node 0) mean            ║
║    - Mirror: 512-x (not -x) for joint, -x for vector streams   ║
║    - Padding: repeat sequence if shorter than max_frame         ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    # Joint stream (most important)
    python src/train_sam.py --stream joint --pretrained /workspace/sam_pretrained/SL-GCN/wlasl_joint-222.pt

    # Bone stream
    python src/train_sam.py --stream bone --pretrained /workspace/sam_pretrained/SL-GCN/wlasl_bone-237.pt

    # All 4 streams in parallel:
    python src/train_sam.py --stream joint --pretrained .../wlasl_joint-222.pt --save_dir /workspace/sam_joint &
    python src/train_sam.py --stream bone --pretrained .../wlasl_bone-237.pt --save_dir /workspace/sam_bone &
    python src/train_sam.py --stream joint_motion --pretrained .../wlasl_joint_motion-245.pt --save_dir /workspace/sam_jm &
    python src/train_sam.py --stream bone_motion --pretrained .../wlasl_bone_motion-241.pt --save_dir /workspace/sam_bm &
"""

import os, sys, json, argparse, logging, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SAM-FT")

# Add SAM-SLR-v2 to path
SAM_ROOT = None
for p in ['/workspace/SLT/SAM-SLR-v2/SL-GCN', 'SAM-SLR-v2/SL-GCN',
          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SAM-SLR-v2', 'SL-GCN')]:
    if os.path.isdir(p):
        SAM_ROOT = p
        break
if SAM_ROOT:
    sys.path.insert(0, SAM_ROOT)

# Fake mediapipe
import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from extract import LABEL_ALIASES

# ============================================================
# SAM's exact bone edges (from gen_bone_data.py, offset -5)
# ============================================================
SAM_BONE_PAIRS = [
    (0, 1), (0, 2),
    (1, 3), (3, 5), (2, 4), (4, 6),
    (7, 8), (7, 9), (7, 11), (7, 13), (7, 15),
    (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (17, 19), (17, 21), (17, 23), (17, 25),
    (19, 20), (21, 22), (23, 24), (25, 26),
    (5, 7), (6, 17),
]

# SAM's flip index (from feeder.py)
SAM_FLIP_INDEX = np.concatenate(
    ([0, 2, 1, 4, 3, 6, 5],
     [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), axis=0)

MAX_FRAME = 150


def generate_bone(joint_data):
    """Generate bone data from joint data. Matches gen_bone_data.py exactly.
    joint_data: [T, 27, 3] -> bone: [T, 27, 3]"""
    bone = np.zeros_like(joint_data)
    for v1, v2 in SAM_BONE_PAIRS:
        bone[:, v2, :] = joint_data[:, v2, :] - joint_data[:, v1, :]
    return bone


def generate_motion(data):
    """Generate temporal motion. Matches gen_motion_data.py exactly.
    data: [T, 27, 3] -> motion: [T, 27, 3]"""
    T = data.shape[0]
    motion = np.zeros_like(data)
    if T > 1:
        motion[:-1] = data[1:] - data[:-1]
    # Last frame = 0 (matches gen_motion_data.py: fp_sp[:, :, T - 1, :, :] = 0)
    return motion


def pad_sequence(data, max_frame=MAX_FRAME):
    """Pad by repeating. Matches sign_gendata.py exactly.
    data: [T, 27, 3] -> [max_frame, 27, 3]"""
    T = data.shape[0]
    if T >= max_frame:
        return data[:max_frame]
    # Repeat and pad (matches sign_gendata.py)
    rest = max_frame - T
    num_repeats = int(np.ceil(rest / T))
    pad = np.concatenate([data for _ in range(num_repeats)], axis=0)[:rest]
    return np.concatenate([data, pad], axis=0)


class SAMDataset(Dataset):
    """Dataset that loads our extracted SAM-27 .npy files.
    Matches SAM's feeder.py behavior exactly."""

    def __init__(self, data_path, manifest, label_to_idx, stream='joint',
                 augment=True, window_size=120):
        self.label_to_idx = label_to_idx
        self.stream = stream
        self.augment = augment
        self.window_size = window_size
        self.is_vector = stream in ('bone', 'joint_motion', 'bone_motion')
        self.data_path = Path(data_path)

        # Load all samples
        self.samples = []
        skipped = 0
        for fname, label in manifest.items():
            if label not in label_to_idx:
                skipped += 1
                continue
            fpath = self.data_path / fname
            if not fpath.exists():
                skipped += 1
                continue
            self.samples.append((fname, label_to_idx[label]))

        log.info(f"SAMDataset [{stream}]: {len(self.samples)} samples, {skipped} skipped, "
                 f"window={window_size}, is_vector={self.is_vector}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]

        # Load raw joint data [T, 27, 3]
        joint_data = np.load(self.data_path / fname).astype(np.float32)

        # Generate stream-specific data
        if self.stream == 'joint':
            data = joint_data.copy()
        elif self.stream == 'bone':
            data = generate_bone(joint_data)
        elif self.stream == 'joint_motion':
            data = generate_motion(joint_data)
        elif self.stream == 'bone_motion':
            data = generate_motion(generate_bone(joint_data))
        else:
            data = joint_data.copy()

        # Pad to MAX_FRAME (repeat, matching sign_gendata.py)
        data = pad_sequence(data, MAX_FRAME)

        # Convert to SAM format: [3, T, 27, 1] (C, T, V, M)
        data = data.transpose(2, 0, 1)  # [3, T, 27]
        data = np.expand_dims(data, axis=-1)  # [3, T, 27, 1]

        # Random choose window (matching feeder.py random_choose)
        if self.augment:
            C, T, V, M = data.shape
            if T > self.window_size:
                begin = random.randint(0, T - self.window_size)
                data = data[:, begin:begin + self.window_size, :, :]
            elif T < self.window_size:
                # Auto pad with zeros
                pad = np.zeros((C, self.window_size - T, V, M), dtype=np.float32)
                data = np.concatenate([data, pad], axis=1)

        # Random mirror (matching feeder.py exactly)
        if self.augment and random.random() > 0.5:
            assert data.shape[2] == 27
            data = data[:, :, SAM_FLIP_INDEX, :]
            if self.is_vector:
                data[0, :, :, :] = -data[0, :, :, :]
            else:
                data[0, :, :, :] = 512 - data[0, :, :, :]

        # Normalization (matching feeder.py exactly)
        assert data.shape[0] == 3
        if self.is_vector:
            # For vector streams: center only node 0
            data[0, :, 0, :] = data[0, :, 0, :] - data[0, :, 0, 0].mean(axis=0)
            data[1, :, 0, :] = data[1, :, 0, :] - data[1, :, 0, 0].mean(axis=0)
        else:
            # For joint stream: center all nodes on nose (node 0) mean
            data[0, :, :, :] = data[0, :, :, :] - data[0, :, 0, 0].mean(axis=0)
            data[1, :, :, :] = data[1, :, :, :] - data[1, :, 0, 0].mean(axis=0)

        # Random shift (matching feeder.py)
        if self.augment:
            if self.is_vector:
                data[0, :, 0, :] += random.random() * 20 - 10.0
                data[1, :, 0, :] += random.random() * 20 - 10.0
            else:
                data[0, :, :, :] += random.random() * 20 - 10.0
                data[1, :, :, :] += random.random() * 20 - 10.0

        return torch.from_numpy(data).float(), torch.tensor(label, dtype=torch.long)

    def class_weights(self, temperature=0.5):
        labels = [s[1] for s in self.samples]
        counts = Counter(labels)
        weights = torch.tensor([len(labels) / counts[s[1]] for s in self.samples], dtype=torch.float32)
        return weights ** temperature


def evaluate(model, loader, device):
    model.eval()
    correct, correct_5, total, loss_sum = 0, 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', enabled=False):
                logits = model(x)
                loss_sum += F.cross_entropy(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            _, top5 = logits.topk(5, dim=1)
            correct_5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
            total += y.size(0)
    return {
        'val_loss': loss_sum / max(len(loader), 1),
        'acc': 100 * correct / max(total, 1),
        'top5': 100 * correct_5 / max(total, 1),
    }


def train_stream(
    stream='joint',
    pretrained_path=None,
    data_path='ASL_landmarks_sam27',
    save_dir='/workspace/sam_joint',
    epochs=50,
    batch_size=128,
    lr_head=1.4e-3,
    lr_backbone=1.4e-4,
    freeze_epochs=12,
    patience=30,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Stream-specific settings (matching SAM configs exactly)
    if stream == 'joint':
        window_size = 120
        is_vector = False
    else:
        window_size = 100
        is_vector = True

    log.info(f"Stream: {stream} | window: {window_size} | is_vector: {is_vector}")
    log.info(f"Device: {device} | Pretrained: {pretrained_path}")

    # Load manifest
    manifest_path = Path(data_path) / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    unique_labels = sorted(set(manifest.values()))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    num_classes = len(unique_labels)
    log.info(f"Classes: {num_classes}, Samples in manifest: {len(manifest)}")

    # Create dataset
    full_ds = SAMDataset(data_path, manifest, label_to_idx,
                         stream=stream, augment=True, window_size=window_size)

    # Create non-augmented version for val/test
    val_manifest = manifest  # same manifest, different dataset instance

    # Split indices
    indices = list(range(len(full_ds)))
    labels_list = [full_ds.samples[i][1] for i in indices]
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=labels_list)
    temp_labels = [labels_list[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=temp_labels)

    train_ds = Subset(full_ds, train_idx)

    # Val/test with no augmentation
    val_ds_full = SAMDataset(data_path, manifest, label_to_idx,
                             stream=stream, augment=False, window_size=window_size)
    val_ds = Subset(val_ds_full, val_idx)
    test_ds = Subset(val_ds_full, test_idx)

    log.info(f"Split: Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    # Sampler
    all_weights = full_ds.class_weights(temperature=0.5)
    train_weights = all_weights[train_idx]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True, persistent_workers=True)

    # Create model (must import from SAM's directory)
    from model.decouple_gcn_attn import Model as SLGCN
    model = SLGCN(num_class=num_classes, num_point=27, num_person=1,
                  graph='graph.sign_27.Graph',
                  graph_args={'labeling_mode': 'spatial'},
                  groups=16, block_size=41, in_channels=3).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_params:,}")

    # Load pretrained weights
    if pretrained_path and os.path.exists(pretrained_path):
        log.info(f"Loading pretrained: {pretrained_path}")
        pretrained = torch.load(pretrained_path, map_location=device, weights_only=False)
        model_dict = model.state_dict()
        loaded, skipped_keys = 0, []
        for k, v in pretrained.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded += 1
            else:
                skipped_keys.append(k)
        model.load_state_dict(model_dict)
        log.info(f"Loaded {loaded} params, skipped {len(skipped_keys)}: {skipped_keys}")
    else:
        log.warning("No pretrained weights — training from scratch!")

    # Optimizer: separate LR for backbone vs head
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head},
    ], momentum=0.9, weight_decay=1e-4, nesterov=True)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 32], gamma=0.1)

    # Phase 1: Freeze backbone
    log.info(f"Phase 1: Freeze backbone for {freeze_epochs} epochs")
    for param in backbone_params:
        param.requires_grad = False

    best_acc, trigger_times = 0.0, 0
    BEST_CKPT = save_dir / 'best_model.pth'

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Unfreeze at freeze_epochs
        if epoch == freeze_epochs + 1:
            log.info("Phase 2: Unfreezing backbone")
            for param in backbone_params:
                param.requires_grad = True

        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', enabled=False):
                logits = model(x, keep_prob=0.9)
                loss = F.cross_entropy(logits, y, label_smoothing=0.1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)

        # Validation
        val_m = evaluate(model, val_loader, device)
        val_acc, val_top5, val_loss = val_m['acc'], val_m['top5'], val_m['val_loss']

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)
        eta_str = f"{int(eta//3600)}h{int((eta%3600)//60):02d}m"
        log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | "
                 f"LR: {optimizer.param_groups[1]['lr']:.2e} | "
                 f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                 f"Acc: {val_acc:.2f}% | Top5: {val_top5:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch, 'best_acc': best_acc,
                'num_classes': num_classes, 'stream': stream,
                'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label,
                'model_type': 'SAM-SLR-v2',
            }, BEST_CKPT)
            log.info(f"  ✨ Best! ({best_acc:.2f}%)")
        else:
            trigger_times += 1

        if trigger_times >= patience:
            log.info(f"🛑 Early stopping at epoch {epoch}")
            break

    # Test
    log.info(f"🏆 Best: {best_acc:.2f}%")
    best = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    test_m = evaluate(model, test_loader, device)
    log.info(f"🧪 Test: Top-1 {test_m['acc']:.2f}% | Top-5 {test_m['top5']:.2f}%")

    # Per-class accuracy
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
        miss_str = ", ".join(f"{p}({c})" for p, c in misses)
        log.info(f"  {lbl}: {acc:.1f}% ({per_class_correct.get(lbl,0)}/{per_class_total[lbl]}) -> {miss_str}")

    # Save results
    with open(save_dir / 'confusion_matrix.json', 'w') as f:
        json.dump(confusion, f, indent=2)
    # Save predictions for ensemble later
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y)
    all_logits = torch.cat(all_logits, 0)
    all_labels = torch.cat(all_labels, 0)
    torch.save({'logits': all_logits, 'labels': all_labels, 'stream': stream},
               save_dir / 'test_logits.pt')
    log.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--stream', default='joint',
                   choices=['joint', 'bone', 'joint_motion', 'bone_motion'])
    p.add_argument('--pretrained', default=None)
    p.add_argument('--data_path', default='ASL_landmarks_sam27')
    p.add_argument('--save_dir', default='/workspace/sam_joint')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--freeze_epochs', type=int, default=20)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    args = p.parse_args()

    train_stream(
        stream=args.stream,
        pretrained_path=args.pretrained,
        data_path=args.data_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
    )
