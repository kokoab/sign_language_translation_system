#!/usr/bin/env python3
"""Train the frozen high-resolution hand-crop MobileCLIP2 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

# The hand replay model is small, but it still uses unified memory on Apple silicon.
# Establish conservative allocator watermarks before torch initializes MPS. Callers
# may lower these values, but an uncapped default is not allowed for this trainer.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.model_hand_mobileclip2_v17 import (
        HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17, make_checkpoint,
    )
    from active.v17.schema_hand_mobileclip2_v17 import HandMobileCLIP2V17Config, schema_fingerprint
    from active.v17.train_stage_1_v17 import ExponentialMovingAverage, load_rejections
    from active.v17.train_stage_1_mobileclip2_v17 import select_device
    from active.v17.extract_hand_rgb_supplement_v17 import selection_items
else:
    from .model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17, make_checkpoint
    from .schema_hand_mobileclip2_v17 import HandMobileCLIP2V17Config, schema_fingerprint
    from .train_stage_1_v17 import ExponentialMovingAverage, load_rejections
    from .train_stage_1_mobileclip2_v17 import select_device
    from .extract_hand_rgb_supplement_v17 import selection_items


LOG = logging.getLogger("stage1_hand_mobileclip2_v17")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_exact_hand_finetune(
    model, checkpoint_path, manifest_path, expected_schema, label_to_index
):
    """Strictly restore the selected hand expert for replay adaptation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_hand_mobileclip2_v17":
        raise ValueError("fine-tune checkpoint is not a v17 hand checkpoint")
    if checkpoint.get("manifest_sha256") != sha256_file(Path(manifest_path)):
        raise ValueError("fine-tune checkpoint Citizen manifest mismatch")
    if checkpoint.get("schema_fingerprint") != expected_schema:
        raise ValueError("fine-tune checkpoint hand schema mismatch")
    if checkpoint.get("label_to_index") != label_to_index:
        raise ValueError("fine-tune checkpoint label mapping mismatch")
    if checkpoint.get("model_config") != model.config.to_dict():
        raise ValueError("fine-tune checkpoint model config mismatch")
    provenance = checkpoint.get("training_data_provenance", {})
    if (
        checkpoint.get("test_evaluated") is not False
        or provenance.get("test_evaluated") is not False
    ):
        raise ValueError("fine-tune checkpoint lacks sealed-test provenance")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "mode": "exact_selected_checkpoint_replay_finetune",
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_validation_metrics": checkpoint.get("validation_metrics", {}),
        "strict_state_dict": True,
        "test_evaluated": False,
    }


class HandMobileCLIP2Dataset(Dataset):
    def __init__(self, root, split, manifest_path, rejection_path=None, *, cache=True):
        if split not in ("train", "val"):
            raise ValueError("the Citizen test split is sealed")
        root, manifest_path = Path(root), Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = sorted(manifest["classes"], key=lambda item: item["class_index"])
        self.label_to_index = {str(item["canonical_label"]): int(item["class_index"]) for item in classes}
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.num_classes = len(classes)
        self.expected_schema = schema_fingerprint(HandMobileCLIP2V17Config())
        rejected = load_rejections(Path(rejection_path) if rejection_path else None)
        self.files = []
        targets = []
        for label, target in self.label_to_index.items():
            class_root = root / split / label
            if not class_root.is_dir():
                raise FileNotFoundError(class_root)
            selected = [
                path for path in sorted(class_root.glob("*.hand_mobileclip2_v17.npz"))
                if (split, label, path.name.removesuffix(".hand_mobileclip2_v17.npz") + ".mp4") not in rejected
            ]
            if not selected:
                raise ValueError(f"no usable {split} samples for {label}")
            self.files.extend(selected)
            targets.extend([target] * len(selected))
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.source_name = "citizen"
        self._cache = [self._load(path) for path in self.files] if cache else None

    def _load(self, path):
        with np.load(path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            valid = payload["valid"].astype(np.bool_)
            boxes = payload["boxes_normalized"].astype(np.float32)
            metadata = json.loads(str(payload["metadata_json"]))
        if embeddings.shape != (16, 3, 512) or valid.shape != (16, 3) or boxes.shape != (16, 3, 4):
            raise ValueError(f"{path}: hand embedding shape mismatch")
        if metadata.get("schema_fingerprint") != self.expected_schema:
            raise ValueError(f"{path}: hand embedding schema mismatch")
        if not np.isfinite(embeddings).all() or not np.isfinite(boxes).all():
            raise ValueError(f"{path}: non-finite hand features")
        if not np.all(embeddings[~valid] == 0):
            raise ValueError(f"{path}: invalid hand views must remain zero")
        return tuple(torch.from_numpy(value.copy()) for value in (embeddings, valid, boxes))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        values = self._cache[index] if self._cache is not None else self._load(self.files[index])
        return (*values, self.targets[index])

    def balanced_subset(self, samples_per_class):
        remaining = {index: samples_per_class for index in range(self.num_classes)}
        selected = []
        for index, target in enumerate(self.targets.tolist()):
            if remaining[target] > 0:
                selected.append(index); remaining[target] -= 1
        if any(remaining.values()):
            raise ValueError("subset exceeds a class count")
        return Subset(self, selected)


class HandMobileCLIP2SupplementDataset(HandMobileCLIP2Dataset):
    """Exact-manifest loader for train-only SemLex or Tier-A local embeddings."""

    def __init__(self, root, source, selection_manifest, label_to_index, *, cache=True):
        self.source_name = source
        self.label_to_index = dict(label_to_index)
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.num_classes = len(self.label_to_index)
        self.expected_schema = schema_fingerprint(HandMobileCLIP2V17Config())
        items, _ = selection_items(Path(selection_manifest), source)
        self.files = []
        targets = []
        root = Path(root)
        for item in items:
            if item.label not in self.label_to_index:
                raise ValueError(f"unknown supplement label: {item.label}")
            path = (
                root / source / item.label
                / f"{item.item_id}.hand_mobileclip2_v17.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            self.files.append(path)
            targets.append(self.label_to_index[item.label])
        self.targets = torch.tensor(targets, dtype=torch.long)
        self._cache = [self._load(path) for path in self.files] if cache else None


class HandMobileCLIP2LocalValidationDataset(HandMobileCLIP2Dataset):
    """Strict familiar-signer local validation; never eligible for training."""

    def __init__(self, root, selection_manifest, label_to_index, *, cache=True):
        self.source_name = "local_validation"
        self.label_to_index = dict(label_to_index)
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.num_classes = len(self.label_to_index)
        self.expected_schema = schema_fingerprint(HandMobileCLIP2V17Config())
        items, _ = selection_items(
            Path(selection_manifest), "local_deep_clean_val"
        )
        self.files = []
        targets = []
        root = Path(root)
        for item in items:
            if item.label not in self.label_to_index:
                raise ValueError(f"unknown local validation label: {item.label}")
            path = root / item.label / f"{item.item_id}.hand_mobileclip2_v17.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            self.files.append(path)
            targets.append(self.label_to_index[item.label])
        self.targets = torch.tensor(targets, dtype=torch.long)
        self._cache = [self._load(path) for path in self.files] if cache else None

    def _load(self, path):
        values = HandMobileCLIP2Dataset._load(self, path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        if (
            metadata.get("source") != "local_deep_clean_val"
            or metadata.get("split")
            != "validation_nonsigner_disjoint_user_approved"
            or metadata.get("training_eligible") is not False
            or metadata.get("test_accessed") is not False
        ):
            raise ValueError(f"local validation hand provenance mismatch: {path}")
        return values


def source_balanced_weights(datasets, margins):
    if len(datasets) != len(margins) or any(value <= 0 for value in margins):
        raise ValueError("source margins must be positive and aligned")
    total_margin = float(sum(margins))
    weights = []
    for dataset, margin in zip(datasets, margins):
        targets = dataset.targets.numpy()
        classes, counts = np.unique(targets, return_counts=True)
        count_by_class = dict(zip(classes.tolist(), counts.tolist()))
        source_mass = float(margin) / total_margin
        weights.extend(
            source_mass / (len(classes) * count_by_class[int(target)])
            for target in targets
        )
    return torch.tensor(weights, dtype=torch.double)


def checkpoint_improves_citizen_then_local_tie(
    citizen_top1: float,
    local_top1: float | None,
    best_citizen_top1: float,
    best_local_top1: float,
) -> bool:
    """Keep Citizen primary; local may decide only an exact Citizen tie."""
    return citizen_top1 > best_citizen_top1 or (
        citizen_top1 == best_citizen_top1
        and local_top1 is not None
        and local_top1 > best_local_top1
    )


def augment(embeddings, valid, boxes):
    output, output_valid, output_boxes = embeddings.clone(), valid.clone(), boxes.clone()
    batch, frames = output.shape[:2]
    device = output.device
    if torch.rand((), device=device) < 0.65:
        base = torch.linspace(0, 1, frames, device=device)
        for sample in range(batch):
            rate = 0.84 + 0.32 * torch.rand((), device=device)
            offset = (torch.rand((), device=device) - 0.5) * 0.08
            indices = (((base - 0.5) * rate + 0.5 + offset).clamp(0, 1) * (frames - 1)).round().long()
            output[sample] = output[sample].index_select(0, indices)
            output_valid[sample] = output_valid[sample].index_select(0, indices)
            output_boxes[sample] = output_boxes[sample].index_select(0, indices)
    if torch.rand((), device=device) < 0.35:
        drop = (torch.rand_like(output_valid.float()) < 0.08) & output_valid
        output_valid &= ~drop
        output[drop] = 0
        output_boxes[drop] = 0
    if torch.rand((), device=device) < 0.5:
        noise = torch.randn_like(output) * 0.005
        output = F.normalize(output + noise * output_valid.unsqueeze(-1), dim=-1) * output_valid.unsqueeze(-1)
    return output, output_valid, output_boxes


def supervised_contrastive(features, targets, temperature=0.10):
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = targets[:, None].eq(targets[None, :]) & ~eye
    if not positive.any():
        return features.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * ~eye
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    valid_rows = positive.any(dim=1)
    mean_positive = (log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return -mean_positive[valid_rows].mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); logits_all=[]; targets_all=[]; loss_sum=0
    for embeddings, valid, boxes, targets in loader:
        logits = model(embeddings.to(device), valid.to(device), boxes.to(device))
        if device.type == "mps": torch.mps.synchronize()
        logits_all.append(logits.cpu()); targets_all.append(targets); loss_sum += float(F.cross_entropy(logits.cpu(), targets)) * len(targets)
    logits = torch.cat(logits_all); targets = torch.cat(targets_all)
    predicted = logits.argmax(1); top5=logits.topk(5,dim=1).indices
    confusion=np.zeros((model.config.num_classes,model.config.num_classes),dtype=np.int64)
    np.add.at(confusion,(targets.numpy(),predicted.numpy()),1)
    tp=np.diag(confusion).astype(float); precision=tp/np.maximum(confusion.sum(0),1); recall=tp/np.maximum(confusion.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return {"loss":loss_sum/len(targets),"top1":100*float((predicted==targets).float().mean()),"top5":100*float((top5==targets[:,None]).any(1).float().mean()),"macro_f1":100*float(f1.mean()),"samples":float(len(targets))}


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=select_device(args.device)
    if device.type == "mps":
        if not 0.0 < args.mps_memory_fraction <= 0.25:
            raise ValueError("MPS memory fraction must be in (0, 0.25]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    if args.local_data_root and not args.no_cache:
        raise ValueError(
            "local deep-clean hand replay must use --no-cache to keep host memory bounded"
        )
    train_set=HandMobileCLIP2Dataset(args.data_root,"train",args.manifest,args.rejections,cache=not args.no_cache)
    val_set=HandMobileCLIP2Dataset(args.data_root,"val",args.manifest,args.rejections,cache=not args.no_cache)
    train_sets=[train_set]
    if args.semlex_data_root or args.semlex_manifest:
        if not args.semlex_data_root or not args.semlex_manifest:
            raise ValueError("SemLex data root and manifest must be provided together")
        train_sets.append(HandMobileCLIP2SupplementDataset(args.semlex_data_root,"semlex",args.semlex_manifest,train_set.label_to_index,cache=not args.no_cache))
    if args.local_data_root or args.local_manifest:
        if not args.local_data_root or not args.local_manifest:
            raise ValueError("local data root and manifest must be provided together")
        train_sets.append(HandMobileCLIP2SupplementDataset(args.local_data_root,args.local_source,args.local_manifest,train_set.label_to_index,cache=not args.no_cache))
    has_local_validation = bool(
        args.local_validation_data_root or args.local_validation_manifest
    )
    if has_local_validation and not (
        args.local_validation_data_root and args.local_validation_manifest
    ):
        raise ValueError(
            "local validation data root and manifest must be provided together"
        )
    local_validation_set = (
        HandMobileCLIP2LocalValidationDataset(
            args.local_validation_data_root,
            args.local_validation_manifest,
            train_set.label_to_index,
            cache=not args.no_cache,
        )
        if has_local_validation else None
    )
    train_data=ConcatDataset(train_sets) if len(train_sets)>1 else train_set
    val_data=val_set; dim,depth,epochs=args.dim,args.depth,args.epochs
    if args.smoke:
        train_data=train_set.balanced_subset(2); val_data=val_set.balanced_subset(1); dim,depth,epochs=64,1,1; args.max_train_batches=2
    requested={"citizen":args.citizen_margin,"semlex":args.semlex_margin,args.local_source:args.local_margin}
    sampler=None
    if len(train_sets)>1 and not args.smoke:
        weights=source_balanced_weights(train_sets,[requested[dataset.source_name] for dataset in train_sets])
        sampler=WeightedRandomSampler(weights,len(train_data),replacement=True,generator=torch.Generator().manual_seed(args.seed))
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=0)
    val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=0)
    local_validation_loader=(
        DataLoader(
            local_validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if local_validation_set is not None else None
    )
    config=HandMobileCLIP2Stage1Config(num_classes=train_set.num_classes,dim=dim,depth=depth,heads=args.heads if dim%args.heads==0 else 4)
    model=HandMobileCLIP2Stage1V17(config).to(device)
    initialization=None
    initial_metrics=None
    initial_local_metrics=None
    if args.fine_tune_from:
        initialization=initialize_exact_hand_finetune(
            model,args.fine_tune_from,args.manifest,train_set.expected_schema,
            train_set.label_to_index,
        )
        initial_metrics=evaluate(model,val_loader,device)
        initial_local_metrics=(
            evaluate(model,local_validation_loader,device)
            if local_validation_loader is not None else None
        )
        initialization["initial_validation_metrics"]=initial_metrics
        initialization["initial_local_validation_metrics"]=initial_local_metrics
        LOG.info("exact_fine_tune=%s initial_top1=%.2f",args.fine_tune_from,initial_metrics["top1"])
    ema=ExponentialMovingAverage(model,args.ema_decay)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    warmup=max(1,min(args.warmup_epochs,epochs))
    def scale(epoch):
        if epoch<warmup:return (epoch+1)/warmup
        progress=(epoch-warmup)/max(epochs-warmup,1);return .02+.98*.5*(1+np.cos(np.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
    args.output.mkdir(parents=True,exist_ok=True); history=[];best=initial_metrics["top1"] if initial_metrics else -1.;best_local=(initial_local_metrics["top1"] if initial_local_metrics else -1.);stale=0
    provenance={
        "sources":{dataset.source_name:len(dataset) for dataset in train_sets},
        "source_margins":(
            {dataset.source_name:requested[dataset.source_name] for dataset in train_sets}
            if len(train_sets)>1 else {"citizen":1.0}
        ),
        "semlex_manifest_sha256":sha256_file(args.semlex_manifest) if args.semlex_manifest else None,
        "local_manifest_sha256":sha256_file(args.local_manifest) if args.local_manifest else None,
        "local_validation_manifest_sha256":sha256_file(args.local_validation_manifest) if args.local_validation_manifest else None,
        "local_validation_samples":len(local_validation_set) if local_validation_set is not None else 0,
        "local_validation_signer_disjoint":False if local_validation_set is not None else None,
        "local_validation_signer_overlap_user_approved":True if local_validation_set is not None else None,
        "local_validation_training_eligible":False if local_validation_set is not None else None,
        "citizen_test_accessed":False,
        "semlex_test_accessed":False,
        "local_test_accessed":False,
        "test_evaluated":False,
        "initialization":initialization,
        "optimization":{
            "mode":"balanced_replay_domain_adaptation" if initialization else "from_scratch",
            "checkpoint_selection":"citizen_official_validation_top1_then_exact_tie_local_top1",
            "local_validation_used_for_selection":"exact_citizen_top1_ties_only",
            "seed":args.seed,
            "maximum_epochs":epochs,
            "patience":args.patience,
            "peak_learning_rate":args.lr,
            "warmup_epochs":args.warmup_epochs,
            "batch_size":args.batch_size,
            "cache_features":not args.no_cache,
            "mps_memory_fraction":args.mps_memory_fraction if device.type == "mps" else None,
            "sampler":"class_and_source_balanced_with_replacement" if sampler is not None else "shuffle",
        },
    }
    if initial_metrics is not None:
        state={key:value.detach().cpu().clone() for key,value in ema.shadow.items()}
        checkpoint=make_checkpoint(model,state,epoch=0,validation_metrics=initial_metrics,label_to_index=train_set.label_to_index,manifest_sha256=sha256_file(args.manifest),schema_fingerprint=train_set.expected_schema);checkpoint["training_data_provenance"]=provenance
        if initial_local_metrics is not None:
            checkpoint["local_validation_metrics"]=initial_local_metrics
        temporary=args.output/"best_model.pth.tmp";torch.save(checkpoint,temporary);temporary.replace(args.output/"best_model.pth")
        (args.output/"initialization_metrics.json").write_text(json.dumps(initial_metrics,indent=2)+"\n")
        if initial_local_metrics is not None:
            (args.output/"initialization_local_metrics.json").write_text(json.dumps(initial_local_metrics,indent=2)+"\n")
    LOG.info("device=%s train=%d val=%d local_val=%d parameters=%d sources=%s",device,len(train_data),len(val_data),len(local_validation_set) if local_validation_set is not None else 0,model.parameter_count,provenance["sources"])
    for epoch in range(1,epochs+1):
        model.train(); total=seen=0;started=time.monotonic()
        for batch_index,(embeddings,valid,boxes,targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index>=args.max_train_batches:break
            embeddings,valid,boxes=augment(embeddings.to(device),valid.to(device),boxes.to(device));targets=targets.to(device)
            optimizer.zero_grad(set_to_none=True); features=model.forward_features(embeddings,valid,boxes); logits=model.classifier(features)
            ce=F.cross_entropy(logits,targets,label_smoothing=args.label_smoothing); contrast=supervised_contrastive(features,targets)
            loss=ce+args.contrastive_weight*contrast;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();ema.update(model)
            total+=float(loss.detach())*len(targets);seen+=len(targets)
        live={k:v.detach().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema.shadow);metrics=evaluate(model,val_loader,device);local_metrics=(evaluate(model,local_validation_loader,device) if local_validation_loader is not None else None);model.load_state_dict(live);scheduler.step()
        row={"epoch":epoch,"train_loss":total/max(seen,1),**metrics,"lr":optimizer.param_groups[0]["lr"],"seconds":time.monotonic()-started}
        if local_metrics is not None:
            row.update({f"local_val_{key}":value for key,value in local_metrics.items()})
        history.append(row)
        LOG.info("epoch=%d train_loss=%.4f val_loss=%.4f top1=%.2f top5=%.2f macro_f1=%.2f seconds=%.1f",epoch,row["train_loss"],row["loss"],row["top1"],row["top5"],row["macro_f1"],row["seconds"])
        if local_metrics is not None:
            LOG.info("epoch=%d local_top1=%.2f local_top5=%.2f local_macro_f1=%.2f",epoch,local_metrics["top1"],local_metrics["top5"],local_metrics["macro_f1"])
        selected=checkpoint_improves_citizen_then_local_tie(
            metrics["top1"],
            local_metrics["top1"] if local_metrics is not None else None,
            best,
            best_local,
        )
        if selected:
            best=metrics["top1"];best_local=local_metrics["top1"] if local_metrics is not None else best_local;stale=0;state={k:v.detach().cpu().clone() for k,v in ema.shadow.items()}
            checkpoint=make_checkpoint(model,state,epoch=epoch,validation_metrics=metrics,label_to_index=train_set.label_to_index,manifest_sha256=sha256_file(args.manifest),schema_fingerprint=train_set.expected_schema);checkpoint["training_data_provenance"]=provenance
            if local_metrics is not None:
                checkpoint["local_validation_metrics"]=local_metrics
            temporary=args.output/"best_model.pth.tmp";torch.save(checkpoint,temporary);temporary.replace(args.output/"best_model.pth")
        else:stale+=1
        (args.output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        if stale>=args.patience:LOG.info("early stopping after %d stale epochs",stale);break
    result={"best_validation_top1":best,"best_local_validation_top1_at_selected_checkpoint":best_local if local_validation_set is not None else None,"epochs_completed":len(history),"parameters":model.parameter_count,"device":str(device),"training_data_provenance":provenance,"test_evaluated":False}
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");return result


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("data/local/citizen100_v17/hand_mobileclip2_s0"));parser.add_argument("--manifest",type=Path,default=Path("active/v17/citizen100_manifest.json"));parser.add_argument("--rejections",type=Path,default=Path("data/local/citizen100_v17/rejections.csv"));parser.add_argument("--output",type=Path,default=Path("artifacts/models/stage1_v17_hand_mobileclip2_frozen"))
    parser.add_argument("--epochs",type=int,default=160);parser.add_argument("--batch-size",type=int,default=64);parser.add_argument("--lr",type=float,default=3e-4);parser.add_argument("--weight-decay",type=float,default=.03);parser.add_argument("--warmup-epochs",type=int,default=8);parser.add_argument("--label-smoothing",type=float,default=.1);parser.add_argument("--contrastive-weight",type=float,default=.10);parser.add_argument("--patience",type=int,default=30);parser.add_argument("--ema-decay",type=float,default=.999);parser.add_argument("--dim",type=int,default=256);parser.add_argument("--depth",type=int,default=3);parser.add_argument("--heads",type=int,default=8);parser.add_argument("--device",default="auto");parser.add_argument("--mps-memory-fraction",type=float,default=.12);parser.add_argument("--seed",type=int,default=1701);parser.add_argument("--no-cache",action="store_true");parser.add_argument("--smoke",action="store_true");parser.add_argument("--max-train-batches",type=int,default=0,help=argparse.SUPPRESS);parser.add_argument("--fine-tune-from",type=Path,help="Strict selected hand checkpoint for balanced replay adaptation")
    parser.add_argument("--semlex-data-root",type=Path);parser.add_argument("--semlex-manifest",type=Path);parser.add_argument("--local-data-root",type=Path);parser.add_argument("--local-manifest",type=Path);parser.add_argument("--local-source",choices=("local_tier_a","local_deep_clean"),default="local_tier_a");parser.add_argument("--citizen-margin",type=float,default=.45);parser.add_argument("--semlex-margin",type=float,default=.45);parser.add_argument("--local-margin",type=float,default=.10)
    parser.add_argument("--local-validation-data-root",type=Path)
    parser.add_argument("--local-validation-manifest",type=Path)
    return parser


def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s");print(json.dumps(train(build_parser().parse_args()),indent=2))


if __name__=="__main__":main()
