#!/usr/bin/env python3
"""Fine-tune late MobileCLIP2 stages on real hand crops with temporal shift."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_hand_rgb_v17 import decode_packed_crops
    from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
    from active.v17.model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17
    from active.v17.schema_hand_rgb_v17 import CROP_SIZE, HandRGBV17Config, schema_fingerprint
    from active.v17.train_stage_1_hand_mobileclip2_v17 import supervised_contrastive
    from active.v17.train_stage_1_v17 import load_rejections
else:
    from .extract_hand_rgb_v17 import decode_packed_crops
    from .extract_mobileclip2_v17 import build_encoder, select_device
    from .model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17
    from .schema_hand_rgb_v17 import CROP_SIZE, HandRGBV17Config, schema_fingerprint
    from .train_stage_1_hand_mobileclip2_v17 import supervised_contrastive
    from .train_stage_1_v17 import load_rejections


LOG = logging.getLogger("finetune_hand_mobileclip2_v17")
VISUAL_FRAMES = 8
VIEWS = 3


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HandRGBPixelDataset(Dataset):
    def __init__(self, root, split, manifest_path, rejection_path=None):
        if split not in ("train", "val"):
            raise ValueError("the Citizen test split is sealed")
        self.root, self.split = Path(root), split
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        classes = sorted(manifest["classes"], key=lambda item: item["class_index"])
        self.label_to_index = {str(item["canonical_label"]): int(item["class_index"]) for item in classes}
        self.num_classes = len(classes)
        rejected = load_rejections(Path(rejection_path) if rejection_path else None)
        self.files = []
        targets = []
        for label, target in self.label_to_index.items():
            class_root = self.root / split / label
            selected = [
                path for path in sorted(class_root.glob("*.hand_rgb_v17.npz"))
                if (split, label, path.name.removesuffix(".hand_rgb_v17.npz") + ".mp4") not in rejected
            ]
            if not selected:
                raise ValueError(f"no usable {split} crops for {label}")
            self.files.extend(selected); targets.extend([target] * len(selected))
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.expected_schema = schema_fingerprint(HandRGBV17Config())

    def __len__(self): return len(self.files)

    def _frame_indices(self):
        boundaries = np.linspace(0, 16, VISUAL_FRAMES + 1, dtype=int)
        if self.split == "train":
            return np.asarray([
                np.random.randint(boundaries[index], max(boundaries[index] + 1, boundaries[index + 1]))
                for index in range(VISUAL_FRAMES)
            ])
        return np.rint(np.linspace(0, 15, VISUAL_FRAMES)).astype(int)

    def __getitem__(self, index):
        path = self.files[index]
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            if metadata.get("schema_fingerprint") != self.expected_schema:
                raise ValueError(f"{path}: crop schema mismatch")
            frames = self._frame_indices()
            crops = decode_packed_crops(payload["jpeg_blob"], payload["jpeg_offsets"], CROP_SIZE)[frames]
            valid = payload["valid"][frames].astype(np.bool_)
            boxes = payload["boxes_normalized"][frames].astype(np.float32)
        pixels = torch.from_numpy(crops.copy()).permute(0, 1, 4, 2, 3).float().div_(255.0)
        return pixels, torch.from_numpy(valid), torch.from_numpy(boxes), self.targets[index]

    def balanced_subset(self, count):
        remaining={index:count for index in range(self.num_classes)};selected=[]
        for index,target in enumerate(self.targets.tolist()):
            if remaining[target]>0:selected.append(index);remaining[target]-=1
        if any(remaining.values()):raise ValueError("subset exceeds class count")
        return Subset(self,selected)


def temporal_shift(value: torch.Tensor, valid: torch.Tensor, fold_div: int = 8) -> torch.Tensor:
    """Bidirectional TSM over time, independently for every hand view."""
    batch, frames, views = valid.shape
    _, channels, height, width = value.shape
    structured = value.reshape(batch, frames, views, channels, height, width).permute(0, 2, 1, 3, 4, 5)
    mask = valid.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    structured = structured * mask
    output = torch.zeros_like(structured)
    fold = channels // fold_div
    output[:, :, :-1, :fold] = structured[:, :, 1:, :fold]
    output[:, :, 1:, fold:2 * fold] = structured[:, :, :-1, fold:2 * fold]
    output[:, :, :, 2 * fold:] = structured[:, :, :, 2 * fold:]
    output *= mask
    return output.permute(0, 2, 1, 3, 4, 5).reshape_as(value)


class FineTunedHandMobileCLIP2V17(nn.Module):
    def __init__(self, visual, temporal_head: HandMobileCLIP2Stage1V17):
        super().__init__()
        self.visual = visual
        self.temporal_head = temporal_head
        self._freeze_early_visual()

    def _freeze_early_visual(self):
        for module in (self.visual.trunk.stem, *list(self.visual.trunk.stages[:3])):
            module.requires_grad_(False); module.eval()
        self.visual.trunk.stages[3].requires_grad_(True)
        self.visual.trunk.final_conv.requires_grad_(True)
        self.visual.trunk.head.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in (self.visual.trunk.stem, *list(self.visual.trunk.stages[:3])):
            module.eval()
        return self

    def encode_visual(self, pixels, valid):
        batch, frames, views, channels, height, width = pixels.shape
        flattened = pixels.reshape(batch * frames * views, channels, height, width)
        with torch.no_grad():
            value = self.visual.trunk.stem(flattened)
            for stage in self.visual.trunk.stages[:3]: value = stage(value)
        value = temporal_shift(value, valid)
        value = self.visual.trunk.stages[3](value)
        value = self.visual.trunk.final_conv(value)
        value = self.visual.trunk.forward_head(value)
        value = F.normalize(value, dim=-1).reshape(batch, frames, views, -1)
        return value * valid.unsqueeze(-1)

    def forward_features(self, pixels, valid, boxes):
        embeddings8 = self.encode_visual(pixels, valid)
        # Repeat real observations to the frozen 16-position temporal-head contract.
        embeddings = embeddings8.repeat_interleave(2, dim=1)
        valid16 = valid.repeat_interleave(2, dim=1)
        boxes16 = boxes.repeat_interleave(2, dim=1)
        return self.temporal_head.forward_features(embeddings, valid16, boxes16)

    def forward(self, pixels, valid, boxes):
        return self.temporal_head.classifier(self.forward_features(pixels, valid, boxes))


def augment_pixels(pixels, valid, boxes):
    device=pixels.device;batch=pixels.shape[0]
    if torch.rand((),device=device)<.5:
        pixels=torch.flip(pixels,dims=(-1,));pixels=pixels[:,:,torch.tensor([1,0,2],device=device)]
        valid=valid[:,:,torch.tensor([1,0,2],device=device)];boxes=boxes[:,:,torch.tensor([1,0,2],device=device)].clone()
        old_x0=boxes[...,0].clone();boxes[...,0]=1-boxes[...,2];boxes[...,2]=1-old_x0
    gain=.85+.30*torch.rand(batch,1,1,3,1,1,device=device)
    bias=(torch.rand(batch,1,1,3,1,1,device=device)-.5)*.08
    pixels=(pixels*gain+bias).clamp(0,1)
    if torch.rand((),device=device)<.35:
        pixels=(pixels+torch.randn_like(pixels)*.015).clamp(0,1)
    pixels*=valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return pixels,valid,boxes


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();logits_all=[];targets_all=[];loss_sum=0
    for pixels,valid,boxes,targets in loader:
        logits=model(pixels.to(device),valid.to(device),boxes.to(device));
        if device.type=="mps":torch.mps.synchronize()
        logits=logits.cpu();logits_all.append(logits);targets_all.append(targets);loss_sum+=float(F.cross_entropy(logits,targets))*len(targets)
    logits=torch.cat(logits_all);targets=torch.cat(targets_all);pred=logits.argmax(1);top5=logits.topk(5,dim=1).indices
    confusion=np.zeros((model.temporal_head.config.num_classes,)*2,dtype=np.int64);np.add.at(confusion,(targets.numpy(),pred.numpy()),1)
    tp=np.diag(confusion).astype(float);precision=tp/np.maximum(confusion.sum(0),1);recall=tp/np.maximum(confusion.sum(1),1);f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return {"loss":loss_sum/len(targets),"top1":100*float((pred==targets).float().mean()),"top5":100*float((top5==targets[:,None]).any(1).float().mean()),"macro_f1":100*float(f1.mean()),"samples":float(len(targets))}


def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);device=select_device(args.device)
    train_set=HandRGBPixelDataset(args.data_root,"train",args.manifest,args.rejections);val_set=HandRGBPixelDataset(args.data_root,"val",args.manifest,args.rejections)
    train_data,val_data=train_set,val_set;epochs=args.epochs
    if args.smoke:train_data=train_set.balanced_subset(1);val_data=val_set.balanced_subset(1);epochs=1;args.max_train_batches=2
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,persistent_workers=args.workers>0)
    val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,persistent_workers=args.workers>0)
    clip_model,_=build_encoder(device);visual=clip_model.visual
    checkpoint=torch.load(args.frozen_head,map_location="cpu",weights_only=False)
    if checkpoint.get("format")!="slt_stage1_hand_mobileclip2_v17":raise ValueError("frozen hand-head checkpoint format mismatch")
    head=HandMobileCLIP2Stage1V17(HandMobileCLIP2Stage1Config(**checkpoint["model_config"]));head.load_state_dict(checkpoint["model_state_dict"])
    model=FineTunedHandMobileCLIP2V17(visual,head).to(device)
    visual_parameters=[p for p in model.visual.parameters() if p.requires_grad];head_parameters=list(model.temporal_head.parameters())
    optimizer=torch.optim.AdamW([{"params":visual_parameters,"lr":args.backbone_lr},{"params":head_parameters,"lr":args.head_lr}],weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
    args.output.mkdir(parents=True,exist_ok=True);history=[];best=-1.;stale=0
    LOG.info("device=%s train=%d val=%d trainable=%d",device,len(train_data),len(val_data),sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(1,epochs+1):
        model.train();total=seen=0;started=time.monotonic()
        for batch_index,(pixels,valid,boxes,targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index>=args.max_train_batches:break
            pixels,valid,boxes=augment_pixels(pixels.to(device),valid.to(device),boxes.to(device));targets=targets.to(device)
            optimizer.zero_grad(set_to_none=True);features=model.forward_features(pixels,valid,boxes);logits=model.temporal_head.classifier(features)
            loss=F.cross_entropy(logits,targets,label_smoothing=.1)+args.contrastive_weight*supervised_contrastive(features,targets)
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();total+=float(loss.detach())*len(targets);seen+=len(targets)
        metrics=evaluate(model,val_loader,device);scheduler.step();row={"epoch":epoch,"train_loss":total/max(seen,1),**metrics,"seconds":time.monotonic()-started};history.append(row)
        LOG.info("epoch=%d train_loss=%.4f val_loss=%.4f top1=%.2f top5=%.2f macro_f1=%.2f seconds=%.1f",epoch,row["train_loss"],row["loss"],row["top1"],row["top5"],row["macro_f1"],row["seconds"])
        if metrics["top1"]>best:
            best=metrics["top1"];stale=0;checkpoint_out={"format":"slt_stage1_hand_mobileclip2_finetuned_v17","format_version":1,"epoch":epoch,"validation_metrics":metrics,"manifest_sha256":sha256_file(args.manifest),"crop_schema_fingerprint":train_set.expected_schema,"model_config":model.temporal_head.config.to_dict(),"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"label_to_index":train_set.label_to_index,"visual_frames":VISUAL_FRAMES,"views":VIEWS,"test_evaluated":False};temporary=args.output/"best_model.pth.tmp";torch.save(checkpoint_out,temporary);temporary.replace(args.output/"best_model.pth")
        else:stale+=1
        (args.output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        if stale>=args.patience:LOG.info("early stopping after %d stale epochs",stale);break
    result={"best_validation_top1":best,"epochs_completed":len(history),"test_evaluated":False};(args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");return result


def build_parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,default=Path("data/local/citizen100_v17/hand_rgb"));p.add_argument("--manifest",type=Path,default=Path("active/v17/citizen100_manifest.json"));p.add_argument("--rejections",type=Path,default=Path("data/local/citizen100_v17/rejections.csv"));p.add_argument("--frozen-head",type=Path,default=Path("artifacts/models/stage1_v17_hand_mobileclip2_frozen/best_model.pth"));p.add_argument("--output",type=Path,default=Path("artifacts/models/stage1_v17_hand_mobileclip2_finetuned"));p.add_argument("--epochs",type=int,default=20);p.add_argument("--patience",type=int,default=6);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--workers",type=int,default=2);p.add_argument("--backbone-lr",type=float,default=1e-5);p.add_argument("--head-lr",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=.03);p.add_argument("--contrastive-weight",type=float,default=.05);p.add_argument("--device",default="auto");p.add_argument("--seed",type=int,default=1701);p.add_argument("--smoke",action="store_true");p.add_argument("--max-train-batches",type=int,default=0,help=argparse.SUPPRESS);return p


def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s");print(json.dumps(train(build_parser().parse_args()),indent=2))


if __name__=="__main__":main()
