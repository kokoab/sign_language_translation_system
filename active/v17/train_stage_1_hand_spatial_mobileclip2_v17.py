#!/usr/bin/env python3
"""Fine-tune MobileCLIP2 pooling/projection after spatial temporal shift."""

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
    repo_root=Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:sys.path.insert(0,str(repo_root))
    from active.v17.extract_mobileclip2_v17 import build_encoder,select_device
    from active.v17.fine_tune_hand_mobileclip2_v17 import temporal_shift
    from active.v17.model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config,HandMobileCLIP2Stage1V17
    from active.v17.schema_hand_spatial_mobileclip2_v17 import HandSpatialMobileCLIP2V17Config,schema_fingerprint
    from active.v17.train_stage_1_hand_mobileclip2_v17 import supervised_contrastive
    from active.v17.train_stage_1_v17 import load_rejections
else:
    from .extract_mobileclip2_v17 import build_encoder,select_device
    from .fine_tune_hand_mobileclip2_v17 import temporal_shift
    from .model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config,HandMobileCLIP2Stage1V17
    from .schema_hand_spatial_mobileclip2_v17 import HandSpatialMobileCLIP2V17Config,schema_fingerprint
    from .train_stage_1_hand_mobileclip2_v17 import supervised_contrastive
    from .train_stage_1_v17 import load_rejections


LOG=logging.getLogger("stage1_hand_spatial_mobileclip2_v17")


def sha256_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HandSpatialDataset(Dataset):
    def __init__(self,root,split,manifest_path,rejection_path=None):
        if split not in ("train","val"):raise ValueError("the Citizen test split is sealed")
        root=Path(root);manifest=json.loads(Path(manifest_path).read_text());classes=sorted(manifest["classes"],key=lambda x:x["class_index"]);self.label_to_index={str(x["canonical_label"]):int(x["class_index"]) for x in classes};self.num_classes=len(classes);rejected=load_rejections(Path(rejection_path) if rejection_path else None);self.expected_schema=schema_fingerprint(HandSpatialMobileCLIP2V17Config());self.files=[];targets=[]
        for label,target in self.label_to_index.items():
            selected=[path for path in sorted((root/split/label).glob("*.hand_spatial_mobileclip2_v17.npz")) if (split,label,path.name.removesuffix(".hand_spatial_mobileclip2_v17.npz")+".mp4") not in rejected]
            if not selected:raise ValueError(f"no usable {split} spatial maps for {label}")
            self.files.extend(selected);targets.extend([target]*len(selected))
        self.targets=torch.tensor(targets,dtype=torch.long)
    def __len__(self):return len(self.files)
    def __getitem__(self,index):
        path=self.files[index]
        with np.load(path,allow_pickle=False) as z:
            metadata=json.loads(str(z["metadata_json"]));maps=z["spatial_maps"];valid=z["valid"].astype(np.bool_);boxes=z["boxes_normalized"].astype(np.float32)
        if metadata.get("schema_fingerprint")!=self.expected_schema or maps.shape!=(16,3,512,8,8):raise ValueError(f"{path}: spatial schema/shape mismatch")
        if not np.isfinite(maps).all() or not np.all(maps[~valid]==0):raise ValueError(f"{path}: invalid spatial values")
        return torch.from_numpy(maps.copy()),torch.from_numpy(valid),torch.from_numpy(boxes),self.targets[index]
    def balanced_subset(self,count):
        remaining={i:count for i in range(self.num_classes)};selected=[]
        for i,target in enumerate(self.targets.tolist()):
            if remaining[target]>0:selected.append(i);remaining[target]-=1
        if any(remaining.values()):raise ValueError("subset exceeds class count")
        return Subset(self,selected)


class SpatialTemporalMobileCLIP2V17(nn.Module):
    def __init__(self,final_conv,visual_head,temporal_head,residual_shift=False):
        super().__init__();self.final_conv=final_conv;self.visual_head=visual_head;self.temporal_head=temporal_head;self.residual_shift=residual_shift
        self.shift_gate=nn.Parameter(torch.zeros(())) if residual_shift else None
    def forward_features(self,maps,valid,boxes):
        batch,frames,views=valid.shape;base=maps.reshape(batch*frames*views,512,8,8);shifted=temporal_shift(base,valid)
        value=base+torch.tanh(self.shift_gate)*(shifted-base) if self.residual_shift else shifted
        value=self.final_conv(value);value=self.visual_head(value);value=F.normalize(value,dim=-1).reshape(batch,frames,views,512)*valid.unsqueeze(-1)
        return self.temporal_head.forward_features(value,valid,boxes)
    def forward(self,maps,valid,boxes):return self.temporal_head.classifier(self.forward_features(maps,valid,boxes))


def augment(maps,valid,boxes):
    maps=maps.clone();valid=valid.clone();boxes=boxes.clone();device=maps.device;batch,frames=maps.shape[:2]
    if torch.rand((),device=device)<.65:
        base=torch.linspace(0,1,frames,device=device)
        for sample in range(batch):
            rate=.84+.32*torch.rand((),device=device);offset=(torch.rand((),device=device)-.5)*.08;indices=(((base-.5)*rate+.5+offset).clamp(0,1)*(frames-1)).round().long();maps[sample]=maps[sample].index_select(0,indices);valid[sample]=valid[sample].index_select(0,indices);boxes[sample]=boxes[sample].index_select(0,indices)
    if torch.rand((),device=device)<.35:
        drop=(torch.rand_like(valid.float())<.06)&valid;valid&=~drop;maps[drop]=0;boxes[drop]=0
    if torch.rand((),device=device)<.5:maps=maps+torch.randn_like(maps)*.003*valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return maps,valid,boxes


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();logits_all=[];targets_all=[];loss_sum=0
    for maps,valid,boxes,targets in loader:
        logits=model(maps.to(device).float(),valid.to(device),boxes.to(device));
        if device.type=="mps":torch.mps.synchronize()
        logits=logits.cpu();logits_all.append(logits);targets_all.append(targets);loss_sum+=float(F.cross_entropy(logits,targets))*len(targets)
    logits=torch.cat(logits_all);targets=torch.cat(targets_all);pred=logits.argmax(1);top5=logits.topk(5,dim=1).indices;classes=model.temporal_head.config.num_classes;conf=np.zeros((classes,classes),dtype=np.int64);np.add.at(conf,(targets.numpy(),pred.numpy()),1);tp=np.diag(conf).astype(float);precision=tp/np.maximum(conf.sum(0),1);recall=tp/np.maximum(conf.sum(1),1);f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return {"loss":loss_sum/len(targets),"top1":100*float((pred==targets).float().mean()),"top5":100*float((top5==targets[:,None]).any(1).float().mean()),"macro_f1":100*float(f1.mean()),"samples":float(len(targets))}


def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);device=select_device(args.device);train_set=HandSpatialDataset(args.data_root,"train",args.manifest,args.rejections);val_set=HandSpatialDataset(args.data_root,"val",args.manifest,args.rejections);train_data,val_data=train_set,val_set;epochs=args.epochs
    if args.smoke:train_data=train_set.balanced_subset(1);val_data=val_set.balanced_subset(1);epochs=1;args.max_train_batches=2
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,persistent_workers=args.workers>0);val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,persistent_workers=args.workers>0)
    clip,_=build_encoder(device);frozen=torch.load(args.frozen_head,map_location="cpu",weights_only=False)
    if frozen.get("format")!="slt_stage1_hand_mobileclip2_v17":raise ValueError("frozen hand checkpoint mismatch")
    head=HandMobileCLIP2Stage1V17(HandMobileCLIP2Stage1Config(**frozen["model_config"]));head.load_state_dict(frozen["model_state_dict"])
    model=SpatialTemporalMobileCLIP2V17(clip.visual.trunk.final_conv,clip.visual.trunk.head,head,args.residual_shift).to(device);del clip
    visual_parameters=list(model.final_conv.parameters())+list(model.visual_head.parameters())+([model.shift_gate] if model.shift_gate is not None else [])
    optimizer=torch.optim.AdamW([{"params":visual_parameters,"lr":args.visual_lr},{"params":model.temporal_head.parameters(),"lr":args.head_lr}],weight_decay=args.weight_decay);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6);args.output.mkdir(parents=True,exist_ok=True);history=[];best=-1.;stale=0
    LOG.info("device=%s train=%d val=%d trainable=%d",device,len(train_data),len(val_data),sum(p.numel() for p in model.parameters()))
    if args.residual_shift and not args.smoke:
        metrics=evaluate(model,val_loader,device);row={"epoch":0,"train_loss":None,**metrics,"seconds":0.0};history.append(row);best=metrics["top1"]
        out={"format":"slt_stage1_hand_spatial_mobileclip2_v17","format_version":2,"epoch":0,"validation_metrics":metrics,"manifest_sha256":sha256_file(args.manifest),"schema_fingerprint":train_set.expected_schema,"model_config":model.temporal_head.config.to_dict(),"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"label_to_index":train_set.label_to_index,"residual_shift":True,"initialize_from":str(args.frozen_head),"initialize_from_sha256":sha256_file(args.frozen_head),"test_evaluated":False};temporary=args.output/"best_model.pth.tmp";torch.save(out,temporary);temporary.replace(args.output/"best_model.pth")
        LOG.info("epoch=0 exact residual baseline top1=%.2f top5=%.2f",metrics["top1"],metrics["top5"])
    for epoch in range(1,epochs+1):
        model.train();total=seen=0;started=time.monotonic()
        for batch_index,(maps,valid,boxes,targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index>=args.max_train_batches:break
            maps,valid,boxes=augment(maps.to(device).float(),valid.to(device),boxes.to(device));targets=targets.to(device);optimizer.zero_grad(set_to_none=True);features=model.forward_features(maps,valid,boxes);logits=model.temporal_head.classifier(features);loss=F.cross_entropy(logits,targets,label_smoothing=.1)+args.contrastive_weight*supervised_contrastive(features,targets);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();total+=float(loss.detach())*len(targets);seen+=len(targets)
        metrics=evaluate(model,val_loader,device);scheduler.step();row={"epoch":epoch,"train_loss":total/max(seen,1),**metrics,"seconds":time.monotonic()-started};history.append(row);LOG.info("epoch=%d train_loss=%.4f val_loss=%.4f top1=%.2f top5=%.2f macro_f1=%.2f seconds=%.1f",epoch,row["train_loss"],row["loss"],row["top1"],row["top5"],row["macro_f1"],row["seconds"])
        if metrics["top1"]>best:
            best=metrics["top1"];stale=0;out={"format":"slt_stage1_hand_spatial_mobileclip2_v17","format_version":2 if args.residual_shift else 1,"epoch":epoch,"validation_metrics":metrics,"manifest_sha256":sha256_file(args.manifest),"schema_fingerprint":train_set.expected_schema,"model_config":model.temporal_head.config.to_dict(),"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"label_to_index":train_set.label_to_index,"residual_shift":args.residual_shift,"initialize_from":str(args.frozen_head),"initialize_from_sha256":sha256_file(args.frozen_head),"test_evaluated":False};temporary=args.output/"best_model.pth.tmp";torch.save(out,temporary);temporary.replace(args.output/"best_model.pth")
        else:stale+=1
        (args.output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        if stale>=args.patience:LOG.info("early stopping after %d stale epochs",stale);break
    result={"best_validation_top1":best,"epochs_completed":sum(int(row["epoch"]>0) for row in history),"parameters":sum(p.numel() for p in model.parameters()),"residual_shift":args.residual_shift,"test_evaluated":False};(args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");return result


def build_parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,default=Path("data/local/citizen100_v17/hand_spatial_mobileclip2_s0"));p.add_argument("--manifest",type=Path,default=Path("active/v17/citizen100_manifest.json"));p.add_argument("--rejections",type=Path,default=Path("data/local/citizen100_v17/rejections.csv"));p.add_argument("--frozen-head",type=Path,default=Path("artifacts/models/stage1_v17_hand_mobileclip2_frozen/best_model.pth"));p.add_argument("--output",type=Path,default=Path("artifacts/models/stage1_v17_hand_mobileclip2_spatial"));p.add_argument("--epochs",type=int,default=80);p.add_argument("--patience",type=int,default=15);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--workers",type=int,default=2);p.add_argument("--visual-lr",type=float,default=2e-5);p.add_argument("--head-lr",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=.03);p.add_argument("--contrastive-weight",type=float,default=.05);p.add_argument("--device",default="auto");p.add_argument("--seed",type=int,default=1701);p.add_argument("--smoke",action="store_true");p.add_argument("--residual-shift",action="store_true");p.add_argument("--max-train-batches",type=int,default=0,help=argparse.SUPPRESS);return p


def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s")
    print(json.dumps(train(build_parser().parse_args()),indent=2))


if __name__=="__main__":main()
