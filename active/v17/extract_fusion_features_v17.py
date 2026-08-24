#!/usr/bin/env python3
"""Extract aligned landmark or hand-spatial pooled features for gated fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root=Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:sys.path.insert(0,str(repo_root))
    from active.v17.extract_mobileclip2_v17 import build_encoder,select_device
    from active.v17.model_v17 import SLTStage1V17,Stage1V17Config
    from active.v17.model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config,HandMobileCLIP2Stage1V17
    from active.v17.train_stage_1_v17 import Citizen100V17Dataset
    from active.v17.train_stage_1_hand_spatial_mobileclip2_v17 import HandSpatialDataset,SpatialTemporalMobileCLIP2V17
else:
    from .extract_mobileclip2_v17 import build_encoder,select_device
    from .model_v17 import SLTStage1V17,Stage1V17Config
    from .model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config,HandMobileCLIP2Stage1V17
    from .train_stage_1_v17 import Citizen100V17Dataset
    from .train_stage_1_hand_spatial_mobileclip2_v17 import HandSpatialDataset,SpatialTemporalMobileCLIP2V17


def item_id(path: Path) -> str:
    name=path.name
    for suffix in (".hand_spatial_mobileclip2_v17.npz",".v17.npz"):
        if name.endswith(suffix):name=name.removesuffix(suffix);break
    return f"{path.parent.name}/{name}"


@torch.no_grad()
def landmark_features(args,device):
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False);dataset=Citizen100V17Dataset(args.data_root,args.split,args.manifest,args.rejections,cache=True);model=SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]));model.load_state_dict(checkpoint["model_state_dict"]);model.to(device).eval();features=[];logits=[]
    for value,_ in DataLoader(dataset,batch_size=args.batch_size,shuffle=False):
        batch_logits, pooled = model(value.to(device), return_embeddings=True)
        features.append(pooled.cpu().numpy())
        logits.append(batch_logits.cpu().numpy())
    return dataset,np.concatenate(features),np.concatenate(logits)


@torch.no_grad()
def hand_features(args,device):
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False);dataset=HandSpatialDataset(args.data_root,args.split,args.manifest,args.rejections);clip,_=build_encoder(device);head=HandMobileCLIP2Stage1V17(HandMobileCLIP2Stage1Config(**checkpoint["model_config"]));model=SpatialTemporalMobileCLIP2V17(clip.visual.trunk.final_conv,clip.visual.trunk.head,head);model.load_state_dict(checkpoint["model_state_dict"]);model.to(device).eval();features=[];logits=[]
    for maps,valid,boxes,_ in DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers):
        pooled=model.forward_features(maps.to(device).float(),valid.to(device),boxes.to(device));features.append(pooled.cpu().numpy());logits.append(model.temporal_head.classifier(pooled).cpu().numpy())
    return dataset,np.concatenate(features),np.concatenate(logits)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",required=True,choices=("landmark","hand_spatial"));p.add_argument("--split",required=True,choices=("train","val"));p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--manifest",type=Path,default=Path("active/v17/citizen100_manifest.json"));p.add_argument("--rejections",type=Path,default=Path("data/local/citizen100_v17/rejections.csv"));p.add_argument("--output",type=Path,required=True);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--workers",type=int,default=2);p.add_argument("--device",default="auto");args=p.parse_args();device=select_device(args.device)
    dataset,features,logits=landmark_features(args,device) if args.mode=="landmark" else hand_features(args,device)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,features=features.astype(np.float32),logits=logits.astype(np.float32),targets=dataset.targets.numpy(),item_ids=np.asarray([item_id(path) for path in dataset.files]),mode=np.array(args.mode),split=np.array(args.split),checkpoint=np.array(str(args.checkpoint)));print(json.dumps({"mode":args.mode,"split":args.split,"samples":len(dataset),"feature_shape":list(features.shape),"test_accessed":False},indent=2))


if __name__=="__main__":main()
