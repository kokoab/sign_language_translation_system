#!/usr/bin/env python3
"""Train zero-initialized hand-feature residuals over frozen Apple logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset


def load_aligned(landmark_path,hand_path):
    with np.load(landmark_path,allow_pickle=False) as z:l={key:z[key] for key in ("features","logits","targets","item_ids")}
    with np.load(hand_path,allow_pickle=False) as z:h={key:z[key] for key in ("features","logits","targets","item_ids")}
    lookup={str(value):index for index,value in enumerate(h["item_ids"])}
    if set(lookup)!=set(map(str,l["item_ids"])):raise ValueError("fusion item IDs differ")
    order=np.asarray([lookup[str(value)] for value in l["item_ids"]]);
    if not np.array_equal(l["targets"],h["targets"][order]):raise ValueError("fusion targets differ")
    return l["features"],h["features"][order],l["logits"],l["targets"],l["item_ids"]


class GatedFeatureResidual(nn.Module):
    def __init__(self,dim=256,classes=100):
        super().__init__();self.landmark_norm=nn.LayerNorm(dim);self.hand_norm=nn.LayerNorm(dim);self.hand_projection=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Dropout(.2));self.gate=nn.Sequential(nn.Linear(dim*2,dim//4),nn.GELU(),nn.Linear(dim//4,1),nn.Sigmoid());self.residual=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Dropout(.3),nn.Linear(dim,classes));nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,landmark,hand,base_logits):
        landmark=self.landmark_norm(landmark);hand=self.hand_projection(self.hand_norm(hand));combined=torch.cat((landmark,hand),dim=1);return base_logits+self.gate(combined)*self.residual(combined)


@torch.no_grad()
def metrics(model,loader,device):
    model.eval();logits=[];targets=[]
    for landmark,hand,base,target in loader:logits.append(model(landmark.to(device),hand.to(device),base.to(device)).cpu());targets.append(target)
    logits=torch.cat(logits);targets=torch.cat(targets);pred=logits.argmax(1);top5=logits.topk(5,dim=1).indices;classes=logits.shape[1];conf=np.zeros((classes,classes),dtype=np.int64);np.add.at(conf,(targets.numpy(),pred.numpy()),1);tp=np.diag(conf).astype(float);precision=tp/np.maximum(conf.sum(0),1);recall=tp/np.maximum(conf.sum(1),1);f1=2*precision*recall/np.maximum(precision+recall,1e-12);return {"top1":100*float((pred==targets).float().mean()),"top5":100*float((top5==targets[:,None]).any(1).float().mean()),"macro_f1":100*float(f1.mean()),"samples":int(len(targets))},logits.numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--landmark-train",type=Path,required=True);p.add_argument("--hand-train",type=Path,required=True);p.add_argument("--landmark-val",type=Path,required=True);p.add_argument("--hand-val",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("artifacts/models/stage1_v17_hand_feature_fusion"));p.add_argument("--epochs",type=int,default=120);p.add_argument("--patience",type=int,default=20);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--seed",type=int,default=1701);p.add_argument("--device",default="mps");args=p.parse_args();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);device=torch.device(args.device)
    train=load_aligned(args.landmark_train,args.hand_train);val=load_aligned(args.landmark_val,args.hand_val);train_dataset=TensorDataset(*(torch.from_numpy(value) for value in train[:4]));val_dataset=TensorDataset(*(torch.from_numpy(value) for value in val[:4]));train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True);val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False);model=GatedFeatureResidual(classes=train[2].shape[1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.03);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-6);args.output.mkdir(parents=True,exist_ok=True);history=[];best=-1;stale=0
    base_top1=100*float((val[2].argmax(1)==val[3]).mean())
    for epoch in range(1,args.epochs+1):
        model.train();total=seen=0;started=time.monotonic()
        for landmark,hand,base,target in train_loader:
            landmark,hand,base,target=landmark.to(device),hand.to(device),base.to(device),target.to(device);optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(landmark,hand,base),target,label_smoothing=.05);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();total+=float(loss.detach())*len(target);seen+=len(target)
        result,predictions=metrics(model,val_loader,device);scheduler.step();row={"epoch":epoch,"train_loss":total/seen,**result,"seconds":time.monotonic()-started};history.append(row)
        if result["top1"]>best:best=result["top1"];stale=0;torch.save({"format":"slt_stage1_hand_feature_fusion_v17","epoch":epoch,"validation_metrics":result,"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"test_evaluated":False},args.output/"best_model.pth")
        else:stale+=1
        (args.output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        if epoch==1 or epoch%10==0:print(json.dumps(row))
        if stale>=args.patience:break
    checkpoint=torch.load(args.output/"best_model.pth",map_location=device,weights_only=False);model.load_state_dict(checkpoint["model_state_dict"]);final,predictions=metrics(model,val_loader,device);apple=val[2].argmax(1);fused=predictions.argmax(1);targets=val[3];paired={"both_correct":int(((apple==targets)&(fused==targets)).sum()),"apple_only":int(((apple==targets)&(fused!=targets)).sum()),"fusion_only":int(((apple!=targets)&(fused==targets)).sum()),"both_wrong":int(((apple!=targets)&(fused!=targets)).sum())};result={"apple_baseline_top1":base_top1,"best_fusion":final,"paired":paired,"epochs_completed":len(history),"test_evaluated":False};(args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
