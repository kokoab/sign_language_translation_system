import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from active.v17.pretrained_boundary_v17 import *
from active.v17.temporal_boundary_v17 import boundary_targets,masked_boundary_loss
manifest=json.loads((UPSTREAM/'prepared_manifest.json').read_text())
counts={}
for row in manifest['records']:
 n=int(np.floor((row['pose_frames']-1)/row['pose_fps']*FPS+1e-7))+1
 y=boundary_targets(np.arange(n)/FPS,row['intervals'],[],False)
 counts[row['split']]=counts.get(row['split'],0)+int((y>=0).any(1).sum())
row=manifest['records'][0];pose=load_pose(row)
tick=time.perf_counter();xs=[window_features(pose,min(i,len(pose.body.data)-1))[0] for i in range(16)]
normalization=(time.perf_counter()-tick)/16
model=PretrainedBoundary().to('mps').eval();x=torch.from_numpy(np.stack(xs)).to('mps');ts=torch.arange(64,device='mps')[None].expand(16,-1)/20
with torch.no_grad():
 for _ in range(2):z=model.project(x);h=model.encode_projected(z,ts);torch.mps.synchronize()
 timing=[]
 for _ in range(5):
  tick=time.perf_counter();z=model.project(x);h=model.encode_projected(z,ts);torch.mps.synchronize();timing.append(time.perf_counter()-tick)
z=z.detach();last_input=z
with torch.no_grad():
 for layer in model.backbone.encoder_attn[:-1]:last_input=layer(last_input,ts)
last_input=last_input.detach();measure={}
for name in ['all_attention','last_attention','head_only']:
 model.train();model.set_adaptation(name!='head_only')
 if name=='last_attention':
  for layer in model.backbone.encoder_attn[:-1]:layer.requires_grad_(False)
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
 target=torch.zeros(16,2,device='mps');target[::3]=1;dur=[]
 for i in range(8):
  tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
  out=model.forward_projected(z,ts) if name=='all_attention' else model.edge(model.backbone.encoder_attn[-1](last_input,ts)[:,TARGET]) if name=='last_attention' else model.edge(h.detach())
  loss=masked_boundary_loss(out,target,torch.ones(2,device='mps'));loss.backward();optimizer.step();torch.mps.synchronize()
  if i>=3:dur.append(time.perf_counter()-tick)
 measure[name]=float(np.median(dur))
result=dict(windows=counts,batch_size=16,normalization_seconds_per_window=normalization,frozen_full_encoder_seconds_per_batch=float(np.median(timing)),training_seconds_per_batch=measure,scope='Disposable synthetic-label timing on real training windows; benchmark weights discarded, no validation fitting.')
Path(__file__).with_name('benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
