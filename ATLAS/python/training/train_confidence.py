"""Experimental actual-window repair; frozen encoders, balanced full fusion-head adaptation.
Known phone data is explicitly training/development, not a fresh accuracy claim.
"""
from pathlib import Path
import sys,os,json,copy,time,random,collections
import numpy as np,torch
import torch.nn.functional as F
D=(Path(__file__).resolve().parents[3] / 'artifacts/reports/my_transition_recovery_v17_20261011');P=D.parent/'temporal_repair_training_v17_20261011';sys.path.insert(0,str(P));sys.path.insert(0,str(D.parents[2]))
from features import BASE,sha
from train import read,predict
from active.v17.export_unified_multimodal_coreml_v17 import load_model

def main():
 recipe=json.loads((D/'confidence_training_recipe.json').read_text());assert recipe['scope']=='confidence_runtime_window_head' and recipe['ready'] and not recipe['promotion_ready']
 for p,h in recipe['pins'].items():assert sha(p)==h, 'Changed pinned input '+p
 torch.set_num_threads(2);device=torch.device('mps');model,base=load_model(BASE);head=model.fusion_head.to(device).eval();labels=sorted(base['label_to_index'],key=base['label_to_index'].get);my=labels.index('MY');like=labels.index('LIKE')
 with np.load(D/'jitter_paired_features.npz') as d:
  ids=d['item_ids'];mask=np.isin(ids,recipe['training_events']);phone=tuple(torch.tensor(d[f'f{i}'][mask],device=device) for i in range(4));py=torch.tensor(d['targets'][mask],device=device);dev=tuple(torch.tensor(d[f'f{i}'][~mask],device=device) for i in range(4));dy=torch.tensor(d['targets'][~mask]);dev_ids=ids[~mask];train_ids=ids[mask]
 assert len(py)>0 and len(dy)>0 and not set(dev_ids)&set(train_ids)
 parts=[read(n) for n in ['citizen_train','semlex_train','local_train']];
 temporal=read('temporal_train');keep=temporal[1]==like;parts.append((tuple(a[keep] for a in temporal[0]),temporal[1][keep],temporal[2][keep.numpy()]));x=tuple(torch.cat([z[0][i] for z in parts]).to(device) for i in range(4));y=torch.cat([z[1] for z in parts]).to(device)
 validations={n:read(n) for n in ['citizen_val','semlex_val','local_val']};validations={n:(tuple(a.to(device) for a in z),t,ids) for n,(z,t,ids) in validations.items()}
 initial=copy.deepcopy(head.state_dict());teacher=predict(head,x).to(device);baseline={n:predict(head,z).argmax(1) for n,(z,t,ids) in validations.items()}
 def assess(epoch,arm):
  metrics={};eligible=True
  for name,(z,t,ids) in validations.items():
   out=predict(head,z);pred=out.argmax(1);bc=torch.bincount(t[baseline[name]==t],minlength=100);nc=torch.bincount(t[pred==t],minlength=100);bad=torch.nonzero(nc<bc).flatten().tolist();eligible &= not bad
   metrics[name]={'correct':int((pred==t).sum()),'n':len(t),'regressed_classes':[labels[i] for i in bad],'previously_correct_lost':int(((baseline[name]==t)&(pred!=t)).sum())};eligible &= not bool(((baseline[name]==t)&(pred!=t)).any())
  out=predict(head,dev);pred=out.argmax(1);train_pred=predict(head,phone).argmax(1);loss=float(F.cross_entropy(out,dy));events={e:int((pred[np.flatnonzero(dev_ids==e)]==dy[np.flatnonzero(dev_ids==e)]).sum()) for e in sorted(set(dev_ids))}
  return dict(epoch=epoch,arm=arm,eligible=bool(eligible),worst_dev_fraction=min(events[e]/sum(dev_ids==e) for e in events),head_drift=float(sum((head.state_dict()[k]-v).square().mean() for k,v in initial.items())),dev_correct=int((pred==dy).sum()),dev_n=len(dy),dev_loss=loss,dev_events=events,train_correct=int((train_pred==py.cpu()).sum()),train_n=len(py),retention=metrics)
 first=assess(0,'baseline');best=(first,copy.deepcopy(initial));diagnostic=(first,copy.deepcopy(initial));hist=[first];started=time.time()
 for p in head.parameters():p.requires_grad_(True)
 last=head.residual[-1];last.weight.requires_grad_(True);last.bias.requires_grad_(True)
 allowed=torch.zeros_like(last.bias);allowed[[my,like]]=1
 weights=[]
 for (z,t,ids) in parts:
  counts=collections.Counter(t.tolist());weights.extend([(0.15/len(counts)/counts[int(v)] + (0.15/counts.get(like,1) if int(v)==like else 0)) if len(counts)>1 else 0.10/len(t) for v in t])
 weights=torch.tensor(weights,dtype=torch.double);phone_weights=torch.tensor([1/len(set(train_ids))/sum(train_ids==i) for i in train_ids],dtype=torch.double)
 for seed in recipe['seeds']:
  for lr in recipe['learning_rates']:
   for share in recipe['phone_shares']:
    head.load_state_dict(initial);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);gen=torch.Generator().manual_seed(seed)
    opt=torch.optim.Adam(head.parameters(),lr=lr);arm=f's{seed}_lr{lr}_p{share}'
    for epoch in range(1,recipe['epochs']+1):
     head.eval()  # Keep dropout deterministic while the fusion head learns.
     for step in range(32):
      ni=round(64*share);ri=torch.multinomial(weights,64-ni,replacement=True,generator=gen).to(device);pi=torch.multinomial(phone_weights,ni,replacement=True,generator=gen).to(device)
      out=head(*(a[ri] for a in x));po=head(*(a[pi] for a in phone))
      ce=F.cross_entropy(out,y[ri]);kd=F.kl_div(F.log_softmax(out/2,1),F.softmax(teacher[ri]/2,1),reduction='batchmean')*4
      loss=(1-share)*(ce+2*kd)+share*F.cross_entropy(po,py[pi])
      loss=loss+.01*((last.weight-initial['residual.6.weight']).square().mean()+(last.bias-initial['residual.6.bias']).square().mean())
      opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
     row=assess(epoch,arm);hist.append(row);score=lambda z:(z['worst_dev_fraction'],z['dev_correct'],-z['dev_loss'],-z['head_drift'])
     if row['eligible'] and score(row)>score(best[0]):best=(row,copy.deepcopy(head.state_dict()))
     if score(row)>score(diagnostic[0]):diagnostic=(row,copy.deepcopy(head.state_dict()))
     print(json.dumps(dict(arm=arm,epoch=epoch,eligible=row['eligible'],dev=row['dev_correct'],train=row['train_correct'],seconds=time.time()-started)),flush=True)
 for name,(row,state) in [('confidence_selected',best),('confidence_diagnostic',diagnostic)]:
  checkpoint=copy.deepcopy(base);checkpoint['head_state_dict']={k:v.cpu() for k,v in state.items()};checkpoint['actual_window_recipe_sha256']=sha(D/'confidence_training_recipe.json');checkpoint['selection']=row;checkpoint['test_evaluated']=False;torch.save(checkpoint,D/(name+'.pth'))
 (D/'confidence_training_results.json').write_text(json.dumps(dict(history=hist,baseline=first,selected=best[0],diagnostic=diagnostic[0],seconds=time.time()-started,encoders_unchanged=True,all_other_word_output_rows_unchanged=False),indent=2)+'\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
