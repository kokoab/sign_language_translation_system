"""Bounded fusion-head experiment on verified frozen features and grouped native positives."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.8');os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.4')
from pathlib import Path
import json,copy,collections,time,random,sys
import numpy as np,torch
import torch.nn.functional as F
from features import O,R,BASE,sha
from active.v17.export_unified_multimodal_coreml_v17 import load_model

def validate_recipe(recipe):
 if recipe.get('format')!='slt_temporal_repair_training_recipe_v2' or not recipe.get('recipe_training_ready'):raise ValueError('unadmitted recipe')
 declared=json.loads((O/'declared_recipe.json').read_text())
 if recipe['training']!=declared['training'] or recipe['selection']!=declared['selection']:raise ValueError('recipe changed after declaration')
 required=[str(BASE),str(O/'train.py'),str(O/'materialization.json'),str(O/'isolated_feature_manifest.json'),str(O/'isolated_role_check.json'),str(O/'exact_core_contract.json')]+[str(O/'features'/f'{n}.npz') for n in ['citizen_train','citizen_val','semlex_train','semlex_val','local_train','local_val','temporal_train','temporal_validation']]
 if not all(p in recipe['pins'] for p in required):raise ValueError('missing pinned input')
 for p,h in recipe['pins'].items():
  if sha(p)!=h:raise ValueError('changed pinned input: '+p)
 if recipe['input_summary']['events_attempted']!=144:raise ValueError('incomplete input audit')
 if recipe.get('promotion_ready') or recipe['canonical_training_ready']:raise ValueError('scope escalation')
 return True

def read(name):
 with np.load(O/'features'/f'{name}.npz',allow_pickle=False) as d:
  keep=np.ones(len(d['targets']),dtype=bool)
  if name in ['semlex_train','semlex_val']:
   audit=json.loads((O/'isolated_role_check.json').read_text())['semlex'];exclusions=audit['excluded_train_ids' if name.endswith('_train') else 'excluded_validation_ids'];keep=~np.isin(d['item_ids'],exclusions)
  return tuple(torch.from_numpy(d[f'f{i}'][keep]) for i in range(4)),torch.from_numpy(d['targets'][keep]),d['item_ids'][keep]

def predict(head,x):
 head.eval()
 with torch.inference_mode():return torch.cat([head(*(a[i:i+256] for a in x)).cpu() for i in range(0,len(x[0]),256)])

def main():
 recipe=json.loads((O/'training_recipe.json').read_text());validate_recipe(recipe);torch.set_num_threads(2);device=torch.device('mps');model,base=load_model(BASE);labels=sorted(base['label_to_index'],key=base['label_to_index'].get);assert len(labels)==100
 trainsets=['citizen_train','semlex_train','local_train','temporal_train'];parts=[read(s) for s in trainsets]
 x=tuple(torch.cat([p[0][i] for p in parts]).to(device) for i in range(4));y=torch.cat([p[1] for p in parts]).to(device)
 source=torch.cat([torch.full((len(p[1]),),i) for i,p in enumerate(parts)]).to(device);weights=[]
 shares=[.7/3]*3+[.3]
 for part,share in zip(parts,shares):
  target=part[1].tolist();ids=part[2].tolist();event_class={i:t for i,t in zip(ids,target)};class_events=collections.Counter(event_class.values());views=collections.Counter(ids)
  weights += [share/len(class_events)/class_events[t]/views[i] for t,i in zip(target,ids)]
 weights=torch.tensor(weights,dtype=torch.double);assert abs(weights.sum().item()-1)<1e-8
 validations={}
 for name in ['citizen_val','semlex_val','local_val','temporal_validation']:
  z,t,ids=read(name);validations[name]=(tuple(a.to(device) for a in z),t,ids)
 head=model.fusion_head.to(device).eval();initial=copy.deepcopy(head.state_dict());teacher=predict(head,x).to(device);baseline={}
 for name,(z,t,ids) in validations.items():
  logits=predict(head,z);baseline[name]={'prediction':logits.argmax(1),'correct':torch.bincount(t[logits.argmax(1)==t],minlength=100)}
 assert len(set(validations['citizen_val'][1].tolist()))==100
 temporal_ids=validations['temporal_validation'][2];groups={i:np.flatnonzero(temporal_ids==i) for i in dict.fromkeys(temporal_ids.tolist())}
 def snapshot(epoch,loss):
  stats={};eligible=True
  for name,(z,t,ids) in validations.items():
   logits=predict(head,z);pred=logits.argmax(1);correct=torch.bincount(t[pred==t],minlength=100)
   if name!='temporal_validation':
    worse=torch.nonzero(correct<baseline[name]['correct']).flatten().tolist();eligible &= not worse
    stats[name]={'correct':int((pred==t).sum()),'n':len(t),'per_class_correct':correct.tolist(),'regressed_classes':[labels[i] for i in worse],'predictions':pred.tolist()}
   else:
    ce=F.cross_entropy(logits,t,reduction='none');events=[{'id':k,'correct_views':int((pred[ix]==t[ix]).sum()),'views':len(ix),'loss':float(ce[ix].mean())} for k,ix in groups.items()]
    stats[name]={'events':events,'all_views_correct_events':sum(r['correct_views']==r['views'] for r in events),'event_mean_loss':sum(r['loss'] for r in events)/len(events)}
  return {'epoch':epoch,'loss':loss,'eligible':bool(eligible),'metrics':stats}
 def score(row):
  m=row['metrics']['temporal_validation'];return (m['all_views_correct_events'],-m['event_mean_loss'])
 baseline_row=snapshot(0,None);best=dict(row=baseline_row,head=copy.deepcopy(initial),arm='epoch0');diagnostic=copy.deepcopy(best);all_runs=[];started=time.time()
 for seed in recipe['training']['seeds']:
  for lr in recipe['training']['head_learning_rates']:
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);head.load_state_dict(initial);arm=f'seed{seed}_lr{lr:g}';anchors=[p.detach().clone() for p in head.parameters()]
   optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=6);generator=torch.Generator().manual_seed(seed);history=[baseline_row]
   for epoch in range(1,7):
    head.train();indices=torch.multinomial(weights,6000,replacement=True,generator=generator);total=0.
    for start in range(0,len(indices),64):
     ix=indices[start:start+64].to(device);output=head(*(a[ix] for a in x));replay=source[ix]<3
     hard=F.cross_entropy(output,y[ix]);kd=F.kl_div(F.log_softmax(output[replay]/2,1),F.softmax(teacher[ix][replay]/2,1),reduction='batchmean')*4 if replay.any() else output.sum()*0
     anchor=.01*sum((p-a).square().mean() for p,a in zip(head.parameters(),anchors));loss=hard+kd+anchor
     optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);optimizer.step();total+=loss.item()*len(ix)
    scheduler.step();row=snapshot(epoch,total/6000);history.append(row)
    if row['eligible'] and score(row)>score(best['row']):best=dict(row=row,head=copy.deepcopy(head.state_dict()),arm=arm)
    if score(row)>score(diagnostic['row']):diagnostic=dict(row=row,head=copy.deepcopy(head.state_dict()),arm=arm)
    print(json.dumps({'arm':arm,'epoch':epoch,'eligible':row['eligible'],'temporal_robust':score(row)[0],'temporal_loss':-score(row)[1],'regressed':{n:len(m.get('regressed_classes',[])) for n,m in row['metrics'].items()},'seconds':time.time()-started}),flush=True)
   all_runs.append({'arm':arm,'seed':seed,'lr':lr,'history':history})
 for name,selection in [('selected',best),('diagnostic',diagnostic)]:
  checkpoint=copy.deepcopy(base);checkpoint['head_state_dict']={k:v.detach().cpu() for k,v in selection['head'].items()};checkpoint['epoch']=selection['row']['epoch'];checkpoint['temporal_repair']={'recipe_sha256':sha(O/'training_recipe.json'),'arm':selection['arm'],'selection':selection['row'],'promotion_ready':False};checkpoint['canonical_phrase_manifest_sha256']=recipe['canonical_phrase_manifest_sha256'];checkpoint['test_evaluated']=False
  for key in ['landmark_model_state_dict','hand_model_state_dict']:assert all(torch.equal(v,base[key][k]) for k,v in checkpoint[key].items())
  torch.save(checkpoint,O/f'{name}.pth')
 result={'runs':all_runs,'baseline':baseline_row,'selected':{'arm':best['arm'],'row':best['row']},'diagnostic':{'arm':diagnostic['arm'],'row':diagnostic['row']},'seconds':time.time()-started,'encoders_unchanged':True,'checkpoints':{n:sha(O/f'{n}.pth') for n in ['selected','diagnostic']}}
 (O/'training_results.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Training complete; selected',best['arm'],'epoch',best['row']['epoch'],flush=True)
if __name__=='__main__':main()
