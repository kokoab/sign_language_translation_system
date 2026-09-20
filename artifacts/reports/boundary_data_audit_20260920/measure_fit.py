from pathlib import Path
import sys,runpy,json,collections,hashlib,time
import numpy as np
import torch
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT)); torch.set_num_threads(2)
m=runpy.run_path('artifacts/reports/boundary_phase_v17_20260920/run_experiment.py',run_name='diagnose_boundary')
payload=json.load(open(m['MANIFEST'])); model,labels=m['load_base']()
ck=torch.load(m['MODEL'],map_location='cpu',weights_only=False)
model.base.load_state_dict(ck['base_model_state_dict']); model.phase_head.load_state_dict(ck['phase_head_state_dict']); model.to('mps').eval()
results={}; hashes={}; conflicts=[]
for role in ('train','validation'):
 rows,rejected=m['context_samples'](payload,labels,role)
 metrics={}; correct=collections.Counter(); totals=collections.Counter(); confusion=collections.defaultdict(lambda:np.zeros((3,3),dtype=int))
 with torch.inference_mode():
  for start in range(0,len(rows),64):
   rs=rows[start:start+64]; x=torch.from_numpy(np.stack([r.features for r in rs]).astype(np.float32)).to('mps')
   g,p=model(x); gp=g.argmax(1).cpu().tolist(); pp=p.argmax(1).cpu().tolist()
   for r,a,b in zip(rs,gp,pp):
    key=r.source; confusion[key][r.phase,b]+=1
    if r.target>=0:totals[key]+=1;correct[key]+=a==r.target
    h=hashlib.sha256(r.features.tobytes()).hexdigest(); target=(r.phase,r.target)
    if h in hashes and hashes[h][0]!=target: conflicts.append([hashes[h][1],r.identity])
    else:hashes[h]=(target,r.identity)
 for source,cm in confusion.items():
  metrics[source]={'samples':int(cm.sum()),'phase_accuracy':float(cm.trace()/cm.sum()),'phase_confusion':cm.tolist(),'known_correct':correct[source],'known_total':totals[source],'known_accuracy':correct[source]/max(1,totals[source])}
 results[role]=metrics
 print(role,json.dumps(metrics),flush=True)
results['conflicting_identical_features_count']=len(conflicts); results['conflict_examples']=conflicts[:10]
Path('artifacts/reports/boundary_data_audit_20260920/measured_fit.json').write_text(json.dumps(results,indent=2)+'\n')
