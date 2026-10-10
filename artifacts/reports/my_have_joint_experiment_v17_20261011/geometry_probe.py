"""Describe palm extension/pinch overlap; no training and no test split access."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[2];M=R/'artifacts/reports/my_like_regression_v17_20261011'
labels=json.loads((M/'rollback_review/base_comparison.json').read_text())['current']['validation']['citizen_val']['labels']
def features(x):
 result=[]
 for hand in [x[:,:21],x[:,21:42]]:
  mask=(hand[:,:,4]>.3).sum(-1)>=15
  if mask.sum()<3:continue
  h=hand[mask,:,:2];size=np.linalg.norm(h[:,9]-h[:,0],axis=-1)+1e-6
  straight=[]
  for m,t in [(5,8),(9,12),(13,16),(17,20)]:
   chain=sum(np.linalg.norm(h[:,i+1]-h[:,i],axis=-1) for i in range(m,t))
   straight.append(np.linalg.norm(h[:,t]-h[:,m],axis=-1)/(chain+1e-6))
  pinch=np.linalg.norm(h[:,4]-h[:,12],axis=-1)/size
  wrist=np.linalg.norm(h[:,None,0]-h[None,:,0],axis=-1).max()/np.median(size)
  droop=(h[:,[8,12,16],1].mean(-1)-h[:,0,1])/size
  result.append(dict(wrist_motion=float(wrist),droop=float(np.percentile(droop,90)-np.percentile(droop,10)),straight=float(np.percentile(np.min(straight,axis=0),50)),middle=float(np.percentile(straight[1],25)),pinch_min=float(np.percentile(pinch,10)),pinch_change=float(np.percentile(pinch,90)-np.percentile(pinch,10))))
 return result
out={'validation':{},'phone':[]}
C=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run/raw_cache_citizen_variant'
for split in ['citizen_val','semlex_val','local_val']:
 with np.load(C/(split+'.npz'),allow_pickle=False) as z:
  y=z['targets'];idx=np.flatnonzero(np.isin(y,[labels.index('MY'),labels.index('LIKE')]))
  xx=z['landmarks'];ids=z['item_ids']
  out['validation'][split]=[dict(id=str(ids[i]),target=labels[y[i]],hands=features(xx[i])) for i in idx]
for fname in [M/'dump.json.inputs.json',M/'rollback_review/contacts.inputs.json']:
 for x in json.loads(fname.read_text()):
  out['phone'].append(dict(source=fname.name,id=x.get('id'),span=[x.get('start'),x.get('end')],hands=features(np.array(x['landmarks']).reshape(32,61,5))))
(P/'geometry_features.json').write_text(json.dumps(out))
for target in ['MY','LIKE']:
 rows=[h for rr in out['validation'].values() for r in rr if r['target']==target for h in r['hands']]
 print(target,'hands',len(rows),{k:np.quantile([h[k] for h in rows],[0,.1,.5,.9,1]).round(3).tolist() for k in rows[0]})
print('first contacts',out['phone'][:4])
