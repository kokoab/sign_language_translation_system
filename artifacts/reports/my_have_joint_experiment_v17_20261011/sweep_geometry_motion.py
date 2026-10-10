from pathlib import Path
import json,itertools,numpy as np,runpy
P=Path(__file__).resolve().parent;R=P.parents[2];M=R/'artifacts/reports/my_like_regression_v17_20261011/rollback_review'
ns=runpy.run_path(str(P/'geometry_probe.py'));features=ns['features'];labels=ns['labels']
rows=[];phone=ns['out']['phone']; C=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run/raw_cache_citizen_variant'
for split in ['citizen_val','semlex_val','local_val']:
 with np.load(M/f'current_{split}_scores.npz') as z: yy=z['targets'];ll=z['logits'];ids=z['item_ids'];pred=ll.argmax(-1)
 with np.load(C/(split+'.npz')) as z: xx=z['landmarks']
 for i in np.flatnonzero(pred==labels.index('LIKE')):
  rank=int(np.where(np.argsort(-ll[i])==labels.index('MY'))[0][0])+1
  rows.append(dict(split=split,id=str(ids[i]),target=labels[yy[i]],my_rank=rank,hands=features(xx[i])))
results=[]
for straight,middle,pinch,change,rank,motion,droop in itertools.product([.97,.98,.99,.995],[.95,.97,.98,.99],[.4,.6,.8,1.],[.3,.5,.7],[3,5,100],[.5,.75,1.,1.5,100.],[.25,.5,.75,1.,100.]):
 def passes(r):
  return r.get('my_rank',1)<=rank and bool(r['hands']) and all(h['straight']>=straight and h['middle']>=middle and h['pinch_min']>=pinch and h['pinch_change']<=change and h['wrist_motion']<=motion and h['droop']<=droop for h in r['hands'])
 changes=[r for r in rows if passes(r)]
 results.append(dict(motion=motion,droop=droop,straight=straight,middle=middle,pinch=pinch,change=change,rank=rank,changed=len(changes),false_my=sum(r['target']!='MY' for r in changes),fixed_my=sum(r['target']=='MY' for r in changes),first_late=passes(phone[1]),first_early=passes(phone[0]),phone_hits=sum(passes(r) for r in phone[4:])))
(P/'geometry_motion_sweep.json').write_text(json.dumps(dict(rows=rows,sweep=results)))
print('variants',len(results),'current LIKE spans',len(rows))
print('safe late rescue',sum(r['first_late'] and r['false_my']==0 for r in results))
print('late rescue best',sorted([r for r in results if r['first_late']],key=lambda r:(r['false_my'],-r['fixed_my']))[:3])
