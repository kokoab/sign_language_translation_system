"""Export existing validation observations without changing precision or input times."""
from pathlib import Path
import sys,json,pickle,base64,copy
import cv2,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R))
D=P/'validation_traces';D.mkdir(exist_ok=True);items=[]
for dataset in ['pair_val','citizen_val','tune']:
 for path in sorted((Path('/Users/frnzlo/slt_go_split_cache')/dataset).glob('*.pkl')):
  item=pickle.load(path.open('rb'));assert '/test/' not in item['meta']['video']
  cap=cv2.VideoCapture(item['meta']['video']);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release();assert w and h
  obs,hands=list(item['obs']),list(item['hands'])
  if dataset!='tune':
   last=obs[-1]
   for j in range(20):o=copy.copy(last);o.seconds=last.seconds+.05*(j+1);obs.append(o);hands.append(hands[-1])
  rows=[]
  for o,hand in zip(obs,hands):
   d=o.detection
   row=dict(s=o.seconds,w=w,h=h,b=np.c_[d.body_xy,d.body_confidence].tolist(),f=np.c_[d.face_xy,d.face_confidence].tolist(),ff=bool(o.face_for_features),ef32=base64.b64encode(hand[0].astype('<f4').tobytes()).decode(),v=hand[1].astype(float).ravel().tolist(),x=hand[2].astype(float).ravel().tolist())
   for key,name in [('l','left'),('r','right')]:
    v=o.assigned[name]
    if v is not None:row[key]=np.c_[v.xy,v.confidence].tolist();row[key+'c']=v.chirality;row[key+'s']=float(v.score)
   rows.append(row)
  dest=D/(dataset+'__'+path.stem+'.json');dest.write_text(json.dumps(dict(frames=rows),separators=(',',':')))
  items.append(dict(id=path.stem,dataset=dataset,reference=item['meta']['target'],path=str(dest)))
 print(dataset,sum(r['dataset']==dataset for r in items),flush=True)
(P/'validation_traces.json').write_text(json.dumps(items))
