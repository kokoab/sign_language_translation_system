import sys,json,pickle,cv2
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from active.v17.stage1_window_v17 import raw_observation_features
HERE=Path(__file__).resolve().parent
CACHE=Path('/Users/frnzlo/slt_go_split_cache')
def rise(raw,s,e):
    result=[]
    for st in (0,21):
        a=raw[max(0,s-6):s+1];b=raw[s:e+1]
        a=a[(a[:,st,4]>=.3)&(a[:,st+9,4]>=.3)]
        b=b[(b[:,st,4]>=.3)&(b[:,st+9,4]>=.3)]
        if len(a)<2 or len(b)<2:continue
        size=np.median(np.linalg.norm(b[:,st+9,:2]-b[:,st,:2],axis=1))
        if size<1e-3:continue
        result.append(float((np.median(a[:3,st,1])-np.median(b[-3:,st,1]))/size))
    return result
if __name__=='__main__':
 d=json.load(open(HERE/'guarded_regression_pair_val_tune_citizen_val.json'))
 for row in d['pair_val:closed_merge_odds']['rows']:
  if row['target']==['HAVE'] or not any(w['gloss']=='HAVE' for w in row['words']):continue
  item=pickle.load(open(CACHE/'pair_val'/(row['key']+'.pkl'),'rb'))
  cap=cv2.VideoCapture(item['meta']['video']);_,fr=cap.read();cap.release()
  for o in item['obs']:o.frame=np.broadcast_to(np.zeros((1,1,1),np.uint8),fr.shape)
  raw,t=raw_observation_features(item['obs'])
  for w in row['words']:
   if w['gloss']=='HAVE':print(row['key'],w['start_frame'],w['end_frame'],rise(raw,w['start_frame'],min(w['end_frame'],len(raw)-1)))
 trace=json.load(open(HERE/'take11_trace.json'))['frames'];raw=[]
 for r in trace:
  a=np.zeros((61,5))
  for key,st in [('l',0),('r',21)]:
   if key in r:
    p=np.array(r[key]);a[st:st+21,0]=p[:,0]*r['w']/r['h'];a[st:st+21,1]=p[:,1];a[st:st+21,4]=p[:,2]
  raw.append(a)
 raw=np.asarray(raw)
 for s,e in [(82,88),(140,142),(231,237),(417,425),(568,572)]:print('take11',s,e,rise(raw,s,e))
