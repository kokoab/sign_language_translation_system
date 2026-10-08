"""Live phone trace (Documents/live_reel_traces/<stamp>.json) -> slim decoder-input cache, so the exact live
inputs (observations + hand-crop embeddings the phone used) can be decoded offline:
trace_to_slim.py TRACE.json VIDEO.mp4 -> CACHE/live_trace/<stamp>.pkl"""
import base64,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Volumes/secret/SLT/SLT')
from active.v17.extract_v17 import FrameDetection,HandDetection
from scripts.live_isolated_v17 import ObservedFrame
tr,video=sys.argv[1],sys.argv[2]
rows=[r for r in json.load(open(tr))['frames'] if not r.get('start') and not r.get('held')]
obs,hands=[],[]
def hand(r,k):
    if k not in r:return None
    p=np.array(r[k],np.float32);return HandDetection(xy=p[:,:2],confidence=p[:,2],chirality=r.get(k+'c','unknown'),score=float(r.get(k+'s',0)))
for r in rows:
    b=np.array(r['b'],np.float32);f=np.array(r['f'],np.float32);l,rt=hand(r,'l'),hand(r,'r')
    det=FrameDetection(hands=[h for h in (l,rt) if h is not None],body_xy=b[:,:2],body_confidence=b[:,2],face_xy=f[:,:2],face_confidence=f[:,2])
    obs.append(ObservedFrame(frame=None,detection=det,assigned={'left':l,'right':rt},seconds=float(r['s']),motion=0.,hand_quality=0.,face_quality=0.,face_for_features=bool(r['ff'])))
    e=np.frombuffer(base64.b64decode(r['e']),np.float16).astype(np.float32).reshape(3,512) if 'e' in r else np.zeros((3,512),np.float32)
    hands.append((e,np.array(r.get('v',[0,0,0]),np.float32),np.array(r.get('x',[0]*12),np.float32).reshape(3,4)))
out=Path('/Users/frnzlo/slt_go_split_cache/live_trace');out.mkdir(parents=True,exist_ok=True)
pickle.dump(dict(obs=obs,hands=hands,meta=dict(video=video,target=None)),open(out/(Path(tr).stem+'.pkl'),'wb'));print('frames',len(obs))
