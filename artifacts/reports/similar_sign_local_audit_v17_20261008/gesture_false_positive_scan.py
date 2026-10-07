"""Scan validation videos (Citizen val, local val) at 20 Hz with Apple Vision.

Per class: frames where the old two-open-palm pose, the new right-open/left-fist-above-shoulder
pose, and any fist hand occur, plus runs that would trigger a 1 s hold. Validation only.
"""
import glob,json,sys,collections
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import AppleVisionDetector,assign_hands
from scripts.live_reel_stage1_v17 import is_finish_hand,is_fist_hand,is_control_gesture_pose
OUT=Path(__file__).resolve().parent/'gesture_false_positive_scan.json'
det=AppleVisionDetector()
videos=[('citizen',p) for p in sorted(glob.glob(str(ROOT/'data/local/citizen100_v17/raw/val/*/*')))]
videos+=[('local',p) for p in sorted(glob.glob(str(ROOT/'data/local/local_deep_clean_v17/raw/val/*/*')))]
assert not any('/test/' in p for _,p in videos)
stats=collections.defaultdict(lambda:collections.Counter())
for n,(src,path) in enumerate(videos):
    cls=Path(path).parent.name;cap=cv2.VideoCapture(path);fps=cap.get(cv2.CAP_PROP_FPS) or 30
    prev={'left':None,'right':None};shoulder=None;st=-9;i=0;nt=0.0;proc=0;run={'old':0,'new':0};best={'old':0,'new':0}
    while True:
        ok,frame=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;body=proc%8==0;proc+=1
        s=max(frame.shape[:2]);frame=frame if s<=1280 else cv2.resize(frame,None,fx=1280/s,fy=1280/s)
        d=det.detect(frame,include_body=body,include_face=False,include_hands=True);a=assign_hands(d.hands,prev)
        for k in ('left','right'):prev[k]=a[k].xy[0].copy() if a[k] is not None and a[k].confidence[0]>0 else prev[k]
        if body and (d.body_confidence[:2]>0).all():shoulder=float(d.body_xy[:2,1].mean());st=t
        o=bool(is_finish_hand(a['left']) and is_finish_hand(a['right']) and np.linalg.norm(a['left'].xy[0]-a['right'].xy[0])>=0.12)
        w=is_control_gesture_pose(a,shoulder if t-st<=1.5 else None)
        c=stats[(src,cls)];c['frames']+=1;c['old']+=o;c['new']+=w;c['fist_any']+=bool(is_fist_hand(a['left']) or is_fist_hand(a['right']))
        for k,v in (('old',o),('new',w)):
            run[k]=run[k]+1 if v else 0;best[k]=max(best[k],run[k])
    c=stats[(src,cls)];c['clips']+=1
    for k in ('old','new'):
        c[k+'_clips']+=best[k]>0;c[k+'_trigger_clips']+=best[k]>=20
    if n%200==0:print(n,len(videos),flush=True)
OUT.write_text(json.dumps({f'{s}:{c}':dict(v) for (s,c),v in sorted(stats.items())},indent=1)+'\n');print('done')
