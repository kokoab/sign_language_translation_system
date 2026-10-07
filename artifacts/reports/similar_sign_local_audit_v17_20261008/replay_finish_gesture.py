"""Replay a phone take through Apple Vision at 20 Hz and compare finish-gesture detectors.

old: two upright open palms (current app/desktop rule).
new: signer's right hand open/upright, left hand a fist, both wrists above the shoulder line
     (last body detection within 1.5 s), from scripts/live_reel_stage1_v17.is_finish_gesture.
Reports every observed run (start, duration) and whether it would trigger (>= 1 s hold).
"""
import json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import AppleVisionDetector,assign_hands
from scripts.live_reel_stage1_v17 import is_finish_hand,is_control_gesture_pose
video=Path(sys.argv[1]);out=Path(sys.argv[2])
def old(assigned):
    l,r=assigned['left'],assigned['right']
    return bool(is_finish_hand(l) and is_finish_hand(r) and np.linalg.norm(l.xy[0]-r.xy[0])>=0.12)
det=AppleVisionDetector();cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
prev={'left':None,'right':None};shoulder=None;shoulder_t=-9;rows=[];next_t=0.0;i=0;processed=0
while True:
    ok,frame=cap.read()
    if not ok:break
    t=i/fps;i+=1
    if t+1e-9<next_t:continue
    next_t+=0.05
    body=processed%8==0;processed+=1
    small=frame if max(frame.shape[:2])<=1280 else cv2.resize(frame,None,fx=1280/max(frame.shape[:2]),fy=1280/max(frame.shape[:2]))
    d=det.detect(small,include_body=body,include_face=False,include_hands=True)
    a=assign_hands(d.hands,prev)
    for k in ('left','right'):prev[k]=a[k].xy[0].copy() if a[k] is not None and a[k].confidence[0]>0 else prev[k]
    if body and (d.body_confidence[:2]>0).all():shoulder=float(d.body_xy[:2,1].mean());shoulder_t=t
    s=shoulder if t-shoulder_t<=1.5 else None
    rows.append(dict(t=round(t,3),old=old(a),new=is_control_gesture_pose(a,s)))
def runs(key):
    res=[];start=None
    for r in rows+[dict(t=rows[-1]['t']+1,old=False,new=False)]:
        if r[key] and start is None:start=r['t']
        if not r[key] and start is not None:res.append(dict(start=start,seconds=round(r['t']-start,2)));start=None
    return res
report=dict(video=str(video),frames=len(rows),
            old=dict(frames=sum(r['old'] for r in rows),runs=runs('old'),triggers=sum(x['seconds']>=1.0 for x in runs('old'))),
            new=dict(frames=sum(r['new'] for r in rows),runs=runs('new'),triggers=sum(x['seconds']>=1.0 for x in runs('new'))))
out.write_text(json.dumps(report,indent=1)+'\n')
print(json.dumps({k:(v if k not in('old','new') else {kk:vv for kk,vv in v.items() if kk!='runs'}) for k,v in report.items()}))
print('old runs',report['old']['runs'][:40]);print('new runs',report['new']['runs'][:40])
