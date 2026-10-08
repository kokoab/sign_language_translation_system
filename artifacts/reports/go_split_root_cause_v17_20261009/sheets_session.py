"""sheets_session.py VIDEO WORDS.json TAG [STEP]: 4 fps contact sheets annotated with the phone's words."""
import json,sys
from pathlib import Path
import cv2,numpy as np
v,wf,tag=sys.argv[1:4];step=float(sys.argv[4]) if len(sys.argv)>4 else .25
words=json.load(open(wf));OUT=Path(__file__).parent/'sheets'
cap=cv2.VideoCapture(v);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));th=[]
for t in np.arange(0,n/fps,step):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps));ok,f=cap.read()
    if not ok:break
    f=cv2.resize(f[0:640,240:1040],(300,240));cv2.putText(f,f'{t:.2f}',(4,18),0,.55,(0,255,255),2)
    lab=[w['gloss'] for w in words if w['start_seconds']-.05<=t<=w['end_seconds']+.05]
    if lab:cv2.putText(f,' '.join(lab),(4,230),0,.6,(0,0,255),2)
    th.append(f)
for k in range(0,len(th),32):
    c=th[k:k+32];c+=[np.zeros_like(th[0])]*(32-len(c))
    cv2.imwrite(str(OUT/f'{tag}_{k*step:06.1f}.jpg'),np.vstack([np.hstack(c[r*8:(r+1)*8]) for r in range(4)]),[cv2.IMWRITE_JPEG_QUALITY,72])
print(len(th))
