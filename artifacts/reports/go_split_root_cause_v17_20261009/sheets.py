"""Contact sheets of the user's phone takes (4 frames/s) annotated with the phone's committed words."""
import json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');P=ROOT/'data/local/phone_session_pull_v17_20261009'
OUT=Path(__file__).resolve().parent/'sheets';OUT.mkdir(exist_ok=True)
sess=json.loads((P/'live_reel_sessions/20261009_082836.json').read_text())
# phone words per take (seconds are relative to each take's start event)
takes={};cur=None
for e in sess['events']:
    if e['event']=='video_start':cur=e['file'];takes[cur]=[]
    elif e['event']=='word' and cur:takes[cur].append(e['word'])
json.dump(takes,open(OUT.parent/'phone_words.json','w'),indent=1)
STEP=0.25;COLS=8;ROWS=4
for name,words in takes.items():
    cap=cv2.VideoCapture(str(P/'live_reel_videos'/name));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur=n/fps;times=np.arange(0,dur,STEP);thumbs=[]
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps));ok,f=cap.read()
        if not ok:break
        f=cv2.resize(f,(320,180))
        lab=[w['gloss'] for w in words if w['start_seconds']-0.05<=t<=w['end_seconds']+0.05]
        cv2.putText(f,f'{t:.2f}',(4,16),0,0.5,(0,255,255),1)
        if lab:cv2.putText(f,' '.join(lab),(4,172),0,0.6,(0,0,255),2)
        thumbs.append(f)
    for k in range(0,len(thumbs),COLS*ROWS):
        chunk=thumbs[k:k+COLS*ROWS];chunk+= [np.zeros_like(thumbs[0])]*(COLS*ROWS-len(chunk))
        sheet=np.vstack([np.hstack(chunk[r*COLS:(r+1)*COLS]) for r in range(ROWS)])
        cv2.imwrite(str(OUT/f'{name[:-4]}_{k*STEP:06.1f}s.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,80])
    print(name,len(thumbs),[ (w['gloss'],round(w['start_seconds'],2),round(w['end_seconds'],2)) for w in words])
