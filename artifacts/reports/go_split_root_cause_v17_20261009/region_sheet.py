"""One row per BIO-active region (gaps <= 3 frames bridged): 6 frames across the region, zoomed."""
import pickle,sys
from pathlib import Path
import cv2,numpy as np
take=sys.argv[1];HERE=Path(__file__).resolve().parent
T=pickle.load(open(HERE/'trace'/f'20261009_082836_{take}.pkl','rb'));bio=T['min3']['bio'];sec=T['min3']['seconds']
n=max(bio)+1;P=np.exp(np.stack([bio[t] if t in bio else np.log([1e-6,1,1e-6,1e-6]) for t in range(n)]));act=P[:,2]+P[:,3]
on=act>=.3;regions=[];t=0
while t<n:
    if on[t]:
        u=t
        while u+1<n and (on[u+1] or (u+2<n and on[u+2]) or (u+3<n and on[u+3])):u+=1
        if u-t>=4:regions.append((t,u))
        t=u+1
    else:t+=1
cap=cv2.VideoCapture(str(Path('/Volumes/secret/SLT/SLT/data/local/phone_session_pull_v17_20261009/live_reel_videos')/f'20261009_082836_{take}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
rows=[]
for k,(s,e) in enumerate(regions):
    th=[]
    for t in np.linspace(sec[s]-.1,sec[min(e,len(sec)-1)]+.1,6):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,int(round(t*fps))));ok,f=cap.read()
        if not ok:f=np.zeros((720,1280,3),np.uint8)
        f=cv2.resize(f[0:600,290:990],(280,240));cv2.putText(f,f'{t:.2f}',(4,18),0,0.55,(0,255,255),2);th.append(f)
    r=np.hstack(th);cv2.putText(r,f'R{k} f{s}-{e}',(1500,20),0,0.6,(0,0,255),2);rows.append(r)
for i in range(0,len(rows),8):
    cv2.imwrite(str(HERE/'sheets'/f'regions_{take}_{i//8}.jpg'),np.vstack(rows[i:i+8]),[cv2.IMWRITE_JPEG_QUALITY,75])
print(len(regions),[(k,s,e,round(sec[s],2)) for k,(s,e) in enumerate(regions)])
