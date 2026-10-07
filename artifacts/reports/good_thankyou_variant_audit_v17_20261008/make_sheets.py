"""Contact sheets (5 frames per clip: 15-75% of the video) for visual variant review."""
import csv,sys,subprocess
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent;SHEETS=OUT/'sheets';SHEETS.mkdir(exist_ok=True)
gloss=sys.argv[1];per=int(sys.argv[2]) if len(sys.argv)>2 else 10
rows=[r for r in csv.DictReader((OUT/'screen.csv').open()) if r['gloss']==gloss and r['raw']]
def frames(path):
    cap=cv2.VideoCapture(str(ROOT/path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images=[]
    if n<=0:
        allf=[];ok=True
        while ok:
            ok,f=cap.read()
            if ok:allf.append(f)
        pick=[allf[int(len(allf)*q)] for q in (0.15,0.3,0.45,0.6,0.75)] if allf else []
    else:
        pick=[]
        for q in (0.15,0.3,0.45,0.6,0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int((n-1)*q));ok,f=cap.read()
            if ok:pick.append(f)
    for f in pick:
        h,w=f.shape[:2];s=120/h;images.append(cv2.resize(f,(int(w*s),120)))
    return images
index=[]
for start in range(0,len(rows),per):
    strips=[]
    for i,r in enumerate(rows[start:start+per],start):
        ims=frames(r['raw'])
        strip=np.hstack(ims) if ims else np.zeros((120,800,3),np.uint8)
        strip=cv2.copyMakeBorder(strip,22,0,0,max(0,820-strip.shape[1]),cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(strip,f"#{i} {r['source']} screen={r['screen']}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        strips.append(strip[:, :max(820,strip.shape[1])])
        index.append(dict(n=i,archive=r['archive'],raw=r['raw'],source=r['source'],screen=r['screen']))
    width=max(s.shape[1] for s in strips)
    sheet=np.vstack([cv2.copyMakeBorder(s,0,4,0,width-s.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255)) for s in strips])
    cv2.imwrite(str(SHEETS/f'{gloss}_{start//per:02d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,80])
with (SHEETS/f'{gloss}_index.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
print(len(rows),'clips ->',(len(rows)+per-1)//per,'sheets')
