"""Side-by-side sheets: Citizen train clips vs a seeded sample of local train clips per class."""
import sys,glob,random,cv2,numpy as np
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent/'sheets'
def strip(path,label):
    cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ims=[]
    for q in (0.2,0.35,0.5,0.65,0.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(max(n-1,0)*q));ok,f=cap.read()
        if ok:h,w=f.shape[:2];ims.append(cv2.resize(f,(int(w*100/h),100)))
    s=np.hstack(ims) if ims else np.zeros((100,600,3),np.uint8)
    s=cv2.copyMakeBorder(s,18,3,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    cv2.putText(s,label,(3,13),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,200) if label.startswith('CIT') else (0,0,0),1);return s
for cls in sys.argv[1:]:
    rng=random.Random(cls)
    cit=sorted(glob.glob(str(ROOT/f'data/local/citizen100_v17/raw/train/{cls}/*')))[:3]
    loc=sorted(glob.glob(str(ROOT/f'data/local/local_deep_clean_v17/raw/train/{cls}/*')));loc=rng.sample(loc,min(7,len(loc)))
    rows=[strip(p,f'CITIZEN {cls} {Path(p).name[:30]}') for p in cit]+[strip(p,f'LOCAL {cls} {Path(p).name[:30]}') for p in loc]
    W=max(r.shape[1] for r in rows)
    cv2.imwrite(str(OUT/f'{cls}.jpg'),np.vstack([cv2.copyMakeBorder(r,0,0,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255)) for r in rows]),[cv2.IMWRITE_JPEG_QUALITY,78])
    print(cls,len(cit),len(loc))
