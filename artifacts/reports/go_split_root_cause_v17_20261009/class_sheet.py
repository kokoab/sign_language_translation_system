"""class_sheet.py SRCDIR LABEL N -> sheets/class_<LABEL>_<tag>.jpg: N clips, 8 evenly spaced frames each."""
import glob,sys
from pathlib import Path
import cv2,numpy as np
src,label,n=sys.argv[1],sys.argv[2],int(sys.argv[3]);tag=sys.argv[4] if len(sys.argv)>4 else 'x'
rows=[]
for v in sorted(glob.glob(f'{src}/{label}/*'))[:n]:
    cap=cv2.VideoCapture(v);m=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));th=[]
    for k in np.linspace(0,m-1,8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(k));ok,f=cap.read()
        if not ok:f=np.zeros((180,240,3),np.uint8)
        h,w=f.shape[:2];s=180/h;f=cv2.resize(f,(int(w*s),180));f=f[:, max(0,(f.shape[1]-240)//2):][:, :240]
        if f.shape[1]<240:f=cv2.copyMakeBorder(f,0,0,0,240-f.shape[1],cv2.BORDER_CONSTANT)
        th.append(f)
    r=np.hstack(th);cv2.putText(r,Path(v).name[:30],(4,14),0,0.45,(0,255,255),1);rows.append(r)
out=Path(__file__).parent/'sheets'/f'class_{label}_{tag}.jpg';cv2.imwrite(str(out),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,75]);print(out)
