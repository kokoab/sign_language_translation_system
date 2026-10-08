"""zoom.py VIDEO START END [STEP] -> sheets/zoom_<take>_<start>.jpg, 6 columns, center crop, labelled time."""
import sys
from pathlib import Path
import cv2,numpy as np
v,a,b=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]);step=float(sys.argv[4]) if len(sys.argv)>4 else 0.1
cap=cv2.VideoCapture(v);fps=cap.get(cv2.CAP_PROP_FPS);th=[]
for t in np.arange(a,b,step):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(t*fps)));ok,f=cap.read()
    if not ok:break
    f=cv2.resize(f[0:720,240:1040],(400,360));cv2.putText(f,f'{t:.2f}',(4,22),0,0.7,(0,255,255),2);th.append(f)
C=6;th+=[np.zeros_like(th[0])]*((-len(th))%C)
out=Path(__file__).parent/'sheets'/f'zoom_{Path(v).stem[-2:]}_{a:06.2f}.jpg'
cv2.imwrite(str(out),np.vstack([np.hstack(th[i:i+C]) for i in range(0,len(th),C)]),[cv2.IMWRITE_JPEG_QUALITY,80]);print(out)
