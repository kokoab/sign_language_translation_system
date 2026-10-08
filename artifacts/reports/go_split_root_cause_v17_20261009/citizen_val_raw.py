"""Citizen val (validation only) per-frame raw observation features from the slim cache, for the boundary
student's isolated single-run validation: CACHE/isolated_val_raw/citizen/<label>/<clip>.npz."""
import pickle,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.stage1_window_v17 import raw_observation_features
C=Path('/Users/frnzlo/slt_go_split_cache');OUT=C/'isolated_val_raw/citizen'
for f in sorted((C/'citizen_val').glob('*.pkl')):
    item=pickle.load(open(f,'rb'));label=item['meta']['target'][0]
    ok,fr=cv2.VideoCapture(item['meta']['video']).read();view=np.broadcast_to(np.zeros((1,1,1),np.uint8),fr.shape)
    for o in item['obs']:o.frame=view
    raw,times=raw_observation_features(item['obs'])
    p=OUT/label/(f.stem+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,raw=raw.astype(np.float16),times=times)
print('done',len(list(OUT.glob('*/*.npz'))))
