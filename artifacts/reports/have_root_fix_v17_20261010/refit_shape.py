"""Correct the rotation using only the original isolated training inputs; no phone/phrase fitting."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from active.v17.segmental_runtime_v17 import HandshapePair
HERE=Path(__file__).resolve().parent
CACHE=Path('/Users/frnzlo/slt_go_split_cache')
def canon(hand,left):
    xy=hand[:,:2].astype(np.float64)-hand[0,:2]
    v=xy[9];size=np.linalg.norm(v)
    if size<1e-3:return None
    a=np.arctan2(-v[0],-v[1]);c,s=np.cos(a),np.sin(a)
    xy=xy@np.array([[c,-s],[s,c]]).T/size
    assert np.allclose(xy[9],[0,-1],atol=1e-6)
    if left:xy[:,0]*=-1
    return xy[1:].ravel()
def features(path):
    raw=np.load(path)['raw'].astype(np.float32)
    act=np.flatnonzero(raw[:,:42,4].max(1)>.3)
    if len(act)<4:return []
    a,b=act[0],act[-1];n=b-a
    out=[]
    for f in raw[a+n//4:a+3*n//4+1]:
        for st,left in ((0,True),(21,False)):
            h=f[st:st+21]
            if h[:,4].min()<.3:continue
            feat=canon(h,left)
            if feat is not None:out.append(feat)
    return out
X,y,inputs=[],[],[]
for label in ('HAVE','ANGRY'):
    for path in sorted((CACHE/'isolated_raw').glob(f'*/{label}/*.npz')):
        rows=features(path);X.extend(rows);y.extend([int(label=='HAVE')]*len(rows))
        inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=len(rows),label=label))
sc=StandardScaler().fit(X)
clf=LogisticRegression(C=.3,max_iter=2000,class_weight='balanced').fit(sc.transform(X),y)
m=dict(format='slt_handshape_pair_v17',pair=['HAVE','ANGRY'],positive='HAVE',features='v2: wrist centred, rotate by atan2(-vx,-vy), scale middle MCP, mirror left',trained_on='Original isolated train HAVE/ANGRY middle-half inputs; see shape_inputs.json',mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]))
(HERE/'handshape_correct_rotation.json').write_text(json.dumps(m))
(HERE/'shape_inputs.json').write_text(json.dumps(inputs,indent=2))
old=HandshapePair(ROOT/'artifacts/models/handshape_have_angry_v17/model.json')
new=HandshapePair(HERE/'handshape_correct_rotation.json');new.canon=canon
rows=[]
for path in sorted((CACHE/'isolated_val_raw').glob('*/*/*.npz')):
    if path.parent.name not in ('HAVE','ANGRY','TIRED','HAPPY'):continue
    raw=np.load(path)['raw'].astype(np.float32)
    rows.append(dict(path=str(path),label=path.parent.name,old=old.score(raw),new=new.score(raw)))
(HERE/'shape_validation.json').write_text(json.dumps(rows,indent=2))
print('training clips',len(inputs),'frames',len(y))
for lab in ('HAVE','ANGRY','TIRED','HAPPY'):
    rs=[r for r in rows if r['label']==lab]
    print(lab,len(rs),'P_HAVE>=.5 old/new',*[sum(r[k] is not None and r[k]>=.5 for r in rs) for k in ('old','new')])
