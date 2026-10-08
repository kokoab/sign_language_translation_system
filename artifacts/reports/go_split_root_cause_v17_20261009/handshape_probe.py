"""HAVE vs ANGRY from canonicalised hand shape (wrist-centred, size-normalised, rotated wrist->middle-MCP up,
left hands mirrored), per frame, logistic regression. Train: isolated train clips (Citizen/SemLex/local, middle
half of active frames). Report: Citizen val clips (per-clip mean P(HAVE)) and the user's take-11 spans."""
import glob,json,pickle,sys
import numpy as np
sys.path.insert(0,'/Volumes/secret/SLT/SLT')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def canon(h,left):
    xy=h[:,:2].astype(np.float64)-h[0,:2];v=xy[9];s=np.linalg.norm(v)
    if s<1e-3:return None
    a=np.arctan2(v[0],-v[1]);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);xy=(xy@R.T)/s
    if left:xy[:,0]*=-1
    return xy[1:].ravel()
def frames(raw,a=0,b=None):
    out=[];b=len(raw)-1 if b is None else b
    for f in raw[a:b+1]:
        for st,left in ((0,True),(21,False)):
            h=f[st:st+21]
            if h[:,4].min()<.3:continue
            c=canon(h,left)
            if c is not None:out.append(c)
    return out
def clipframes(path):
    raw=np.load(path)['raw'].astype(np.float32);act=np.flatnonzero(raw[:,:42,4].max(1)>.3)
    if len(act)<4:return []
    a,b=act[0],act[-1];n=b-a;return frames(raw,a+n//4,a+3*n//4)
X,y,g=[],[],[]
for lab,c in ((1,'HAVE'),(0,'ANGRY')):
    for f in glob.glob(f'/Users/frnzlo/slt_go_split_cache/isolated_raw/*/{c}/*.npz'):
        F=clipframes(f);X+=F;y+=[lab]*len(F)
X=np.array(X);y=np.array(y);sc=StandardScaler().fit(X)
clf=LogisticRegression(C=0.3,max_iter=2000,class_weight='balanced').fit(sc.transform(X),y)
print('train frames',len(y),'HAVE',int(y.sum()))
for c in ('HAVE','ANGRY','TIRED','HAPPY'):
    vals=[]
    for f in sorted(glob.glob(f'/Users/frnzlo/slt_go_split_cache/isolated_val_raw/citizen/{c}/*.npz')):
        F=clipframes(f)
        if F:vals.append(round(float(clf.predict_proba(sc.transform(np.array(F)))[:,1].mean()),2))
    print(f'Citizen val {c:6s} P(HAVE) per clip:',vals)
from active.v17.stage1_window_v17 import raw_observation_features
import cv2
item=pickle.load(open('/Users/frnzlo/slt_go_split_cache/takes2/20261009_135024_11.pkl','rb'))
ok,fr=cv2.VideoCapture(item['meta']['video']).read();view=np.broadcast_to(np.zeros((1,1,1),np.uint8),fr.shape)
for o in item['obs']:o.frame=view
raw,t=raw_observation_features(item['obs'])
for w in json.load(open('have_probe_words.json')):
    if w['g'] in ('ANGRY','HAVE','TIRED'):
        F=frames(raw,w['s'],w['e']);p=clf.predict_proba(sc.transform(np.array(F)))[:,1]
        print(f"user {w['g']:6s}@{t[w['s']]:5.2f} (truth HAVE) P(HAVE) mean {p.mean():.2f} max3 {np.sort(p)[-3:].mean():.2f}")
pickle.dump(dict(scaler=sc,clf=clf),open('handshape_have_angry.pkl','wb'))
