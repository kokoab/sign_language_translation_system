"""HAVE vs TIRED: how much the hands roll during the sign. Per frame, per hand: angle of the wrist->middle-MCP
vector; roll = range (p90 - p10) of that angle over the frames where the hand is visible, averaged over hands.
Also droop = change of fingertip-minus-wrist height (fingers turning down)."""
import numpy as np
def ok(f,st):return (f[st:st+21,4]>=.3).sum()>=15
def roll(raw):
    raw=np.asarray(raw,np.float32);vals=[];droops=[]
    for st in (0,21):
        fr=[f for f in raw if ok(f,st)]
        if len(fr)<3:continue
        ang=np.unwrap(np.array([np.arctan2(f[st+9,0]-f[st,0],-(f[st+9,1]-f[st,1])) for f in fr]))
        vals.append(np.percentile(ang,90)-np.percentile(ang,10))
        size=np.median([np.linalg.norm(f[st+9,:2]-f[st,:2]) for f in fr])+1e-6
        tip=np.array([f[[st+8,st+12,st+16],1].mean()-f[st,1] for f in fr])/size
        droops.append(tip.max()-tip.min())
    if not vals:return None
    return float(np.mean(vals)),float(np.mean(droops))
