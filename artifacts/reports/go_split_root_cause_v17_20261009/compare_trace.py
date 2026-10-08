"""Live vs recording, frame by frame: compare_trace.py LIVE_TRACE.json REPLAY_TRACE.json.
Aligns frames by time (offset estimated from wrist tracks), then reports hand presence, landmark
differences and hand-crop embedding similarity per frame, plus summaries around committed words."""
import base64,json,sys
import numpy as np
def load(p):
    rows=[r for r in json.load(open(p))['frames'] if not r.get('start')]
    out=[]
    for r in rows:
        d=dict(s=r['s'],held=r.get('held',False))
        for k in ('l','r'):d[k]=np.array(r[k],np.float32) if k in r else None
        d['b']=np.array(r['b'],np.float32)
        if 'e' in r:d['e']=np.frombuffer(base64.b64decode(r['e']),np.float16).astype(np.float32).reshape(3,512);d['v']=np.array(r['v'])
        out.append(d)
    return out
live,rep=load(sys.argv[1]),load(sys.argv[2])
def track(rows):
    t=np.array([r['s'] for r in rows]);y=np.array([np.nan if r['r'] is None else r['r'][0,1] for r in rows]);return t,y
tl,yl=track(live);tr,yr=track(rep)
best=None
for off in np.arange(-1.5,1.5,0.01):
    yi=np.interp(tl+off,tr,np.nan_to_num(yr,nan=-1));m=~np.isnan(yl)&(yi>=0)
    if m.sum()<50:continue
    err=np.nanmean(np.abs(yl[m]-yi[m]))
    if best is None or err<best[0]:best=(err,off)
print('frames live',len(live),'replay',len(rep),'offset (replay = live + off) %.2f s, wrist-y error %.4f'%(best[1],best[0]))
off=best[1];pairs=[]
for a in live:
    j=int(np.argmin(np.abs(tr-(a['s']+off))))
    if abs(tr[j]-(a['s']+off))<0.03:pairs.append((a,rep[j]))
pres=np.array([[a[k] is not None for k in 'lr']+[b[k] is not None for k in 'lr'] for a,b in pairs])
print('aligned frames',len(pairs))
print('hand present  live L/R %.2f %.2f   replay L/R %.2f %.2f'%tuple(pres.mean(0)))
dl=[np.abs(a[k][:,:2]-b[k][:,:2]).mean() for a,b in pairs for k in 'lr' if a[k] is not None and b[k] is not None]
dc=[np.abs(a[k][:,2]-b[k][:,2]).mean() for a,b in pairs for k in 'lr' if a[k] is not None and b[k] is not None]
print('landmark xy |diff| median %.4f p90 %.4f (image-normalised); confidence |diff| median %.3f'%(np.median(dl),np.percentile(dl,90),np.median(dc)))
cos=[]
for a,b in pairs:
    if 'e' in a and 'e' in b:
        for v in range(3):
            if a['v'][v]>.5 and b['v'][v]>.5:
                x,y=a['e'][v],b['e'][v];cos.append(float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9)))
cos=np.array(cos);print('hand-crop embedding cosine live vs replay: median %.3f p10 %.3f min %.3f (n=%d)'%(np.median(cos),np.percentile(cos,10),cos.min(),len(cos)))
held=sum(a['held'] for a in live);print('live frames held by finish gesture',held)
