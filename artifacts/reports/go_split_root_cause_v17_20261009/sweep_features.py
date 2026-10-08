"""Sweep of HAVE/ANGRY-like failure modes for all 100 classes on validation clips (Citizen/SemLex/local val raw
caches; never test), installed fragment recognizer, feature level (no decoder):
  full        whole clip
  lm_only     hand-crop inputs masked (what landmarks alone say)
  partial     prefix/suffix 30%/50% and middle 50% views (resampled to 32/16 frames)
  next/prev   the clip's core (15-85%) followed by the start (15-30%) of a clip of every class, or preceded by
              the end (70-85%) of one: synthetic transitions (HAVE + start of BIG -> ANGRY?)
Output sweep/feature_sweep.npz + sweep/feature_sweep.json."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
HERE=Path(__file__).resolve().parent;OUT=HERE/'sweep'
C=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
def main(ckpt=None,tag='feature_sweep',transitions=True):
    global labels,model,X,src,N,T,res
    labels=json.loads((ROOT/'artifacts/coreml/live_reel_labels.json').read_text())
    model,_=load_model(HERE/'run_frag_0.3/span_recognizer/best_model.pth');model.to('mps').eval()
    X={k:[] for k in ('landmarks','hand_embeddings','hand_valid','hand_boxes','targets')};src=[]
    for s in ('citizen','semlex','local'):
        d=np.load(C/f'{s}_val.npz')
        for k in X:X[k].append(d[k])
        src+= [s]*len(d['targets'])
    X={k:np.concatenate(v) for k,v in X.items()};src=np.array(src);N=len(src);T=X['targets']
    print('clips',N,flush=True)
    @torch.inference_mode()
    def predict(lm,he,hv,hb,bs=256):
        out=[]
        for i in range(0,len(lm),bs):
            x=[torch.from_numpy(np.ascontiguousarray(a[i:i+bs])).to('mps') for a in (lm.astype(np.float32),he.astype(np.float32),hv.astype(bool),hb.astype(np.float32))]
            out.append(torch.softmax(model(*x).float(),-1).cpu().numpy())
        return np.concatenate(out)
    def take(lm,he,hv,hb,pos_lm,pos_h):
        """Resample by fractional positions (0..1) of the 32 landmark / 16 hand frames (nearest; xyz linear)."""
        n=lm.shape[0]
        p=pos_lm*31;lo=np.floor(p).astype(int).clip(0,31);hi=(lo+1).clip(max=31);w=(p-lo)[...,None,None]
        ar=np.arange(n)[:,None]
        o=lm[ar,lo].copy();o[...,:3]=lm[ar,lo][...,:3]*(1-w)+lm[ar,hi][...,:3]*w;o[...,3:]=lm[ar,np.rint(p).astype(int).clip(0,31)][...,3:]
        q=np.rint(pos_h*15).astype(int).clip(0,15)
        return o,he[ar,q],hv[ar,q],hb[ar,q]
    def window(a,b,n):
        u=np.linspace(0,1,32)[None].repeat(n,0);v=np.linspace(0,1,16)[None].repeat(n,0);return a+(b-a)*u,a+(b-a)*v
    res={'targets':T,'src':src}
    lm,he,hv,hb=X['landmarks'],X['hand_embeddings'],X['hand_valid'],X['hand_boxes']
    res['full']=predict(lm,he,hv,hb)
    res['lm_only']=predict(lm,np.zeros_like(he),np.zeros_like(hv),np.zeros_like(hb))
    res['hand_only']=predict(np.zeros_like(lm),he,hv,hb)
    for name,(a,b) in {'prefix30':(0,.3),'prefix50':(0,.5),'suffix30':(.7,1),'suffix50':(.5,1),'middle50':(.25,.75)}.items():
        pl,ph=window(a,b,N);res[name]=predict(*take(lm,he,hv,hb,pl,ph));print(name,flush=True)
    core=window(.15,.85,N);res['core']=predict(*take(lm,he,hv,hb,*core))
    # transitions: up to 6 clips per class as X (Citizen first), one representative clip per class as Y
    rng=np.random.default_rng(20261009)
    xs=[];ys=[]
    for c in range(100):
        idx=np.flatnonzero(T==c);idx=idx[np.argsort(src[idx]!='citizen',kind='stable')]
        xs+=list(idx[:6])
        full_ok=idx[res['full'][idx].argmax(1)==c]   # a Y clip the model reads correctly
        ys.append(int(full_ok[0]) if len(full_ok) else int(idx[0]))
    xs=np.array(xs);ys=np.array(ys)
    def compose(xi,yi,mode,k=.15):
        """core of X with the start of Y appended (next) or the end of Y prepended (prev)."""
        n=len(xi);u=np.linspace(0,1,32);v=np.linspace(0,1,16)
        def build(grid):
            if mode=='next':
                inx=grid<1-k;px=np.where(inx,.15+.70*grid/(1-k),0);py=np.where(inx,0,.15+.15*(grid-(1-k))/k)
            else:
                inx=grid>=k;px=np.where(inx,.15+.70*(grid-k)/(1-k),0);py=np.where(inx,0,.70+.15*grid/k)
            return inx,px,py
        il,pxl,pyl=build(u);ih,pxh,pyh=build(v)
        ox=take(lm[xi],he[xi],hv[xi],hb[xi],np.repeat(pxl[None],n,0),np.repeat(pxh[None],n,0))
        oy=take(lm[yi],he[yi],hv[yi],hb[yi],np.repeat(pyl[None],n,0),np.repeat(pyh[None],n,0))
        return tuple(np.where((il if a.shape[1]==32 else ih).reshape((1,-1)+(1,)*(a.ndim-2)),a,b) for a,b in zip(ox,oy))
    for mode in ('next','prev'):
        P=np.zeros((len(xs),100,100),np.float16)
        for j,y in enumerate(ys):
            P[:,j]=predict(*compose(xs,np.full(len(xs),y),mode))
        res[mode]=P;print(mode,flush=True)
    res['trans_x']=xs;res['trans_y']=ys
    np.savez_compressed(OUT/'feature_sweep.npz',**res);print('saved',flush=True)
if __name__=='__main__':main()
