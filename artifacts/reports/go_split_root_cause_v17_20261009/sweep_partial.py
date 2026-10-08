"""Partial-read sweep for one recognizer: sweep_partial.py CHECKPOINT TAG. Full + 5 partial views of the
Citizen/SemLex/local val clips (never test); confident (>=0.5) wrong reads of fragments of clips read
correctly whole. Writes sweep/partial_TAG.json."""
import json,sys,collections
from pathlib import Path
import numpy as np,torch
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
HERE=Path(__file__).resolve().parent
C=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
L=json.loads((ROOT/'artifacts/coreml/live_reel_labels.json').read_text())
ckpt,tag=sys.argv[1],sys.argv[2]
model,_=load_model(Path(ckpt));model.to('mps').eval()
X={k:[] for k in ('landmarks','hand_embeddings','hand_valid','hand_boxes','targets')}
for s in ('citizen','semlex','local'):
    d=np.load(C/f'{s}_val.npz')
    for k in X:X[k].append(d[k])
X={k:np.concatenate(v) for k,v in X.items()};T=X['targets'];N=len(T)
@torch.inference_mode()
def predict(lm,he,hv,hb,bs=256):
    out=[]
    for i in range(0,len(lm),bs):
        x=[torch.from_numpy(np.ascontiguousarray(a[i:i+bs])).to('mps') for a in (lm.astype(np.float32),he.astype(np.float32),hv.astype(bool),hb.astype(np.float32))]
        out.append(torch.softmax(model(*x).float(),-1).cpu().numpy())
    return np.concatenate(out)
def view(a,b):
    lm,he,hv,hb=X['landmarks'],X['hand_embeddings'],X['hand_valid'],X['hand_boxes'];ar=np.arange(N)[:,None]
    p=(a+(b-a)*np.linspace(0,1,32))[None].repeat(N,0)*31;lo=np.floor(p).astype(int).clip(0,31);hi=(lo+1).clip(max=31);w=(p-lo)[...,None,None]
    o=lm[ar,lo].copy();o[...,:3]=lm[ar,lo][...,:3]*(1-w)+lm[ar,hi][...,:3]*w;o[...,3:]=lm[ar,np.rint(p).astype(int).clip(0,31)][...,3:]
    q=np.rint((a+(b-a)*np.linspace(0,1,16))[None].repeat(N,0)*15).astype(int).clip(0,15)
    return o,he[ar,q],hv[ar,q],hb[ar,q]
full=predict(X['landmarks'],X['hand_embeddings'],X['hand_valid'],X['hand_boxes']);ok=full.argmax(1)==T
res={'full_top1':float(ok.mean()),'views':{}};pairs=collections.Counter()
for name,(a,b) in {'prefix30':(0,.3),'prefix50':(0,.5),'suffix30':(.7,1),'suffix50':(.5,1),'middle50':(.25,.75)}.items():
    P=predict(*view(a,b));arg=P.argmax(1);conf=P.max(1)
    wrong=ok&(arg!=T)&(conf>=.5);own=ok&(arg==T)&(conf>=.5);silent=ok&(conf<.5)
    res['views'][name]=dict(wrong_confident=int(wrong.sum()),own_confident=int(own.sum()),below_threshold=int(silent.sum()))
    for t,z in zip(T[wrong],arg[wrong]):pairs[(L[t],L[z])]+=1
res['wrong_total']=sum(v['wrong_confident'] for v in res['views'].values())
watch=[('YEAR','MAKE'),('ANSWER','GO'),('DIFFERENT','GO'),('PLEASE','MY'),('OUR','MY'),('WE','I'),('SORRY','I'),('FAMILY','IMPORTANT'),('FAMILY','WAIT'),('MAN','FATHER'),('YOUR','MOTHER'),('LEARN','SICK'),('LEARN','LESS'),('TIRED','HAVE'),('DRINK','MOTHER')]
res['watch']={f'{a}->{b}':pairs[(a,b)] for a,b in watch};res['top']=[(f'{a}->{b}',n) for (a,b),n in pairs.most_common(15)]
(HERE/'sweep'/f'partial_{tag}.json').write_text(json.dumps(res,indent=1))
print(tag,'full val top1 %.4f'%res['full_top1'],'confident wrong fragment reads',res['wrong_total']);print(' views',res['views']);print(' watch',res['watch'])
