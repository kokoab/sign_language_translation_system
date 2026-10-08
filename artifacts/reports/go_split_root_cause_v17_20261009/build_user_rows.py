"""User-domain training rows (recognizer inputs) from the user's phone sessions with known content:
135024_11 and 201823 ("I HAVE BIG" sentences and HAVE alone), 095533 (HAVE alone). Session 100646 is held out.
Committed spans of the installed decoder: I->I, BIG/LANGUAGE->BIG, HAVE/ANGRY/TIRED/HAPPY->HAVE; in I..BIG gaps
with no committed HAVE the most HAVE-like span is added as HAVE. Each span with +-2 frame jitter.
Output: artifacts/generated/user_rows_have_v17_20261010.npz"""
import json,pickle,sys,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_variants as ev
v=json.load(open('variants_installed_now.json'))['installed_now'];rt=ev.runtime(v);rec=rt.recognizer;L=rec.labels;H=L.index('HAVE')
MAP={'I':'I','BIG':'BIG','LANGUAGE':'BIG','HAVE':'HAVE','ANGRY':'HAVE','TIRED':'HAVE','HAPPY':'HAVE'}
sessions=[('takes2','20261009_135024_11'),('takes3','20261009_201823_01'),('takes4','20261010_095533_01')]
rows={k:[] for k in ('landmarks','hand_embeddings','hand_valid','hand_boxes')};T=[];IDS=[];cnt=collections.Counter()
def add(obs,hands,sec,s,e,label,tag):
    for ds in (-2,-1,0,1,2):
        for de in (-2,-1,0,1,2):
            a,b=s+ds,e+de
            if a<0 or b>=len(obs) or b-a<3:continue
            ids=[i for i in range(len(obs)) if sec[a]-.1<=sec[i]<=sec[b]+.1];x=rec.inputs([obs[i] for i in ids],[hands[i] for i in ids])
            if x is None:continue
            for k,val in zip(rows,x):rows[k].append(val)
            T.append(L.index(label));IDS.append(f'user:{tag}:{label}:{s}_{e}_{ds}_{de}'[:42])
for cache,stem in sessions:
    item=pickle.load(open(f'/Users/frnzlo/slt_go_split_cache/{cache}/{stem}.pkl','rb'));ev.restore_frames(item)
    obs,hands=item['obs'],item['hands'];sec=np.array([o.seconds for o in obs]);cache_d={}
    words=ev.run_video(rt,item,cache_d)
    for w in words:
        lab=MAP.get(w['g'])
        if lab:add(obs,hands,sec,w['s'],w['e'],lab,stem[-6:]);cnt[(stem[-9:],lab)]+=1
    if 'I' in [w['g'] for w in words] and cache!='takes4':
        for i,w in enumerate(words):
            if w['g']!='I':continue
            nxt=next((x for x in words[i+1:] if x['g'] in ('BIG','LANGUAGE')),None)
            if nxt is None or nxt['s']-w['e']>40 or any(MAP.get(x['g'])=='HAVE' for x in words[i+1:] if x['s']<nxt['s']):continue
            best=None
            for a in range(w['e']+1,nxt['s']-2):
                for b in range(a+3,min(nxt['s'],a+16)):
                    ids=[k for k in range(len(obs)) if sec[a]-.1<=sec[k]<=sec[b]+.1];x=rec.inputs([obs[k] for k in ids],[hands[k] for k in ids])
                    if x is None:continue
                    l=rec.logits([x])[0][:100];p=np.exp(l-l.max());p/=p.sum()
                    if best is None or p[H]>best[0]:best=(p[H],a,b)
            if best and best[0]>.2:add(obs,hands,sec,best[1],best[2],'HAVE',stem[-6:]+'gap');cnt[(stem[-9:],'HAVE_gap')]+=1
out={k:np.stack(v) for k,v in rows.items()};out['hand_embeddings']=out['hand_embeddings'].astype(np.float16);out['hand_valid']=out['hand_valid'].astype(bool)
out['targets']=np.asarray(T,np.int64);out['item_ids']=np.asarray(IDS)
np.savez('/Volumes/secret/SLT/SLT/artifacts/generated/user_rows_have_v17_20261010.npz',**out)
print('spans',dict(cnt));print('rows',len(T),collections.Counter(L[t] for t in T))
