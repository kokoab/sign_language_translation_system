"""Per active region of a traced take: Mac words (min3/min6), phone words, B-probability profile,
and recognizer top-3 on the whole region vs the committed pieces."""
import json,pickle,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
take=sys.argv[1];T=pickle.load(open(HERE/'trace'/f'20261009_082836_{take}.pkl','rb'));labels=T['labels']
phone=json.load(open(HERE/'phone_words.json'))[f'20261009_082836_{take}.mp4']
r=T['min3'];bio=r['bio'];sec=np.array(r['seconds']);n=max(bio)+1
P=np.exp(np.stack([bio.get(t,np.log(np.array([1e-6,1,1e-6,1e-6]))) for t in range(n)]))
act=P[:,2]+P[:,3]
def top(s,e,k=3):
    l=T['min3']['cache'].get((s,e))
    if l is None:l=T['min6']['cache'].get((s,e))
    if l is None:return None
    w=np.asarray(l[:100],np.float64);p=np.exp(w-w.max());p/=p.sum();o=np.argsort(-p)[:k]
    return ' '.join(f'{labels[i]}:{p[i]:.2f}' for i in o)
def nearest(s,e):
    keys=set(T['min3']['cache'])|set(T['min6']['cache'])
    best=max(keys,key=lambda k:(min(e,k[1])-max(s,k[0])+1)/(max(e,k[1])-min(s,k[0])+1))
    return best
on=act>=.3;regions=[];t=0
while t<n:
    if on[t]:
        u=t
        while u+1<n and (on[u+1] or (u+2<n and on[u+2]) or (u+3<n and on[u+3])):u+=1
        regions.append((t,u));t=u+1
    else:t+=1
for s,e in regions:
    if e-s<2:continue
    print(f'== region f{s}-{e} ({sec[s]:.2f}-{sec[e]:.2f}s, {e-s+1} fr)')
    print('   B  :',' '.join(f'{x:.1f}'[1:] if x<1 else '1.' for x in P[s:e+1,2]))
    print('   act:',' '.join(f'{x:.1f}'[1:] if x<1 else '1.' for x in act[s:e+1]))
    for tag in ('min3','min6'):
        ws=[w for w in T[tag]['words'] if w['end_frame']>=s-2 and w['start_frame']<=e+2]
        print(f'   mac {tag}:',' | '.join(f"{w['gloss']} f{w['start_frame']}-{w['end_frame']} q{w['score']:.2f}: [{top(w['start_frame'],w['end_frame'])}]" for w in ws))
    ph=[w for w in phone if w['end_seconds']>=sec[s]-.1 and w['start_seconds']<=sec[e]+.1]
    print('   phone:',' | '.join(f"{w['gloss']} {w['start_seconds']:.2f}-{w['end_seconds']:.2f} q{w['score']:.2f}" for w in ph))
    k=nearest(s,e);print(f'   whole region ~ span {k}: [{top(*k)}]')
