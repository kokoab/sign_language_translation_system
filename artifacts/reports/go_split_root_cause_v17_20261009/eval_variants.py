"""Decoder variants on cached decoder inputs (slim_trace.py): installed v2 recognizer, phone word mode
(letters hidden, letter gate 0.95). Span logits are cached per video (pure function of the cached inputs),
so variants only pay for new spans. Sets: citizen_val (isolated, 378), tune (89 local continuous phrases,
WER), takes (user phone takes: GO / words). Usage: eval_variants.py OUT.json SETS VARIANTS.json"""
import copy,json,pickle,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.av_boundary_v17 import AVBoundary
from active.v17.segmental_runtime_v17 import SegmentalRuntime,build_runtime
from scripts.evaluate_temporal_boundary_v17 import edit_counts
CACHE=Path('/Users/frnzlo/slt_go_split_cache')
CFG=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/user_take_replay/config_v2_installed.json'
words=build_runtime(CFG,device='mps').words
def boundary():
    p=torch.load(ROOT/words.config['boundary'],map_location='cpu',weights_only=False);b=AVBoundary(**p['config']);b.load_state_dict(p['state_dict']);return b.eval()
_recs={}
def recognizer(path):
    if path is None:return words.recognizer
    if path not in _recs:
        base=json.loads(CFG.read_text());cfg=dict(base,recognizer=path,letter_head=v_head.get(path,base['letter_head']))
        tmp=Path(__file__).resolve().parent/f'config_{abs(hash(path))%10**8}.json';tmp.write_text(json.dumps(cfg,indent=1))
        _recs[path]=build_runtime(tmp,device='mps').words.recognizer
    return _recs[path]
v_head={}
def runtime(v):
    cfg=copy.deepcopy(words.config)
    for k,x in v.get('decoder',{}).items():cfg['decoder'][k]=x
    for k,x in v.get('stream',{}).items():cfg['stream'][k]=x
    if v.get('letter_head'):v_head[v['recognizer']]=v['letter_head']
    if v.get('boundary'):cfg['boundary']=v['boundary']
    return SegmentalRuntime(boundary_from(cfg['boundary']),recognizer(v.get('recognizer')),cfg,mode='both',letter_gate=0.95)
def boundary_from(path):
    p=torch.load(ROOT/path,map_location='cpu',weights_only=False);b=AVBoundary(**p['config']);b.load_state_dict(p['state_dict']);return b.eval()
_shapes={}
def restore_frames(item):
    import cv2
    v=item['meta']['video']
    if v not in _shapes:
        ok,f=cv2.VideoCapture(v).read();_shapes[v]=f.shape
    view=np.broadcast_to(np.zeros((1,1,1),np.uint8),_shapes[v])
    for o in item['obs']:o.frame=view
def run_video(rt,item,cache):
    rt.reset();rt.raw_cache=cache;out=[]
    obs,hands=item['obs'],item['hands']
    if item.get('pad'):
        import copy as c
        obs=list(obs);hands=list(hands)
        for k in range(20):
            o=c.copy(obs[-1]);o.seconds=obs[-1].seconds+0.05*(k+1);obs.append(o);hands.append(hands[-1])
    for o,h in zip(obs,hands):out+=rt.observe(o,hands=h)
    out+=rt.finish()
    return [dict(g=w['gloss'],s=w['start_frame'],e=w['end_frame'],c=w['commit_frame'],q=round(w['score'],3)) for w in out if not w['gloss'].startswith('FS_')]
def main(out_path,sets,variants):
    variants=json.loads(Path(variants).read_text()) if variants.endswith('.json') else json.loads(variants)
    res=json.loads(Path(out_path).read_text()) if Path(out_path).exists() else {}
    for name in sets.split(','):
        files=sorted((CACHE/name).glob('*.pkl'))
        tags={k:v.get('cache_tag','v2') for k,v in variants.items()}
        rts={k:runtime(v) for k,v in variants.items() if f'{name}:{k}' not in res}
        if not rts:continue
        per={k:[] for k in rts}
        for f in files:
            item=pickle.load(open(f,'rb'));restore_frames(item);item['pad']=name in ('citizen_val','pair_val')
            secs=np.array([o.seconds for o in item['obs']]);safe=len(secs)<2 or np.diff(secs).max()<=.26
            for tag in sorted({tags[k] for k in rts}):
                sc=CACHE/(name+'_spans'+('' if tag=='v2' else '_'+tag));sc.mkdir(exist_ok=True)
                cp=sc/f.name;cache=pickle.load(open(cp,'rb')) if (safe and cp.exists()) else {}
                n0=len(cache)
                for k,rt in rts.items():
                    if tags[k]!=tag:continue
                    per[k].append(dict(key=f.stem,target=item['meta']['target'],words=run_video(rt,item,cache if safe else {})))
                if safe and len(cache)>n0:
                    tmp=cp.with_suffix(f'.{__import__("os").getpid()}.tmp');pickle.dump(cache,open(tmp,'wb'));tmp.replace(cp)
        for k,rows in per.items():
            res[f'{name}:{k}']=dict(summary=summarize(name,rows),rows=rows)
            print(name,k,json.dumps(res[f'{name}:{k}']['summary']),flush=True)
        Path(out_path).write_text(json.dumps(res))
def latency(rows):
    lat=[(w['c']-w['e'])/20 for r in rows for w in r['words']]
    return dict(lat_median=round(float(np.median(lat)),3),lat_p90=round(float(np.percentile(lat,90)),3)) if lat else {}
def summarize(name,rows):
    return {**summarize_(name,rows),**latency(rows)}
def summarize_(name,rows):
    if name in ('citizen_val','pair_val'):
        hyp=[[w['g'] for w in r['words']] for r in rows];t=[r['target'][0] for r in rows]
        return dict(n=len(rows),correct=sum(a in h for a,h in zip(t,hyp)),exact=sum(h==[a] for a,h in zip(t,hyp)),
                    silent=sum(not h for h in hyp),extra=sum(max(0,len(h)-1) for h in hyp))
    if name=='tune':
        tot=dict(S=0,D=0,I=0,C=0,R=0,H=0)
        for r in rows:
            h=[w['g'] for w in r['words']];m=edit_counts(r['target'],h)
            for a,b in (('S','substitutions'),('D','deletions'),('I','insertions'),('C','correct'),('R','references')):tot[a]+=m[b]
            tot['H']+=len(h)
        return dict(wer=round((tot['S']+tot['D']+tot['I'])/tot['R'],4),**tot)
    out={}
    for r in rows:
        h=[w['g'] for w in r['words']]
        out[r['key'][-2:]]=dict(n=len(h),HAVE=h.count('HAVE'),ANGRY=h.count('ANGRY'),GO=h.count('GO'),BIG=h.count('BIG'),LANGUAGE=h.count('LANGUAGE'),other=len(h)-h.count('GO')-h.count('BIG')-h.count('LANGUAGE'),text=' '.join(h))
    return out
if __name__=='__main__':main(*sys.argv[1:])
