"""Decoder grid (segment bonus c x minimum span frames) on the installed v2 recognizer, phone word mode
(letter gate 0.95, letters hidden). Inputs: the user's three takes (insertions) and Citizen val (378).
Shared Vision pass, hand encodings and span logits. Resumable per clip; low priority."""
import copy,glob,json,resource,sys
from pathlib import Path
import cv2,torch
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.av_boundary_v17 import AVBoundary
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.segmental_runtime_v17 import SegmentalRuntime,build_runtime
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
E=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008';OUT=E/'decoder_grid';OUT.mkdir(exist_ok=True)
GRID=[(-1.5,3),(-1.0,3),(-0.5,3),(0.0,3),(-1.5,5),(-1.0,5),(-0.5,5),(-1.5,6),(-1.0,6)]
words=build_runtime(E/'user_take_replay/config_v2_installed.json',device='mps').words
def boundary():
    p=torch.load(ROOT/words.config['boundary'],map_location='cpu',weights_only=False);b=AVBoundary(**p['config']);b.load_state_dict(p['state_dict']);return b.eval()
rts={}
for c,m in GRID:
    cfg=copy.deepcopy(words.config);cfg['decoder']['c']=c;cfg['stream']['min_span_frames']=m
    rts[f'c{c}_min{m}']=SegmentalRuntime(boundary(),words.recognizer,cfg,mode='both',letter_gate=0.95)
shared={};[setattr(r,'raw_cache',shared) for r in rts.values()]
args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
takes=sorted(glob.glob(str(ROOT/'data/local/phone_session_pull_v17_20261009/live_reel_videos/20261009_082836_0*.mp4')))
cit=sorted(glob.glob(str(ROOT/'data/local/citizen100_v17/raw/val/*/*')));assert not any('/test/' in v for v in cit)
JL=OUT/'results.jsonl';done={json.loads(l)['path'] for l in JL.read_text().splitlines()} if JL.exists() else set()
for n,vid in enumerate(takes+cit):
    rel=str(Path(vid).relative_to(ROOT))
    if rel in done:continue
    cap=cv2.VideoCapture(vid);fps=cap.get(cv2.CAP_PROP_FPS) or 30;wr={'left':None,'right':None};i=0;nt=0.0;pr=0;obs=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;obs.append(observe_stage2_frame(f,t,pr,det,wr,args));pr+=1
    if vid in cit:
        for k in range(20):
            o=copy.copy(obs[-1]);o.seconds=obs[-1].seconds+0.05*(k+1);obs.append(o)
    shared.clear();out={k:[] for k in rts};[r.reset() for r in rts.values()];first=next(iter(rts.values()))
    for o in obs:
        out[next(iter(rts))]+=first.observe(o);h=first.hands[-1]
        for k,r in rts.items():
            if r is not first:out[k]+=r.observe(o,hands=h)
    for k,r in rts.items():out[k]+=r.finish()
    row=dict(path=rel,kind='take' if vid in takes else 'citizen',target=Path(vid).parent.name,
             words={k:[w['gloss'] for w in v if not w['gloss'].startswith('FS_')] for k,v in out.items()})
    with JL.open('a') as fh:fh.write(json.dumps(row)+'\n')
    if n%25==0:print(n,len(takes)+len(cit),'maxrss_GB',round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,2),flush=True)
rows=[json.loads(l) for l in JL.read_text().splitlines()];summary={}
for k in rts:
    c=[r for r in rows if r['kind']=='citizen'];t=[r for r in rows if r['kind']=='take']
    correct=sum(r['target'] in r['words'][k] for r in c);silent=sum(not r['words'][k] for r in c);extra=sum(max(0,len(r['words'][k])-1) for r in c)
    summary[k]=dict(citizen_correct=correct,citizen_silent=silent,citizen_extra=extra,take_words=sum(len(r['words'][k]) for r in t),
                    take_GO=sum(r['words'][k].count('GO') for r in t),take_BIG=sum(r['words'][k].count('BIG') for r in t),
                    take_LANGUAGE=sum(r['words'][k].count('LANGUAGE') for r in t),takes={r['path'].split('/')[-1]:' '.join(r['words'][k]) for r in t})
(OUT/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
for k,v in summary.items():print(k,{x:v[x] for x in v if x!='takes'})
