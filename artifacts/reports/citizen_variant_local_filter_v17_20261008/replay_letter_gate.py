"""Word-mode decoder variants on Citizen validation videos (phone view: letters hidden).

Settings share one Vision pass, one hand-crop encoding and one recognizer call per span:
sink (current phone), letter gates G in GATES (letters compete only when top word p < G), words-only.
Saves per clip (resumable), low priority, logs peak RSS. Citizen val only (realistic rest before/after).
"""
import collections,glob,json,resource,sys
from pathlib import Path
import cv2
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.segmental_runtime_v17 import SegmentalRuntime,build_runtime
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
HERE=Path(__file__).resolve().parent;OUT=HERE/'letter_gate_replay';OUT.mkdir(exist_ok=True)
GATES=(0.95,0.9,0.8,0.7)
cfg_path=HERE/'commit_replay/config_citizen_variant.json'
base=build_runtime(cfg_path,device='mps').words           # phone word decoder: mode 'both' + letter head
cfg=base.config
def _boundary():
    import torch
    from active.v17.av_boundary_v17 import AVBoundary
    payload=torch.load(ROOT/cfg['boundary'],map_location='cpu',weights_only=False)
    b=AVBoundary(**payload['config']);b.load_state_dict(payload['state_dict']);return b.eval()
def make(mode,gate=None):
    return SegmentalRuntime(_boundary(),base.recognizer,cfg,mode=mode,letter_gate=gate)
runtimes={'sink':base}
for g in GATES:runtimes[f'gate_{g}']=make('both',g)
runtimes['words_only']=make('words')
shared={}
for rt in runtimes.values():rt.raw_cache=shared
args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
videos=sorted(glob.glob(str(ROOT/'data/local/citizen100_v17/raw/val/*/*')));assert videos and not any('/test/' in v for v in videos)
JL=OUT/'results.jsonl';done=set()
if JL.exists():done={json.loads(l)['path'] for l in JL.read_text().splitlines()}
for n,path in enumerate(videos):
    rel=str(Path(path).relative_to(ROOT))
    if rel in done:continue
    target=Path(path).parent.name;cap=cv2.VideoCapture(path);fps=cap.get(cv2.CAP_PROP_FPS) or 30
    obs=[];wr={'left':None,'right':None};i=0;nt=0.0;proc=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;obs.append(observe_stage2_frame(frame,t,proc,det,wr,args));proc+=1
    import copy
    for k in range(20):
        o=copy.copy(obs[-1]);o.seconds=obs[-1].seconds+0.05*(k+1);obs.append(o)
    shared.clear();words={k:[] for k in runtimes}
    for rt in runtimes.values():rt.reset()
    for o in obs:
        words['sink']+=base.observe(o);hand=base.hands[-1]
        for k,rt in runtimes.items():
            if k!='sink':words[k]+=rt.observe(o,hands=hand)
    for k,rt in runtimes.items():words[k]+=rt.finish()
    row=dict(path=rel,target=target,words={k:[w['gloss'] for w in v] for k,v in words.items()})
    with JL.open('a') as fh:fh.write(json.dumps(row)+'\n')
    if n%25==0:print(n,len(videos),'maxrss_GB',round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,2),flush=True)
rows=[json.loads(l) for l in JL.read_text().splitlines()]
def view(ws,t):
    w=[g for g in ws if not g.startswith('FS_')];return ('target' if t in w else 'other' if w else 'nothing'),max(0,len(w)-1)
summary={}
for k in runtimes:
    c=collections.Counter();extra=0
    for r in rows:
        o,e=view(r['words'][k],r['target']);c[o]+=1;extra+=e
    summary[k]=dict(clips=len(rows),**c,extra_words=extra)
(OUT/'summary.json').write_text(json.dumps(summary,indent=1)+'\n');print(json.dumps(summary,indent=1))
