"""Replay validation videos through the live segmental decoder (Python reference of the phone path).

Per clip: frames at 20 Hz -> Apple Vision -> boundary student -> span recognizer (+ letter head,
dual word/letter decoder as on the phone) -> committed words. Outcome per clip: target committed,
other word(s) only, or nothing committed; plus the best preview of the target. Run for the August
recognizer (phone before today) and the Citizen-variant recognizer (installed today).
Citizen val + SemLex val only; no test data.
"""
import collections,copy,glob,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.segmental_runtime_v17 import CONFIG,build_runtime
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
HERE=Path(__file__).resolve().parent;OUT=HERE/'commit_replay';OUT.mkdir(exist_ok=True)
base=json.loads(CONFIG.read_text())
new=dict(base,recognizer='artifacts/reports/citizen_variant_local_filter_v17_20261008/run/span_recognizer/best_model.pth',
         letter_head='artifacts/reports/citizen_variant_local_filter_v17_20261008/deploy/letter_head_a/model.pth',letter_threshold=0.6)
CONFIGS={'august_phone':dict(base),'citizen_variant':new}
for c in CONFIGS.values():c['backend']='torch'
classes=set(sys.argv[1].split(',')) if len(sys.argv)>1 and sys.argv[1] else None
videos=[('citizen',p) for p in sorted(glob.glob(str(ROOT/'data/local/citizen100_v17/raw/val/*/*')))]
videos+=[('semlex',p) for p in sorted(glob.glob(str(ROOT/'data/local/semlex_citizen100_val_audit/raw/*/*')))]
videos=[(s,p) for s,p in videos if '/test/' not in p and (classes is None or Path(p).parent.name in classes)]
args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
# Observe each clip once; both runtimes consume the identical observation sequence.
results={k:[] for k in CONFIGS}
runtimes={}
for name,cfg in CONFIGS.items():
    path=OUT/f'config_{name}.json';path.write_text(json.dumps(cfg,indent=1));runtimes[name]=build_runtime(path,device='mps')
for n,(src,path) in enumerate(videos):
    target=Path(path).parent.name;cap=cv2.VideoCapture(path);fps=cap.get(cv2.CAP_PROP_FPS) or 30
    obs=[];wrists={'left':None,'right':None};i=0;nt=0.0;processed=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;obs.append(observe_stage2_frame(frame,t,processed,det,wrists,args));processed+=1
    # 1 s of trailing rest so the decoder can close the last span, as when a signer lowers the hands.
    for k in range(20):
        o=copy.copy(obs[-1]);o.seconds=obs[-1].seconds+0.05*(k+1);obs.append(o)
    for name,rt in runtimes.items():
        rt.reset();words=[];best=0.0
        for o in obs:
            words+=rt.observe(o)
            p=getattr(rt,'preview',None) or getattr(getattr(rt,'words',None),'preview',None)
            if p and p.get('gloss')==target:best=max(best,float(p.get('score',0)))
        words+=rt.finish()
        glosses=[w['gloss'] if isinstance(w,dict) else w.gloss for w in words]
        outcome='target' if target in glosses else ('other' if glosses else 'nothing')
        results[name].append(dict(source=src,target=target,path=str(Path(path).relative_to(ROOT)),words=glosses,outcome=outcome,best_target_preview=best))
    if n%50==0:print(n,len(videos),flush=True)
(OUT/'results.json').write_text(json.dumps(results,indent=1)+'\n')
for name,rows in results.items():
    c=collections.Counter(r['outcome'] for r in rows);print(name,dict(c))
