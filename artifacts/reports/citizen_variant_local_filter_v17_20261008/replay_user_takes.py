"""Replay the user's phone takes through the phone word decoder (letter gate 0.95, letters hidden)
with several recognizers on identical Vision observations and hand-crop encodings."""
import json,sys
from pathlib import Path
import cv2
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
import torch
from active.v17.av_boundary_v17 import AVBoundary
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.segmental_runtime_v17 import CONFIG,SegmentalRuntime,build_runtime
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
E=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008';OUT=E/'user_take_replay';OUT.mkdir(exist_ok=True)
MODELS={'v2_installed':(E/'run_v2/span_recognizer/best_model.pth',E/'deploy_v2/letter_head_a/model.pth'),
        'v4':(E/'run_v4/span_recognizer/best_model.pth',E/'deploy_v4/letter_head_a/model.pth')}
base=json.loads(CONFIG.read_text())
def boundary():
    p=torch.load(ROOT/base['boundary'],map_location='cpu',weights_only=False);b=AVBoundary(**p['config']);b.load_state_dict(p['state_dict']);return b.eval()
rts={}
for name,(rec,head) in MODELS.items():
    cfg=dict(base,backend='torch',recognizer=str(rec.relative_to(ROOT)),letter_head=str(head.relative_to(ROOT)),letter_threshold=0.5)
    path=OUT/f'config_{name}.json';path.write_text(json.dumps(cfg,indent=1))
    words=build_runtime(path,device='mps').words
    rts[name]=SegmentalRuntime(boundary(),words.recognizer,words.config,mode='both',letter_gate=0.95)
args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
result={}
for vid in sys.argv[1:]:
    cap=cv2.VideoCapture(vid);fps=cap.get(cv2.CAP_PROP_FPS) or 30;wr={'left':None,'right':None};i=0;nt=0.0;pr=0
    out={k:[] for k in rts};[r.reset() for r in rts.values()];first=next(iter(rts.values()))
    while True:
        ok,f=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;o=observe_stage2_frame(f,t,pr,det,wr,args);pr+=1
        out[next(iter(rts))]+=first.observe(o);h=first.hands[-1]
        for k,r in rts.items():
            if r is not first:out[k]+=r.observe(o,hands=h)
    for k,r in rts.items():out[k]+=r.finish()
    result[Path(vid).name]={k:[dict(gloss=w['gloss'],start=round(w['start_seconds'],2),end=round(w['end_seconds'],2),score=round(w['score'],2)) for w in v if not w['gloss'].startswith('FS_')] for k,v in out.items()}
    print(Path(vid).name)
    for k,v in result[Path(vid).name].items():print(f'  {k:13s}',' '.join(w['gloss'] for w in v))
(OUT/'results.json').write_text(json.dumps(result,indent=1)+'\n')
