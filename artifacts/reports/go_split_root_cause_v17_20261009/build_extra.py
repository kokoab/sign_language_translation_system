"""Recognizer inputs (live verifier path: Apple Vision 20 Hz + MobileCLIP2 hand crops, whole clip) for extra
train-eligible clips of the zero-local classes: ASLLVD exact-variant (all), ASL-LEX citation (zero-local classes),
ASLLRP segmented HOME (signers BEN/CORY/RACHEL; JONATHAN validation-only and RIT excluded; visually checked).
Output: artifacts/generated/extra_zero_local_v17_20261009.npz in the raw-cache row format."""
import glob,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.segmental_runtime_v17 import build_runtime
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
Z=['WHAT','GOODBYE','HAVE','COME','FIND','HEAR','LISTEN','TALK','SICK','HUNGRY','BIG','HOME','CHILD','SIGN']
labels=json.loads((ROOT/'artifacts/coreml/live_reel_labels.json').read_text())
items=[]
m=json.loads((ROOT/'data/local/asllvd_asllex_v17/exact_variant_manifest.json').read_text())
for v in m['videos'] if 'videos' in m else []:pass
for c in Z:
    items+=[('asllvd',c,p) for p in sorted(glob.glob(str(ROOT/f'data/local/asllvd_asllex_v17/raw/{c}/*')))]
    items+=[('asllex',c,p) for p in sorted(glob.glob(str(ROOT/f'data/local/asllex_citation_videos_v17_20261009/raw/{c}/*')))]
a=json.loads((ROOT/'data/local/asllrp_segmented_citizen100_v17/manifest.json').read_text())['videos']
items+=[('asllrp','HOME',str(ROOT/v['path'])) for v in a if v['canonical_label']=='HOME' and v['split_role']=='train_candidate'
        and v['signer_id'] in ('BENJAMIN_JAMES_BAHAN','CORY','RACHEL')]
print('items',len(items),flush=True)
rec=build_runtime(ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/user_take_replay/config_v2_installed.json',device='mps').words.recognizer
args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
rows={k:[] for k in ('landmarks','hand_embeddings','hand_valid','hand_boxes','targets','item_ids')};skipped=[]
for src,c,p in items:
    cap=cv2.VideoCapture(p);fps=cap.get(cv2.CAP_PROP_FPS) or 30;wr={'left':None,'right':None};i=0;nt=0.0;pr=0;obs=[];hands=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;o=observe_stage2_frame(f,t,pr,det,wr,args);pr+=1;obs.append(o);hands.append(rec.frame_hand(o))
    x=rec.inputs(obs,hands) if len(obs)>=4 else None
    if x is None:skipped.append((src,c,Path(p).name,len(obs)));continue
    for k,v in zip(('landmarks','hand_embeddings','hand_valid','hand_boxes'),x):rows[k].append(v)
    rows['targets'].append(labels.index(c));rows['item_ids'].append(f'{src}:{c}:{Path(p).stem}'[:42])
out={k:np.stack(v) for k,v in rows.items() if k not in('targets','item_ids')}
out['hand_embeddings']=out['hand_embeddings'].astype(np.float16);out['hand_valid']=out['hand_valid'].astype(bool)
out['targets']=np.asarray(rows['targets'],np.int64);out['item_ids']=np.asarray(rows['item_ids'])
np.savez(ROOT/'artifacts/generated/extra_zero_local_v17_20261009.npz',**out)
import collections;print('built',len(out['targets']),collections.Counter(labels[t] for t in out['targets']),'skipped',skipped)
