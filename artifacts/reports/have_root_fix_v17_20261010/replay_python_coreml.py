"""Run the production Python decoder on actual phone/PTS inputs and FP16 Core ML models."""
import base64,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import FrameDetection,HandDetection
from scripts.live_isolated_v17 import ObservedFrame
from active.v17.segmental_runtime_v17 import build_runtime,SegmentalRuntime
HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'config_fixed.json').read_text())
built=build_runtime(HERE/'config_fixed.json',backend='coreml',compute_units='ALL',image_encoder=ROOT/'artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP16.mlpackage')
base=built.words if hasattr(built,'words') else built
print('letter threshold',base.recognizer.letter_threshold,flush=True)
def hand(r,k):
 if k not in r:return None
 p=np.array(r[k],np.float32)
 return HandDetection(xy=p[:,:2],confidence=p[:,2],chirality=r.get(k+'c','unknown'),score=float(r.get(k+'s',0)))
results={}
for name,path in [('take11',HERE/'take11_trace.json'),('live',ROOT/'data/local/phone_session_pull_v17_20261010b/live_reel_traces/20261010_100646.json'),('0955',HERE/'0955_trace.json')]:
 rt=SegmentalRuntime(base.boundary.model,base.recognizer,cfg,mode='both',letter_gate=.95)
 words=[];n=0
 for r in json.loads(path.read_text())['frames']:
  if r.get('start') or r.get('held'):continue
  b=np.array(r['b'],np.float32);f=np.array(r['f'],np.float32);l,right=hand(r,'l'),hand(r,'r')
  det=FrameDetection(hands=[h for h in (l,right) if h is not None],body_xy=b[:,:2],body_confidence=b[:,2],face_xy=f[:,:2],face_confidence=f[:,2])
  o=ObservedFrame(frame=np.broadcast_to(np.zeros((1,1,1),np.uint8),(r['h'],r['w'],3)),detection=det,assigned={'left':l,'right':right},seconds=float(r['s']),motion=0.,hand_quality=0.,face_quality=0.,face_for_features=bool(r['ff']))
  he=np.frombuffer(base64.b64decode(r['e']),np.float16).astype(np.float32).reshape(3,512)
  h=(he,np.array(r['v'],np.float32),np.array(r['x'],np.float32).reshape(3,4))
  words+=rt.observe(o,hands=h);n+=1
 words+=rt.finish();words=[w for w in words if not w['gloss'].startswith('FS_')]
 results[name]=dict(frames=n,hypothesis=[w['gloss'] for w in words],words=words)
 print(name,results[name]['hypothesis'],flush=True)
 (HERE/'python_coreml_replays.json').write_text(json.dumps(results))
