from pathlib import Path
import os,sys,json,time,hashlib,argparse
os.environ['GLOG_minloglevel']='2'
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from PIL import Image
from active.v17.extract_v17 import AppleVisionDetector,read_video_frames,extract_frames_v17,rotate_frame_clockwise
from active.v17.schema_v17 import V17Config
from active.v17.mediapipe_full_v17 import MediaPipeFullDetector,MediaPipeFullV17Config,DetectionMemo
from active.v17.extract_hand_rgb_v17 import extract_clip,decode_packed_crops
from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.model_v17 import SLTStage1V17,Stage1V17Config
from active.v17 import coreml_runtime_v17 as cml
ap=argparse.ArgumentParser();ap.add_argument('family',choices=['apple','mediapipe']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
out=ROOT/'artifacts/reports/capstone_mac_comparison_v17_20261007'
mp=a.family=='mediapipe';device='mps';torch.set_num_threads(1)
prefix=ROOT/'artifacts/models'
base=prefix/'mp_stage1_v17_orientation_robust_v1_20261004/best_model.pth' if mp else ROOT/'artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/best_model.pth'
unified=prefix/('mp_stage1_v17_unified_multimodal_student_v1_20261004/best_model.pth' if mp else 'stage1_v17_unified_multimodal_student_v1/best_model.pth')
span=prefix/('mp_span_recognizer_v17_local_a_20261004/best_model.pth' if mp else 'span_recognizer_v17_local_a/best_model.pth')
ck=torch.load(base,map_location='cpu',weights_only=False);lm=SLTStage1V17(Stage1V17Config(**ck['model_config']));lm.load_state_dict(ck['model_state_dict']);lm.eval().to(device)
models={'base':lm,'fusion':load_model(unified)[0].to(device),'span':load_model(span)[0].to(device)}
config=MediaPipeFullV17Config() if mp else V17Config()
detector=MediaPipeFullDetector(config) if mp else AppleVisionDetector(config.minimum_point_confidence)
encoder=cml.load(ROOT/'artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage','ALL')
encoder.predict({'image':Image.fromarray(np.zeros((256,256,3),np.uint8))})
paths=sorted((ROOT/'data/local/citizen100_v17/landmarks/val').glob('*/*.v17.npz'))
if a.limit:paths=paths[:a.limit]
assert paths and all('test' not in p.parts for p in paths)
def sync():torch.mps.synchronize()
def infer(stage,x,e,v,b):
 sync();t=time.perf_counter()
 with torch.inference_mode():
  y=models[stage](x) if stage=='base' else models[stage](x,e,v,b)
 sync();return (time.perf_counter()-t)*1000
zeros=[torch.zeros(shape,device=device,dtype=dtype) for shape,dtype in [((1,32,61,5),torch.float32),((1,16,3,512),torch.float32),((1,16,3),torch.bool),((1,16,3,4),torch.float32)]]
for _ in range(5):
 for stage in models:infer(stage,*zeros)
rows=[]
for idx,p in enumerate(paths):
 if mp and detector.calls >= 1000:detector.renew()
 raw=ROOT/'data/local/citizen100_v17/raw/val'/p.parent.name/(p.name.removesuffix('.v17.npz')+'.mp4')
 with np.load(p) as z:metadata=json.loads(str(z['metadata_json']))
 frames,video_meta=read_video_frames(raw,config.maximum_source_frames,config.maximum_image_side,rotation='auto')
 angle=float(metadata.get('vision_coarse_rotation_clockwise') or 0)
 if angle:frames=[rotate_frame_clockwise(f,angle) for f in frames]
 d=DetectionMemo(detector) if mp else detector
 start=time.perf_counter();result=extract_frames_v17(frames,config,detector=d,metadata=video_meta);extract_ms=1000*(time.perf_counter()-start)
 if result is None:
  rows.append(dict(path=str(p.relative_to(ROOT)),failed=True,extraction_ms=extract_ms));continue
 landmark=ROOT/'data/local/mediapipe_full_v17_20261003/landmarks/citizen100_v17/landmarks/val'/p.parent.name/p.name if mp else p
 if mp:d.reset_sequence()
 start=time.perf_counter();arrays,_,_=extract_clip(raw,landmark,d,HandRGBV17Config())
 crops=decode_packed_crops(arrays['jpeg_blob'],arrays['jpeg_offsets'],256)
 emb=np.zeros((16,3,512),np.float32)
 for fi,vi in np.argwhere(arrays['valid']):
  emb[fi,vi]=np.asarray(encoder.predict({'image':Image.fromarray(crops[fi,vi])})['embedding'],np.float32).reshape(512)
 hand_ms=1000*(time.perf_counter()-start)
 t=time.perf_counter();x=torch.from_numpy(result.features.astype(np.float32)[None]).to(device);e=torch.from_numpy(emb[None]).to(device);v=torch.from_numpy(arrays['valid'][None]).to(device);b=torch.from_numpy(arrays['boxes_normalized'].astype(np.float32)[None]).to(device);sync();transfer=1000*(time.perf_counter()-t)
 # Same staged pre-processing and two warmed inference calls per graph.
 costs={stage:float(np.median([infer(stage,x,e,v,b) for _ in range(2)])) for stage in models}
 row=dict(path=str(p.relative_to(ROOT)),failed=False,extraction_ms=extract_ms,hand_preparation_ms=hand_ms,transfer_ms=transfer,inference_ms=costs,total_ms={stage:extract_ms+transfer+(0 if stage=='base' else hand_ms)+cost for stage,cost in costs.items()})
 rows.append(row)
 if idx%20==0:print(a.family,idx+1,len(paths),row['total_ms'],flush=True)
 (out/(a.family+('_smoke' if a.limit else '')+'.json')).write_text(json.dumps(dict(family=a.family,checkpoints=[str(v.relative_to(ROOT)) for v in [base,unified,span]],device=device,rows=rows),indent=2))
print('COMPLETE',a.family,len(rows),flush=True)
