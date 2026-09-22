"""Real approved-video end-to-end smoke, without opening camera or a window."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import cv2
from scripts.app_shell_v17 import parser
from scripts.live_renz_v17 import BufferedRenz
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.approved_phrase_data_v17 import digest
args=parser().parse_args(['--renz-buffered','--no-speech','--naturalizer','literal'])
model=BufferedRenz(args);detector=AppleVisionDetector(args.minimum_point_confidence)
manifest=json.loads((ROOT/'data/local/combined_dataset_v17_20260922/manifest.json').read_text())
row=next(r for r in manifest['records'] if r['source']=='asllrp_contiguous' and r['role']=='validation')
assert digest(ROOT/row['video_path'])==row['video_sha256']
cap=cv2.VideoCapture(str(ROOT/row['video_path']));fps=cap.get(cv2.CAP_PROP_FPS);obs=[];wrists={'left':None,'right':None};index=0;deadline=0.
while True:
 ok,frame=cap.read()
 if not ok:break
 t=index/fps;index+=1
 if t+1e-6<deadline:continue
 obs.append(observe_stage2_frame(frame,t,len(obs),detector,wrists,args));deadline=max(deadline+1/args.processing_fps,t)
cap.release();result=model.predict(obs,-1,obs[-1].seconds)
result.update(device=model.device,source=row['video_path'],frames=len(obs),source_seconds=obs[-1].seconds)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['device','inference_seconds','frames','source_seconds']}))
