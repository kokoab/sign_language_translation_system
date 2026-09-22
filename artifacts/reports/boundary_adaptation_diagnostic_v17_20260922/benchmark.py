"""Synchronized batch-one boundary timings on a real unchanged calibration pose."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts import evaluate_boundary_expanded_v17 as e
OUT=Path(__file__).parent

def main():
 torch.set_num_threads(2)
 assert (OUT/'evaluation.json').exists() and (OUT/'calibration.json').exists(),'benchmark after inference workers finish'
 if (OUT/'benchmark.json').exists():raise FileExistsError('preserve benchmark')
 rows=json.loads(e.PREPARED.read_text())['records'];row=next(r for r in rows if r['split']=='calibration' and r['pose_frames']>100)
 assert e.digest(ROOT/row['pose_path'])==row['pose_sha256'];pose=e.load_pose(row)
 times=torch.tensor((np.arange(e.FRAMES)-e.TARGET)[None]/e.FPS,dtype=torch.float32,device='mps')
 results=[]
 for spec in e.ARMS:
  model=e.PretrainedBoundary().to('mps').eval()
  if spec['checkpoint']:model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
  for readout in (('bio',) if not spec['checkpoint'] else ('bio','edges')):
   durations=[]
   with torch.inference_mode():
    for i in range(35):
     torch.mps.synchronize();tick=time.perf_counter()
     x,_=e.window_features(pose,min(53+i,len(pose.body.data)-1))
     h=model.encode_projected(model.project(torch.from_numpy(x[None]).to('mps')),times)
     value=model.backbone.sign_bio_head(h).log_softmax(-1) if readout=='bio' else model.edge(h).sigmoid()
     value.cpu();torch.mps.synchronize()
     if i>=5:durations.append(time.perf_counter()-tick)
   results.append(dict(arm=spec['name'],readout=readout,samples=len(durations),median_ms=float(np.median(durations)*1000),p95_ms=float(np.percentile(durations,95)*1000)))
 e.atomic(OUT/'benchmark.json',dict(status='complete',results=results,lookahead_ms=500,
  pose_sha256=row['pose_sha256'],limitation='Batch1 real cached pose: normalization, temporal CNN, attention, readout and CPU output. Excludes MediaPipe frontend, Reel, camera scheduling and thermals; no live/iPhone claim.'))
 print(json.dumps(results))

if __name__=='__main__':main()
