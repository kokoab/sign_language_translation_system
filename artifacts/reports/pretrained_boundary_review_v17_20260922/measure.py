"""Disposable profiling only; existing checkpoints/caches remain untouched."""
import sys,json,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from active.v17.pretrained_boundary_v17 import *
from active.v17.temporal_boundary_v17 import masked_boundary_loss
from scripts.train_pretrained_boundary_v17 import CACHE,REPORT as RUN
from active.v17.approved_phrase_data_v17 import digest
OUT=Path(__file__).parent
state_path=ROOT/json.loads((RUN/'training_results.json').read_text())['runs'][0]['checkpoint']
state=torch.load(state_path,map_location='cpu',weights_only=False)['model_state_dict']
times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32,device='mps')

def stats(values):return dict(n=len(values),median_ms=float(np.median(values)*1000),p95_ms=float(np.percentile(values,95)*1000))
def save(result):(OUT/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
def timing(fn,n=30):
 for _ in range(4):fn();torch.mps.synchronize()
 values=[]
 for _ in range(n):
  torch.mps.synchronize();start=time.perf_counter();fn();torch.mps.synchronize();values.append(time.perf_counter()-start)
 return stats(values)

def main():
 result=dict(checkpoint_sha256=digest(state_path),scope='Desktop synthetic scheduling/real cached input microbenchmarks; not camera or iPhone end-to-end latency. Disposable benchmark optimizer updates not saved.')
 rows=json.loads((UPSTREAM/'prepared_manifest.json').read_text())['records'];row=next(r for r in rows if r['split']=='train' and r['pose_frames']>200)
 pose=load_pose(row);target=min(70,len(pose.body.data)-1)
 model=PretrainedBoundary().to('mps').eval();model.load_state_dict(state,strict=True)
 x_np,_=window_features(pose,target);x=torch.from_numpy(x_np)[None].to('mps')
 with torch.inference_mode():
  projected=model.project(x)
  result['batch1_normalization']=timing(lambda:window_features(pose,target))
  result['batch1_cnn']=timing(lambda:model.project(x))
  result['batch1_attention_head']=timing(lambda:model.forward_projected(projected,times))
  result['batch1_forward']=timing(lambda:model(x,times))
  def complete():return model(torch.from_numpy(window_features(pose,target)[0])[None].to('mps'),times)
  result['batch1_normalize_transfer_forward']=timing(complete)
  # Perturb a neighboring frame while the center stays fixed: CNN is temporal.
  other=x.clone();other[:,TARGET-2]+=1
  result['cnn_neighbor_effect_on_target_maxabs']=float((model.project(other)[:,TARGET]-projected[:,TARGET]).abs().max().cpu())
 save(result)
 # Real frozen Reel classification concurrently, including its CoreML hand encoder.
 from scripts.evaluate_temporal_boundary_v17 import evaluation_rows,arguments,observations
 from scripts.live_reel_stage1_v17 import build_components
 from scripts.live_boundary_v17 import classify_interval
 from active.v17.extract_v17 import AppleVisionDetector
 evalrow=evaluation_rows()[0];args=arguments(evalrow['video_path'],OUT/'unused_sessions');args.no_motion_trim=True
 reel=build_components(args)['classifier'];reel.args.no_motion_trim=reel.full.args.no_motion_trim=True
 obs=observations(evalrow,args,AppleVisionDetector(args.minimum_point_confidence));event=evalrow['events'][0]
 def recognize():return classify_interval(reel,obs,dict(start_seconds=event['start'],end_seconds=event['end']),args,.1)
 recognize();stop=threading.Event();reel_times=[];started=threading.Event()
 def worker():
  started.set()
  while not stop.is_set():
   tick=time.perf_counter();recognize();reel_times.append(time.perf_counter()-tick)
 with ThreadPoolExecutor(max_workers=1) as executor:
  future=executor.submit(worker);started.wait()
  try:
   with torch.inference_mode():result['batch1_with_reel_contention']=timing(complete,60)
  finally:stop.set();future.result()
 result['concurrent_reel_calls']=stats(reel_times);save(result)
 del reel,obs,projected,model;torch.mps.empty_cache()
 xmap=np.load(CACHE/'train_x.npy',mmap_mode='r');ymap=np.load(CACHE/'train_y.npy',mmap_mode='r')
 indices=np.random.default_rng(123).permutation(len(xmap))[:128*14].reshape(14,128)
 weights=torch.tensor(np.clip((ymap==0).sum(0)/np.maximum((ymap==1).sum(0),1),1,20),dtype=torch.float32,device='mps')
 profiles={}
 for variant in ('memmap_original','memmap_fewer_syncs','resident_fp32_fewer_syncs','resident_fp16_roundtrip_fewer_syncs'):
  resident=None
  if variant.startswith('resident'):resident=np.array(xmap,dtype=np.float16 if 'fp16' in variant else np.float32)
  model=PretrainedBoundary().to('mps');model.load_state_dict(state,strict=True);model.train();model.set_adaptation(True)
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=1e-4)
  durations=[];gather=[];gpu_total=torch.zeros((),device='mps')
  for i,ids in enumerate(indices):
   torch.mps.synchronize();tick=time.perf_counter();a=time.perf_counter()
   xx=xmap[ids] if resident is None else resident[ids].astype(np.float32,copy=False)
   yy=np.asarray(ymap[ids]);gather.append(time.perf_counter()-a)
   xt=torch.from_numpy(xx).to('mps');yt=torch.from_numpy(yy).to('mps');optimizer.zero_grad(set_to_none=True)
   logits=model.forward_projected(xt,times)
   if variant=='memmap_original':
    loss=masked_boundary_loss(logits,yt,weights)
    if not torch.isfinite(loss):raise ValueError('loss')
   else:
    valid=yt>=0;loss=(F.binary_cross_entropy_with_logits(logits,yt.clamp_min(0),pos_weight=weights,reduction='none')*valid).sum()/valid.sum()
   loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   if not (torch.isfinite(norm)&torch.isfinite(loss)):raise ValueError('grad/loss')
   optimizer.step()
   if variant=='memmap_original':n=int((yt>=0).sum());total=float(loss.detach().cpu())*n
   else:gpu_total+=loss.detach()*int((yy>=0).sum())
   torch.mps.synchronize()
   if i>=4:durations.append(time.perf_counter()-tick)
  profiles[variant]=dict(step=stats(durations),gather=stats(gather[4:]))
  result['training_profiles']=profiles;save(result)
  del model,optimizer,resident;torch.mps.empty_cache()
 # FP16 storage roundtrip is a precision change, not a lossless speed trick.
 model=PretrainedBoundary().to('mps').eval();model.load_state_dict(state,strict=True)
 with torch.inference_mode():
  vals=np.array(xmap[indices[0]]);a=model.forward_projected(torch.from_numpy(vals).to('mps'),times).sigmoid().cpu().numpy()
  b=model.forward_projected(torch.from_numpy(vals.astype('float16').astype('float32')).to('mps'),times).sigmoid().cpu().numpy()
 result['fp16_roundtrip_probe']=dict(max_probability_difference=float(np.abs(a-b).max()),threshold_flips=int(((a>=.5)!=(b>=.5)).sum()),windows=len(vals),scope='Training windows only; not a WER-equivalence test.')
 save(result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
