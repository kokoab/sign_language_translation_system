"""Frozen published Renz I3D/MS-TCN -> current Reel, approved development only."""
import sys,json,time,hashlib,subprocess,importlib.util
from pathlib import Path
import numpy as np,torch,cv2
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
EDGE='--edge-pad' in sys.argv
if EDGE:OUT=OUT/'edge_padded';OUT.mkdir(exist_ok=True)
VENDOR=ROOT/'artifacts/vendor/renz_sign_segmentation'
from scripts.diagnose_combined_transitions_v17 import selected_rows
from active.v17.approved_phrase_data_v17 import digest,verify_manifest

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def spans(labels):
 out=[];start=None
 for i,value in enumerate(list(labels)+[1]):
  if value==0 and start is None:start=i
  if value!=0 and start is not None:out.append((start,i));start=None
 return out

def distance(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  old=d;d=[i]
  for j,y in enumerate(b,1):d.append(min(d[-1]+1,old[j]+1,old[j-1]+(x!=y)))
 return d[-1]

def self_check():
 assert spans([1,0,0,1,0])==[(1,3),(4,5)]
 assert spans([1,1])==[] and spans([0])==[(0,1)]
 assert distance(['A','B'],['A','X','B'])==1

def main():
 self_check();verify_manifest();torch.set_num_threads(2)
 device='cpu' # Published Conv3D path; CPU avoids unsupported MPS operations.
 i3d=module('renz_i3d',VENDOR/'demo/models/i3d.py').InceptionI3d(num_classes=981,num_in_frames=16,include_embds=True)
 weights=ROOT/'artifacts/models/renz_pretrained_v17'
 state=torch.load(weights/'i3d_kinetics_bslcp.pth.tar',map_location='cpu',weights_only=True)['state_dict']
 i3d.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);i3d.eval()
 seg=module('renz_mstcn',VENDOR/'demo/models/mstcn.py').MultiStageModel(4,10,64,1024,2)
 seg.load_state_dict(torch.load(weights/'mstcn_bslcp_i3d_bslcp.model',map_location='cpu',weights_only=True),strict=True);seg.eval()
 manifest=ROOT/'data/local/combined_dataset_v17_20260922/manifest.json';curated=ROOT/'artifacts/reports/clean_boundary_subset_20260920/curated_manifest.json'
 rows,events,cores=selected_rows(json.loads(manifest.read_text()),json.loads(curated.read_text()));rows=[r for r in rows if r['role']=='validation']
 provenance=dict(vendor_commit=subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip(),sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),manifest,curated,*weights.iterdir()]},settings=dict(fps=25,frames=16,stride=1,threshold=.5,segment_chunk=100,device=device,edge_pad=EDGE),deviations=['Aspect-preserving square padding before published256resize/224center crop; upstream stretches non-square input','Safe ffmpeg resampling to pipe instead of overwriting source','CPU rather than CUDA; strict state dictionary loading','Correct batched feature assignment; preserve midpoint time mapping'],limitations=['Offline noncausal transfer; no streaming latency claim','12 previously used development videos, not independent test','Source-language BSL weights with no ASL tuning'])
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 from scripts.live_reel_stage1_v17 import parser,ReelCascadeClassifier,VerifiedCommitLock
 from scripts.live_stage2_ctc_v17 import observe_stage2_frame
 from active.v17.extract_v17 import AppleVisionDetector
 args=parser().parse_args(['--no-display','--no-speech','--naturalizer','literal']);reel=ReelCascadeClassifier(args);detector=AppleVisionDetector(args.minimum_point_confidence)
 results=[];begin=time.monotonic()
 for row in rows:
  path=ROOT/row['video_path'];assert digest(path)==row['video_sha256'];t0=time.monotonic()
  cache=OUT/(hashlib.sha256(row['source_item_id'].encode()).hexdigest()[:12]+'.npz')
  if cache.exists():
   with np.load(cache) as z:features=z['features'];prob=z['boundary_probability'];nframes=int(z['nframes'])
  else:
   # Decode/resample once; all resize operations preserve aspect ratio.
   raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vf','fps=25,scale=256:256:force_original_aspect_ratio=decrease:flags=bilinear,pad=256:256:(ow-iw)/2:(oh-ih)/2,crop=224:224','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
   frames=np.frombuffer(raw,np.uint8).reshape(-1,224,224,3).copy();nframes=len(frames)
   if EDGE:frames=np.pad(frames,((8,7),(0,0),(0,0),(0,0)),mode='edge')
   if nframes<16:frames=np.concatenate([frames,np.repeat(frames[-1:],16-nframes,axis=0)])
   features=[]
   with torch.inference_mode():
    for start in range(0,len(frames)-15,4):
     starts=range(start,min(start+4,len(frames)-15))
     batch=np.stack([frames[j:j+16] for j in starts])
     x=torch.from_numpy(batch).permute(0,4,1,2,3).float()/255-.5
     features.extend(i3d(x)['embds'].reshape(len(batch),1024).numpy())
   features=np.asarray(features);prob=[]
   with torch.inference_mode():
    for start in range(0,len(features),100):
     x=torch.from_numpy(features[start:start+100].T.copy())[None]
     prob.extend(seg(x,torch.ones_like(x))[-1].softmax(1)[0,1].numpy().tolist())
   prob=np.asarray(prob);assert np.isfinite(features).all() and np.isfinite(prob).all()
   np.savez_compressed(cache,features=features,boundary_probability=prob,nframes=nframes)
  # Class1 is boundary; class0 contiguous regions are sign proposals.
  offset=0 if EDGE else 8
  regions=[((a+offset)/25,(b+offset)/25) for a,b in spans(prob>.5)]
  cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);obs=[];deadline=0.;index=0;wrists={'left':None,'right':None}
  while True:
   ok,frame=cap.read()
   if not ok:break
   seconds=index/fps;index+=1
   if seconds+1e-6<deadline:continue
   obs.append(observe_stage2_frame(frame,seconds,len(obs),detector,wrists,args));deadline=max(deadline+1/args.processing_fps,seconds)
  cap.release();candidates=[];hyp=[]
  for start,end in regions:
   chosen=[o for o in obs if start<=o.seconds<end]
   if len(chosen)<4:candidates.append(dict(start=start,end=end,skipped='fewer_than_four_observations'));continue
   p=reel.classify(chosen);v=reel.verify(chosen);agreement=p['model_score'] if p.get('candidate_gloss')==v.get('candidate_gloss') else 0.
   commit=bool(p.get('accepted') and v.get('accepted')) and VerifiedCommitLock(args.commit_hits,args.instant_commit_score).update(str(v.get('candidate_gloss')),max(v['model_score'],agreement),proposal=str(p.get('candidate_gloss')),proposal_score=p['model_score'],minimum_score=args.commit_score)
   if commit:hyp.append(v['candidate_gloss'])
   candidates.append(dict(start=start,end=end,proposal=p.get('candidate_gloss'),verifier=v.get('candidate_gloss'),proposal_score=p['model_score'],verifier_score=v['model_score'],commit=commit))
  gt=cores[row['source_item_id']];ious=[]
  for c in gt:
   ious.append(max([max(0,min(c['end'],b)-max(c['start'],a))/(max(c['end'],b)-min(c['start'],a)) for a,b in regions] or [0]))
  ref=row['target_sequence']
  result=dict(id=row['source_item_id'],reference=ref,hypothesis=hyp,edit_distance=distance(ref,hyp),candidates=candidates,core_best_iou=ious,seconds=time.monotonic()-t0,boundary_frames=int((prob>.5).sum()),feature_windows=len(prob))
  results.append(result);(OUT/'results.json').write_text(json.dumps(dict(results=results,complete=len(results)==len(rows),elapsed_seconds=time.monotonic()-begin),indent=2)+'\n')
  print(json.dumps({k:result[k] for k in ['id','reference','hypothesis','seconds','feature_windows']}),flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
