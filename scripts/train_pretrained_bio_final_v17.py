"""Recipe-pinned, BIO-preserving last-block adaptation with recognition selection."""
import argparse,json,sys,time,shutil,subprocess,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_boundary_expanded_v17 as e
from scripts import train_pretrained_boundary_v17 as prior
from scripts.train_temporal_boundary_v17 import atomic
from active.v17.approved_phrase_data_v17 import verify_manifest,digest
from active.v17.pretrained_boundary_v17 import PretrainedBoundary,Pose,load_pose,window_features,FPS,FRAMES,TARGET
OUT=ROOT/'artifacts/reports/pretrained_bio_final_v17_20260922'
CACHE=ROOT/'artifacts/cache/pretrained_bio_final_v17_20260922'
MODELS=ROOT/'artifacts/models/pretrained_bio_final_v17_20260922'
RECIPE=ROOT/'active/v17/pretrained_bio_final_recipe_20260922.json'
LOCAL=ROOT/'artifacts/reports/boundary_local_calibration_v17_20260922'
AUG=ROOT/'artifacts/cache/pretrained_boundary_augmented_v17_20260922/manifest.json'

def targets(times,intervals):
 y=np.full(len(times),-1,np.int64);coverage=np.zeros(len(times),np.int64)
 for start,end in intervals:
  if not np.isfinite([start,end]).all() or end<=start:raise ValueError('invalid interval')
  ids=np.flatnonzero((times>=start-1e-8)&(times<=end+1e-8))
  if len(ids):y[ids]=3;y[ids[0]]=2;coverage[ids]+=1
 y[coverage!=1]=-1
 return y

def eligible(s,b):
 return s['wer']<b['wer'] and s['correct']>=b['correct'] and s['insertions']<=b['insertions'] and s['retained_baseline_correct_events']>=.98*b['baseline_correct_events']

def setup_model():
 model=PretrainedBoundary().to('mps').eval();model.requires_grad_(False)
 model.backbone.encoder_attn[-1].requires_grad_(True);model.backbone.sign_bio_head.requires_grad_(True)
 return model

def prefix(model,x,times):
 for layer in model.backbone.encoder_attn[:-1]:x=layer(x,times)
 return x

def logits(model,z,times):
 return model.backbone.sign_bio_head(model.backbone.encoder_attn[-1](z,times)[:,TARGET])

def prepare():
 if RECIPE.exists():raise FileExistsError('recipe exists')
 canonical=verify_manifest();aug=json.loads(AUG.read_text());base=json.loads((prior.BASE_CACHE/'manifest.json').read_text())
 prepared=json.loads(prior.PREPARED.read_text());rows={r['item']:r for r in prepared['records']}
 local=json.loads((LOCAL/'manifest.json').read_text());reserved={r['video_sha256'] for v in local['phases'].values() for r in v}
 expanded=json.loads((e.REPORT/'tcn_comparison/evaluation_membership.json').read_text());reserved|={r['video_sha256'] for r in expanded['videos']}
 ys=[];counts={};offset=0
 for span in base['spans']:
  r=rows[span['item']]
  if span['split']!='train':continue
  if r['split']!='train' or r['video_sha256'] in reserved or span['offset']!=offset:raise ValueError('training identity/split mismatch')
  if digest(ROOT/r['video_path'])!=r['video_sha256'] or digest(ROOT/r['pose_path'])!=r['pose_sha256']:raise ValueError('training source changed')
  n=int(np.floor((r['pose_frames']-1)/r['pose_fps']*FPS+1e-7))+1
  indices=np.asarray(span['target_indices']);y=targets(np.arange(n)/FPS,r['intervals'])[indices]
  if len(indices)!=span['count']:raise ValueError('target count mismatch')
  ys.extend(y.tolist());offset+=len(y)
 for value in (-1,2,3):counts[str(value)]=ys.count(value)
 assert len(ys)==base['counts']['train'] and counts['2']>0 and counts['3']>0
 atomic(OUT/'targets.json',dict(values=ys,counts=counts))
 evaluation=[]
 for phase,items in local['phases'].items():
  for r in items:
   assert digest(ROOT/r['video_path'])==r['video_sha256'] and digest(ROOT/r['feature_path'])==r['feature_sha256']
   path=LOCAL/(phase+'_poses')/(hashlib.sha256(r['source_item_id'].encode()).hexdigest()[:20]+'.pose')
   with path.open('rb') as f:pose=Pose.read(f)
   evaluation.append(dict(item=r['source_item_id'],pose_path=str(path.relative_to(ROOT)),pose_sha256=digest(path),pose_fps=pose.body.fps,pose_frames=len(pose.body.data),video_sha256=r['video_sha256']))
 atomic(OUT/'evaluation_poses.json',dict(records=evaluation))
 files=[Path(e.__file__),LOCAL/'completion.json',OUT/'PLAN.md',prior.PREPARED,AUG,prior.BASE_CACHE/'manifest.json',LOCAL/'manifest.json',OUT/'targets.json',OUT/'evaluation_poses.json',e.REPORT/'tcn_comparison/evaluation_membership.json',Path(__file__),e.SPLIT,e.COMBINED]
 pins={**prior.pins(),**aug['files'],**{str(p.relative_to(ROOT)):digest(p) for p in files}}
 for r in evaluation:pins[r['pose_path']]=r['pose_sha256']
 recipe=dict(format='slt_pretrained_bio_final_v1',training_ready=True,canonical=canonical,hashes=pins,seed=17621,
  maximum_epochs=80,evaluate_every=4,patience_epochs=12,head_lr=2e-5,block_lr=1e-5,kl_weight=2.,weight_decay=1e-4,
  batch_size=128,targets='B=2 first observed inside; I=3 through inclusive end; overlap/unknown=-1. No invented O labels.',
  adaptation='last attention block and original BIO head; clean plus2 existing augmented variants; frozen BIO KL prior on training windows',
  selection='eligible lower whole-video calibrationWER, no correct-count loss, no extra insertions, >=98% baseline retained. Frozen epoch0 fallback.',
  confirmation='existing local30 after selection; reused development, not fresh independent test',counts=counts,
  context=dict(fps=20,frames=64,target=53,lookahead_seconds=.5),promoted=False)
 atomic(RECIPE,recipe)
 print(json.dumps(dict(status='prepared',targets=counts,recipe_sha256=digest(RECIPE))))

def verify():
 r=json.loads(RECIPE.read_text());assert r['training_ready'] and verify_manifest()['sha256']==r['canonical']['sha256']
 for path,sha in r['hashes'].items():
  if digest(ROOT/path)!=sha:raise ValueError('pinned input/code changed: '+path)
 return r

def preflight(r):
 model=setup_model();times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32,device='mps')
 x=torch.from_numpy(np.array(np.load(prior.BASE_CACHE/'train_x.npy',mmap_mode='r')[:128])).to('mps')
 with torch.no_grad():
  span=next(s for s in json.loads((prior.BASE_CACHE/'manifest.json').read_text())['spans'] if s['split']=='train' and s['count']>=4)
  row=next(r for r in json.loads(prior.PREPARED.read_text())['records'] if r['item']==span['item'])
  pose=load_pose(row);ids=np.linspace(0,span['count']-1,4,dtype=int)
  raw=torch.from_numpy(np.stack([window_features(pose,span['target_indices'][i])[0] for i in ids])).to('mps')
  original=torch.from_numpy(np.array(np.load(prior.BASE_CACHE/'train_x.npy',mmap_mode='r')[span['offset']+ids])).to('mps')
  torch.testing.assert_close(model.project(raw),original,rtol=2e-4,atol=2e-4)
  z=prefix(model,x,times);a=model.backbone.sign_bio_head(model.encode_projected(x,times));b=logits(model,z,times)
  torch.testing.assert_close(a,b);teacher=b.detach().softmax(-1)
 y=torch.tensor(json.loads((OUT/'targets.json').read_text())['values'][:128],device='mps');valid=y>=0
 tick=time.perf_counter();loss=F.cross_entropy(logits(model,z,times)[valid],y[valid]);loss.backward();torch.mps.synchronize()
 trainable=[p for p in model.parameters() if p.requires_grad]
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
 assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
 atomic(OUT/'preflight.json',dict(status='passed',recipe_sha256=digest(RECIPE),trainable_parameters=sum(p.numel() for p in trainable),prefix_parity=True,backward_seconds=time.perf_counter()-tick,optimizer_steps=0))

def run(r):
 CACHE.mkdir(parents=True);MODELS.mkdir(parents=True)
 ys=np.asarray(json.loads((OUT/'targets.json').read_text())['values']);n=len(ys)
 if shutil.disk_usage(CACHE).free<3*n*FRAMES*384*4+2*1024**3:raise OSError('insufficient cache space with2GiB reserve')
 times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32,device='mps')
 torch.manual_seed(r['seed']);np.random.seed(r['seed']);model=setup_model()
 aug=json.loads(AUG.read_text());paths=[prior.BASE_CACHE/'train_x.npy',*[ROOT/p for p in aug['augmented_files']]]
 variants=[];priors=[]
 with torch.no_grad():
  for v,path in enumerate(paths):
   source=np.load(path,mmap_mode='r');assert source.shape==(n,FRAMES,384)
   zmap=np.lib.format.open_memmap(CACHE/f'prefix_{v}.npy',mode='w+',dtype='float32',shape=source.shape)
   pmap=np.lib.format.open_memmap(CACHE/f'prior_{v}.npy',mode='w+',dtype='float32',shape=(n,4))
   for i in range(0,n,128):
    z=prefix(model,torch.from_numpy(np.array(source[i:i+128])).to('mps'),times)
    zmap[i:i+128]=z.cpu().numpy();pmap[i:i+128]=logits(model,z,times).softmax(-1).cpu().numpy()
   zmap.flush();pmap.flush();variants.append(zmap);priors.append(pmap);print('cached variant',v,flush=True)
 e.PREPARED=OUT/'evaluation_poses.json'
 local=json.loads((LOCAL/'manifest.json').read_text());cal=local['phases']['calibration']
 args=e.arguments(cal[0]['video_path'],OUT/'sessions');args.no_motion_trim=True
 reel=e.build_components(args)['classifier'];reel.args.no_motion_trim=reel.full.args.no_motion_trim=True
 assert reel.provenance()==json.loads((LOCAL/'completion.json').read_text())['reel']
 detector=e.AppleVisionDetector(args.minimum_point_confidence)
 sys.path.insert(0,str(e.UPSTREAM));from probe import upstream_helpers
 decode=upstream_helpers();cache=e.cached_inputs(cal,model,args,detector,OUT/'unused')
 def recognize(items,inputs):
  model.eval()
  _,records,_,_=e.run_arm(dict(name='bio_final',checkpoint=None,recipe=None),items,inputs,model,args,reel,times,decode,readout='bio')
  return records
 baseline=recognize(cal,cache);base=e.summarize(baseline,baseline);atomic(OUT/'baseline_calibration.json',dict(summary=base,records=baseline))
 def save(path,epoch,summary):
  atomic_state=dict(format=r['format'],recipe_sha256=digest(RECIPE),epoch=epoch,model_state_dict=model.state_dict(),calibration_summary=summary,readout='bio')
  tmp=path.with_suffix('.tmp');torch.save(atomic_state,tmp);tmp.replace(path)
 save(MODELS/'selected.pth',0,base);best=base;best_epoch=0;best_trained=float('inf');history=[]
 optimizer=torch.optim.AdamW([dict(params=model.backbone.encoder_attn[-1].parameters(),lr=r['block_lr']),dict(params=model.backbone.sign_bio_head.parameters(),lr=r['head_lr'])],weight_decay=r['weight_decay'])
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=1,factor=.5)
 scheduler.step(base['wer'])
 w=torch.ones(4,device='mps');w[2]=min(5.,float(np.sqrt((ys==3).sum()/max(1,(ys==2).sum()))))
 for epoch in range(1,r['maximum_epochs']+1):
  tick=time.perf_counter();model.eval();model.backbone.encoder_attn[-1].train();model.backbone.sign_bio_head.train();total=0.
  for ids in np.array_split(np.random.permutation(n),int(np.ceil(n/128))):
   choices=np.random.randint(3,size=len(ids));z=np.empty((len(ids),FRAMES,384),np.float32);p=np.empty((len(ids),4),np.float32)
   for v in range(3):
    mask=choices==v;z[mask]=variants[v][ids[mask]];p[mask]=priors[v][ids[mask]]
   z=torch.from_numpy(z).to('mps');teacher=torch.from_numpy(p).to('mps');y=torch.tensor(ys[ids],device='mps');valid=y>=0
   optimizer.zero_grad(set_to_none=True);out=logits(model,z,times)
   supervised=F.cross_entropy(out[valid],y[valid],weight=w) if bool(valid.any()) else out.sum()*0
   loss=supervised+r['kl_weight']*F.kl_div(out.log_softmax(-1),teacher,reduction='batchmean')
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
   optimizer.step();total+=float(loss.detach().cpu())*len(ids)
  record=dict(epoch=epoch,loss=total/n,training_seconds=time.perf_counter()-tick)
  if epoch%r['evaluate_every']==0:
   records=recognize(cal,cache);s=e.summarize(records,baseline);scheduler.step(s['wer']);record['calibration']=s
   atomic(OUT/f'calibration_epoch_{epoch}.json',dict(summary=s,records=records))
   if s['wer']<best_trained:best_trained=s['wer'];save(MODELS/'best_trained.pth',epoch,s)
   if eligible(s,base) and s['wer']<best['wer']:
    best=s;best_epoch=epoch;save(MODELS/'selected.pth',epoch,s)
  history.append(record);atomic(OUT/'history.json',history);print(json.dumps(record),flush=True)
  if epoch-best_epoch>=r['patience_epochs']:break
 selected=torch.load(MODELS/'selected.pth',map_location='cpu',weights_only=False);assert selected['recipe_sha256']==digest(RECIPE)
 atomic(OUT/'selection.json',dict(epoch=selected['epoch'],summary=best,checkpoint=str((MODELS/'selected.pth').relative_to(ROOT)),sha256=digest(MODELS/'selected.pth'),locked_before_confirmation=True))
 del cache;torch.mps.empty_cache()
 # The model's first3blocks/CNN remain frozen, so selected/frozen share confirmation inputs.
 confirmation=local['phases']['confirmation'];model=setup_model();inputs=e.cached_inputs(confirmation,model,args,detector,OUT/'unused')
 original=recognize(confirmation,inputs);model.load_state_dict(selected['model_state_dict']);chosen=recognize(confirmation,inputs) if selected['epoch'] else original
 b=e.summarize(original,original);s=e.summarize(chosen,original)
 atomic(OUT/'confirmation.json',dict(baseline=b,selected=s,records=chosen,baseline_records=original,improved=eligible(s,b) if selected['epoch'] else False))
 verify()
 lines=['# BIO-preserving final planned adaptation','',f"Trained {epoch} epochs. Selected epoch {selected['epoch']} (0 means frozen fallback). No automatic deployment.",'',
  '| Split | Frozen WER | Selected WER | Frozen correct | Selected correct | Frozen insertions | Selected insertions |','|---|---:|---:|---:|---:|---:|---:|']
 for name,a,z in [('calibration',base,best),('confirmation',b,s)]:lines.append(f"| {name} | {a['wer']:.2%} | {z['wer']:.2%} | {a['correct']}/{a['references']} | {z['correct']}/{z['references']} | {a['insertions']} | {z['insertions']} |")
 lines+=['','selected.pth is the recognition-selected result; best_trained.pth preserves the best actually fine-tuned candidate even if frozen wins.',
  'Only last attention block and original BIO head adapted. Unknown/overlapping labels ignored; no invented background. Frozen BIO KL is a regularizer, not human ground truth. Original decoder/500msfuture preserved.',
  'Existing59/30local partitions, familiar3signers/sixphrases/15glosses; reused development, not fresh independent test. Expanded72 untouched. No accuracy guarantee or mobile readiness claim.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preflight','run']);a=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if a.action=='prepare':prepare()
 elif a.action=='preflight':preflight(verify())
 else:
  started=time.perf_counter();message='BIO adaptation failed; inspect completion.json.'
  try:
   r=verify();preflight(r);run(r)
   atomic(OUT/'completion.json',dict(status='complete',elapsed_seconds=time.perf_counter()-started,promoted=False))
   message='BIO adaptation and confirmation complete; report ready.'
  except BaseException:
   atomic(OUT/'completion.json',dict(status='failed',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started));raise
  finally:
   subprocess.run([str(ROOT/'venv/bin/python'),'scripts/index_large_artifacts_v17.py'],cwd=ROOT,check=False)
   subprocess.run(['osascript','-e','display notification '+json.dumps(message)+' with title "SLT BIO adaptation"'],check=False)
