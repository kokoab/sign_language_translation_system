"""Dedicated, hash-gated pretrained boundary adaptation; cached frozen blocks."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from active.v17.approved_phrase_data_v17 import digest,verify_manifest
from active.v17.pretrained_boundary_v17 import (UPSTREAM,WEIGHTS,PretrainedBoundary,load_pose,window_features,FPS,FRAMES,TARGET,should_stop)
from active.v17.temporal_boundary_v17 import boundary_targets,masked_boundary_loss
from scripts.train_temporal_boundary_v17 import atomic,COMBINED,CURATED
REPORT=ROOT/'artifacts/reports/pretrained_boundary_finetune_v17_20260922'
RECIPE=ROOT/'active/v17/pretrained_boundary_recipe_20260922.json'
CACHE=ROOT/'artifacts/cache/pretrained_boundary_finetune_v17_20260922'
MODELS=ROOT/'artifacts/models/pretrained_boundary_finetune_v17_20260922'
PREPARED=UPSTREAM/'prepared_manifest.json'
BASE_CACHE=CACHE
AUGMENTED=False


def select_run(augmented=False):
    global REPORT,RECIPE,CACHE,MODELS,AUGMENTED
    AUGMENTED=augmented
    name='pretrained_boundary_augmented_v17_20260922' if augmented else 'pretrained_boundary_finetune_v17_20260922'
    REPORT=ROOT/'artifacts/reports'/name;CACHE=ROOT/'artifacts/cache'/name;MODELS=ROOT/'artifacts/models'/name
    RECIPE=ROOT/'active/v17'/('pretrained_boundary_augmented_recipe_20260922_v2.json' if augmented else 'pretrained_boundary_recipe_20260922.json')


def converged(epoch,best_epoch,recipe):
    return epoch>=recipe['maximum_epochs'] or (epoch>=recipe['minimum_epochs'] and
        epoch-max(best_epoch,recipe['warmup_epochs'])>=recipe['patience'])



def pins():
    files=[ROOT/'active/v17/pretrained_boundary_v17.py',ROOT/'active/v17/temporal_boundary_v17.py',Path(__file__),
           ROOT/'scripts/evaluate_pretrained_boundary_v17.py',ROOT/'scripts/live_boundary_v17.py',
           ROOT/'scripts/live_reel_stage1_v17.py',ROOT/'scripts/live_isolated_v17.py',
           ROOT/'scripts/evaluate_temporal_boundary_v17.py',ROOT/'scripts/train_temporal_boundary_v17.py',UPSTREAM/'check.py',UPSTREAM/'probe.py',UPSTREAM/'prepare.py',
           ROOT/'scripts/live_stage2_ctc_v17.py',ROOT/'scripts/app_shell_v17.py',
           *sorted((UPSTREAM/'source').rglob('*.py')),
           *[p for p in (UPSTREAM/'dependencies').rglob('*') if p.is_file() and '__pycache__' not in p.parts],
           *[WEIGHTS/n for n in ('model.safetensors','config.json')]]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def contract():
    if RECIPE.exists():raise FileExistsError('Do not overwrite a training contract')
    verified=verify_manifest()
    validation=json.loads((UPSTREAM/'validation.json').read_text())
    prepared=json.loads(PREPARED.read_text())
    if validation['status']!='passed' or prepared['status']!='complete' or len(prepared['records'])!=1121:
        raise ValueError('complete validated inputs required')
    recipe=dict(format='slt_pretrained_boundary_v1',training_ready=True,entrypoint=str(Path(__file__).relative_to(ROOT)),
                canonical=verified,inputs={str(PREPARED.relative_to(ROOT)):digest(PREPARED),
                prepared['source_manifest']:prepared['source_sha256'],
                str((UPSTREAM/'validation.json').relative_to(ROOT)):digest(UPSTREAM/'validation.json')},
                code_sha256=pins(),seeds=[17621,17622],maximum_epochs=120,minimum_epochs=40,patience=20,
                minimum_improvement=.0001,warmup_epochs=5,batch_size=128,cache_batch_size=16,
                head_lr=.001,attention_lr=.00005,weight_decay=.0001,gradient_clip=1.,
                context=dict(fps=FPS,frames=FRAMES,target=TARGET,lookahead_seconds=.5,
                             normalization='bounded window only; zero-confidence prefix/explicitEOF padding',
                             eof='score available evidence; never force END'),
                supervision='start/end positives,negatives only inside approved intervals; unknown masked',
                adaptation='new2-outputhead then all4attentionblocks; pretrained CNN frozen',
                selection='minimum fixed-weight BCE on parent-held training calibration; never validation',
                promotion=False,test_accessed=False)
    if AUGMENTED:
        base=json.loads((BASE_CACHE/'manifest.json').read_text())
        if base['status']!='complete':raise ValueError('original cache incomplete')
        recipe.update(format='slt_pretrained_boundary_augmented_v2',seeds=[17621],maximum_epochs=80,
                      minimum_epochs=0,patience=8,scheduler_patience=3,
                      augmentation=dict(variants=2,seed=176210,sensor_fps=[15,20],observation_dropout=[0,.15],
                        kind='past-only observation hold before normalization; source clock/targets unchanged'),
                      base_cache=str(BASE_CACHE.relative_to(ROOT)))
        recipe['inputs'][str((BASE_CACHE/'manifest.json').relative_to(ROOT))]=digest(BASE_CACHE/'manifest.json')
        recipe['inputs'].update(base['files'])
        source_inputs=json.loads((ROOT/prepared['source_manifest']).read_text())['inputs']
        for path in (COMBINED,CURATED,UPSTREAM.parent/'asl_temporal_boundary_v17_20260922/baseline_oracle.json'):
            name=str(path.relative_to(ROOT));sha=digest(path)
            if path in (COMBINED,CURATED) and source_inputs.get(name)!=sha:raise ValueError('evaluation/training source mismatch: '+name)
            recipe['inputs'][name]=sha
    atomic(RECIPE,recipe)
    return recipe


def verified_recipe(recipe_path=None):
    recipe=json.loads((recipe_path or RECIPE).read_text())
    if not recipe['training_ready'] or recipe['entrypoint']!=str(Path(__file__).relative_to(ROOT)):
        raise ValueError('wrong/unready dedicated contract')
    if recipe['code_sha256']!=pins():raise ValueError('pinned code/weights changed')
    for name,sha in recipe['inputs'].items():
        if digest(ROOT/name)!=sha:raise ValueError('input contract changed: '+name)
    if verify_manifest()['sha256']!=recipe['canonical']['sha256']:raise ValueError('canonical contract changed')
    return recipe


def preflight(recipe):
    row=next(r for r in json.loads(PREPARED.read_text())['records'] if r['split']=='train')
    if digest(ROOT/row['pose_path'])!=row['pose_sha256']:raise ValueError('pose changed')
    pose=load_pose(row)
    x=np.stack([window_features(pose,i)[0] for i in range(min(4,len(pose.body.data)))])
    times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32,device='mps')
    model=PretrainedBoundary().to('mps');model.train();model.set_adaptation(True)
    features=torch.from_numpy(x).to('mps')
    with torch.no_grad():projected=model.project(features)
    loss=model.forward_projected(projected,times).square().mean();loss.backward()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in trainable):raise ValueError('invalid gradients')
    if any(p.grad is not None for n,p in model.named_parameters() if not p.requires_grad):raise ValueError('frozen parameter got gradients')
    result=dict(status='passed',loss=float(loss.detach().cpu()),optimizer_steps=0,
                trainable_parameters=sum(p.numel() for n,p in trainable),recipe_sha256=digest(RECIPE))
    atomic(REPORT/'preflight.json',result);print(json.dumps(result),flush=True)



def cached_batch(x,y,ids,variants=()):
    xx=np.array(x[ids])
    if len(variants)>1:
        choices=np.random.randint(len(variants),size=len(ids))
        for v in range(1,len(variants)):
            mask=choices==v
            if mask.any():xx[mask]=variants[v][ids[mask]]
    return xx,np.array(y[ids])


def benchmark(recipe):
    """Disposable real-cache training steps; no checkpoint or dataset writes."""
    base=ROOT/recipe.get('base_cache',str(BASE_CACHE.relative_to(ROOT)))
    meta=json.loads((base/'manifest.json').read_text())
    rows={r['item']:r for r in json.loads(PREPARED.read_text())['records']}
    model=PretrainedBoundary().to('mps').eval()
    times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32,device='mps')
    parity=[];projection_seconds=[]
    with torch.inference_mode():
        for split in ('train','calibration','validation'):
            span=next(s for s in meta['spans'] if s['split']==split and s['count']>=4)
            pose=load_pose(rows[span['item']]);positions=np.linspace(0,span['count']-1,4,dtype=int)
            ids=np.array(span['target_indices'])[positions]
            x=np.stack([window_features(pose,j)[0] for j in ids])
            projected=model.project(torch.from_numpy(x).to('mps')).cpu().numpy()
            cached=np.load(base/f'{split}_x.npy',mmap_mode='r')[span['offset']+positions]
            np.testing.assert_allclose(projected,cached,rtol=2e-4,atol=2e-4)
            parity.append(dict(split=split,max_abs=float(np.abs(projected-cached).max())))
        pose=load_pose(rows[next(s['item'] for s in meta['spans'] if s['split']=='train' and s['count']>16)])
        for repeat in range(8):
            torch.mps.synchronize();tick=time.perf_counter()
            x=np.stack([window_features(pose,j,augmentation_seed=1000+repeat*16+j)[0] for j in range(16)])
            xt=torch.from_numpy(x).to('mps');z=model.project(xt)
            torch.testing.assert_close(model(xt,times),model.forward_projected(z,times))
            z.cpu();torch.mps.synchronize()
            if repeat>=2:projection_seconds.append(time.perf_counter()-tick)
    model.train();model.set_adaptation(True)
    optimizer=torch.optim.AdamW([{'params':model.edge.parameters(),'lr':recipe['head_lr']},
            {'params':model.backbone.encoder_attn.parameters(),'lr':recipe['attention_lr']}],weight_decay=recipe['weight_decay'])
    x=np.load(base/'train_x.npy',mmap_mode='r');y=np.load(base/'train_y.npy',mmap_mode='r')
    weights=torch.tensor(np.clip((y==0).sum(0)/np.maximum((y==1).sum(0),1),1,20),dtype=torch.float32,device='mps')
    rng=np.random.default_rng(17621);durations=[]
    for i in range(32):
        torch.mps.synchronize();tick=time.perf_counter();ids=rng.choice(len(x),recipe['batch_size'],replace=False)
        bx,by=cached_batch(x,y,ids,[x,x,x]);xx=torch.from_numpy(bx).to('mps');yy=torch.from_numpy(by).to('mps')
        optimizer.zero_grad(set_to_none=True);loss=masked_boundary_loss(model.forward_projected(xx,times),yy,weights)
        if not torch.isfinite(loss):raise ValueError('benchmark loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],recipe['gradient_clip'])
        if not torch.isfinite(norm):raise ValueError('benchmark gradient')
        optimizer.step();int((yy>=0).sum());float(loss.detach().cpu());torch.mps.synchronize()
        if i>=4:durations.append(time.perf_counter()-tick)
    cx=np.load(base/'calibration_x.npy',mmap_mode='r');cy=np.load(base/'calibration_y.npy',mmap_mode='r');model.eval()
    torch.mps.synchronize();tick=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(cx),recipe['batch_size']):
            xx=torch.from_numpy(np.array(cx[start:start+recipe['batch_size']])).to('mps')
            yy=torch.from_numpy(np.array(cy[start:start+recipe['batch_size']])).to('mps')
            loss=masked_boundary_loss(model.forward_projected(xx,times),yy,weights)
            int((yy>=0).sum());float(loss.cpu())
    torch.mps.synchronize();cal_seconds=time.perf_counter()-tick
    result=dict(recipe_sha256=digest(RECIPE),status='passed',clean_cache_parity=parity,
                augmented_direct_cache_parity=True,step_median_seconds=float(np.median(durations)),
                step_p95_seconds=float(np.percentile(durations,95)),calibration_seconds=cal_seconds,
                estimated_epoch_seconds=float(np.median(durations))*int(np.ceil(len(x)/recipe['batch_size']))+cal_seconds,
                augmentation_batch_seconds=float(np.median(projection_seconds)),
                limitation='Real original float32 cache; excludes extra variant paging, checkpoint I/O and sustained thermal slowdown. Projection timing includes parity forwards so is conservative. Disposable weights discarded.')
    atomic(REPORT/'benchmark.json',result);print(json.dumps(result),flush=True)


def build_augmented_cache(recipe):
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'manifest.json').exists():raise FileExistsError('augmented cache already exists')
    base_dir=ROOT/recipe['base_cache'];base=json.loads((base_dir/'manifest.json').read_text())
    variants=recipe['augmentation']['variants'];n=base['counts']['train']
    required=variants*n*FRAMES*384*4+2*1024**3
    if shutil.disk_usage(CACHE).free<required:raise OSError('insufficient space for float32 variants plus 2GiB reserve')
    rows={r['item']:r for r in json.loads(PREPARED.read_text())['records']}
    arrays=[np.lib.format.open_memmap(CACHE/f'train_x_aug{v+1}.npy',mode='w+',dtype='float32',shape=(n,FRAMES,384)) for v in range(variants)]
    labels={split:np.load(base_dir/f'{split}_y.npy',mmap_mode='r') for split in base['counts']}
    model=PretrainedBoundary().to('mps').eval();started=time.perf_counter();covered=Counter()
    with torch.inference_mode():
        for number,span in enumerate(base['spans']):
            row=rows[span['item']];split=row['split'];offset=span['offset'];ids=np.array(span['target_indices'])
            if split!=span['split'] or row['parent']!=span['parent'] or offset!=covered[split]:raise ValueError('source/split/offset mismatch')
            if digest(ROOT/row['video_path'])!=row['video_sha256'] or digest(ROOT/row['pose_path'])!=row['pose_sha256']:raise ValueError('source changed')
            length=int(np.floor((row['pose_frames']-1)/row['pose_fps']*FPS+1e-7))+1
            y=boundary_targets(np.arange(length)/FPS,row['intervals'],[],False)
            if not np.array_equal(ids,np.flatnonzero((y>=0).any(1))) or not np.array_equal(y[ids],labels[split][offset:offset+len(ids)]):raise ValueError('cached targets/clock mismatch')
            covered[split]+=len(ids)
            if split!='train':continue
            pose=load_pose(row)
            for start in range(0,len(ids),recipe['cache_batch_size']):
                batch=ids[start:start+recipe['cache_batch_size']]
                for v,xx in enumerate(arrays):
                    x=np.stack([window_features(pose,int(j),augmentation_seed=recipe['augmentation']['seed']+v*n+offset+start+k)[0] for k,j in enumerate(batch)])
                    z=model.project(torch.from_numpy(x).to('mps')).cpu().numpy()
                    if not np.isfinite(z).all():raise ValueError('nonfinite augmented projection')
                    xx[offset+start:offset+start+len(batch)]=z
            if (number+1)%100==0:print('augmented cache',number+1,len(base['spans']),flush=True)
    if dict(covered)!=base['counts']:raise ValueError('cache coverage mismatch')
    for x in arrays:x.flush()
    paths=[CACHE/f'train_x_aug{v+1}.npy' for v in range(variants)]
    value=dict(base,status='complete',recipe_sha256=digest(RECIPE),elapsed_seconds=time.perf_counter()-started,
               base_cache=recipe['base_cache'],augmentation=recipe['augmentation'],
               augmented_files=[str(p.relative_to(ROOT)) for p in paths],files={**base['files'],**{str(p.relative_to(ROOT)):digest(p) for p in paths}})
    atomic(CACHE/'manifest.json',value);atomic(REPORT/'cache_summary.json',{k:v for k,v in value.items() if k!='spans'})
    return value


def build_cache(recipe):
    if recipe.get('augmentation'):return build_augmented_cache(recipe)
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'manifest.json').exists():raise FileExistsError('cache exists; inspect before repeating')
    rows=json.loads(PREPARED.read_text())['records'];counts=Counter();plans=[]
    for row in rows:
        if digest(ROOT/row['video_path'])!=row['video_sha256'] or digest(ROOT/row['pose_path'])!=row['pose_sha256']:
            raise ValueError('video/pose changed: '+row['item'])
        n=int(np.floor((row['pose_frames']-1)/row['pose_fps']*FPS+1e-7))+1
        y=boundary_targets(np.arange(n)/FPS,row['intervals'],[],False)
        indices=np.flatnonzero((y>=0).any(1));split=row['split']
        plans.append((row,indices,y[indices],counts[split]));counts[split]+=len(indices)
    arrays={}
    for split,n in counts.items():
        arrays[split]=(np.lib.format.open_memmap(CACHE/f'{split}_x.npy',mode='w+',dtype='float32',shape=(n,FRAMES,384)),
                       np.lib.format.open_memmap(CACHE/f'{split}_y.npy',mode='w+',dtype='float32',shape=(n,2)))
    model=PretrainedBoundary().to('mps').eval();started=time.perf_counter()
    with torch.inference_mode():
        for i,(row,indices,y,offset) in enumerate(plans):
            pose=load_pose(row);xx,yy=arrays[row['split']];yy[offset:offset+len(y)]=y
            for start in range(0,len(indices),recipe['cache_batch_size']):
                ids=indices[start:start+recipe['cache_batch_size']]
                x=np.stack([window_features(pose,int(j))[0] for j in ids])
                z=model.project(torch.from_numpy(x).to('mps')).cpu().numpy()
                xx[offset+start:offset+start+len(ids)]=z
            if (i+1)%100==0:print('cache',i+1,len(plans),flush=True)
    for x,y in arrays.values():x.flush();y.flush()
    paths=list(CACHE.glob('*.npy'))
    value=dict(status='complete',recipe_sha256=digest(RECIPE),counts=dict(counts),elapsed_seconds=time.perf_counter()-started,
               files={str(p.relative_to(ROOT)):digest(p) for p in paths},
               spans=[dict(item=r['item'],split=r['split'],parent=r['parent'],offset=o,count=len(idx),target_indices=idx.tolist()) for r,idx,y,o in plans])
    atomic(CACHE/'manifest.json',value);atomic(REPORT/'cache_summary.json',{k:v for k,v in value.items() if k!='spans'})
    return value


def train(recipe):
    meta=json.loads((CACHE/'manifest.json').read_text())
    if meta['status']!='complete' or meta['recipe_sha256']!=digest(RECIPE):raise ValueError('wrong cache contract')
    for name,sha in meta['files'].items():
        if digest(ROOT/name)!=sha:raise ValueError('cache changed')
    base_dir=ROOT/meta['base_cache'] if 'base_cache' in meta else CACHE
    data={s:(np.load(base_dir/f'{s}_x.npy',mmap_mode='r'),np.load(base_dir/f'{s}_y.npy',mmap_mode='r')) for s in meta['counts']}
    variants=[data['train'][0],*[np.load(ROOT/p,mmap_mode='r') for p in meta.get('augmented_files',[])]]
    train_y=data['train'][1]
    weights=np.clip((train_y==0).sum(0)/np.maximum((train_y==1).sum(0),1),1,20).astype('float32')
    positive_weight=torch.from_numpy(weights).to('mps')
    times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32,device='mps')
    MODELS.mkdir(parents=True,exist_ok=True);runs=[]
    def batches(split,order,augment=False):
        x,y=data[split]
        for start in range(0,len(order),recipe['batch_size']):
            ids=order[start:start+recipe['batch_size']]
            xx,yy=cached_batch(x,y,ids,variants if augment else ())
            yield torch.from_numpy(xx).to('mps'),torch.from_numpy(yy).to('mps')
    def evaluate(model,split):
        model.eval();total=0.;valid=0
        with torch.inference_mode():
            for x,y in batches(split,np.arange(len(data[split][0]))):
                loss=masked_boundary_loss(model.forward_projected(x,times),y,positive_weight)
                n=int((y>=0).sum());total+=float(loss.cpu())*n;valid+=n
        return total/valid
    for seed in recipe['seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=PretrainedBoundary().to('mps');model.set_adaptation(False)
        optimizer=torch.optim.AdamW([{'params':model.edge.parameters(),'lr':recipe['head_lr']},
                  {'params':model.backbone.encoder_attn.parameters(),'lr':recipe['attention_lr']}],weight_decay=recipe['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=recipe.get('scheduler_patience',8),min_lr=1e-6)
        best=float('inf');best_epoch=0;patience_epoch=0;meaningful=float('inf');history=[];started=time.perf_counter()
        for epoch in range(1,recipe['maximum_epochs']+1):
            tick=time.perf_counter();model.train();model.set_adaptation(epoch>recipe['warmup_epochs'])
            total=0.;valid=0
            for x,y in batches('train',np.random.permutation(len(data['train'][0])),augment=True):
                optimizer.zero_grad(set_to_none=True)
                logits=model.forward_projected(x,times);loss=masked_boundary_loss(logits,y,positive_weight)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],recipe['gradient_clip'])
                if not torch.isfinite(norm):raise ValueError('nonfinite training gradient')
                optimizer.step();n=int((y>=0).sum());total+=float(loss.detach().cpu())*n;valid+=n
            calibration=evaluate(model,'calibration')
            if not np.isfinite(calibration):raise ValueError('nonfinite calibration loss')
            scheduler.step(calibration)
            if calibration<best:
                best=calibration;best_epoch=epoch
                path=MODELS/f'seed_{seed}.pth';tmp=path.with_suffix('.tmp')
                torch.save(dict(format=recipe['format'],seed=seed,epoch=epoch,recipe_sha256=digest(RECIPE),
                                model_state_dict=model.state_dict(),calibration_loss=best),tmp);tmp.replace(path)
            if calibration<meaningful-recipe['minimum_improvement']:
                meaningful=calibration;patience_epoch=epoch
            record=dict(epoch=epoch,train_loss=total/valid,calibration_loss=calibration,best_epoch=best_epoch,
                        seconds=time.perf_counter()-tick,adapt_attention=epoch>recipe['warmup_epochs'],lr=[g['lr'] for g in optimizer.param_groups])
            history.append(record);atomic(REPORT/f'history_{seed}.json',dict(seed=seed,history=history))
            print(json.dumps(dict(seed=seed,**record)),flush=True)
            if converged(epoch,patience_epoch,recipe):break
        saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict'],strict=True)
        result=dict(seed=seed,epochs=len(history),selected_epoch=best_epoch,calibration_loss=best,
                    validation_loss=evaluate(model,'validation'),elapsed_seconds=time.perf_counter()-started,
                    checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path),
                    stop_reason='maximum_epochs' if len(history)==recipe['maximum_epochs'] else 'calibration_plateau')
        runs.append(result);atomic(REPORT/'training_results.json',dict(runs=runs,promoted=False))
    return runs


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['contract','preflight','benchmark','run']);parser.add_argument('--augmented',action='store_true');args=parser.parse_args()
    select_run(args.augmented)
    REPORT.mkdir(parents=True,exist_ok=True)
    if args.action=='contract':contract()
    elif args.action=='preflight':preflight(verified_recipe())
    elif args.action=='benchmark':benchmark(verified_recipe())
    else:
        started=time.perf_counter();message='Pretrained ASL boundary run failed; inspect completion.json.'
        try:
            recipe=verified_recipe();preflight(recipe);build_cache(recipe);train(recipe)
            verified_recipe()
            from scripts.evaluate_pretrained_boundary_v17 import evaluate
            evaluate(report=REPORT,recipe_path=RECIPE);verified_recipe()
            atomic(REPORT/'completion.json',dict(status='complete',elapsed_seconds=time.perf_counter()-started,promoted=False))
            subprocess.run([str(ROOT/'venv/bin/python'),'scripts/index_large_artifacts_v17.py'],cwd=ROOT,check=True)
            message='Pretrained ASL boundary training and paired evaluation complete; results ready.'
        except BaseException:
            atomic(REPORT/'completion.json',dict(status='failed',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started));raise
        finally:subprocess.run(['osascript','-e','display notification '+json.dumps(message)+' with title "SLT pretrained adaptation"'],check=False)
