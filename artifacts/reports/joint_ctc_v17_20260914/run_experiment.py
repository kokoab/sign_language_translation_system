"""Bounded, provenance-checked frozen-versus-joint CTC comparison."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from active.v17.joint_ctc_v17 import JointCTC, chunks, sequence_targets, ctc_loss, decode, core_crop
from active.v17 import train_stage1_window_v17 as old
from active.v17.stage1_window_v17 import normalize_time_window, window_sample_times
from active.v17.model_reel_emission_v17 import pool_stage1_encoded
from active.v17.train_stage_1_phrase_adapt_v17 import load_features
from active.v17.train_stage_2_other_ctc_v17 import _edit_operations
from scripts.evaluate_stage1_window_v17 import failed_gates, word_latency

REPORT = Path(__file__).resolve().parent
PRIOR = ROOT/'artifacts/reports/o5s5_augmented_v17_20260914'
MANIFEST = PRIOR/'combined_supervision.json'
BASE = ROOT/'artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth'
CACHE = ROOT/'artifacts/generated/joint_ctc_v17_20260914/data.pt'
MODELS = ROOT/'artifacts/models/joint_ctc_v17_20260914'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def save(name, value):
    path = REPORT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def emit(value):
    print(json.dumps(value), flush=True)


def tensor(value):
    return torch.as_tensor(value, device=DEVICE)


def fresh():
    torch.manual_seed(17111)
    payload = torch.load(BASE, map_location='cpu', weights_only=False)
    return JointCTC(old._load_start(payload).base).to(DEVICE)


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prepare():
    if CACHE.exists():
        raise FileExistsError('preparation cache already exists; use preflight/train')
    old.verify_development_freeze(PRIOR/'development_freeze.json', BASE, MANIFEST)
    frozen = json.loads((PRIOR/'development_freeze.json').read_text())
    payload = torch.load(BASE, map_location='cpu', weights_only=False)
    labels = {k:int(v) for k,v in payload['label_to_index'].items() if int(v)<100}
    manifest = json.loads(MANIFEST.read_text())
    signers = {role:{r['signer_id'] for r in manifest['rows'] if r['role']==role}
               for role in ('train','validation')}
    assert not signers['train'] & signers['validation']
    data = {'labels':labels, 'sequences':[], 'cores':[], 'replay':{}, 'background':{}, 'evaluation':[]}
    audit = {'sequence_rejections':[], 'core_rejections':[], 'input_hashes':{},
             'manifest_sha256':old._sha256(MANIFEST), 'base_sha256':old._sha256(BASE),
             'train_validation_signer_disjoint':True, 'protected_test_accessed':False}
    for n,row in enumerate(manifest['rows']):
        path = ROOT/row['archive_path']
        raw,times = old._load_sequence(path)
        audit['input_hashes'][str(path.relative_to(ROOT))] = old._sha256(path)
        if row['all_signs_annotated']:
            value = chunks(raw,times)
            targets = sequence_targets(row,labels)
            required = len(targets)+sum(a==b for a,b in zip(targets,targets[1:]))
            if required>value.length:
                raise ValueError('infeasible sequence '+row['source_item_id'])
            data['sequences'].append(dict(value=value,targets=targets,role=row['role'],
                source=row['source'],identity=row['source_item_id']))
        elif row['source']=='o5s5':
            for i,event in enumerate(row['intervals']):
                if event['label'] not in labels:
                    continue
                identity=f"{row['source_item_id']}:{i}:{event['start_seconds']}:{event['end_seconds']}"
                try:
                    core,count=core_crop(raw,times,event)
                except ValueError as error:
                    audit['core_rejections'].append(dict(identity=identity,reason=str(error),role=row['role']))
                    continue
                center=(event['start_seconds']+event['end_seconds'])/2
                context_end=min(float(times[-1]),center+.535)
                context,_=normalize_time_window(raw,times,context_end,1.07)
                retained=times[(times>=context_end-1.07-1e-9)&(times<=context_end+1e-9)]
                sampled=window_sample_times(retained,context_end,1.07)
                mask=(sampled>=event['start_seconds'])&(sampled<=event['end_seconds'])
                data['cores'].append(dict(features=core,context=context,mask=mask,
                    target=labels[event['label']],role=row['role'],identity=identity,
                    signer=row['signer_id'],source_frames=count))
        if (n+1)%200==0:
            emit({'prepared_manifest_rows':n+1})
    for role in ('train','validation'):
        rows=frozen['isolated_replay'][role]
        for row in rows:
            path=ROOT/row['path']
            if old._protected(path) or old._sha256(path)!=row['sha256']:
                raise ValueError('protected or changed replay '+str(path))
            audit['input_hashes'][row['path']]=row['sha256']
        data['replay'][role]=dict(features=np.stack([load_features(ROOT/r['path']) for r in rows]),
            targets=np.array([r['target'] for r in rows]),sources=[r['source'] for r in rows],
            identities=[r['path'] for r in rows])
        context,_=old.load_context_samples(MANIFEST,labels,role)
        bg=[r for r in context if r.category=='background']
        identities=[r.identity+':'+hashlib.sha256(r.features.tobytes()).hexdigest() for r in bg]
        assert len(set(identities))==len(bg)
        data['background'][role]=dict(features=np.stack([r.features for r in bg]),identities=identities)
    evaluation=json.loads((PRIOR/'evaluation_manifest.json').read_text())
    for row in evaluation['rows']:
        path=ROOT/row['archive_path']
        if row['role']!='validation' or old._protected(path) or old._sha256(path)!=row['archive_sha256']:
            raise ValueError('evaluation provenance mismatch')
        raw,times=old._load_sequence(path)
        data['evaluation'].append(dict(row=row,value=chunks(raw,times)))
        audit['input_hashes'][str(path.relative_to(ROOT))]=row['archive_sha256']
    base=fresh().base.eval()
    with torch.inference_mode():
        x=data['replay']['train']['features']
        data['teacher_logits']=torch.cat([base(tensor(v)).cpu() for v in np.array_split(x, max(1,int(np.ceil(len(x)/64))))])
    audit['counts']={
        'sequences':dict(Counter(r['role']+'/'+r['source'] for r in data['sequences'])),
        'cores':dict(Counter(r['role'] for r in data['cores'])),
        'core_rejections':len(audit['core_rejections']),
        'static_single_observation_cores':sum(r['source_frames']==1 for r in data['cores']),
        'replay':{r:len(v['targets']) for r,v in data['replay'].items()},
        'background':{r:len(v['features']) for r,v in data['background'].items()},
        'evaluation_recordings':len(data['evaluation']),
        'sequence_chunks':sum(len(r['value'].features) for r in data['sequences']),
        'max_chunk_duration_seconds':max(float(t[-1]-t[0]) for r in data['sequences'] for t in r['value'].times),
        'max_sequence_tokens':max(r['value'].length for r in data['sequences'])}
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,CACHE)
    audit['cache_sha256']=old._sha256(CACHE)
    audit['code_hashes']={str(p.relative_to(ROOT)):old._sha256(p) for p in
        [Path(__file__),ROOT/'active/v17/joint_ctc_v17.py',ROOT/'test/test_joint_ctc_v17.py',REPORT/'PLAN.md']}
    save('data_audit.json',audit)
    emit(audit['counts'])


def load_data():
    audit=json.loads((REPORT/'data_audit.json').read_text())
    if old._sha256(CACHE)!=audit['cache_sha256'] or old._sha256(BASE)!=audit['base_sha256']:
        raise ValueError('prepared cache or initialization changed')
    return torch.load(CACHE,map_location='cpu',weights_only=False)


@torch.inference_mode()
def isolated(model, data):
    replay=data['replay']['validation']
    model.eval()
    logits=torch.cat([model.base(tensor(x)).cpu() for x in np.array_split(replay['features'],22)])
    predictions=logits.argmax(-1).numpy()
    return {source:float(np.mean(predictions[np.array(replay['sources'])==source]
        ==replay['targets'][np.array(replay['sources'])==source])) for source in ('citizen','semlex')}


@torch.inference_mode()
def core_metrics(model,data,contextual=False):
    model.eval(); output={}
    for role in ('train','validation'):
        rows=[r for r in data['cores'] if r['role']==role]
        predictions=[]
        for begin in range(0,len(rows),64):
            batch=rows[begin:begin+64]
            x=tensor(np.stack([r['context' if contextual else 'features'] for r in batch]))
            if contextual:
                enc,active=model.base.encode(x)
                active=active & tensor(np.stack([r['mask'] for r in batch]))
                logits=model.base.classifier(pool_stage1_encoded(model.base,enc,active))
            else:
                logits=model.base(x)
            predictions.extend(logits.argmax(-1).cpu().tolist())
        correct=Counter(r['target'] for r,p in zip(rows,predictions) if p==r['target'])
        totals=Counter(r['target'] for r in rows)
        output[role]=dict(total=len(rows),correct=sum(correct.values()),
            accuracy=sum(correct.values())/len(rows),macro_accuracy=float(np.mean([correct[k]/v for k,v in totals.items()])),
            rows=[dict(identity=r['identity'],signer=r['signer'],target=r['target'],prediction=p) for r,p in zip(rows,predictions)])
    return output


def preflight(data):
    model=fresh();model.eval()
    available=[r for r in data['sequences'] if r['role']=='train' and 1<=len(r['targets'])<=3
               and len(r['value'].features)<=5 and any(t<101 for t in r['targets'])]
    small=[];seen=set()
    for row in sorted(available,key=lambda r:r['value'].length):
        if row['targets'] not in seen:
            small.append(row);seen.add(row['targets'])
        if len(small)==4:break
    assert len(small)==4
    logits,lengths=model.sequences([r['value'] for r in small])
    loss=ctc_loss(logits,[r['targets'] for r in small],lengths);loss.backward()
    gradient=sum(float(p.grad.detach().abs().sum().cpu()) for p in model.base.parameters() if p.grad is not None)
    assert gradient>0
    result={'gradient_l1':gradient,'initial_ctc_loss':float(loss.detach().cpu()),
        'train_only_ids':[r['identity'] for r in small], 'raw_core':core_metrics(model,data),
        'contextual_core':core_metrics(model,data,True),'isolated_baseline':isolated(model,data),
        'protected_test_accessed':False}
    model.zero_grad(set_to_none=True);model.base.requires_grad_(False)
    logits,lengths=model.sequences([r['value'] for r in small])
    ctc_loss(logits,[r['targets'] for r in small],lengths).backward()
    result['frozen_encoder_has_no_gradient']=all(p.grad is None for p in model.base.parameters())
    assert result['frozen_encoder_has_no_gradient']
    model=fresh();model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[]
    for step in range(1,201):
        optimizer.zero_grad(set_to_none=True)
        logits,lengths=model.sequences([r['value'] for r in small])
        loss=ctc_loss(logits,[r['targets'] for r in small],lengths)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%10==0:
            with torch.inference_mode():
                logits,lengths=model.sequences([r['value'] for r in small])
                paths=logits.argmax(-1).cpu().tolist()
                pred=[tuple(k for k,_ in itertools.groupby(p[:n]) if k) for p,n in zip(paths,lengths)]
            exact=sum(p==r['targets'] for p,r in zip(pred,small))
            record=dict(step=step,loss=float(loss.detach().cpu()),exact=exact,total=4)
            history.append(record);emit({'tiny_fit':record})
            if exact==4 and record['loss']<.5:break
    result['tiny_fit']=history;result['passed']=history[-1]['exact']==4 and history[-1]['loss']<.5
    # Compare the real initialized model on identical tensors, not only toy layers.
    model=fresh().eval();x=torch.from_numpy(data['replay']['validation']['features'][:8])
    with torch.inference_mode():
        gpu=model.tokens(x.to(DEVICE)).cpu();model.cpu();cpu=model.tokens(x)
    result['cpu_mps']={'max_abs_logit_error':float((gpu-cpu).abs().max()),
        'same_token_argmax':int((gpu.argmax(-1)==cpu.argmax(-1)).sum()),'tokens':256}
    save('preflight.json',result)
    if not result['passed']:raise RuntimeError('tiny-fit preflight failed; no full training launched')
    emit({'preflight_passed':True,'gradient_l1':gradient,'tiny_fit':history[-1]})


def schedule(data,epoch):
    groups=defaultdict(list)
    for i,row in enumerate(data['sequences']):
        if row['role']=='train':groups['sequence/'+row['source']].append(i)
    groups['core']=[i for i,r in enumerate(data['cores']) if r['role']=='train']
    groups['replay']=list(range(len(data['replay']['train']['targets'])))
    groups['background']=list(range(len(data['background']['train']['features'])))
    limits={'core':16,'replay':32,'background':16}
    steps=max(int(np.ceil(len(v)/limits.get(k,8))) for k,v in groups.items())
    rng=np.random.default_rng(17111+epoch)
    batches={k:[x.tolist() for x in np.array_split(rng.permutation(v),steps)] for k,v in sorted(groups.items())}
    for k,v in groups.items():assert sorted(sum(batches[k],[]))==sorted(v)
    return batches,steps


def train_epoch(model,data,optimizer,epoch):
    batches,steps=schedule(data,epoch)
    seq_keys=[k for k in batches if k.startswith('sequence/')]
    counts=Counter(); sums=Counter();started=time.perf_counter()
    torch.manual_seed(17111+epoch)
    model.train();model.base.eval() # identical encoder dropout contract in both arms
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);losses={}
        for key in seq_keys:
            selected=[data['sequences'][i] for i in batches[key][step]]
            if not selected:continue
            logits,lengths=model.sequences([r['value'] for r in selected])
            losses[key]=ctc_loss(logits,[r['targets'] for r in selected],lengths)
        indices=batches['replay'][step]
        if indices:
            r=data['replay']['train'];logits=model.base(tensor(r['features'][indices]))
            ce=F.cross_entropy(logits,tensor(r['targets'][indices]))
            kl=F.kl_div(F.log_softmax(logits/2,-1),F.softmax(data['teacher_logits'][indices].to(DEVICE)/2,-1),reduction='batchmean')*4
            losses['replay']=ce+kl
        indices=batches['core'][step]
        if indices:
            rows=[data['cores'][i] for i in indices]
            losses['core']=F.cross_entropy(model.base(tensor(np.stack([r['features'] for r in rows]))),
                                          tensor(np.array([r['target'] for r in rows])))
        indices=batches['background'][step]
        if indices:
            logits=model.tokens(tensor(data['background']['train']['features'][indices]))
            losses['background']=F.cross_entropy(logits.flatten(0,1),torch.zeros(logits.shape[0]*32,dtype=torch.long,device=DEVICE))
        # Equal epoch-level source contribution despite different source populations.
        total=sum(value * steps/sum(bool(b) for b in batches[key])
                  * (.2 if key=='background' else 1/len(seq_keys) if key in seq_keys else 1)
                  for key,value in losses.items())
        if not torch.isfinite(total):raise ValueError('nonfinite total loss')
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        for key,value in losses.items():sums[key]+=float(value.detach().cpu());counts[key]+=1
    return dict(steps=steps,seconds=time.perf_counter()-started,
        losses={k:sums[k]/counts[k] for k in sums},
        coverage={k:sum(map(len,v)) for k,v in batches.items()},
        schedule_sha256=hashlib.sha256(json.dumps(batches,sort_keys=True).encode()).hexdigest())


@torch.inference_mode()
def evaluate(model,data,arm,epoch):
    model.eval();labels={v+1:k for k,v in data['labels'].items()}
    results=[];started=time.perf_counter()
    for begin in range(0,len(data['evaluation']),8):
        batch=data['evaluation'][begin:begin+8]
        logits,lengths=model.sequences([r['value'] for r in batch]);paths=logits.argmax(-1).cpu().tolist()
        for item,path,length in zip(batch,paths,lengths):
            row,value=item['row'],item['value'];path=path[:length]
            final=[labels[t] for t in decode(path)]
            operations=Counter(o['operation'] for o in _edit_operations(row['reference'],final))
            updates=[];end_token=0
            for keep,end in zip(value.keep,value.ends):
                end_token+=int(keep.sum())
                updates.append(dict(seconds=end,hypothesis=[labels[t] for t in decode(path[:end_token])]))
            intervals=[r for r in row.get('intervals',[]) if r['label'] in data['labels']]
            latency=word_latency(intervals,updates) if row.get('verified_reference_intervals') else None
            results.append(dict(item_id=row['source_item_id'],source=row['source'],reference=row['reference'],
                final=final,operations=dict(operations),source_latency=latency,updates=updates))
    metrics={}
    for source in sorted({r['source'] for r in results}):
        rows=[r for r in results if r['source']==source];counts=Counter()
        for row in rows:counts.update(row['operations'])
        total=sum(len(r['reference']) for r in rows)
        edits=sum(counts[k] for k in ('substitution','deletion','insertion'))
        metrics[source]=dict(samples=len(rows),reference_tokens=total,wer_percent=100*edits/max(1,total),
            substitutions=counts['substitution'],deletions=counts['deletion'],insertions=counts['insertion'])
    background=data['background']['validation']['features']
    paths=model.tokens(tensor(background)).argmax(-1).cpu().tolist()
    false_emissions=sum(bool(decode(p)) for p in paths)
    latencies=[r['source_latency'] for r in results if r['source_latency'] is not None]
    delays=[d for l in latencies for d in l['delays_seconds']]
    isolated_result=isolated(model,data)
    baseline=json.loads((REPORT/'baseline.json').read_text())
    connected=metrics['asllrp_other_ctc']
    candidate=dict(connected_wer=connected['wer_percent'],connected_insertions=connected['insertions'],
        connected_deletions=connected['deletions'],familiar_wer=metrics['local_phrases']['wer_percent'],
        citizen_accuracy=isolated_result['citizen'],semlex_accuracy=isolated_result['semlex'],
        transition_false_emissions=false_emissions,matched_pool=len(results)==334,
        complete_streaming=True,median_delay_seconds=None,latency_includes_runtime=False)
    report=dict(arm=arm,epoch=epoch,metrics=metrics,rows=results,isolated=isolated_result,
        raw_core=core_metrics(model,data),candidate=candidate,baseline=baseline,
        failed_gates=failed_gates(candidate,baseline),transition_false_emissions=false_emissions,
        source_clock_latency=dict(median_delay_seconds=float(np.median(delays)) if delays else None,
            p95_delay_seconds=float(np.percentile(delays,95)) if delays else None,
            missed_signs=sum(l['missed_signs'] for l in latencies),
            total_signs=sum(l['total_signs'] for l in latencies)),
        evaluation_seconds=time.perf_counter()-started,protected_test_accessed=False,
        latency_includes_runtime=False)
    return report


def train(data,arm):
    pre=json.loads((REPORT/'preflight.json').read_text())
    assert pre['passed']
    freeze=json.loads((REPORT/'training_freeze.json').read_text())
    for path,digest in freeze['code_and_recipe_sha256'].items():
        if old._sha256(ROOT/path)!=digest:
            raise ValueError('training code/recipe changed after freeze: '+path)
    if freeze['data_audit_sha256']!=old._sha256(REPORT/'data_audit.json'):
        raise ValueError('training data audit changed after freeze')
    model=fresh();initial_encoder=state_hash(model.base);initial_head=state_hash(model.head)
    if arm=='frozen':model.base.requires_grad_(False)
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.base.parameters() if p.requires_grad],'lr':1e-5},
        {'params':model.head.parameters(),'lr':3e-4}],weight_decay=1e-4)
    directory=MODELS/arm;directory.mkdir(parents=True,exist_ok=False)
    history=[]
    for epoch in range(1,13):
        training=train_epoch(model,data,optimizer,epoch)
        report=evaluate(model,data,arm,epoch)
        path=directory/f'epoch_{epoch:02d}.pth'
        torch.save(dict(format='slt_joint_ctc_experiment_v17',arm=arm,epoch=epoch,
            base_model_config=asdict(model.base.config),state_dict={k:v.cpu() for k,v in model.state_dict().items()},
            label_to_index=data['labels'],training=training,seed=17111,
            manifest_sha256=old._sha256(MANIFEST),initial_encoder_sha256=initial_encoder,
            initial_head_sha256=initial_head,encoder_sha256=state_hash(model.base),
            training_freeze_sha256=old._sha256(REPORT/'training_freeze.json')),path)
        report.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=old._sha256(path),training=training)
        save(f'{arm}/epoch_{epoch:02d}.json',report)
        history.append(dict(epoch=epoch,**report['candidate'],failed_gates=report['failed_gates'],
            train_seconds=training['seconds'],lg_raw_core=report['raw_core']['validation']['accuracy']))
        save(f'{arm}/summary.json',history)
        emit({'arm':arm,**history[-1]})
    if arm=='frozen':assert state_hash(model.base)==initial_encoder
    else:assert state_hash(model.base)!=initial_encoder


def baseline(data):
    freeze=json.loads((PRIOR/'development_freeze.json').read_text())
    ctc=json.loads((ROOT/freeze['ctc_baseline']).read_text())['metrics']['asllrp_other_ctc']
    familiar=json.loads((PRIOR/'older_reel_familiar/summary.json').read_text())
    gaps=json.loads((PRIOR/'transitions/baseline.json').read_text())
    own=isolated(fresh(),data)
    value=dict(connected_wer=ctc['wer_percent'],connected_insertions=ctc['insertions'],
        connected_deletions=ctc['deletions'],familiar_wer=familiar['wer_percent'],
        citizen_accuracy=own['citizen'],semlex_accuracy=own['semlex'],
        transition_false_emissions=gaps['ctc_false_emissions'])
    save('baseline.json',value);emit({'baseline':value})


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','frozen','joint'])
    args=parser.parse_args()
    if args.action=='prepare':prepare();return
    data=load_data()
    if args.action=='preflight':baseline(data);preflight(data)
    else:train(data,args.action)


if __name__=='__main__':main()
