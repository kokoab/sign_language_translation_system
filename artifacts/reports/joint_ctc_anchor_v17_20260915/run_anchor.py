"""Repair continuous CTC alignment using verified positive event anchors."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time

REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[2]
PRIOR=REPORT.parent/'joint_ctc_v17_20260914'
spec=importlib.util.spec_from_file_location('prior_joint_experiment',PRIOR/'run_experiment.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch,np,F=old.torch,old.np,old.F
from active.v17.joint_ctc_supervision_v17 import positive_ctc_loss
from active.v17.joint_ctc_anchors_v17 import event_anchors, anchor_loss
DEVICE=old.DEVICE
MODELS=ROOT/'artifacts/models/joint_ctc_anchor_v17_20260915'


def save(name,value):
    path=REPORT/name;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def load_data():
    data=old.load_data()
    audit=json.loads((PRIOR/'data_audit.json').read_text())
    assert old.old._sha256(old.MANIFEST)==audit['manifest_sha256']
    manifest=json.loads(old.MANIFEST.read_text())
    rows={(r['source'],r['source_item_id']):r for r in manifest['rows'] if r['role']=='train'}
    counts=Counter();populations=defaultdict(lambda:[0,0])
    for row in data['sequences']:
        if row['role']!='train':continue
        source=rows[(row['source'],row['identity'])]
        assert source['all_signs_annotated'] is True
        assert old.sequence_targets(source,data['labels'])==row['targets']
        value=row['value'];times=np.concatenate([t[k] for t,k in zip(value.times,value.keep)])
        row['anchors']=event_anchors(times,source['intervals'],row['targets'])
        counts['events']+=len(row['targets']);counts['anchors']+=len(row['anchors'])
        counts['known']+=sum(t<101 for _,t in row['anchors'])
        for _,target in row['anchors']:populations[row['source']][int(target==101)]+=1
    data['anchor_populations']=dict(populations)
    save('anchor_audit.json',dict(counts,populations=dict(populations),protected_test_accessed=False,validation_anchors_used=False))
    return data


def train_epoch(model,data,optimizer,epoch):
    batches,steps=old.schedule(data,epoch)
    keys=[k for k in batches if k.startswith('sequence/')]
    counts=Counter();sums=Counter();started=time.perf_counter()
    torch.manual_seed(17111+epoch);model.train();model.base.eval()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);losses={}
        for key in keys:
            rows=[data['sequences'][i] for i in batches[key][step]]
            if rows:
                logits,lengths=model.sequences([r['value'] for r in rows])
                losses[key]=old.ctc_loss(logits,[r['targets'] for r in rows],lengths)
                losses[key+'_anchor']=anchor_loss(logits,[r['anchors'] for r in rows],data['anchor_populations'][rows[0]['source']])
        replay_indices=batches['replay'][step];core_indices=batches['core'][step]
        replay=data['replay']['train'];cores=[data['cores'][i] for i in core_indices]
        positive_features=[];positive_targets=[]
        if replay_indices:
            positive_features.extend(replay['features'][replay_indices]);positive_targets.extend(replay['targets'][replay_indices])
        positive_features.extend(r['features'] for r in cores);positive_targets.extend(r['target'] for r in cores)
        x=old.tensor(np.stack(positive_features));targets=old.tensor(np.array(positive_targets))
        encoded,active=model.base.encode(x)
        isolated_logits=model.base.classifier(old.pool_stage1_encoded(model.base,encoded,active))
        ctc_logits=model.head(torch.cat((encoded,model.base.classifier(encoded)),dim=-1))
        n=len(replay_indices)
        if n:
            ce=F.cross_entropy(isolated_logits[:n],targets[:n])
            kl=F.kl_div(F.log_softmax(isolated_logits[:n]/2,-1),
                F.softmax(data['teacher_logits'][replay_indices].to(DEVICE)/2,-1),reduction='batchmean')*4
            losses['replay']=ce+kl
            losses['replay_ctc']=positive_ctc_loss(ctc_logits[:n],targets[:n])
            losses['replay_anchor']=F.cross_entropy(ctc_logits[:n,16],targets[:n]+1)
        if cores:
            losses['core']=F.cross_entropy(isolated_logits[n:],targets[n:])
            losses['core_ctc']=positive_ctc_loss(ctc_logits[n:],targets[n:])
            losses['core_anchor']=F.cross_entropy(ctc_logits[n:,16],targets[n:]+1)
        indices=batches['background'][step]
        if indices:
            logits=model.tokens(old.tensor(data['background']['train']['features'][indices]))
            losses['background']=F.cross_entropy(logits.flatten(0,1),torch.zeros(logits.shape[0]*32,dtype=torch.long,device=DEVICE))
        total=0
        for key,value in losses.items():
            group=key.removesuffix('_anchor')
            if group in ('replay_ctc','core_ctc'):group=group.removesuffix('_ctc')
            weight=.2 if key=='background' else 1/len(keys) if group in keys else 1
            denominator=1 if key.endswith('_anchor') and group in keys else sum(bool(b) for b in batches[group])
            total=total+value*steps/denominator*weight
        if not torch.isfinite(total):raise ValueError('nonfinite repaired training loss')
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        for key,value in losses.items():sums[key]+=float(value.detach().cpu());counts[key]+=1
    return dict(steps=steps,seconds=time.perf_counter()-started,losses={k:sums[k]/counts[k] for k in sums},
        coverage={k:sum(map(len,v)) for k,v in batches.items()},
        schedule_sha256=hashlib.sha256(json.dumps(batches,sort_keys=True).encode()).hexdigest())


@torch.inference_mode()
def training_metrics(model,data):
    model.eval();counts=Counter()
    rows=[r for r in data['sequences'] if r['role']=='train']
    for begin in range(0,len(rows),8):
        batch=rows[begin:begin+8];logits,lengths=model.sequences([r['value'] for r in batch])
        for row,path,length in zip(batch,logits.argmax(-1).cpu().tolist(),lengths):
            pred=old.decode(path[:length]);truth=[t for t in row['targets'] if t<101]
            counts.update(o['operation'] for o in old._edit_operations(truth,pred))
            counts['known_targets']+=len(truth);counts['known_emissions']+=len(pred)
            counts['blank_steps']+=path[:length].count(0);counts['tokens']+=length
    counts['known_recall']=counts['match']/counts['known_targets']
    counts['known_wer']=sum(counts[k] for k in ('substitution','deletion','insertion'))/counts['known_targets']
    replay=data['replay']['train'];correct=0
    for begin in range(0,len(replay['targets']),64):
        paths=model.tokens(old.tensor(replay['features'][begin:begin+64])).argmax(-1).cpu().tolist()
        correct+=sum(old.decode(p)==[int(t)+1] for p,t in zip(paths,replay['targets'][begin:begin+64]))
    counts['isolated_ctc_correct']=correct;counts['isolated_ctc_total']=len(replay['targets'])
    return dict(counts)


def freeze():
    paths=[Path(__file__),REPORT/'PLAN.md',ROOT/'active/v17/joint_ctc_supervision_v17.py',
        ROOT/'test/test_joint_ctc_supervision_v17.py',ROOT/'active/v17/joint_ctc_anchors_v17.py',
        ROOT/'test/test_joint_ctc_anchors_v17.py']
    prior=json.loads((PRIOR/'training_freeze.json').read_text())
    hashes=dict(prior['code_and_recipe_sha256'])
    hashes.update({str(p.relative_to(ROOT)):old.old._sha256(p) for p in paths})
    for p,h in hashes.items():assert old.old._sha256(ROOT/p)==h,p
    value=dict(seed=17111,epochs=12,prior_freeze_sha256=old.old._sha256(PRIOR/'training_freeze.json'),
        code_sha256=hashes,data_audit_sha256=old.old._sha256(PRIOR/'data_audit.json'),
        correction='single-token positive CTC plus sparse verified event anchors and positive midpoint CE; original data and recipe otherwise')
    if (REPORT/'training_freeze.json').exists():
        assert json.loads((REPORT/'training_freeze.json').read_text())==value
    else:save('training_freeze.json',value)


def train(data):
    freeze();model=old.fresh();initial=old.state_hash(model.base)
    optimizer=torch.optim.AdamW([{'params':model.base.parameters(),'lr':1e-5},
        {'params':model.head.parameters(),'lr':3e-4}],weight_decay=1e-4)
    MODELS.mkdir(parents=True,exist_ok=True);history=[];start=1
    completed=sorted(MODELS.glob('epoch_*.pth'))
    if completed:
        payload=torch.load(completed[-1],map_location='cpu',weights_only=False)
        assert payload['training_freeze_sha256']==old.old._sha256(REPORT/'training_freeze.json')
        model.load_state_dict(payload['state_dict']);optimizer.load_state_dict(payload['optimizer_state_dict'])
        start=payload['epoch']+1
        summary=REPORT/'training_summary.json'
        history=json.loads(summary.read_text())[:start-1] if summary.exists() else []
        # A process can stop after checkpoint rename but before the summary write.
        for path in completed[len(history):]:
            row=torch.load(path,map_location='cpu',weights_only=False)
            history.append(dict(epoch=row['epoch'],training=row['training'],metrics=row['training_metrics'],
                                checkpoint_sha256=old.old._sha256(path)))
        assert [r['epoch'] for r in history]==list(range(1,start))
    for epoch in range(start,13):
        stats=train_epoch(model,data,optimizer,epoch);metrics=training_metrics(model,data)
        path=MODELS/f'epoch_{epoch:02d}.pth';temp=path.with_suffix('.tmp')
        torch.save(dict(format='slt_joint_ctc_anchor_repair_v17',epoch=epoch,seed=17111,
            base_model_config=asdict(model.base.config),state_dict={k:v.cpu() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),label_to_index=data['labels'],training=stats,
            initial_encoder_sha256=initial,encoder_sha256=old.state_hash(model.base),
            training_metrics=metrics,training_freeze_sha256=old.old._sha256(REPORT/'training_freeze.json')),temp)
        temp.replace(path)
        history.append(dict(epoch=epoch,training=stats,metrics=metrics,checkpoint_sha256=old.old._sha256(path)))
        save('training_summary.json',history);old.emit(history[-1])


def evaluate(data):
    freeze()
    old.REPORT=REPORT
    save('baseline.json',json.loads((PRIOR/'baseline.json').read_text()))
    model=old.fresh();path=MODELS/'epoch_12.pth'
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['format']=='slt_joint_ctc_anchor_repair_v17' and p['epoch']==12 and p['seed']==17111
    assert p['training_freeze_sha256']==old.old._sha256(REPORT/'training_freeze.json')
    history=json.loads((REPORT/'training_summary.json').read_text())
    assert len(history)==12 and history[-1]['checkpoint_sha256']==old.old._sha256(path)
    model.load_state_dict(p['state_dict'])
    result=old.evaluate(model,data,'anchor_ctc_repair',12)
    result.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=old.old._sha256(path))
    save('evaluation.json',result);old.emit({k:result[k] for k in ('candidate','failed_gates','source_clock_latency')})


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','evaluate'])
    args=parser.parse_args();data=load_data()
    {'train':train,'evaluate':evaluate}[args.action](data)
