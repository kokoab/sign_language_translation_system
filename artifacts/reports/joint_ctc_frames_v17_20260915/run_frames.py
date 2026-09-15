"""Train verified interval and guarded-gap labels on the actual continuous sequence path."""
import argparse
import importlib.util
import json
import hashlib
from collections import Counter
import time
from pathlib import Path

REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[2]
PREVIOUS=REPORT.parent/'joint_ctc_balanced_v17_20260915'
ANCHOR=REPORT.parent/'joint_ctc_anchor_v17_20260915'
spec=importlib.util.spec_from_file_location('anchor_training',ANCHOR/'run_anchor.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from active.v17.joint_ctc_loss_balance_v17 import frame_normalized_ctc_loss
from active.v17.joint_ctc_supervision_v17 import positive_ctc_loss
from active.v17.joint_ctc_frame_supervision_v17 import sequence_frame_targets, sequence_frame_loss
torch=m.torch
old,np,F,DEVICE=m.old,m.np,m.F,m.DEVICE
MODELS=ROOT/'artifacts/models/joint_ctc_frames_v17_20260915'
START=ROOT/'artifacts/models/joint_ctc_balanced_v17_20260915/epoch_24.pth'
sha=m.old.old._sha256
m.REPORT=REPORT
# Retain the reviewed per-frame CTC normalization.
m.old.ctc_loss=frame_normalized_ctc_loss

def normalized_positive(logits,targets):
    return positive_ctc_loss(logits,targets)/logits.shape[1]

m.positive_ctc_loss=normalized_positive


def load_data():
    data=m.load_data()
    manifest=json.loads(old.MANIFEST.read_text())
    rows={(r['source'],r['source_item_id']):r for r in manifest['rows'] if r['role']=='train'}
    populations=Counter();audit={}
    for row in data['sequences']:
        if row['role']!='train':continue
        value=row['value'];times=np.concatenate([t[k] for t,k in zip(value.times,value.keep)])
        row['frame_targets']=sequence_frame_targets(times,rows[(row['source'],row['identity'])],data['labels'])
        targets=row['frame_targets'];source=row['source'];populations[source]+=int((targets!=-100).sum())
        counts=audit.setdefault(source,Counter())
        counts.update(known=int(((targets>0)&(targets<101)).sum()),other=int((targets==101).sum()),
                      blank=int((targets==0).sum()),ignored=int((targets==-100).sum()))
    data['frame_populations']=dict(populations)
    m.save('frame_supervision_audit.json',dict(sources=audit,populations=dict(populations),
        blank_guard_seconds=.10,training_only=True,protected_test_accessed=False))
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
                losses[key+'_anchor']=m.anchor_loss(logits,[r['anchors'] for r in rows],data['anchor_populations'][rows[0]['source']])
                losses[key+'_frames']=sequence_frame_loss(logits,[r['frame_targets'] for r in rows],data['frame_populations'][rows[0]['source']])
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
            losses['replay_ctc']=normalized_positive(ctc_logits[:n],targets[:n])
            losses['replay_anchor']=F.cross_entropy(ctc_logits[:n,16],targets[:n]+1)
        if cores:
            losses['core']=F.cross_entropy(isolated_logits[n:],targets[n:])
            losses['core_ctc']=normalized_positive(ctc_logits[n:],targets[n:])
            losses['core_anchor']=F.cross_entropy(ctc_logits[n:,16],targets[n:]+1)
        indices=batches['background'][step]
        if indices:
            logits=model.tokens(old.tensor(data['background']['train']['features'][indices]))
            losses['background']=F.cross_entropy(logits.flatten(0,1),torch.zeros(logits.shape[0]*32,dtype=torch.long,device=DEVICE))
        total=0
        for key,value in losses.items():
            group=key.removesuffix('_anchor').removesuffix('_frames')
            if group in ('replay_ctc','core_ctc'):group=group.removesuffix('_ctc')
            weight=.2 if key=='background' else 1/len(keys) if group in keys else 1
            denominator=1 if key.endswith(('_anchor','_frames')) and group in keys else sum(bool(b) for b in batches[group])
            total=total+value*steps/denominator*weight
        if not torch.isfinite(total):raise ValueError('nonfinite repaired training loss')
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        for key,value in losses.items():sums[key]+=float(value.detach().cpu());counts[key]+=1
    return dict(steps=steps,seconds=time.perf_counter()-started,losses={k:sums[k]/counts[k] for k in sums},
        coverage={k:sum(map(len,v)) for k,v in batches.items()},
        schedule_sha256=hashlib.sha256(json.dumps(batches,sort_keys=True).encode()).hexdigest())


m.train_epoch=train_epoch


def freeze():
    previous=json.loads((PREVIOUS/'training_freeze.json').read_text())
    hashes=dict(previous['code_sha256'])
    for path in (Path(__file__),REPORT/'PLAN.md',ROOT/'active/v17/joint_ctc_loss_balance_v17.py',ROOT/'test/test_joint_ctc_loss_balance_v17.py',ROOT/'active/v17/joint_ctc_frame_supervision_v17.py',ROOT/'test/test_joint_ctc_frame_supervision_v17.py',ROOT/'active/v17/live_transition_supervision_v17.py'):
        hashes[str(path.relative_to(ROOT))]=sha(path)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    history=json.loads((PREVIOUS/'training_summary.json').read_text())
    assert len(history)==12 and history[-1]['checkpoint_sha256']==sha(START)
    value=dict(seed=17111,start_epoch=25,final_epoch=36,start_checkpoint_sha256=sha(START),
        code_sha256=hashes,previous_freeze_sha256=sha(PREVIOUS/'training_freeze.json'),
        correction='add direct CE on every unambiguous verified interval frame and guarded interior gap; otherwise retain frame-balanced recipe and optimizer')
    if (REPORT/'training_freeze.json').exists():assert json.loads((REPORT/'training_freeze.json').read_text())==value
    else:m.save('training_freeze.json',value)


def train(data):
    model=m.old.fresh()
    optimizer=torch.optim.AdamW([{'params':model.base.parameters(),'lr':1e-5},
        {'params':model.head.parameters(),'lr':3e-4}],weight_decay=1e-4)
    MODELS.mkdir(parents=True,exist_ok=True)
    completed=sorted(MODELS.glob('epoch_*.pth'))
    path=completed[-1] if completed else START
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if completed:
        assert checkpoint['training_freeze_sha256']==sha(REPORT/'training_freeze.json')
    else:
        assert checkpoint['format']=='slt_joint_ctc_balanced_repair_v17' and checkpoint['epoch']==24
        assert checkpoint['training_freeze_sha256']==sha(PREVIOUS/'training_freeze.json')
    model.load_state_dict(checkpoint['state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial=checkpoint['initial_encoder_sha256'];history=[]
    for path in completed:
        p=torch.load(path,map_location='cpu',weights_only=False)
        history.append(dict(epoch=p['epoch'],training=p['training'],metrics=p['training_metrics'],checkpoint_sha256=sha(path)))
    assert [r['epoch'] for r in history]==list(range(25,checkpoint['epoch']+1))
    for epoch in range(checkpoint['epoch']+1,37):
        stats=m.train_epoch(model,data,optimizer,epoch);metrics=m.training_metrics(model,data)
        path=MODELS/f'epoch_{epoch:02d}.pth';temp=path.with_suffix('.tmp')
        torch.save(dict(format='slt_joint_ctc_frame_repair_v17',epoch=epoch,seed=17111,
            base_model_config=m.asdict(model.base.config),state_dict={k:v.cpu() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),label_to_index=data['labels'],training=stats,
            initial_encoder_sha256=initial,encoder_sha256=m.old.state_hash(model.base),training_metrics=metrics,
            repair_start_checkpoint_sha256=sha(START),training_freeze_sha256=sha(REPORT/'training_freeze.json')),temp)
        temp.replace(path);history.append(dict(epoch=epoch,training=stats,metrics=metrics,checkpoint_sha256=sha(path)))
        m.save('training_summary.json',history);m.old.emit(history[-1])


def evaluate(data):
    path=MODELS/'epoch_36.pth';p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['format']=='slt_joint_ctc_frame_repair_v17' and p['epoch']==36 and p['seed']==17111
    assert p['training_freeze_sha256']==sha(REPORT/'training_freeze.json')
    history=json.loads((REPORT/'training_summary.json').read_text())
    assert len(history)==12 and history[-1]['checkpoint_sha256']==sha(path)
    m.old.REPORT=REPORT
    m.save('baseline.json',json.loads((m.PRIOR/'baseline.json').read_text()))
    model=m.old.fresh();model.load_state_dict(p['state_dict'])
    result=m.old.evaluate(model,data,'verified_frame_ctc_repair',36)
    result.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path))
    m.save('evaluation.json',result);m.old.emit({k:result[k] for k in ('candidate','failed_gates','source_clock_latency')})


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['train','evaluate'])
    args=parser.parse_args();freeze();data=load_data()
    {'train':train,'evaluate':evaluate}[args.action](data)
