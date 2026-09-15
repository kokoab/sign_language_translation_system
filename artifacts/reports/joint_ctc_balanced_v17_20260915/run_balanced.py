"""Continue verified-anchor training with CTC likelihood normalized per output frame."""
import argparse
import importlib.util
import json
from pathlib import Path

REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[2]
PREVIOUS=REPORT.parent/'joint_ctc_anchor_v17_20260915'
spec=importlib.util.spec_from_file_location('anchor_training',PREVIOUS/'run_anchor.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from active.v17.joint_ctc_loss_balance_v17 import frame_normalized_ctc_loss
from active.v17.joint_ctc_supervision_v17 import positive_ctc_loss
torch=m.torch
MODELS=ROOT/'artifacts/models/joint_ctc_balanced_v17_20260915'
START=ROOT/'artifacts/models/joint_ctc_anchor_v17_20260915/epoch_12.pth'
sha=m.old.old._sha256
m.REPORT=REPORT
# Reuse the reviewed training loop; these are the only objective replacements.
m.old.ctc_loss=frame_normalized_ctc_loss

def normalized_positive(logits,targets):
    return positive_ctc_loss(logits,targets)/logits.shape[1]

m.positive_ctc_loss=normalized_positive


def freeze():
    previous=json.loads((PREVIOUS/'training_freeze.json').read_text())
    hashes=dict(previous['code_sha256'])
    for path in (Path(__file__),REPORT/'PLAN.md',ROOT/'active/v17/joint_ctc_loss_balance_v17.py',ROOT/'test/test_joint_ctc_loss_balance_v17.py'):
        hashes[str(path.relative_to(ROOT))]=sha(path)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    history=json.loads((PREVIOUS/'training_summary.json').read_text())
    assert len(history)==12 and history[-1]['checkpoint_sha256']==sha(START)
    value=dict(seed=17111,start_epoch=13,final_epoch=24,start_checkpoint_sha256=sha(START),
        code_sha256=hashes,previous_freeze_sha256=sha(PREVIOUS/'training_freeze.json'),
        correction='CTC NLL per valid output frame instead of per target; all other objectives and optimizer state retained')
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
        assert checkpoint['format']=='slt_joint_ctc_anchor_repair_v17' and checkpoint['epoch']==12
        assert checkpoint['training_freeze_sha256']==sha(PREVIOUS/'training_freeze.json')
    model.load_state_dict(checkpoint['state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial=checkpoint['initial_encoder_sha256'];history=[]
    for path in completed:
        p=torch.load(path,map_location='cpu',weights_only=False)
        history.append(dict(epoch=p['epoch'],training=p['training'],metrics=p['training_metrics'],checkpoint_sha256=sha(path)))
    assert [r['epoch'] for r in history]==list(range(13,checkpoint['epoch']+1))
    for epoch in range(checkpoint['epoch']+1,25):
        stats=m.train_epoch(model,data,optimizer,epoch);metrics=m.training_metrics(model,data)
        path=MODELS/f'epoch_{epoch:02d}.pth';temp=path.with_suffix('.tmp')
        torch.save(dict(format='slt_joint_ctc_balanced_repair_v17',epoch=epoch,seed=17111,
            base_model_config=m.asdict(model.base.config),state_dict={k:v.cpu() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),label_to_index=data['labels'],training=stats,
            initial_encoder_sha256=initial,encoder_sha256=m.old.state_hash(model.base),training_metrics=metrics,
            repair_start_checkpoint_sha256=sha(START),training_freeze_sha256=sha(REPORT/'training_freeze.json')),temp)
        temp.replace(path);history.append(dict(epoch=epoch,training=stats,metrics=metrics,checkpoint_sha256=sha(path)))
        m.save('training_summary.json',history);m.old.emit(history[-1])


def evaluate(data):
    path=MODELS/'epoch_24.pth';p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['format']=='slt_joint_ctc_balanced_repair_v17' and p['epoch']==24 and p['seed']==17111
    assert p['training_freeze_sha256']==sha(REPORT/'training_freeze.json')
    history=json.loads((REPORT/'training_summary.json').read_text())
    assert len(history)==12 and history[-1]['checkpoint_sha256']==sha(path)
    m.old.REPORT=REPORT
    m.save('baseline.json',json.loads((m.PRIOR/'baseline.json').read_text()))
    model=m.old.fresh();model.load_state_dict(p['state_dict'])
    result=m.old.evaluate(model,data,'frame_balanced_ctc_repair',24)
    result.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path))
    m.save('evaluation.json',result);m.old.emit({k:result[k] for k in ('candidate','failed_gates','source_clock_latency')})


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['train','evaluate'])
    args=parser.parse_args();freeze();data=m.load_data()
    {'train':train,'evaluate':evaluate}[args.action](data)
