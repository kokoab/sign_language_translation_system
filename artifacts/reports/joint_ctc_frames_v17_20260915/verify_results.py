"""Check repair provenance and independently measure the actual CTC output."""
import importlib.util
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frames_run',HERE/'run_frames.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
m=n.m
torch,np=m.torch,m.np
sha=m.old.old._sha256


def verify():
    torch.set_num_threads(2)
    n.freeze();data=n.load_data()
    audit=json.loads((m.PRIOR/'data_audit.json').read_text())
    for path,digest in audit['input_hashes'].items():
        assert not m.old.old._protected(m.ROOT/path),path
        assert sha(m.ROOT/path)==digest,path
    histories={}
    for directory in ('joint_ctc_repair_v17_20260914','joint_ctc_anchor_v17_20260915','joint_ctc_balanced_v17_20260915','joint_ctc_frames_v17_20260915'):
        report=HERE.parent/directory
        freeze=json.loads((report/'training_freeze.json').read_text())
        for path,digest in freeze['code_sha256'].items():assert sha(m.ROOT/path)==digest,path
        history=json.loads((report/'training_summary.json').read_text())
        assert [r['epoch'] for r in history]==list(range(25,37) if directory==HERE.name else range(13,25) if directory=='joint_ctc_balanced_v17_20260915' else range(1,13))
        for row in history:
            path=m.ROOT/'artifacts/models'/directory/f"epoch_{row['epoch']:02d}.pth"
            assert sha(path)==row['checkpoint_sha256']
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==row['epoch'] and checkpoint['seed']==17111
            assert checkpoint['training_freeze_sha256']==sha(report/'training_freeze.json')
            assert checkpoint['training_metrics']==row['metrics']
            assert checkpoint['optimizer_state_dict']['state']
            batches,steps=m.old.schedule(data,row['epoch'])
            assert row['training']['steps']==steps==110
            assert row['training']['schedule_sha256']==hashlib.sha256(json.dumps(batches,sort_keys=True).encode()).hexdigest()
            assert row['training']['coverage']=={k:sum(map(len,v)) for k,v in batches.items()}
        histories[directory]=history
    report=json.loads((HERE/'evaluation.json').read_text())
    path=m.ROOT/report['checkpoint'];assert sha(path)==report['checkpoint_sha256']
    expected={r['row']['source_item_id']:r['row'] for r in data['evaluation']}
    assert len(report['rows'])==len(expected)==334
    assert {r['item_id'] for r in report['rows']}==set(expected)
    for row in report['rows']:
        assert row['reference']==expected[row['item_id']]['reference']
        ops=Counter(o['operation'] for o in m.old._edit_operations(row['reference'],row['final']))
        assert dict(ops)==row['operations']
    for source,metrics in report['metrics'].items():
        rows=[r for r in report['rows'] if r['source']==source];counts=Counter()
        for r in rows:counts.update(r['operations'])
        total=sum(len(r['reference']) for r in rows)
        assert metrics['reference_tokens']==total
        for plural,singular in [('deletions','deletion'),('insertions','insertion'),('substitutions','substitution')]:
            assert metrics[plural]==counts[singular]
        assert abs(metrics['wer_percent']-100*sum(counts[k] for k in ('deletion','insertion','substitution'))/total)<1e-9
    assert report['failed_gates']==m.old.failed_gates(report['candidate'],report['baseline'])
    model=m.old.fresh();initial=m.old.state_hash(model.base)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert checkpoint['initial_encoder_sha256']==initial
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    assert m.old.state_hash(model.base)==checkpoint['encoder_sha256']!=initial
    training=m.training_metrics(model,data)
    assert training==histories[HERE.name][-1]['metrics']
    isolated=Counter();core={}
    with torch.inference_mode():
        replay=data['replay']['validation']
        for begin in range(0,len(replay['targets']),64):
            paths=model.tokens(m.old.tensor(replay['features'][begin:begin+64])).argmax(-1).cpu().tolist()
            for i,p in enumerate(paths,begin):
                source=replay['sources'][i]
                isolated[source+'_total']+=1
                isolated[source+'_correct']+=int(m.old.decode(p)==[int(replay['targets'][i])+1])
                isolated[source+'_empty']+=int(not m.old.decode(p))
        for role in ('train','validation'):
            rows=[r for r in data['cores'] if r['role']==role];counts=Counter()
            for begin in range(0,len(rows),64):
                batch=rows[begin:begin+64]
                paths=model.tokens(m.old.tensor(np.stack([r['features'] for r in batch]))).argmax(-1).cpu().tolist()
                for r,p in zip(batch,paths):
                    counts['total']+=1;counts['correct']+=int(m.old.decode(p)==[r['target']+1])
                    counts['empty']+=int(not m.old.decode(p))
            core[role]=dict(counts)
        sample=[r['value'] for r in data['evaluation'][:8]]
        gpu,lengths=model.sequences(sample);gpu=gpu.cpu()
        # A prefix must produce the same earlier outputs as the full causal head.
        value=sample[0]
        from active.v17.joint_ctc_v17 import Chunks
        end=min(2,len(value.features))
        prefix=Chunks(value.features[:end],value.times[:end],value.keep[:end],value.ends[:end])
        partial,plen=model.sequences([prefix])
        prefix_error=float((partial[0,:plen[0]].cpu()-gpu[0,:plen[0]]).abs().max())
        assert torch.equal(partial[0,:plen[0]].argmax(-1).cpu(),gpu[0,:plen[0]].argmax(-1))
        model.cpu();cpu,_=model.sequences(sample)
    agree=sum(int((gpu[i,:n].argmax(-1)==cpu[i,:n].argmax(-1)).sum()) for i,n in enumerate(lengths))
    assert agree==sum(lengths)
    result=dict(verified_input_files=len(audit['input_hashes']),verified_checkpoints=48,
        epochs_per_correction=12,optimizer_updates_per_correction=1320,final_model_optimizer_updates=3960,matched_full_coverage_schedules=True,
        final_training_recomputed=training,development_recordings=334,
        nonempty_development_recordings=sum(bool(r['final']) for r in report['rows']),
        isolated_single_clip_ctc=dict(isolated),raw_core_ctc=core,
        cpu_mps=dict(recordings=8,tokens=sum(lengths),agree=agree,max_logit_difference=float((gpu-cpu).abs().max())),
        prefix_consistency=dict(tokens=plen[0],max_logit_difference=prefix_error),
        failed_promotion_gates=report['failed_gates'],protected_test_accessed=False,
        runtime_latency_verified=False,defaults_changed=False,confirmation_run=False,
        checkpoint_sha256=sha(path),verification_script_sha256=sha(Path(__file__)))
    m.save('verification.json',result);m.old.emit(result)


if __name__=='__main__':verify()
