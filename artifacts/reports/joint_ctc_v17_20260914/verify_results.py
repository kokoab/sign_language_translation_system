"""Verify completed arms and diagnose their final CTC paths without retraining."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import runpy
import time

HERE=Path(__file__).resolve().parent
run=runpy.run_path(str(HERE/'run_experiment.py'))
torch,np=run['torch'],run['np']
ROOT,MODELS,DEVICE=run['ROOT'],run['MODELS'],run['DEVICE']
save,sha=run['save'],run['old']._sha256
torch.set_num_threads(2)


def verify():
    audit=json.loads((HERE/'data_audit.json').read_text())
    freeze=json.loads((HERE/'training_freeze.json').read_text())
    for path,digest in freeze['code_and_recipe_sha256'].items():
        assert sha(ROOT/path)==digest,path
    for path,digest in audit['input_hashes'].items():
        assert not run['old']._protected(ROOT/path),path
        assert sha(ROOT/path)==digest,path
    data=run['load_data']()
    initial=run['fresh']()
    initial_encoder=run['state_hash'](initial.base)
    initial_head=run['state_hash'](initial.head)
    del initial
    reports={};rows=[];checkpoints=[]
    for arm in ('frozen','joint'):
        reports[arm]=[]
        for epoch in range(1,13):
            report=json.loads((HERE/arm/f'epoch_{epoch:02d}.json').read_text())
            path=ROOT/report['checkpoint']
            assert sha(path)==report['checkpoint_sha256']
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['initial_encoder_sha256']==initial_encoder
            assert checkpoint['initial_head_sha256']==initial_head
            assert checkpoint['training_freeze_sha256']==sha(HERE/'training_freeze.json')
            assert checkpoint['arm']==arm and checkpoint['epoch']==epoch
            model=run['fresh']();model.load_state_dict(checkpoint['state_dict'])
            actual=run['state_hash'](model.base)
            assert actual==checkpoint['encoder_sha256']
            assert (actual==initial_encoder)==(arm=='frozen')
            del model,checkpoint
            batches,steps=run['schedule'](data,epoch)
            assert report['training']['steps']==steps==110
            for key,value in batches.items():
                assert report['training']['coverage'][key]==sum(map(len,value))
            ids=[r['item_id'] for r in report['rows']]
            expected=[r['row']['source_item_id'] for r in data['evaluation']]
            assert len(ids)==len(set(ids))==334 and set(ids)==set(expected)
            for source,metrics in report['metrics'].items():
                selected=[r for r in report['rows'] if r['source']==source]
                counts=Counter()
                for r in selected:
                    counts.update(o['operation'] for o in run['_edit_operations'](r['reference'],r['final']))
                assert metrics['deletions']==counts['deletion']
                assert metrics['insertions']==counts['insertion']
                assert metrics['substitutions']==counts['substitution']
                assert metrics['reference_tokens']==sum(len(r['reference']) for r in selected)
            assert report['failed_gates']==run['failed_gates'](report['candidate'],report['baseline'])
            reports[arm].append(report)
            rows.append(dict(arm=arm,epoch=epoch,**{k:v for k,v in report['candidate'].items()
                if k not in ('median_delay_seconds','latency_includes_runtime','complete_streaming','matched_pool')},
                o5s5_train_core=report['raw_core']['train']['accuracy'],
                lg_core=report['raw_core']['validation']['accuracy'],
                steps=steps,train_seconds=report['training']['seconds']))
            checkpoints.append(dict(arm=arm,epoch=epoch,path=str(path.relative_to(ROOT)),sha256=sha(path),encoder_sha256=actual))
    for a,b in zip(reports['frozen'],reports['joint']):
        assert a['training']['schedule_sha256']==b['training']['schedule_sha256']
        assert a['training']['coverage']==b['training']['coverage']
    with (HERE/'epoch_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    diagnostics={}
    for arm in ('frozen','joint'):
        model=run['fresh']()
        checkpoint=torch.load(MODELS/arm/'epoch_12.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        group_stats=defaultdict(Counter)
        with torch.inference_mode():
            for role in ('train','validation'):
                selected=[r for r in data['sequences'] if r['role']==role]
                for begin in range(0,len(selected),8):
                    batch=selected[begin:begin+8]
                    logits,lengths=model.sequences([r['value'] for r in batch])
                    paths=logits.argmax(-1).cpu().tolist()
                    for row,path,length in zip(batch,paths,lengths):
                        c=group_stats[role+'/'+row['source']];path=path[:length]
                        pred=run['decode'](path);truth=[t for t in row['targets'] if t<101]
                        c.update(tokens=length,blank_steps=path.count(0),other_steps=path.count(101),
                            known_steps=sum(1<=p<=100 for p in path),sequences=1,
                            known_targets=len(truth),known_emissions=len(pred),exact=int(pred==truth),
                            **dict(Counter(o['operation'] for o in run['_edit_operations'](truth,pred))))
            replay=data['replay']['validation'];isolated_ctc=Counter()
            for begin in range(0,len(replay['targets']),64):
                x=run['tensor'](replay['features'][begin:begin+64])
                paths=model.tokens(x).argmax(-1).cpu().tolist()
                for i,path in enumerate(paths,begin):
                    source=replay['sources'][i]
                    isolated_ctc[source+'_total']+=1
                    isolated_ctc[source+'_correct']+=int(run['decode'](path)==[int(replay['targets'][i])+1])
            sample=[r['value'] for r in data['evaluation'][:8]]
            gpu,lengths=model.sequences(sample);gpu=gpu.cpu()
            model.cpu()
            started=time.perf_counter();cpu,_=model.sequences(sample);elapsed=time.perf_counter()-started
        comparisons=[(gpu[i,:n].argmax(-1),cpu[i,:n].argmax(-1)) for i,n in enumerate(lengths)]
        diagnostics[arm]=dict(final_epoch=12,groups={k:dict(v) for k,v in group_stats.items()},
            isolated_single_clip_ctc=dict(isolated_ctc),
            cpu_mps=dict(recordings=8,tokens=sum(lengths),agree=sum(int((a==b).sum()) for a,b in comparisons),
                max_abs_logit_difference=float((gpu-cpu).abs().max()),cpu_batch_seconds=elapsed),
            timing_limitation='Cached features and batched CPU forward; excludes normalization, Vision, scheduling and real-device latency.')
        assert diagnostics[arm]['cpu_mps']['agree']==sum(lengths)
        run['emit']({'verified_final_arm':arm,'diagnostics':diagnostics[arm]})
    result=dict(status='completed_negative_result',epochs_per_arm=12,total_epochs=24,
        optimizer_updates_per_arm=1320,verified_input_files=len(audit['input_hashes']),
        checkpoint_provenance=checkpoints,matched_schedules=True,
        encoder_frozen_verified=True,joint_encoder_changed_verified=True,
        all_epoch_gates_recomputed=True,eligible=False,
        any_epoch_with_known_development_emissions=any(r['final'] for v in reports.values() for ep in v for r in ep['rows']),
        evaluation_recordings_per_epoch=334,total_recording_evaluations=24*334,
        final_diagnostics=diagnostics,protected_test_accessed=False,defaults_changed=False,
        runtime_latency_verified=False,confirmation_run=False,
        total_successful_training_seconds=sum(r['train_seconds'] for r in rows))
    assert not result['any_epoch_with_known_development_emissions']
    assert all(r['failed_gates'] for v in reports.values() for r in v)
    save('verification.json',result)
    run['emit']({k:v for k,v in result.items() if k not in ('checkpoint_provenance','final_diagnostics')})


if __name__=='__main__':verify()
