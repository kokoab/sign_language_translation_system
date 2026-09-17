"""Approved English comparison; OS exit event queue, never training-status polling."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASELINE=HERE.parent/'stage1_direct_translation_20260917'
ASSETS=ROOT/'data/local/english_text_comparison_20260917'
MODELS=Path(os.environ.get('SLT_CHECKPOINT_ROOT',str(ROOT/'artifacts/models')))/HERE.name
REVISION='aadd2ab0ae0c8268c7c9693540e9904811f36177'


def save(name,value):
    p=HERE/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def wait_for_exit(pid):
    """One kernel event wait; also handle the process exiting before registration."""
    with closing(select.kqueue()) as queue:
        event=select.kevent(pid,filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD|select.KQ_EV_ONESHOT,fflags=select.KQ_NOTE_EXIT)
        try:queue.control([event],0,0)
        except ProcessLookupError:return
        result=queue.control(None,1,None)
        if result and result[0].flags & select.KQ_EV_ERROR:
            if result[0].data==errno.ESRCH:return
            raise OSError(result[0].data,'process exit event failed')


def target_bucket(length):
    if length<1:raise ValueError('empty target')
    for size in [16,32,48,64,96,128]:
        if length<=size:return size
    raise ValueError('target exceeds prepared bucket range; never truncate')


def prepare():
    from transformers import AutoTokenizer
    trim=load_module(HERE/'vocab_trim.py','english_vocab_trim')
    original=json.loads((BASELINE/'manifest.json').read_text())
    texts=[r['reference'] for r in original['train']]
    prefix='Translate sign language video to English: '
    trim.prepare_vocab(ROOT/'data/local/unisign_asl_baseline_20260916/mt5-base',
                       texts+[prefix],ASSETS/'trimmed-mt5')
    tokenizer=AutoTokenizer.from_pretrained(str(ASSETS/'bart-base'),local_files_only=True)
    lengths=[len(tokenizer(t)['input_ids']) for t in texts]
    for n in lengths:target_bucket(n)
    assets={str(p.relative_to(ROOT)):digest(p) for parent in [ASSETS/'bart-base',ASSETS/'trimmed-mt5']
            for p in parent.rglob('*') if p.is_file() and '.cache' not in p.parts}
    assert any(n.endswith('model.safetensors') for n in assets), 'BART weights missing'
    files=[Path(__file__),HERE/'vocab_trim.py',ROOT/'test/test_english_comparison_v17.py',
           ROOT/'test/test_english_vocab_trim_v17.py']
    manifest={**original,'comparison_baseline_manifest_sha256':digest(BASELINE/'manifest.json'),
        'comparison_code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},
        'comparison_assets_sha256':assets,'bart_revision':REVISION,
        'target_length_audit':dict(count=len(lengths),max=max(lengths),mean=sum(lengths)/len(lengths)),
        'recipe':dict(seed=17111,epochs=20,warmup_epochs=1,batch_size=2,
            isolated_loss_weight=.5,learning_rates=[1e-5,3e-4,1e-4],
            target_buckets=[16,32,48,64,96,128],source_padding=256,
            mps_cache_release_every_updates=16,dtype='float32',
            generation=dict(num_beams=4,max_new_tokens=100,no_repeat_ngram_size=0,
                            repetition_penalty=1.,length_penalty=1.,early_stopping=False)),
        'approval':'User approved English comparison; no automatic deployment.',
        'limitation':'12 paired development sentences cannot establish accuracy noninferiority.'}
    manifest['baseline_text_sha256']=manifest.pop('text_sha256')
    manifest['baseline_tokenizer_files']=manifest.pop('tokenizer_files')
    manifest['text_initialization']='facebook/bart-base@'+REVISION
    manifest['text_sha256']=digest(ASSETS/'bart-base/model.safetensors')
    assert not (HERE/'manifest.json').exists(), 'do not overwrite frozen experiment'
    save('manifest.json',manifest)
    save('status.json',dict(stage='prepared; waiting for baseline exit'))
    print('Prepared approved comparison; no GPU use or optimizer steps.',flush=True)


def runtime():
    sys.path.insert(0,str(ROOT))
    b=load_module(BASELINE/'run_experiment.py','baseline_helpers')
    import torch
    torch.set_num_threads(4)
    assert b.DEVICE=='mps', 'MPS unavailable; do not silently launch a CPU training run'
    b.check_checkpoint_space()
    m=json.loads((HERE/'manifest.json').read_text())
    assert digest(BASELINE/'manifest.json')==m['comparison_baseline_manifest_sha256']
    for name,h in {**m['comparison_assets_sha256'],**m['comparison_code_sha256']}.items():
        assert digest(ROOT/name)==h,name
    _,data=b.verify_and_load()
    save('provenance.json',dict(manifest_sha256=digest(HERE/'manifest.json'),device=b.DEVICE,
        dtype='float32',torch=torch.__version__,bart_revision=REVISION,
        bart_parameters=146409509,started_at=datetime.now(timezone.utc).isoformat(),
        original_stage1_sha256=m['base_sha256'],citizen_test_accessed=False))
    return b,torch,m,data


def fresh_bart(b,torch):
    from transformers import AutoTokenizer,BartForConditionalGeneration
    b.random.seed(b.SEED);b.np.random.seed(b.SEED);torch.manual_seed(b.SEED)
    base=b._load_start(torch.load(b.BASE,map_location='cpu',weights_only=False)).base
    text,loading=BartForConditionalGeneration.from_pretrained(str(ASSETS/'bart-base'),
        local_files_only=True,output_loading_info=True)
    assert not loading['unexpected_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'],loading
    assert set(loading['missing_keys'])<={'final_logits_bias','lm_head.weight'},loading
    assert text.lm_head.weight is text.model.shared.weight
    assert torch.count_nonzero(text.final_logits_bias)==0
    save('text_loading.json',loading)
    # The released config blocks repeated trigrams; this experiment allows repetition.
    text.generation_config.no_repeat_ngram_size=0
    text.generation_config.repetition_penalty=1.0
    text.generation_config.length_penalty=1.0
    text.generation_config.early_stopping=False
    tokenizer=AutoTokenizer.from_pretrained(str(ASSETS/'bart-base'),local_files_only=True)
    prefix=tokenizer('Translate sign language video to English: ',return_tensors='pt')['input_ids']
    model=b.DirectTranslation(base,text,prefix).to(b.DEVICE)
    assert sum(p.numel() for p in model.parameters())==146409509
    return model,tokenizer


def configure_bart(b,torch):
    def labels(tokenizer,rows):
        raw=tokenizer([r['reference'] for r in rows])['input_ids']
        size=target_bucket(max(map(len,raw)))
        value=torch.full((len(raw),size),-100,dtype=torch.long)
        for i,ids in enumerate(raw):value[i,:len(ids)]=torch.tensor(ids)
        return value.to(b.DEVICE)
    original_release=b.release_unused_mps
    calls=0
    def periodic_release():
        nonlocal calls
        calls+=1
        # Reused training loop calls twice per update; clear every 16 updates.
        return original_release() if calls%32==0 else {}
    b.HERE=HERE;b.MODELS=MODELS;b.labels=labels;b.release_unused_mps=periodic_release


def preflight(b,torch,model,tokenizer,data):
    rows=sorted(data['train'],key=lambda r:(len(r['x']),len(r['reference'])),reverse=True)[:4]
    x,valid=b.tensors(rows[:2]);labels=b.labels(tokenizer,rows[:2]);model.eval()
    with torch.no_grad():before=float(model.translation_loss(x,valid,labels).cpu())
    optimizer=b.optimizer_for(model);records=[];gradient={};model.train()
    for step in range(36):
        batch=rows[2*(step%2):2*(step%2)+2];bx,bv=b.tensors(batch)
        if b.DEVICE=='mps':torch.mps.synchronize()
        begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
        loss=model.translation_loss(bx,bv,b.labels(tokenizer,batch))
        replay=data['isolated_train']
        aux=torch.nn.functional.cross_entropy(model.base(torch.from_numpy(replay['x'][:4]).to(b.DEVICE)),
                                               torch.from_numpy(replay['y'][:4]).to(b.DEVICE))
        total=loss+.5*aux;assert torch.isfinite(total);total.backward()
        if step==0:
            for name,module in [('stage1',model.base),('projection',model.projection),('text',model.text)]:
                gradient[name]=sum(float(p.grad.abs().sum().cpu()) for p in module.parameters() if p.grad is not None)
            assert all(v>0 and b.math.isfinite(v) for v in gradient.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if b.DEVICE=='mps':
            torch.mps.synchronize()
            if (step+1)%16==0:torch.mps.empty_cache()
        records.append(dict(step=step+1,loss=float(total.detach().cpu()),seconds=time.monotonic()-begin,
            driver_bytes=torch.mps.driver_allocated_memory() if b.DEVICE=='mps' else None))
    model.eval()
    with torch.no_grad():after=float(model.translation_loss(x,valid,labels).cpu())
    result=dict(passed=after<before*.95,before=before,after=after,gradient_l1=gradient,records=records,
        median_step_seconds=float(b.np.median([r['seconds'] for r in records[4:]])),
        note='36 maximum-length train-only joint updates; discarded and weights reset; no accuracy evidence.')
    save('preflight.json',result)
    assert result['passed'], 'training-only tiny fit failed; no full run'


def mismatched(b,model,tokenizer,rows):
    switched=[{**r,'x':rows[(i+1)%len(rows)]['x'],'valid':rows[(i+1)%len(rows)]['valid'],
               'visual_source_item_id':rows[(i+1)%len(rows)]['item_id'],
               'visual_source_video':rows[(i+1)%len(rows)]['video']}
              for i,r in enumerate(rows)]
    return b.predict(model,tokenizer,switched)


def work():
    b,torch,manifest,data=runtime()
    summary=json.loads((BASELINE/'summary.json').read_text())
    save('status.json',dict(stage='reviewing completed baseline'))
    if not summary['beats_zero_visual_chrf']:
        (HERE/'REPORT.md').write_text('# English comparison stopped before training\n\nThe completed full hybrid did not beat its zero-visual chrF control. The approved plan requires reviewing visual grounding first; training a smaller text model is not an established correction.\n\nNo BART training or model promotion. Review the baseline REPORT.md before another experiment.\n')
        return 'stopped_baseline_grounding'
    save('status.json',dict(stage='no-retraining vocabulary-trim evaluation'))
    from transformers import T5Tokenizer
    trim=load_module(HERE/'vocab_trim.py','english_vocab_trim')
    model,tokenizer=b.fresh()
    checkpoint=Path(summary['checkpoint']);assert digest(checkpoint)==summary['checkpoint_sha256']
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(payload['state_dict'],strict=True);del payload;b.gc.collect()
    mismatch_full=mismatched(b,model,tokenizer,data['validation']);save('full_mismatched_predictions.json',mismatch_full)
    trim.trim_model(model,ASSETS/'trimmed-mt5')
    tokenizer=T5Tokenizer.from_pretrained(str(ASSETS/'trimmed-mt5'),local_files_only=True,legacy=False)
    rows=data['validation']+data['diagnostics']
    trimmed=b.predict(model,tokenizer,rows);save('trimmed_predictions.json',trimmed)
    trim_zero=b.predict(model,tokenizer,data['validation'],zero_visual=True);save('trimmed_zero_predictions.json',trim_zero)
    trim_info=dict(parameters=sum(p.numel() for p in model.parameters()),metrics=b.metrics(trimmed),
                   zero_visual=b.metrics(trim_zero),source_checkpoint_sha256=summary['checkpoint_sha256'])
    save('trim_summary.json',trim_info)
    del model;b.gc.collect()
    if b.DEVICE=='mps':torch.mps.empty_cache()
    configure_bart(b,torch)
    save('status.json',dict(stage='BART train-only preflight and timing'))
    model,tokenizer=fresh_bart(b,torch);preflight(b,torch,model,tokenizer,data)
    del model;b.gc.collect()
    if b.DEVICE=='mps':torch.mps.empty_cache()
    model,tokenizer=fresh_bart(b,torch)
    initial=b.predict(model,tokenizer,rows);save('initial_predictions.json',initial)
    retention_before=b.isolated(model,data);save('initial_retention.json',retention_before)
    save('status.json',dict(stage='BART fixed 20-epoch training'))
    checkpoint,history=b.train(model,tokenizer,data)
    save('status.json',dict(stage='final evaluation and report'))
    final=b.predict(model,tokenizer,rows);save('final_predictions.json',final)
    zero=b.predict(model,tokenizer,data['validation'],zero_visual=True);save('zero_visual_predictions.json',zero)
    mismatch=mismatched(b,model,tokenizer,data['validation']);save('mismatched_predictions.json',mismatch)
    retention_after=b.isolated(model,data);save('final_retention.json',retention_after)
    full=summary['translation']['trained hybrid'];candidate=b.metrics(final)
    full_retention=json.loads((BASELINE/'final_retention.json').read_text())
    checks=dict(chrf_within_1=candidate['chrf']>=full['chrf']-1,
        bleu_within_half=candidate['bleu']>=full['bleu']-.5,
        isolated_within_1pp=all(retention_after[k]['accuracy']>=max(full_retention[k]['accuracy'],retention_before[k]['accuracy'])-.01 for k in retention_after),
        beats_zero_visual=candidate['chrf']>b.metrics(zero)['chrf'],
        beats_mismatched_visual=candidate['chrf']>b.metrics(mismatch)['chrf'])
    result=dict(translation={'full mT5':full,'trimmed mT5':trim_info['metrics'],
        'initialized BART':b.metrics(initial),'trained BART':candidate,'BART zero visual':b.metrics(zero),
        'BART mismatched visual':b.metrics(mismatch),'full mT5 mismatched visual':b.metrics(mismatch_full)},
        descriptive_checks=checks,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        final_epoch=20,training_seconds=history[-1]['elapsed_seconds'],
        decision='Screening only; 12 correlated sentences cannot establish accuracy noninferiority. No production promotion.',
        citizen_test_accessed=False,repetition_is_a_gate=False)
    save('summary.json',result)
    lines=['# English model comparison results','',result['decision'],'',
        '| Model/control | BLEU | chrF |','| --- | ---: | ---: |']
    for name,scores in result['translation'].items():lines.append(f"| {name} | {scores['bleu']:.3f} | {scores['chrf']:.3f} |")
    lines+=['','## Isolated development retention','',
        '| Subset | Before | Full mT5 | BART |','| --- | ---: | ---: | ---: |']
    for k in retention_before:
        lines.append(f"| {k} | {retention_before[k]['accuracy']:.2%} | {full_retention[k]['accuracy']:.2%} | {retention_after[k]['accuracy']:.2%} |")
    lines+=['','## Descriptive checks','',json.dumps(checks,indent=2),'',
        'No automatic deployment. Expanded independent signer/source evaluation and semantic review remain required. Inherited mT5 pretraining is not certified signer-disjoint. No mobile timing or accuracy claim. Intended repetitions remain allowed.','',
        'BART used the same original Stage-1 checkpoint and training rows, full coverage, joint isolated CE, 20 fixed epochs. BART adds English text pretraining; it does not inherit the ASL mT5 weights. The trimmed checkpoint was evaluated without retraining. Timing/preflight and hashes are saved alongside this report.','',
        '## Video predictions','']
    full_preds={r['item_id']:r['prediction'] for r in json.loads((BASELINE/'final_predictions.json').read_text())}
    trim_preds={r['item_id']:r['prediction'] for r in trimmed}
    for r in final:
        lines += [f"### {r['item_id']}",'',f"[Video]({r['video']})",'',
            f"Reference: {r['reference'] or 'No verified English reference'}",'',
            f"Full mT5: {full_preds[r['item_id']]}",'',f"Trimmed mT5: {trim_preds[r['item_id']]}",'',
            f"BART: {r['prediction']}",'']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return 'completed'


def finish(status):
    save('completion.json',dict(status=status,finished_at=datetime.now(timezone.utc).isoformat()))
    save('status.json',dict(stage=status))
    message=f'English model comparison {status}; see {HERE.name}/'+('FAILURE.md' if status=='failed' else 'REPORT.md')
    log=ROOT/'docs/ground_truth/live-streaming/log.md'
    log.write_text(log.read_text().replace('\n---\n',f'\n---\n\n## {datetime.now().date()} — English comparison exit\n\n{message}. No model promotion; review predictions and controls.\n',1))
    ground=ROOT/'PROJECT_GROUND_TRUTH.md'
    ground.write_text(re.sub(r'^English comparison status: [^\n]*',
                            f'English comparison status: {status}.',ground.read_text(),count=1,flags=re.M))
    subprocess.run([str(ROOT/'venv/bin/python'),str(ROOT/'scripts/index_large_artifacts_v17.py')],cwd=ROOT,check=False)
    subprocess.run(['osascript','-e','on run argv\ndisplay notification (item 1 of argv) with title "SLT experiment"\nend run',message],capture_output=True,timeout=15,check=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--supervise',type=int);p.add_argument('--work',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    elif args.launch:
        assert (HERE/'manifest.json').exists()
        baseline_launch=BASELINE/'ssd_resume_launch.json'
        if not baseline_launch.exists():baseline_launch=BASELINE/'resume_launch.json'
        pid=json.loads(baseline_launch.read_text())['pid']
        command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
        assert str(BASELINE/'run_experiment.py') in command or (BASELINE/'completion.json').exists()
        if str(BASELINE/'run_experiment.py') not in command:pid=0
        with (HERE/'launch.json').open('x') as record,(HERE/'process.log').open('a') as log:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-u',__file__,'--supervise',str(pid)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            json.dump(dict(pid=child.pid,wait_for_pid=pid,launched_at=datetime.now(timezone.utc).isoformat()),record)
        ground=ROOT/'PROJECT_GROUND_TRUTH.md'
        ground.write_text(ground.read_text().replace('English comparison status: preparing.',
                                                    'English comparison status: queued.'))
        print('Queued comparison',child.pid,'on baseline exit event; no polling.')
    elif args.supervise is not None:
        try:
            if args.supervise>0:wait_for_exit(args.supervise)
            completion=json.loads((BASELINE/'completion.json').read_text())
            if completion['status']!='completed':raise RuntimeError('Baseline failed; comparison not started')
            result=subprocess.run([sys.executable,'-u',__file__,'--work'],cwd=ROOT)
            if not (HERE/'completion.json').exists():raise RuntimeError(f'Comparison worker exited {result.returncode}; inspect process.log')
            sys.exit(result.returncode)
        except Exception:
            (HERE/'FAILURE.md').write_text('# Comparison did not complete\n\n```\n'+traceback.format_exc()+'```\n')
            finish('failed')
            sys.exit(1)
    elif args.work:
        status='failed'
        try:status=work()
        except Exception:
            (HERE/'FAILURE.md').write_text('# Comparison failed\n\n```\n'+traceback.format_exc()+'```\n')
            traceback.print_exc()
        finally:finish(status)
        sys.exit(0 if status!='failed' else 1)
