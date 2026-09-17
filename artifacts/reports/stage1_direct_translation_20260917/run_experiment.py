"""Fixed direct-translation pilot. --prepare, --launch; no status polling."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT/'artifacts/generated/stage1_direct_translation_20260917'
MODELS = Path(os.environ.get('SLT_CHECKPOINT_ROOT',str(ROOT/'artifacts/models')))/HERE.name
PRIOR = ROOT/'artifacts/reports/unisign_asl_baseline_20260916'
BASE = ROOT/'artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth'
TEXT_SOURCE = ROOT/'data/local/unisign_asl_baseline_20260916/how2sign_pose_only_slt.pth'
TEXT_CONFIG = TEXT_SOURCE.parent/'mt5-base'
SEED = 17111
EPOCHS = 20
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from transformers import MT5Config, MT5ForConditionalGeneration, T5Tokenizer, Adafactor
from active.v17.direct_translation_v17 import DirectTranslation, epoch_batches
from active.v17.train_stage1_window_v17 import _load_start, _protected
from active.v17.train_stage_1_phrase_adapt_v17 import load_features, sha256
from scripts.extract_how2sign_transition_landmarks_v17 import safe_name

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def save(name,value):
    path=HERE/name;temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def emit(value):
    print(json.dumps(value),flush=True)


def read_archive(path):
    with np.load(path,allow_pickle=False) as p:
        x=p['landmarks'].astype(np.float32)
        valid=p['window_valid'].astype(bool)
        ranges=p['window_source_ranges']
        metadata=json.loads(str(p['metadata_json']))
    if x.ndim!=4 or x.shape[1:]!=(32,61,5) or valid.shape!=(len(x),):
        raise ValueError('invalid continuous v17 archive: '+str(path))
    if not np.isfinite(x).all() or not valid.any() or metadata['schema_fingerprint']!='b872fa3dcc16aab5':
        raise ValueError('invalid feature values/schema: '+str(path))
    assert len(ranges)==len(x) and np.all(ranges[:,1]>ranges[:,0])
    assert np.all(ranges[1:,0]>=ranges[:-1,1]), 'unordered/overlapping source windows'
    return x,valid,metadata


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    if (HERE/'manifest.json').exists():
        raise FileExistsError('Frozen manifest already exists; do not overwrite')
    prior=json.loads((PRIOR/'manifest.json').read_text())
    prior_results={r['item_id']:r for r in json.loads((PRIOR/'results.json').read_text())}
    dev_rows=[]
    for r in prior['recordings']:
        assert sha256(Path(r['video']))==r['video_sha256']
        dev_rows.append(dict(source_item_id=r['item_id'],source='direct_translation_development',
            role='validation',video_path=r['video'],video_sha256=r['video_sha256'],
            source_group=r.get('source_video',r['item_id']),signer_id='development',
            duration_seconds=prior_results[r['item_id']]['duration_seconds'],license='source-specific research use',
            reference=r['reference'],gloss_annotation=r.get('gloss_annotation')))
    extraction_manifest=DATA/'development_sources.json'
    extraction_manifest.write_text(json.dumps(dict(format='continuous_unlabeled_transition_manifest_v17',rows=dev_rows),indent=2)+'\n')
    from scripts.extract_how2sign_transition_landmarks_v17 import run,build_parser,save as save_archive
    args=build_parser().parse_args(['--manifest',str(extraction_manifest),
        '--output-root',str(DATA/'development_landmarks'), '--report',str(HERE/'extraction.json')])
    extraction=run(args)
    extraction['how2sign_validation_accessed']=True
    save('extraction.json',extraction)
    if extraction['failed']:
        raise RuntimeError('Development extraction failed; see extraction.json')
    validation=[];diagnostics=[];input_hashes={}
    for source,row in zip(prior['recordings'],dev_rows):
        path=DATA/'development_landmarks/development'/(safe_name(row['source_item_id'])+'.transition_landmarks_v17.npz')
        # The reused train-only extractor's bookkeeping flags must reflect this dev use.
        with np.load(path,allow_pickle=False) as payload:
            arrays={k:payload[k] for k in payload.files if k!='metadata_json'}
            metadata=json.loads(str(payload['metadata_json']))
        metadata['how2sign_validation_accessed']=source['reference'] is not None
        metadata['evaluation_only']=True
        save_archive(path,arrays,metadata)
        read_archive(path)
        input_hashes[str(path.relative_to(ROOT))]=sha256(path)
        value=dict(item_id=source['item_id'],archive=str(path.relative_to(ROOT)),
            video=source['video'],reference=source['reference'],gloss_annotation=source.get('gloss_annotation'),
            source_video=source.get('source_video'),video_sha256=source['video_sha256'])
        (validation if source['reference'] is not None else diagnostics).append(value)
    assert len(validation)==12 and len(diagnostics)==9
    manifest_path=ROOT/'active/v17/how2sign_transition_manifest_v17.json'
    original=json.loads(manifest_path.read_text());manifest_hash=sha256(manifest_path)
    train=[];excluded=[]
    for row in original['rows']:
        if row['role']!='train' or not row['sentence'].strip():
            raise ValueError('Unlabelled or non-training row in train manifest')
        if row['signer_id'] in {'how2sign:1','how2sign:2'}:
            excluded.append(dict(item_id=row['source_item_id'],reason='held-out adaptation signer'))
            continue
        path=ROOT/'data/local/how2sign_transition_landmarks_v17'/row['signer_id'].replace(':','_')/(safe_name(row['source_item_id'])+'.transition_landmarks_v17.npz')
        if not path.exists():
            excluded.append(dict(item_id=row['source_item_id'],reason='previous extraction has no usable archive'))
            continue
        _,_,metadata=read_archive(path)
        assert metadata['manifest_sha256']==manifest_hash
        assert metadata['video_sha256']==row['video_sha256'] and metadata['role']=='train'
        assert metadata['source_item_id']==row['source_item_id']
        input_hashes[str(path.relative_to(ROOT))]=sha256(path)
        train.append(dict(item_id=row['source_item_id'],archive=str(path.relative_to(ROOT)),
            video=str(ROOT/row['video_path']),reference=row['sentence'],signer=row['signer_id']))
    assert len(train)>900
    train_ids={r['item_id'].split(':',1)[-1] for r in train}
    assert not train_ids & {r['item_id'].rsplit('-',2)[0] for r in validation}
    frozen_path=ROOT/'artifacts/reports/o5s5_augmented_v17_20260914/development_freeze.json'
    replay=json.loads(frozen_path.read_text())['isolated_replay']
    paths=[]
    for split in ['train','validation']:
        current=set()
        for row in replay[split]:
            path=ROOT/row['path']
            assert not _protected(path) and sha256(path)==row['sha256']
            assert 0<=row['target']<100
            if row['source']=='citizen':
                assert ('/train/' if split=='train' else '/val/') in row['path']
            elif row['source']=='semlex':
                assert ('/semlex_citizen100_train_audit/' if split=='train'
                        else '/semlex_citizen100_val_audit/') in row['path']
            else:
                raise ValueError('Unexpected isolated replay source')
            input_hashes[row['path']]=row['sha256'];current.add(row['path'])
        paths.append(current)
    assert not paths[0]&paths[1]
    assert sha256(TEXT_SOURCE)=='1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d'
    files=[Path(__file__),ROOT/'active/v17/direct_translation_v17.py',ROOT/'active/v17/model_v17.py',
           ROOT/'test/test_direct_translation_v17.py',ROOT/'docs/superpowers/plans/2026-09-17-stage1-direct-translation.md',
           ROOT/'scripts/extract_how2sign_transition_landmarks_v17.py',ROOT/'active/v17/extract_v17.py']
    value=dict(train=train,validation=validation,diagnostics=diagnostics,isolated_replay=replay,
        exclusions=excluded,input_hashes=input_hashes,
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
        base_sha256=sha256(BASE),text_sha256=sha256(TEXT_SOURCE),
        source_manifest_sha256=manifest_hash,replay_manifest_sha256=sha256(frozen_path),
        tokenizer_files={p.name:sha256(p) for p in TEXT_CONFIG.iterdir() if p.is_file()},
        seed=SEED,epochs=EPOCHS,isolated_loss_weight=.5,
        counts=dict(train=len(train),validation=len(validation),diagnostics=len(diagnostics),
                    replay={k:len(v) for k,v in replay.items()},train_signers=dict(Counter(r['signer'] for r in train))),
        protected_test_accessed=False,
        limitation='New adaptation excludes evaluation signers 1/2, but inherited ASL mT5 pretraining is not certified signer-disjoint. Fixed 12-utterance development slice; no full-system generalization claim.')
    save('manifest.json',value);emit(value['counts'])
    return value


def verify_and_load():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,digest in {**manifest['input_hashes'],**manifest['code_sha256']}.items():
        assert sha256(ROOT/name)==digest,name
    assert sha256(BASE)==manifest['base_sha256'] and sha256(TEXT_SOURCE)==manifest['text_sha256']
    for name,digest in manifest['tokenizer_files'].items():
        assert sha256(TEXT_CONFIG/name)==digest
    data={}
    for split in ['train','validation','diagnostics']:
        data[split]=[]
        for row in manifest[split]:
            x,valid,_=read_archive(ROOT/row['archive'])
            data[split].append(dict(**row,x=x,valid=valid))
    for split,rows in manifest['isolated_replay'].items():
        data['isolated_'+split]=dict(x=np.stack([load_features(ROOT/r['path']) for r in rows]),
            y=np.array([r['target'] for r in rows]),sources=np.array([r['source'] for r in rows]))
    return manifest,data


def fresh():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    base_payload=torch.load(BASE,map_location='cpu',weights_only=False)
    base=_load_start(base_payload).base
    config=MT5Config.from_pretrained(str(TEXT_CONFIG),local_files_only=True)
    text=MT5ForConditionalGeneration(config)
    payload=torch.load(TEXT_SOURCE,map_location='cpu',weights_only=True)['model']
    state={k[len('mt5_model.'):]:v for k,v in payload.items() if k.startswith('mt5_model.')}
    assert state
    text.load_state_dict(state,strict=True)
    del payload,state,base_payload
    tokenizer=T5Tokenizer.from_pretrained(str(TEXT_CONFIG),local_files_only=True,legacy=False)
    prefix=tokenizer('Translate sign language video to English: ',return_tensors='pt')['input_ids']
    model=DirectTranslation(base,text,prefix).to(DEVICE)
    gc.collect()
    return model,tokenizer


def tensors(rows):
    return ([torch.from_numpy(r['x']).to(DEVICE) for r in rows],
            [torch.from_numpy(r['valid']).to(DEVICE) for r in rows])


def labels(tokenizer,rows):
    # Frozen data audit found a maximum of 70 tokens. Mask padding; never truncate.
    ids=tokenizer([r['reference'] for r in rows],padding='max_length',max_length=70,return_tensors='pt')['input_ids']
    assert ids.shape[1]==70, 'target exceeds frozen padding bound'
    ids[ids==tokenizer.pad_token_id]=-100
    return ids.to(DEVICE)


def optimizer_for(model):
    return Adafactor([{'params':model.base.parameters(),'lr':1e-5},
                      {'params':model.projection.parameters(),'lr':3e-4},
                      {'params':model.text.parameters(),'lr':1e-4}],
                     relative_step=False,scale_parameter=False,warmup_init=False,weight_decay=1e-4)


def release_unused_mps():
    if DEVICE!='mps':return {}
    torch.mps.synchronize()
    before=torch.mps.driver_allocated_memory()
    torch.mps.empty_cache()
    return dict(driver_before=before,driver_after=torch.mps.driver_allocated_memory(),
                live_bytes=torch.mps.current_allocated_memory())


def restore(model,optimizer):
    recovery=os.environ.get('SLT_RECOVERY_RECORD')
    if recovery:
        record=json.loads(Path(recovery).read_text())
        checkpoint=Path(record['checkpoint']);frozen=Path(record['source_manifest']);epoch=record['epoch']
        assert sha256(checkpoint)==record['checkpoint_sha256'], 'recovery checkpoint hash changed'
        assert sha256(frozen)==record['manifest_sha256'], 'recovery source manifest changed'
        assert json.loads(frozen.read_text())['input_hashes']==json.loads((HERE/'manifest.json').read_text())['input_hashes']
    else:
        checkpoint=MODELS/'epoch_01_before_resume.pth';frozen=HERE/'failed_attempt_01/manifest.json';epoch=1
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert payload['format']=='slt_stage1_direct_translation_v17' and payload['epoch']==epoch
    assert payload['manifest_sha256']==sha256(frozen) and payload['seed']==SEED
    model.load_state_dict(payload['state_dict'],strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    history=payload['history']
    assert 0<epoch<EPOCHS and [r['epoch'] for r in history]==list(range(1,epoch+1))
    del payload
    gc.collect();release_unused_mps()
    return history


def check_checkpoint_space(startup=False):
    root=Path(os.environ.get('SLT_CHECKPOINT_ROOT',str(ROOT/'artifacts/models')))
    volume=os.environ.get('SLT_CHECKPOINT_VOLUME')
    if volume and (not os.path.ismount(volume) or not root.resolve().is_relative_to(Path(volume).resolve())):
        raise RuntimeError('Checkpoint SSD is missing or output path is outside its mount')
    if not root.is_dir() or not MODELS.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Checkpoint root missing or model destination outside it')
    # Largest frozen checkpoint is 2.37 GB; 4 GiB leaves room for an atomic save.
    if shutil.disk_usage(root).free<(8 if startup else 4)*2**30:
        raise RuntimeError('Insufficient checkpoint space: need 8 GiB at startup, 4 GiB during training')
    if shutil.disk_usage(HERE).free<(8 if startup else 1)*2**30:
        raise RuntimeError('Insufficient internal space: need 8 GiB at startup, 1 GiB for running reports')


def memory_check():
    """Full-model joint updates at the longest admitted inputs; discard all updates."""
    torch.set_num_threads(4)
    _,data=verify_and_load();model,tokenizer=fresh();optimizer=optimizer_for(model)
    restore(model,optimizer);model.train()
    selected=sorted(data['train'],key=lambda r:(len(r['x']),len(r['reference'])),reverse=True)[:4]
    records=[]
    for step in range(4):
        batch=selected[2*(step%2):2*(step%2)+2];x,valid=tensors(batch)
        optimizer.zero_grad(set_to_none=True)
        loss=model.translation_loss(x,valid,labels(tokenizer,batch))
        replay=data['isolated_train']
        aux=torch.nn.functional.cross_entropy(model.base(torch.from_numpy(replay['x'][:4]).to(DEVICE)),
                                              torch.from_numpy(replay['y'][:4]).to(DEVICE))
        total=loss+.5*aux;total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        memory=release_unused_mps()
        optimizer.step()
        assert torch.isfinite(total)
        records.append(dict(step=step+1,loss=float(total.detach().cpu()),source_tokens=[len(r['x'])*32 for r in batch],
                            before_update=memory,after_update=release_unused_mps()))
        save('memory_check.json',dict(passed=len(records)==4,records=records,
            note='Real joint updates on training-only maximum-length inputs; discarded; resume still starts from saved epoch 1.'))
    emit({'memory_check_passed':True,'steps':len(records)})


@torch.no_grad()
def isolated(model,data):
    model.eval();d=data['isolated_validation'];pred=[]
    for i in range(0,len(d['y']),32):
        pred.extend(model.base(torch.from_numpy(d['x'][i:i+32]).to(DEVICE)).argmax(-1).cpu().tolist())
    pred=np.array(pred)
    return {source:dict(correct=int((pred[d['sources']==source]==d['y'][d['sources']==source]).sum()),
        total=int((d['sources']==source).sum()),accuracy=float((pred[d['sources']==source]==d['y'][d['sources']==source]).mean()))
        for source in sorted(set(d['sources']))}


@torch.no_grad()
def predict(model,tokenizer,rows,zero_visual=False):
    model.eval();output=[]
    for r in rows:
        x,valid=tensors([r]);begin=time.monotonic()
        tokens=model.translate(x,valid,zero_visual=zero_visual)
        text=tokenizer.batch_decode(tokens,skip_special_tokens=True)[0]
        output.append({**{k:v for k,v in r.items() if k not in {'x','valid'}},
            'prediction':text,'generation_seconds':time.monotonic()-begin})
    return output


def metrics(rows):
    from sacrebleu.metrics import BLEU,CHRF
    paired=[r for r in rows if r['reference'] is not None]
    pred=[r['prediction'] for r in paired];refs=[[r['reference'] for r in paired]]
    bleu=BLEU(tokenize='13a');chrf=CHRF()
    return dict(count=len(paired),bleu=bleu.corpus_score(pred,refs).score,
                chrf=chrf.corpus_score(pred,refs).score,
                bleu_signature=str(bleu.get_signature()),chrf_signature=str(chrf.get_signature()))


def preflight(model,tokenizer,data):
    small=sorted(data['train'],key=lambda r:(len(r['reference'].split()),r['item_id']))[:4]
    x,valid=tensors(small[:2]);target=labels(tokenizer,small[:2]);model.eval()
    before=float(model.translation_loss(x,valid,target).detach().cpu())
    optimizer=optimizer_for(model)
    history=[];gradient={}
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        batch=small[2*(step%2):2*(step%2)+2]
        bx,bv=tensors(batch)
        loss=model.translation_loss(bx,bv,labels(tokenizer,batch))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite real-model preflight loss')
        loss.backward()
        if step==0:
            for name,module in [('stage1',model.base),('projection',model.projection),('text',model.text)]:
                gradient[name]=sum(float(p.grad.abs().sum().cpu()) for p in module.parameters() if p.grad is not None)
            assert all(v>0 and math.isfinite(v) for v in gradient.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();history.append(float(loss.detach().cpu()))
    with torch.no_grad():after=float(model.translation_loss(x,valid,target).cpu())
    value=dict(before=before,after=after,history=history,gradient_l1=gradient,
        training_ids=[r['item_id'] for r in small],passed=after<before*.95,
        note='Train-only wiring/optimization check, not a semantic accuracy result; weights reset before main run.')
    save('preflight.json',value)
    if not value['passed']:raise RuntimeError('Tiny-fit loss did not decrease; no full training')
    del optimizer
    model.zero_grad(set_to_none=True)


def train(model,tokenizer,data,resume=False):
    check_checkpoint_space()
    MODELS.mkdir(parents=True,exist_ok=True)
    optimizer=optimizer_for(model);history=[]
    if resume:history=restore(model,optimizer)
    previous_seconds=history[-1]['elapsed_seconds'] if history else 0
    replay=data['isolated_train'];start=time.monotonic()
    for epoch in range(len(history)+1,EPOCHS+1):
        check_checkpoint_space()
        torch.manual_seed(SEED+epoch)
        warmup=epoch==1
        model.base.requires_grad_(not warmup);model.text.requires_grad_(not warmup)
        model.train()
        if warmup:model.base.eval();model.text.eval()
        batches=epoch_batches(len(data['train']),len(replay['y']),SEED+epoch)
        stats=dict(epoch=epoch,translation_loss_sum=0.,isolated_loss_sum=0.,steps=0,
                   translation_examples=0,isolated_examples=0,warmup=warmup)
        for translation_indices,replay_indices in batches:
            batch=[data['train'][i] for i in translation_indices];x,valid=tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            loss=model.translation_loss(x,valid,labels(tokenizer,batch))
            aux=torch.zeros((),device=DEVICE)
            if replay_indices:
                rx=torch.from_numpy(replay['x'][replay_indices]).to(DEVICE)
                ry=torch.from_numpy(replay['y'][replay_indices]).to(DEVICE)
                aux=torch.nn.functional.cross_entropy(model.base(rx),ry)
            total=loss+.5*aux
            if not torch.isfinite(total):raise RuntimeError('Nonfinite joint training loss')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            release_unused_mps()
            optimizer.step()
            stats['translation_loss_sum']+=float(loss.detach().cpu())
            stats['isolated_loss_sum']+=float(aux.detach().cpu())
            stats['steps']+=1;stats['translation_examples']+=len(batch);stats['isolated_examples']+=len(replay_indices)
            release_unused_mps()
        assert stats['translation_examples']==len(data['train']) and stats['isolated_examples']==len(replay['y'])
        stats['translation_loss']=stats['translation_loss_sum']/stats['steps']
        stats['isolated_loss']=stats['isolated_loss_sum']/stats['steps']
        stats['elapsed_seconds']=previous_seconds+time.monotonic()-start
        history.append(stats)
        checkpoint=MODELS/'latest.pth';tmp=MODELS/'latest.filehandle.tmp.pth'
        model.zero_grad(set_to_none=True)
        check_checkpoint_space()
        # Direct filename writes failed on this ExFAT SSD; the full-size file-handle probe passed.
        with tmp.open('wb') as stream:
            torch.save(dict(format='slt_stage1_direct_translation_v17',epoch=epoch,
                base_config=asdict(model.base.config),text_config=model.text.config.to_dict(),
                prefix_ids=model.prefix_ids.cpu(),state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),seed=SEED,
                manifest_sha256=sha256(HERE/'manifest.json'),history=history),stream)
            stream.flush();os.fsync(stream.fileno())
        tmp.replace(checkpoint)
        save('training_history.json',history);emit(stats)
    return checkpoint,history


def write_report(summary,initial,final,control,retention_before,retention_after):
    old={r['item_id']:r for r in json.loads((PRIOR/'results.json').read_text())}
    initial_map={r['item_id']:r['prediction'] for r in initial}
    zero_map={r['item_id']:r['prediction'] for r in control}
    lines=['# Stage 1 direct-translation experiment','',
        '**Completed:** our Stage 1 encoder -> projection -> mT5, with isolated classification retained. No CTC or repeat suppression.','',
        '| System | BLEU | chrF |','|---|---:|---:|']
    for name,score in summary['translation'].items():
        lines.append(f"| {name} | {score['bleu']:.2f} | {score['chrf']:.2f} |")
    lines+=['','Scores cover the same 12 paired development utterances. Nine short/webcam recordings remain qualitative; their gloss labels are not English reference translations.','',
        '## Isolated-sign retention','','| Source | Before | After |','|---|---:|---:|']
    for name in retention_before:
        a,b=retention_before[name],retention_after[name]
        lines.append(f"| {name} ({a['total']} clips) | {100*a['accuracy']:.2f}% | {100*b['accuracy']:.2f}% |")
    lines+=['','These are existing development subsets, not official Citizen test accuracy.','',
        '## Interpretation','',summary['decision'],'',
        'The zero-visual control preserves sequence lengths and masks but replaces projected motion features with zeros. Similar outputs/scores would weaken evidence that translation uses the signing. Repetition is not an automatic rejection gate.','',
        'Training adaptation excludes evaluation signers 1 and 2; inherited mT5 pretraining is not certified signer-disjoint. The checkpoint is initialized from the released Uni-Sign ASL text component, with our own visual encoder. This does not reproduce Uni-Sign’s pose encoder or constitute a fresh large-scale pretraining run.','',
        'Evaluation is at utterance completion. No streaming, iPhone, or unseen-domain accuracy claim. No live defaults changed.','',
        '## Predictions and references','']
    for row in final:
        ident=row['item_id']
        lines += [f'### {ident}','',f"Reference: {row['reference'] if row['reference'] is not None else 'No verified English reference'}",'',
                  f"Before training: {initial_map[ident]}",'',f"After training: {row['prediction']}",'',
                  f"Unchanged Uni-Sign: {old[ident]['prediction']}",'']
        if ident in zero_map:lines += [f'Zero visual features: {zero_map[ident]}','']
        lines += [f"[Video]({row['video']})",'']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')


def work(resume=False):
    started=datetime.now(timezone.utc).isoformat();status='failed'
    try:
        torch.set_num_threads(4)
        check_checkpoint_space(startup=True)
        save('status.json',dict(stage='verifying frozen inputs'))
        manifest,data=verify_and_load()
        save('status.json',dict(stage='loading model'))
        model,tokenizer=fresh()
        provenance=dict(stage1_checkpoint=str(BASE),stage1_sha256=manifest['base_sha256'],
            text_checkpoint=str(TEXT_SOURCE),text_sha256=manifest['text_sha256'],
            text_initialization='strict complete mt5_model.* state from released How2Sign Uni-Sign',
            manifest_sha256=sha256(HERE/'manifest.json'),device=DEVICE,dtype='float32',
            parameters=sum(p.numel() for p in model.parameters()),seed=SEED,epochs=EPOCHS,
            torch=torch.__version__,protected_test_accessed=False,counts=manifest['counts'])
        recovery=os.environ.get('SLT_RECOVERY_RECORD')
        provenance['resumed_from_epoch']=(json.loads(Path(recovery).read_text())['epoch'] if recovery else 1) if resume else None
        provenance['checkpoint_root']=str(MODELS)
        if recovery:provenance['recovery_record_sha256']=sha256(Path(recovery))
        save('provenance.json',provenance)
        rows=data['validation']+data['diagnostics']
        if resume:
            assert json.loads((HERE/'memory_check.json').read_text())['passed']
            initial=json.loads((HERE/'initial_predictions.json').read_text())
            retention_before=json.loads((HERE/'initial_retention.json').read_text())
        else:
            save('status.json',dict(stage='train-only gradient and tiny-fit preflight'))
            preflight(model,tokenizer,data)
            del model;gc.collect()
            if DEVICE=='mps':torch.mps.empty_cache()
            model,tokenizer=fresh()
            save('status.json',dict(stage='initialized hybrid evaluation'))
            initial=predict(model,tokenizer,rows);save('initial_predictions.json',initial)
            retention_before=isolated(model,data);save('initial_retention.json',retention_before)
        save('status.json',dict(stage='20-epoch training'))
        checkpoint,history=train(model,tokenizer,data,resume=resume)
        save('status.json',dict(stage='final evaluation and visual control'))
        final=predict(model,tokenizer,rows);save('final_predictions.json',final)
        control=predict(model,tokenizer,data['validation'],zero_visual=True);save('zero_visual_predictions.json',control)
        retention_after=isolated(model,data);save('final_retention.json',retention_after)
        before,after,zero=metrics(initial),metrics(final),metrics(control)
        unchanged=json.loads((PRIOR/'summary.json').read_text())
        improved=after['chrf']>before['chrf'] and after['bleu']>before['bleu']
        grounded=after['chrf']>zero['chrf']
        retention=all(retention_after[k]['accuracy']>=retention_before[k]['accuracy']-.02 for k in retention_before)
        decision=('The pilot improves both paired text metrics over its initialization and beats the zero-visual chrF control, while isolated accuracy stays within two percentage points. This supports a larger independent evaluation, not deployment.'
                  if improved and grounded and retention else
                  'The pilot does not satisfy all three descriptive checks: improvement over initialization, higher chrF than the zero-visual control, and isolated retention within two percentage points. Inspect the outputs and training fit before deciding on another run; no automatic promotion.')
        summary=dict(translation={'initialized hybrid':before,'trained hybrid':after,'zero-visual control':zero,
                        'unchanged Uni-Sign':{k:unchanged[k] for k in ['bleu','chrf']}},
            improved_over_initial=improved,beats_zero_visual_chrf=grounded,isolated_retention_within_2pp=retention,
            decision=decision,final_epoch=EPOCHS,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            training_seconds=history[-1]['elapsed_seconds'],repetition_is_a_gate=False)
        save('summary.json',summary)
        write_report(summary,initial,final,control,retention_before,retention_after)
        status='completed'
    except Exception:
        (HERE/'FAILURE.md').write_text('# Direct-translation experiment failed\n\nNo model promotion.\n\n```\n'+traceback.format_exc()+'```\n')
        traceback.print_exc()
    finally:
        finish(status,started)
    return 0 if status=='completed' else 1


def finish(status,started):
    save('completion.json',dict(status=status,started_at=started,finished_at=datetime.now(timezone.utc).isoformat()))
    save('status.json',dict(stage=status))
    report='REPORT.md' if status=='completed' else 'FAILURE.md'
    message=f'Stage 1 direct translation {status}. See {HERE.name}/{report}'
    history_path=ROOT/'docs/ground_truth/live-streaming/log.md'
    entry=f'\n## {datetime.now().strftime("%Y-%m-%d")} — direct-translation worker exit\n\n{message}. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.\n'
    history_path.write_text(history_path.read_text().replace('\n---\n','\n---\n'+entry,1))
    ground=ROOT/'PROJECT_GROUND_TRUTH.md'
    ground.write_text(ground.read_text().replace('Direct-translation experiment status: running.',
        f'Direct-translation experiment status: {status}; review its {report}.'))
    subprocess.run([str(ROOT/'venv/bin/python'),str(ROOT/'scripts/index_large_artifacts_v17.py')],cwd=ROOT,check=False)
    subprocess.run(['osascript','-e','on run argv\ndisplay notification (item 1 of argv) with title "SLT experiment"\nend run',message],capture_output=True,timeout=15,check=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--supervise',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--memory-check',action='store_true')
    args=parser.parse_args()
    if args.memory_check:memory_check()
    elif args.prepare:prepare()
    elif args.launch:
        assert (HERE/'manifest.json').exists()
        recovery=bool(os.environ.get('SLT_RECOVERY_RECORD'))
        launch_name='ssd_resume_launch.json' if recovery else ('resume_launch.json' if args.resume else 'launch.json')
        check_checkpoint_space(startup=True)
        if args.resume:
            assert not (HERE/launch_name).exists()
            assert json.loads((HERE/'memory_check.json').read_text())['passed']
            assert json.loads((HERE/'completion.json').read_text())['status']=='failed'
            for name in ['completion.json','FAILURE.md']:
                (HERE/name).rename(HERE/('failed_attempt_02' if recovery else 'failed_attempt_01')/('at_resume_'+name))
        with (HERE/launch_name).open('x') as record,(HERE/'process.log').open('a') as log:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-u',__file__,'--supervise']+(['--resume'] if args.resume else []),cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            json.dump(dict(pid=child.pid,launched_at=datetime.now(timezone.utc).isoformat()),record)
        ground=ROOT/'PROJECT_GROUND_TRUTH.md'
        ground.write_text(ground.read_text().replace('Direct-translation experiment status: prepared.',
                                                   'Direct-translation experiment status: running.'))
        print('Launched',child.pid,'with exit reporting and notification; no polling.')
    elif args.supervise:
        started=datetime.now(timezone.utc).isoformat()
        # One OS-level wait, no polling. Catch fatal signals that bypass Python finally.
        result=subprocess.run([sys.executable,'-u',__file__]+(['--resume'] if args.resume else []),cwd=ROOT)
        if not (HERE/'completion.json').exists():
            (HERE/'FAILURE.md').write_text(f'# Worker terminated\n\nExit code: {result.returncode}. See process.log and the last saved status/checkpoint. No model promotion.\n')
            finish('failed',started)
        sys.exit(result.returncode)
    else:sys.exit(work(resume=args.resume))
