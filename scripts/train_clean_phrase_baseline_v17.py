"""Two-seed frozen-Stage1 baseline using only the approved 283/211 phrase split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from active.v17.train_unified_streaming_ctc_v17 import phrase_sequences, encode, Sequences, collate, collapse_ctc
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.model_unified_streaming_ctc_v17 import UnifiedStreamingCTCConfig, UnifiedStreamingCTCHeadV17
from scripts.train_youtube_motion_pilot_v17 import align_tokens

REPORT=ROOT/'artifacts/reports/clean_phrase_baseline_v17_20260921'
MODELS=ROOT/'artifacts/models/clean_phrase_baseline_v17_20260921'
DATA=ROOT/'active/v17/approved_phrase_manifest_20260921_v2.json'
RUN_MANIFEST=ROOT/'active/v17/clean_phrase_baseline_manifest_20260921.json'
BASE=ROOT/'artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/best_model.pth'
RECIPE=dict(seeds=[17321,17322],epochs=18,batch_size=16,learning_rate=.002,weight_decay=.0001,
            rolling_stride=4,window_frames=8,device='mps',ctc_device='cpu',mps_memory_fraction=.35,threads=2,hidden_dim=128,blocks=3,dropout=.1,
            objective='mean per-sequence CTC loss; shuffled clips once per epoch; no source reweighting',
            selection='mean local/ASLLRP-contiguous known WER; earliest epoch on tie',
            stage1_frozen=True,isolated_replay=False,standalone_blank=False,alignment_loss=False)


def save(name,value):
    path=REPORT/name; temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def summarize(rows):
    by_source={}
    for row in rows:
        b=by_source.setdefault(row['source'],dict(samples=0,exact=0,blank_only=0,other_emitted=0,
                                                substitutions=0,deletions=0,insertions=0,known_tokens=0))
        expected=[t for t in row['expected'] if t!=101]; predicted=[t for t in row['predicted'] if t!=101]
        edits,_=align_tokens(expected,predicted)
        b['samples']+=1;b['exact']+=row['expected']==row['predicted'];b['blank_only']+=not row['predicted']
        b['other_emitted']+=101 in row['predicted'];b['known_tokens']+=len(expected)
        for key,value in edits.items():b[key]+=value
    total={k:sum(b[k] for b in by_source.values()) for k in next(iter(by_source.values()))}
    for b in list(by_source.values())+[total]:
        b['known_wer']=(b['substitutions']+b['deletions']+b['insertions'])/max(b['known_tokens'],1)
        b['exact_accuracy']=b['exact']/b['samples'];b['blank_only_rate']=b['blank_only']/b['samples']
    return dict(by_source=by_source,overall=total,
                selection_score=sum(by_source[s]['known_wer'] for s in ('local_phrases','asllrp_contiguous'))/2)


@torch.inference_mode()
def evaluate(model,samples):
    model.eval();rows=[]
    for i in range(0,len(samples),RECIPE['batch_size']):
        b=collate(samples[i:i+RECIPE['batch_size']]);paths=model(b['evidence'].to('mps')).argmax(-1).cpu().numpy()
        for sample,length,path in zip(b['samples'],b['lengths'],paths):
            rows.append(dict(identity=sample.identity,source=sample.source,expected=list(sample.targets),
                             predicted=list(collapse_ctc(path[:int(length)]))))
    return summarize(rows),rows


def prepare():
    if RUN_MANIFEST.exists() or (REPORT/'evidence.pt').exists():raise FileExistsError('prepared baseline already exists')
    REPORT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(RECIPE['threads'])
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required by user')
    torch.mps.set_per_process_memory_fraction(RECIPE['mps_memory_fraction'])
    provenance=verify_manifest(DATA);m=json.loads(DATA.read_text())
    base=torch.load(BASE,map_location='cpu',weights_only=False)
    labels=base['label_to_index'];stage1=SLTStage1V17(Stage1V17Config(**base['model_config']))
    stage1.load_state_dict(base['model_state_dict']);stage1.requires_grad_(False);stage1.eval()
    samples={};start=time.monotonic()
    for role in ('train','validation'):
        raw=[]
        for path in m['roots'].values():raw+=phrase_sequences(ROOT/path,role,4,8,labels=labels)
        expected={r['identity'] for r in m['admitted'] if r['role']==role}
        if len(raw)!=len(expected) or {r.identity for r in raw}!=expected:raise ValueError('admitted identity mismatch')
        samples[role]=encode(stage1,raw,torch.device('mps'),16,'window')
        torch.mps.empty_cache()
    assert len(samples['train'])==283 and len(samples['validation'])==211
    config=UnifiedStreamingCTCConfig(stage1_dim=stage1.config.dim)
    probe=UnifiedStreamingCTCHeadV17(config).to('mps').eval()
    b=collate(samples['train'][:4])
    with torch.no_grad():
        logits=probe(b['evidence'].to('mps')).cpu()
        loss=torch.nn.CTCLoss(blank=0,zero_infinity=False)(logits.log_softmax(-1).transpose(0,1),b['targets'],b['lengths'],b['target_lengths'])
        if not torch.isfinite(loss):raise ValueError('nonfinite preflight CTC')
    cache=REPORT/'evidence.pt';torch.save(dict(samples=samples,head_config=config.to_dict(),labels=labels),cache)
    code=[Path(__file__),ROOT/'active/v17/model_unified_streaming_ctc_v17.py',
          ROOT/'active/v17/train_unified_streaming_ctc_v17.py',ROOT/'scripts/train_youtube_motion_pilot_v17.py',
          ROOT/'active/v17/model_streaming_stage1_head_v17.py',ROOT/'active/v17/train_streaming_tcn_ctc_v17.py']
    manifest={**m,'version':3,'training_ready':True,'blockers':[],
              'training_entrypoint':'scripts/train_clean_phrase_baseline_v17.py','recipe':RECIPE,
              'scope':'Phrase-only diagnostic baseline; not UNKNOWN/live/mobile readiness. General recipes remain blocked.',
              'base_checkpoint':str(BASE.relative_to(ROOT)),'base_checkpoint_sha256':digest(BASE),
              'evidence_cache':str(cache.relative_to(ROOT)),
              'evidence_sha256':{**m['evidence_sha256'],str(DATA.relative_to(ROOT)):digest(DATA),
                                 str(cache.relative_to(ROOT)):digest(cache),str(BASE.relative_to(ROOT)):digest(BASE),
                                 **{str(p.resolve().relative_to(ROOT)):digest(p) for p in code}}}
    RUN_MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n');verified=verify_manifest(RUN_MANIFEST)
    save('preflight.json',dict(status='passed',manifest=verified,counts={k:len(v) for k,v in samples.items()},
                              by_source={k:dict(Counter(s.source for s in v)) for k,v in samples.items()},
                              finite_forward_ctc=float(loss),optimizer_steps=0,stage1_frozen=True,
                              base_sha256=digest(BASE),encoding_seconds=time.monotonic()-start,test_accessed=False))
    print(json.dumps(dict(status='prepared',counts={k:len(v) for k,v in samples.items()},seconds=time.monotonic()-start)),flush=True)


def train():
    provenance=verify_manifest(RUN_MANIFEST);m=json.loads(RUN_MANIFEST.read_text())
    if m.get('training_entrypoint')!='scripts/train_clean_phrase_baseline_v17.py' or m['recipe']!=RECIPE or not m['training_ready']:
        raise ValueError('recipe authorization mismatch')
    if MODELS.exists():raise FileExistsError(MODELS)
    MODELS.mkdir(parents=True);torch.set_num_threads(RECIPE['threads'])
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required by user')
    torch.mps.set_per_process_memory_fraction(RECIPE['mps_memory_fraction'])
    cache=torch.load(ROOT/m['evidence_cache'],map_location='cpu',weights_only=False)
    train_samples,validation=cache['samples']['train'],cache['samples']['validation']
    if (len(train_samples),len(validation))!=(283,211):raise ValueError('cache count mismatch')
    results={};start=time.monotonic()
    for seed in RECIPE['seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=UnifiedStreamingCTCHeadV17(UnifiedStreamingCTCConfig(**cache['head_config'])).to('mps')
        optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE['learning_rate'],weight_decay=RECIPE['weight_decay'])
        loader=DataLoader(Sequences(train_samples),batch_size=RECIPE['batch_size'],shuffle=True,collate_fn=collate)
        ctc=torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=False)
        history=[];best=None
        initial,_=evaluate(model,validation)
        for epoch in range(1,RECIPE['epochs']+1):
            model.train();total=0
            for b in loader:
                logits=model(b['evidence'].to('mps'));loss=ctc(logits.cpu().log_softmax(-1).transpose(0,1),b['targets'],b['lengths'],b['target_lengths']).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite training CTC')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
                total+=float(loss.detach())*len(b['samples'])
            metrics,_=evaluate(model,validation)
            row=dict(seed=seed,epoch=epoch,loss=total/len(train_samples),metrics=metrics);history.append(row)
            save('status.json',dict(state='training',seed=seed,epoch=epoch));print(json.dumps(row),flush=True)
            if best is None or metrics['selection_score']<best['score']:
                best=dict(epoch=epoch,score=metrics['selection_score'],state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        model.load_state_dict(best['state']);metrics,predictions=evaluate(model,validation)
        verify_manifest(RUN_MANIFEST)
        output=MODELS/f'seed_{seed}.pth'
        torch.save(dict(format='slt_unified_streaming_ctc_v17',version=1,head_config=cache['head_config'],
                        head_state_dict=best['state'],base_checkpoint=str(BASE),base_checkpoint_sha256=m['base_checkpoint_sha256'],
                        label_to_index=cache['labels'],ctc_blank_index=0,other_index=101,other_label='__OTHER__',
                        dataset_provenance=provenance,recipe=RECIPE,seed=seed,selected_epoch=best['epoch'],validation=metrics,test_accessed=False),output)
        results[str(seed)]=dict(checkpoint=str(output.relative_to(ROOT)),checkpoint_sha256=digest(output),
                               selected_epoch=best['epoch'],initial_validation=initial,validation=metrics,
                               predictions=predictions,history=history)
        save('results.json',dict(results=results,manifest=provenance,elapsed_seconds=time.monotonic()-start,test_accessed=False))
    save('status.json',dict(state='complete',seeds=RECIPE['seeds'],elapsed_seconds=time.monotonic()-start))
    lines=['# Clean phrase baseline results','', 'Reused development validation only; no UNKNOWN, test or live-stream accuracy claim.','']
    for seed,r in results.items():
        q=r['validation']['overall'];lines.append(f"- Seed {seed}, epoch {r['selected_epoch']}: known WER {q['known_wer']:.2%}, exact {q['exact']}/{q['samples']}, deletions {q['deletions']}, blank-only {q['blank_only']}.")
    lines+=['','See results.json for source-level metrics, predictions, initial-head diagnostics and both histories. No automatic promotion.']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare();return
    state='failed'
    try:
        train();state='complete'
    except Exception:
        save('status.json',dict(state='failed',traceback=traceback.format_exc()));raise
    finally:
        message='SLT clean phrase baseline '+state+'. See '+str(REPORT/'status.json')
        n=subprocess.run(['/usr/bin/osascript','-e','display notification '+json.dumps(message)+' with title "SLT baseline"'],capture_output=True,text=True)
        save('notification.json',dict(state=state,returncode=n.returncode,stderr=n.stderr))


if __name__=='__main__':main()
