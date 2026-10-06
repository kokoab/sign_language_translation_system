"""Independent arithmetic, input membership, checkpoint and prediction review."""
import hashlib, json, statistics, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/Volumes/secret/SLT/SLT'); sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent; D=HERE/'downstream_recipe'
from scripts import segmental_lab_v17 as lab
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_unfrozen_phrase_adapt_v17 import tensors

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

torch.set_num_threads(1)
audit=read(D/'audit.json'); recipe=read(D/'recipe.json')
m=read(ROOT/recipe['recipe_manifest']); canonical=read(ROOT/m['canonical_manifest'])
assert sha(ROOT/recipe['recipe_manifest'])==recipe['recipe_sha256']
assert sha(ROOT/m['canonical_manifest'])==m['canonical_sha256']
assert canonical['training_ready'] is False
roles={e['video_sha256']:e['role'] for e in canonical['admitted'] if e['source']=='local_phrases'}
ids={s:{e['video_sha256'] for e in m['entries'] if e['split']==s} for s in ('train','validation')}
test={r['video_sha256'] for r in lab.rows_for('test')}
tune={r['video_sha256'] for r in lab.rows_for('tune')}
assert not ids['train']&ids['validation'] and not (ids['train']|ids['validation'])&test
assert not ids['train']&tune
for e in m['entries']:
    assert roles[e['video_sha256']]==e['split']
    for k in ('rgb','hand'): assert sha(ROOT/e[k])==e[k+'_sha256']
assert len(ids['train'])==179 and len(ids['validation'])==139
N={'citizen':378,'semlex':978,'local':2896}
summary={'manifest_membership_hashes_verified':True,'canonical_training_ready':False,
         'protected_test_features_accessed':False,'phase_arithmetic':[], 'phone':{},'fresh_predictions':{}}
for r in audit['phase_table']:
    v=[100*r[k]/n for k,n in N.items()]
    summary['phase_arithmetic'].append(dict(chain=r['chain'],phase=r['phase'],
         scores=v,mean=sum(v)/3,pooled=100*sum(r[k] for k in N)/sum(N.values())))
for conf in sorted({r['configuration'] for r in audit['phone']}):
    rr=[r for r in audit['phone'] if r['configuration']==conf]
    for r in rr:
        assert len(r['samples_ms'])==r['frames']
        assert sorted(r['samples_ms'])[len(r['samples_ms'])//2]==r['median_ms']
        assert r['thermal_before']==r['thermal_after']==0 and not r['low_power']
    summary['phone'][conf]=statistics.median(r['median_ms'] for r in rr)
sel=read(HERE/'chain_9683_floor361_mildroll/selection.json')['candidates'][0]
paths=[('isolated',ROOT/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'),
       ('local',Path(sel['path'])),('fusion',HERE/'chain_9683_floor361_mildroll/fusion/best_model.pth'),
       ('phrase',D/'chain_9683/phrase_adapt/best_model.pth'),('recognizer',D/'chain_9683/span_recognizer/best_model.pth'),
       ('august_clean_recognizer',D/'august/span_recognizer/best_model.pth')]
device=torch.device('mps'); models={}
for name,p in paths:
    if name in ('isolated','local'):
        c=torch.load(p,map_location='cpu',weights_only=False)
        model=SLTStage1V17(Stage1V17Config(**c['model_config']))
        model.load_state_dict(c['model_state_dict'],strict=True)
    else: model,c=load_model(p)
    models[name]=model.eval().to(device)
    summary['fresh_predictions'][name]={'checkpoint_sha256':sha(p),'domains':{}}
with torch.inference_mode():
    for domain,n in N.items():
        p=ROOT/'artifacts/generated/unfrozen_phrase_adapt_v17'/f'{domain}_val.npz'
        with np.load(p,allow_pickle=False) as z: values=tensors({k:z[k] for k in z.files})
        assert len(values[4])==n
        for j,(name,model) in enumerate(models.items()):
            preds=[]
            for start in range(0,n,32):
                inputs=[v[start:start+32].to(device) for v in values[:4]]
                inputs[1]=inputs[1].float()
                logits=model(inputs[0]) if name in ('isolated','local') else model(*inputs)
                assert torch.isfinite(logits).all()
                preds.extend(logits.argmax(1).cpu().tolist())
            correct=int((np.asarray(preds)==values[4].numpy()).sum())
            expected=audit['phase_table'][j][domain] if j<5 else audit['chains']['august']['recognizer']['domains'][domain]
            assert correct==expected,(name,domain,correct,expected)
            summary['fresh_predictions'][name]['domains'][domain]={'correct':correct,'top1':100*correct/n,'input_sha256':sha(p)}
            print(name,domain,correct,flush=True)
out=HERE/'paper_claim_verification.json'; out.write_text(json.dumps(summary,indent=2)+'\n')
print('PASS',out,flush=True)
