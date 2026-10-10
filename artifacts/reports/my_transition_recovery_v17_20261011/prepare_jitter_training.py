from pathlib import Path
import json,hashlib,numpy as np
D=Path(__file__).resolve().parent
with np.load(D/'paired_features.npz') as a,np.load(D/'jitter_features.npz') as b:
 np.savez(D/'jitter_paired_features.npz',**{k:np.concatenate([a[k],b[k]]) for k in ['f0','f1','f2','f3','targets','item_ids','baseline']})
s=(D/'train_paired.py').read_text().replace('paired_training_recipe.json','jitter_training_recipe.json').replace('paired_runtime_window_head','jittered_runtime_window_head').replace('paired_features.npz','jitter_paired_features.npz').replace('paired_selected','jitter_selected').replace('paired_diagnostic','jitter_diagnostic').replace('paired_training_results.json','jitter_training_results.json')
s=s.replace('head_drift=float(sum(', 'worst_dev_fraction=min(events[e]/sum(dev_ids==e) for e in events),head_drift=float(sum(').replace("score=lambda z:(z['dev_correct'],-z['head_drift'])","score=lambda z:(z['worst_dev_fraction'],z['dev_correct'],-z['head_drift'])")
(D/'train_jitter.py').write_text(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.load(open(D/'paired_training_recipe.json'));r.update(format='slt_jittered_runtime_window_recipe_v4',scope='jittered_runtime_window_head',input_contract='598 actual runtime candidate inputs from the same reviewed eight pairs:220 original zero-capture-offset windows plus378 from1/60and1/30s capture offsets, auxiliary phases0and4. Three physical pairs(223windows) train; otherfive(375windows) develop/select. This is correlated temporal augmentation, not new clips/signers. Frozen encoders, original replay/LIKE negatives and per-example isolated retention unchanged.',selection='Retain every formerly correct isolated item in all three domains. Maximize worst development-event correct fraction, then total development correctness, then minimum head movement. No source video or class membership changes; known phone data remains development.')
r['pins'].pop(str(D/'train_paired.py'))
for p in [D/'train_jitter.py',D/'jitter_paired_features.npz',D/'jitter_features.npz',D/'jitter_metadata.json',D/'jitter_dump_cases.json',D/'jitter_dump.json',D/'encode_jitter.py',D/'jitter_canonical_verifier.log']+list((D/'jitter_windows').glob('*.bin')):r['pins'][str(p)]=sha(p)
(D/'jitter_training_recipe.json').write_text(json.dumps(r,indent=2))
s=(D/'paired_worker.py').read_text().replace('train_paired.py','train_jitter.py').replace('paired_training.log','jitter_training.log').replace('paired_completion.json','jitter_completion.json').replace('Paired-window MY recovery','Timing-augmented MY recovery');(D/'jitter_worker.py').write_text(s)
print('Prepared12timing-augmentedarms:223trainwindows,375developmentwindows; physical event split unchanged')
