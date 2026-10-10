from pathlib import Path
import json,hashlib
import numpy as np
D=Path(__file__).resolve().parent
with np.load(D/'actual_features.npz') as a,np.load(D/'successor_features.npz') as b:
 np.savez(D/'paired_features.npz',**{k:np.concatenate([a[k],b[k]]) for k in ['f0','f1','f2','f3','targets','item_ids','baseline']})
s=(D/'train_actual_balanced.py').read_text().replace('balanced_training_recipe.json','paired_training_recipe.json').replace('actual_window_balanced_head','paired_runtime_window_head').replace('actual_features.npz','paired_features.npz')
s=s.replace("'regressed_classes':[labels[i] for i in bad]}","'regressed_classes':[labels[i] for i in bad],'previously_correct_lost':int(((baseline[name]==t)&(pred!=t)).sum())};eligible &= not bool(((baseline[name]==t)&(pred!=t)).any())")
s=s.replace("pred[np.flatnonzero(dev_ids==e)]==my","pred[np.flatnonzero(dev_ids==e)]==dy[np.flatnonzero(dev_ids==e)]")
s=s.replace('return dict(epoch=epoch,arm=arm,eligible=bool(eligible),','return dict(epoch=epoch,arm=arm,eligible=bool(eligible),head_drift=float(sum((head.state_dict()[k]-v).square().mean() for k,v in initial.items())),')
s=s.replace("score=lambda z:(z['dev_correct'],-z['dev_loss'])","score=lambda z:(z['dev_correct'],-z['head_drift'])")
s=s.replace("('balanced_selected',best),('balanced_diagnostic',diagnostic)","('paired_selected',best),('paired_diagnostic',diagnostic)").replace('balanced_training_results.json','paired_training_results.json')
(D/'train_paired.py').write_text(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((D/'balanced_training_recipe.json').read_text());r.update(format='slt_paired_runtime_window_recipe_v3',scope='paired_runtime_window_head',training_events=['MY_FRIEND_1','MY_NAME_1','MY_EAT','FRIEND_1','NAME_1','EAT_1'],input_contract='220 actual current-runtime candidate inputs:89MY and131successors. Three physical pairs train (MY FRIEND1, MY NAME1, MY EAT); remaining five pairs develop/select. Both phase views grouped byphysicalevent. Successor labels manuallyreviewedfromsourcePTSframes, notmodelpseudo-labels. Sourcevideo andcanonicalmanifest remainunchanged.',selection='No formerly correct isolated validation example may be lost in any domain, in addition to per-class count gate. Maximize correct windows on other five known pairs; break ties by smallest head-parameter movement, avoiding unnecessary overconfidence. All knownphoneexamples remain development, not independentaccuracy.')
r['pins'].pop(str(D/'train_actual_balanced.py'))
for p in [D/'train_paired.py',D/'paired_features.npz',D/'successor_features.npz',D/'successor_metadata.json',D/'successor_review.json',D/'successor_cases.json',D/'encode_successor.py',D/'successor_dump.json']+list((D/'successor_windows').glob('*.bin'))+list((D/'successor_dump_native').glob('*.swift')):r['pins'][str(p)]=sha(p)
(D/'paired_training_recipe.json').write_text(json.dumps(r,indent=2))
w=(D/'balanced_worker.py').read_text().replace('train_actual_balanced.py','train_paired.py').replace('balanced_training.log','paired_training.log').replace('balanced_completion.json','paired_completion.json').replace('Balanced-head MY recovery','Paired-window MY recovery');(D/'paired_worker.py').write_text(w)
print('Prepared12paired-windowtrainingarms;6train-events/10development-events')
