from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent
s=(D/'train_actual_balanced.py').read_text().replace('balanced_training_recipe.json','restored_training_recipe.json').replace('actual_window_balanced_head','historical_MY_NAME_recovery_head')
start=s.index(" with np.load(D/'actual_features.npz') as d:");end=s.index('\n assert len(py)',start)
s=s[:start]+''' with np.load(D/'restored_features.npz') as d:
  train_ids=d['item_ids'];phone=tuple(torch.tensor(d[f'f{i}'],device=device) for i in range(4));py=torch.tensor(d['targets'],device=device)
 with np.load(D/'actual_features.npz') as d:
  dev_ids=d['item_ids'];dev=tuple(torch.tensor(d[f'f{i}'],device=device) for i in range(4));dy=torch.tensor(d['targets'])'''+s[end:]
s=s.replace("('balanced_selected',best),('balanced_diagnostic',diagnostic)","('restored_selected',best),('restored_diagnostic',diagnostic)").replace('balanced_training_results.json','restored_training_results.json')
s=s.replace('Experimental actual-window repair; frozen encoders, balanced full fusion-head adaptation.','Historical MY NAME recovery control; eight authorized sources, frozen encoders.')
(D/'train_restored.py').write_text(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((D/'balanced_training_recipe.json').read_text());c=json.loads((D/'restored_input_contract.json').read_text());r.update(format='slt_authorized_historical_MY_NAME_recipe_v1',scope='historical_MY_NAME_recovery_head',seeds=[27931],phone_shares=[.1,.3],training_events=[x['source_item_id'] for x in c['sources']],development_events='All89knownphonewindows; none used in this arm training.',input_contract='Exactly148cached non-prefix spans (84MY,64NAME) from the eight formerly excluded canonical-approved MY NAME sources, explicitly authorized for training reuse by user. Full input-source/hash audit preserved. No other old split membership restored. All89phonewindows used for development only.',negative_sampling='Same45% general/45% LIKE isolated/10% admitted temporalLIKE replay as balanced experiment; historical MY NAME replaces phone-positive branch at10% or30%.',selection='Per-class isolated retention required, then best89knownphonewindow accuracy/loss. This control tests restoration alone, no phone gradients. Reusedhistoricalvalidation is nowtraining/development, neverfreshaccuracy.')
r['pins'].pop(str(D/'train_actual_balanced.py'));r['pins'][str(D/'train_restored.py')]=sha(D/'train_restored.py');r['pins'].update(c['pins']);r['pins'][str(D/'restored_input_contract.json')]=sha(D/'restored_input_contract.json');r['pins'][str(D/'encode_restored.py')]=sha(D/'encode_restored.py')
(D/'restored_training_recipe.json').write_text(json.dumps(r,indent=2))
w=(D/'balanced_worker.py').read_text().replace('train_actual_balanced.py','train_restored.py').replace('balanced_training.log','restored_training.log').replace('balanced_completion.json','restored_completion.json').replace('Balanced-head MY recovery','Historical-source MY recovery');(D/'restored_worker.py').write_text(w)
print('Prepared six historical-source-only head controls')
