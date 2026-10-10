from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;P=D.parent/'temporal_repair_training_v17_20261011'
s=(D/'train_actual.py').read_text().replace("actual_training_recipe.json","balanced_training_recipe.json").replace("actual_window_MY_LIKE_output_rows","actual_window_balanced_head")
s=s.replace("parts=[read(n) for n in ['citizen_train','semlex_train','local_train']];x=", "parts=[read(n) for n in ['citizen_train','semlex_train','local_train']];\n temporal=read('temporal_train');keep=temporal[1]==like;parts.append((tuple(a[keep] for a in temporal[0]),temporal[1][keep],temporal[2][keep.numpy()]));x=")
s=s.replace(' for p in head.parameters():p.requires_grad_(False)',' for p in head.parameters():p.requires_grad_(True)')
s=s.replace('  weights.extend(', '  weights.extend(') if False else s
s=s.replace("weights.extend([1/3/len(counts)/counts[int(v)] for v in t])", "weights.extend([(0.15/len(counts)/counts[int(v)] + (0.15/counts.get(like,1) if int(v)==like else 0)) if len(counts)>1 else 0.10/len(t) for v in t])")
s=s.replace('torch.optim.Adam([last.weight,last.bias],lr=lr)','torch.optim.Adam(head.parameters(),lr=lr)')
s=s.replace('last.weight.grad.mul_(allowed[:,None]);last.bias.grad.mul_(allowed);torch.nn.utils.clip_grad_norm_([last.weight,last.bias],1);','torch.nn.utils.clip_grad_norm_(head.parameters(),1);')
start=s.index('  # Exact invariance');end=s.index('  checkpoint=',start);s=s[:start]+s[end:]
s=s.replace("('actual_selected',best),('actual_diagnostic',diagnostic)","('balanced_selected',best),('balanced_diagnostic',diagnostic)")
s=s.replace("'actual_training_results.json'","'balanced_training_results.json'").replace('all_other_word_output_rows_unchanged=True','all_other_word_output_rows_unchanged=False')
(D/'train_actual_balanced.py').write_text(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((D/'actual_training_recipe.json').read_text());r.update(format='slt_actual_runtime_window_recipe_v2',scope='actual_window_balanced_head',learning_rates=[.00001,.00003,.0001],phone_shares=[.1,.2],epochs=10,negative_sampling='45% general class-balanced replay across three existing domains;45% LIKE replay from those domains;10% previously admitted LIKE temporal train views. Phone occupies10% or20% of each batch. No validation error mining.',frozen='Complete encoders frozen; fusion head can learn nonlinear correction. Original head continues to feed letter head at export. All-class retention gates remain unchanged.')
r['pins'].pop(str(D/'train_actual.py'));r['pins'][str(D/'train_actual_balanced.py')]=sha(D/'train_actual_balanced.py')
for p in [P/'features/temporal_train.npz',P/'training_recipe.json']:r['pins'][str(p)]=sha(p)
(D/'balanced_training_recipe.json').write_text(json.dumps(r,indent=2))
w=(D/'training_worker.py').read_text().replace("'train_actual.py'","'train_actual_balanced.py'").replace("'actual_training.log'","'balanced_training.log'").replace("'training_completion.json'","'balanced_completion.json'").replace('Actual-window MY recovery','Balanced-head MY recovery');(D/'balanced_worker.py').write_text(w)
print('Prepared12balancedfull-headarms; all prior files preserved')
