"""User-authorized reuse of exactly the eight historical MY NAME sources.
No historical trainer defaults; source identities explicitly intersect canonical admission.
"""
from pathlib import Path
import sys,json,pickle,hashlib,collections
import numpy as np,torch
D=Path(__file__).resolve().parent;P=D.parent/'temporal_repair_training_v17_20261011';R=D.parents[2];sys.path.insert(0,str(P));sys.path.insert(0,str(R))
from features import encode,sha,BASE
from active.v17.export_unified_multimodal_coreml_v17 import load_model
removed=json.loads((D.parent/'sept30_fix_port_v17_20261011/removed_MY_phrase_inputs.json').read_text());recipe=json.loads((R/'active/v17/phrase_segment_recipe_manifest_20261007.json').read_text());canonical=json.loads((R/'active/v17/approved_phrase_manifest_20260921_v2.json').read_text())
# Recipe entries were independently hash-matched to the canonical manifest in the preceding audit.
assert recipe['canonical_sha256']==sha(R/'active/v17/approved_phrase_manifest_20260921_v2.json')
entries={x['video_sha256']:x for x in recipe['entries']};model,base=load_model(BASE);labels=sorted(base['label_to_index'],key=base['label_to_index'].get)
arrays=[[],[],[],[]];ys=[];ids=[];source=[];pins={};keys=['landmarks','hand_embeddings','hand_valid','hand_boxes']
for item in removed:
 entry=entries[item['video_sha256']];assert entry['source_item_id']==item['source_item_id'] and entry['approved_role']=='validation'
 for k in ['rgb','hand']:
  p=R/entry[k];assert sha(p)==entry[k+'_sha256'];pins[str(p)]=sha(p)
 key=hashlib.sha256(item['source_item_id'].encode()).hexdigest()[:20];mp=R/'artifacts/cache/segmental_decoder_v17/spans'/(key+'.pkl');lp=R/'artifacts/cache/segmental_decoder_v17/train_labels'/(key+'.json');pins[str(mp)]=sha(mp);pins[str(lp)]=sha(lp)
 memo=pickle.loads(mp.read_bytes());spans=json.loads(lp.read_text())['spans'];count=collections.Counter()
 for s in spans:
  if s.get('prefix') or s['gloss'] not in ['MY','NAME']:continue
  raw=memo.get(tuple(s['span']));assert raw and raw.get('inputs') is not None
  for dest,k in zip(arrays,keys):dest.append(raw['inputs'][k])
  ys.append(labels.index(s['gloss']));ids.append(item['source_item_id']);count[s['gloss']]+=1
 assert count['MY']==item['MY_spans'];source.append(dict(source_item_id=item['source_item_id'],video_sha256=item['video_sha256'],counts=dict(count),former_role='validation',new_role='training_in_scoped_recovery_only'))
xx=[np.stack(a).astype(np.float32) for a in arrays];assert all(np.isfinite(a).all() for a in xx);torch.set_num_threads(2);device=torch.device('mps');model.to(device).eval();f,delta=encode(model,xx,device)
np.savez(D/'restored_features.npz',**{f'f{i}':a for i,a in enumerate(f)},targets=np.array(ys),item_ids=np.array(ids));pins[str(D/'restored_features.npz')]=sha(D/'restored_features.npz');pins[str(BASE)]=sha(BASE)
(D/'restored_input_contract.json').write_text(json.dumps(dict(scope='Exactly eight historical MY NAME videos explicitly authorized for reuse; no other split changes.',n=len(ys),sources=source,pins=pins,parity_max_abs=delta,no_protected_citizen_test=True),indent=2));print('Restored scoped features',len(ys),'rows',len(source),'sources; parity',delta)
