"""Validate every native view, record exclusions, and prepare role-separated frozen features."""
from pathlib import Path
import json,collections,sys,os
from features import O,R,BASE,sha,encode
import numpy as np,torch
from active.v17.export_unified_multimodal_coreml_v17 import load_model
P=O.parent/'temporal_repair_contract_v17_20261011'
assert json.loads((O/'exact_extraction_completion.json').read_text())['returncode']==0
contract=json.loads((O/'exact_core_contract.json').read_text());sources={r['id']:r for r in contract['records']}
model,checkpoint=load_model(BASE);torch.set_num_threads(2);device=torch.device('mps');model.to(device).eval();labels=checkpoint['label_to_index']
parts={role:[[],[],[],[]] for role in ['train','validation']};metadata={role:[] for role in parts};targets={role:[] for role in parts}
pins={};audit=[];seen=set();parities=[]
for case in json.loads((O/'exact_cases.json').read_text()):
 prefix=O/'exact_native_inputs'/case['id'];meta=prefix.with_suffix('.json');d=json.loads(meta.read_text());pins[str(meta)]=sha(meta);n=sum(r['valid'] for r in d['rows'])
 arrays=[]
 for i,shape in enumerate(d['shapes']):
  path=Path(str(prefix)+f'.{i}.bin');pins[str(path)]=sha(path);a=np.fromfile(path,dtype='<f4').reshape(-1,*shape);assert len(a)==n and np.isfinite(a).all();arrays.append(a)
 assert set(np.unique(arrays[2])).issubset({0.,1.})
 assert np.all((arrays[0][...,3:5]>=0)&(arrays[0][...,3:5]<=1))
 assert {r['id'] for r in d['rows']}=={e['id'] for e in case['events']}
 assert len({(r['id'],r['phase'],r['capture_offset'],r['view']) for r in d['rows']})==72*len(case['events'])
 for event in case['events']:
  ident=event['id'];source=sources[ident];assert ident not in seen;seen.add(ident);rr=[r for r in d['rows'] if r['id']==ident]
  kept=[];rejected=[]
  for row in rr:
   assert all(b>a for a,b in zip(row['selected_seconds'],row['selected_seconds'][1:])),ident
   reason=None;core_frames=sum(row['core'][0]-1e-9<=t<row['core'][1]-1e-9 for t in row.get('retained_seconds',[]))
   if not row['valid']:reason='native_input_unavailable'
   elif core_frames<2 or row['core_hand_frames']<2:reason='fewer_than_two_core_hand_frames'
   elif row.get('core_crop_positions',0)<1:reason='no_retained_core_crop'
   if reason:rejected.append({'phase':row['phase'],'capture_offset':row['capture_offset'],'view':row['view'],'reason':reason})
   else:kept.append(row)
  audit.append({'id':ident,'source':source['source'],'role':source['role'],'label':source['label'],'requested':72,'eligible':len(kept),'excluded':rejected})
  if not kept:continue
  indexes=[r['array_index'] for r in kept];features,parity=encode(model,[a[indexes] for a in arrays],device);parities.append(parity);role=source['role']
  for dest,value in zip(parts[role],features):dest.append(value)
  targets[role]+=[labels[source['label']]]*len(kept)
  metadata[role]+=[dict(id=ident,parent=source['source_video_sha256'],group=source['group_key'],source=source['source'],signer_id=source['signer_id'],label=source['label'],phase=r['phase'],capture_offset=r['capture_offset'],view=r['view']) for r in kept]
assert seen==set(sources)
for role in parts:
 path=O/'features'/f'temporal_{role}.npz';np.savez(path,**{f'f{i}':np.concatenate(a) for i,a in enumerate(parts[role])},targets=np.array(targets[role]),item_ids=np.array([r['id'] for r in metadata[role]]));pins[str(path)]=sha(path)
 out=O/f'temporal_{role}_rows.json';out.write_text(json.dumps(metadata[role],indent=2)+'\n');pins[str(out)]=sha(out)
summary={'events_attempted':len(seen),'events_eligible':sum(r['eligible']>0 for r in audit),'views_requested':len(seen)*72,'views_eligible':sum(r['eligible'] for r in audit),'by_role_label':dict(collections.Counter(f"{r['role']}:{r['label']}" for r in audit if r['eligible'])),'encoder_cache_logit_max_abs':max(parities),'excluded_reasons':dict(collections.Counter(e['reason'] for r in audit for e in r['excluded']))}
(O/'materialization.json').write_text(json.dumps({'summary':summary,'events':audit,'pins':pins},indent=2)+'\n');print(json.dumps(summary,indent=2))
