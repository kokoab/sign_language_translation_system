"""Matched training-recording WER; fixed historical decoder, no training or test inference."""
import json,sys,hashlib,time,copy,pickle
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from scripts import segmental_lab_v17 as lab
from active.v17.export_unified_multimodal_coreml_v17 import load_model
O=Path(__file__).resolve().parent
C=R/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe/chain_9683'
paths={'before_interval':C/'phrase_adapt/best_model.pth','after_interval':C/'span_recognizer/best_model.pth','later_v2':R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/span_recognizer/best_model.pth','partial_sign':R/'artifacts/reports/go_split_root_cause_v17_20261009/run_frag_0.3/span_recognizer/best_model.pth'}
paths.update({'transition_start':R/'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/best_model.pth','transition_finetuned':R/'artifacts/reports/my_transition_recovery_v17_20261011/confidence_selected.pth'})
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=R/'active/v17/phrase_segment_recipe_manifest_20261007.json';m=json.loads(manifest.read_text());allowed={e['video_sha256'] for e in m['entries'] if e['split']=='train'}
rows=[r for r in lab.rows_for('local_train') if r['video_sha256'] in allowed];assert len(rows)==len(allowed)==179
held={r['video_sha256'] for name in ['tune','test'] for r in lab.rows_for(name)};assert not allowed&held
cfg=json.loads((C/'span_recognizer/history.json').read_text())['provenance']['cfg']
torch.set_num_threads(2);device=torch.device('mps');models={};labels=None
for name,p in paths.items():
 model,ck=load_model(p);ls=sorted(ck['label_to_index'],key=ck['label_to_index'].get)
 if labels is not None:assert labels==ls
 labels=ls;models[name]=model.eval().to(device)
result={'scope':'Training-set WER on approved 179 recordings, same fixed historical DGS/candidate decoder for all models. Not current runtime replay, not validation/test performance, no weights updated.','manifest_sha256':sha(manifest),'decoder':cfg,'models':{n:{'path':str(p),'sha256':sha(p)} for n,p in paths.items()},'records':{n:[] for n in paths},'input_hashes':{},'state':'running'}
scorer=None
cache=O/'complete_spans';cache.mkdir(exist_ok=True)
for ri,row in enumerate(rows):
 sign,_=lab.load_dgs(row);bio=lab.bio_at_lookahead(sign,cfg['lookahead'])
 spans=set(lab.candidate_spans(bio))|set(lab.dense_spans(bio))
 cp=cache/(lab.key(row['source_item_id'])+'.pkl')
 memo=pickle.loads(cp.read_bytes()) if cp.exists() else lab.load_memo(row)
 missing=[k for k in spans if k not in memo or (memo[k] is not None and memo[k].get('inputs') is None and not memo[k].get('input_unavailable'))]
 if missing:
  if scorer is None:scorer=lab.SpanScorer()
  obs=scorer.observations(row)
  for k in missing:
   memo[k]=scorer.score(obs,k[0]/lab.FPS,k[1]/lab.FPS)
   if memo[k] is not None and memo[k].get('inputs') is None:
    memo[k]['input_unavailable']=True;memo[k]['v_raw']=None
  cp.write_bytes(pickle.dumps(memo))
 else:
  if not cp.exists():cp.write_bytes(pickle.dumps(memo))
 assert all(k in memo and (memo[k] is None or memo[k].get('inputs') is not None or memo[k].get('input_unavailable')) for k in spans)
 keys=[k for k in spans if memo[k] is not None and memo[k].get('inputs') is not None]
 result['input_hashes'][row['source_item_id']]={'memo':sha(cp),'dgs':sha(lab.CACHE/'dgs'/(lab.key(row['source_item_id'])+'.npz')),'video_sha256':row['video_sha256']}
 names=['landmarks','hand_embeddings','hand_valid','hand_boxes']
 for name,model in models.items():
  new={k:None if v is None else dict(v) for k,v in memo.items()}
  with torch.inference_mode():
   for i in range(0,len(keys),64):
    kk=keys[i:i+64];xx=[torch.from_numpy(np.stack([memo[k]['inputs'][f] for k in kk])).to(device) for f in names];xx=[v.float() if f!='hand_valid' else v for f,v in zip(names,xx)]
    logits=model(*xx).float().cpu().numpy();assert np.isfinite(logits).all()
    for k,v in zip(kk,logits):new[k]['v_raw']=v
  for v in new.values():
   if v is not None and v.get('inputs') is None:v['v_raw']=None
  summary,per=lab.evaluate_decoder([(row,sign,new)],labels,cfg['lookahead'],cfg)
  result['records'][name].append({'counts':summary,**per[0]})
 if ri%10==0 or ri==len(rows)-1:
  result['completed_recordings']=ri+1;(O/'results.json').write_text(json.dumps(result)+'\n');print(ri+1,len(rows),flush=True)
result['summary']={}
for name,rr in result['records'].items():
 totals={k:sum(x['counts'][k] for x in rr) for k in ['S','D','I','refs','hyps','correct']};totals['wer_percent']=100*(totals['S']+totals['D']+totals['I'])/totals['refs'];result['summary'][name]=totals
result['state']='complete';(O/'results.json').write_text(json.dumps(result)+'\n');print(json.dumps(result['summary'],indent=2),flush=True)
