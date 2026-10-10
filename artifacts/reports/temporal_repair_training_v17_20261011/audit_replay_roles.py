from pathlib import Path
import json,numpy as np,collections
from features import O,R,sha
checks={};ids={}
for source in ['citizen','semlex','local']:
 with np.load(O/'features'/f'{source}_train.npz') as a,np.load(O/'features'/f'{source}_val.npz') as b:
  ids[source]=(a['item_ids'].tolist(),b['item_ids'].tolist());overlap=set(ids[source][0])&set(ids[source][1]);checks[source]={'item_id_overlap':len(overlap),'original_train':len(a['targets']),'validation':len(b['targets'])}
  if source!='semlex':assert not overlap
manifest=R/'data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json';train={r['canonical_label']+'/'+r['semlex_video_id']:r for r in json.loads(manifest.read_text())['videos']}
raw={}
for role,items in zip(['train','validation'],ids['semlex']):
 raw[role]={}
 for ident in items:
  path=R/train[ident]['raw_path'] if role=='train' else R/'data/local/semlex_citizen100_val_audit/raw'/(ident+'.webm');digest=sha(path)
  if role=='train':assert digest==train[ident]['sha256']
  raw[role][ident]={'path':str(path),'sha256':digest}
held={r['sha256'] for r in raw['validation'].values()};excluded=[i for i,r in raw['train'].items() if r['sha256'] in held]
final_path=R/'artifacts/reports/supplement_finalization_v17_20260922/semlex.json'
final=json.loads(final_path.read_text())['records']
approved={role:{r['canonical_label']+'/'+r['source_item_id'] for r in final if r['role']==role} for role in ['train','validation']}
assert set(ids['semlex'][0])==approved['train']
excluded_validation=sorted(set(ids['semlex'][1])-approved['validation']);assert len(excluded_validation)==25
assert {raw['validation'][i]['sha256'] for i in excluded_validation}=={raw['train'][i]['sha256'] for i in excluded}
assert not ({r['sha256'] for r in raw['train'].values()} & {r['sha256'] for i,r in raw['validation'].items() if i not in excluded_validation})
checks['semlex'].update(shared_raw_train_ids=sorted(excluded),excluded_train_ids=[],excluded_validation_ids=excluded_validation,admitted_train=len(ids['semlex'][0]),admitted_validation=len(approved['validation']),finalized_manifest=str(final_path),finalized_manifest_sha256=sha(final_path),reason='Reconcile stale978-row validation cache to finalized953 admitted validation rows; preserve all1388 canonical train admissions.',baseline_caveat='Old978-row validation included25 training videos. This run compares incumbent/candidate on identical finalized953 rows; not independent phone validation.')
(O/'isolated_role_check.json').write_text(json.dumps(checks,indent=2)+'\n');(O/'replay_source_hashes.json').write_text(json.dumps(raw,indent=2)+'\n')
print('SemLex finalized roles restored:1388train/953validation; stale validation rows removed:',len(excluded_validation))
