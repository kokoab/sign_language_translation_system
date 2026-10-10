"""Fail-closed verifier for this preparation artifact, not an alternate trainer."""
from pathlib import Path
import json,hashlib,copy,functools
O=Path(__file__).resolve().parent;R=O.parents[2]
@functools.lru_cache(None)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def verify(c,for_training=False):
 require(c['format']=='slt_temporal_repair_contract_v17_v1','format')
 require(not c['native_pilot']['training_loader_enabled'],'pilot arrays include development rows')
 require(not c['training_ready'] and not c['promotion_ready'],'preparation cannot enable training/promotion')
 base=json.loads((O/'source_inventory.json').read_text());source={r['id']:r for r in base['records']}
 review={r['id']:r for r in json.loads((O/'my_visual_review.json').read_text())['records']}
 require(len(c['records'])==len(source) and {r['id'] for r in c['records']}==set(source),'source membership')
 for p,digest in c['pins'].items():require(sha(R/p)==digest,'pin mismatch: '+p)
 require(all(c['pins'].get(p)==h for p,h in base['pins'].items()),'source manifest pins missing')
 require(any('ImageEncoder' in p and p.endswith('weight.bin') for p in c['pins']),'encoder weights pin missing')
 roles={};signers={}
 for r in c['records']:
  b=source[r['id']]
  for k in ['role','label','signer_id','group_key','occurrence_key','source_video_path','source_video_sha256','feature_path','feature_sha256','citizen_asl_lex_code']:
   require(r[k]==b[k],'source identity/role mismatch: '+k)
  require(r['role'] in ['train','validation'],'development cannot enter training')
  require(r['label'] in ['MY','LIKE','HAVE'],'unknown label')
  a,z=r['core_seconds'];require(0<=a<z<=r['duration_seconds']+.04,'core out of bounds')
  expected=review[r['id']]['core_seconds'] if r['label']=='MY' else b['core_seconds']
  require(r['core_seconds']==expected,'reviewed interval changed')
  for pkey,hkey in [('source_video_path','source_video_sha256'),('admitted_video_path','admitted_video_sha256'),('feature_path','feature_sha256')]:require(sha(R/r[pkey])==r[hkey],'source bytes mismatch')
  for g in [r['source_video_sha256'],r['group_key']]:roles.setdefault(g,set()).add(r['role'])
  signers.setdefault((r['source'],r['signer_id']),set()).add(r['role'])
 require(all(len(v)==1 for v in roles.values()),'parent overlap')
 require(all(len(v)==1 for v in signers.values()),'within-source signer overlap')
 for r in c['development_records']:
  require(r['role']=='development_only' and not r['training_admitted'] and not r['independent_evaluation'],'development misuse')
  require(sha(R/r['video_path'])==r['video_sha256'],'development source bytes')
 if for_training:raise ValueError('Draft preparation only: full native cache, executable recipe and admission checks are not complete')
 return True
if __name__=='__main__':
 c=json.loads((O/'contract.json').read_text());verify(c);tests={}
 def reject(name,fn,training=False):
  d=copy.deepcopy(c);fn(d)
  try:verify(d,for_training=training)
  except ValueError as e:tests[name]=str(e)
  else:raise AssertionError('Accepted invalid contract: '+name)
 reject('pin_tampering',lambda d:d['pins'].__setitem__(next(iter(d['pins'])),'0'*64))
 reject('role_reassignment',lambda d:d['records'][0].__setitem__('role','validation'))
 reject('label_reassignment',lambda d:d['records'][0].__setitem__('label','MY'))
 reject('core_out_of_bounds',lambda d:d['records'][0].__setitem__('core_seconds',[-1,99]))
 reject('development_as_training',lambda d:d['records'].append(d['development_records'][0]))
 reject('flag_only_bypass',lambda d:d.__setitem__('training_ready',True))
 reject('training_with_partial_cache',lambda d:None,True)
 prod=json.loads((O.parent/'clean_lineage_recovery_v17_20261011/source_manifest.json').read_text())
 checked=[]
 for r in prod:
  if '/mobile_app/' in r['path']:
   require(sha(r['path'])==r['sha256'],'production file changed');checked.append(r['path'])
 result={'contract_valid':True,'training_allowed':False,'promotion_allowed':False,'negative_checks':tests,'source_events':len(c['records']),'production_files_unchanged':sorted(set(checked))}
 (O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
