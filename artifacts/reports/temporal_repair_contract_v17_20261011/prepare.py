"""Pin existing target supervision and temporal provenance; no training or downloads."""
from pathlib import Path
import sys,json,hashlib,collections,subprocess
from functools import lru_cache
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];sys.path.insert(0,str(R))
from scripts import segmental_lab_v17 as lab
from scripts.build_combined_dataset_v17 import load_features
TARGETS={'MY','LIKE','HAVE'}
def read(p):return json.load(open(R/p))
@lru_cache(None)
def sha(p):
 h=hashlib.sha256()
 with (R/p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
@lru_cache(None)
def pts(path):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(R/path)]))
 t=np.array([float(f['best_effort_timestamp_time']) for f in d['frames']]);assert len(t)>1 and np.all(np.diff(t)>0),path
 return t-t[0]
def bounds(path,start,end):
 t=pts(path);assert 0<=start<=end<len(t),(path,start,end,len(t))
 return [float(t[start]),float(t[end+1] if end+1<len(t) else t[end]+np.median(np.diff(t)))]
combined=read('data/local/combined_dataset_v17_20260922/manifest.json')
scoped=read('active/v17/phrase_segment_recipe_manifest_20261007.json')
allowed={e['video_sha256'] for e in scoped['entries'] if e['split']=='train'}
held=lab.rows_for('test')+lab.rows_for('tune') # Exclusion metadata only.
held_hashes={r['video_sha256'] for r in held}
held_parents={r['source_item_id'].split(':')[1] for r in held if r['source_item_id'].startswith('asllrp:')}
parents={p['filename']:p for f in ['data/local/asllrp_contiguous_phrases_v17/manifest.json','data/local/asllrp_other_ctc_v17/manifest.json'] for p in read(f)['parents']}
segmented={r['sha256']:r for r in read('data/local/asllrp_segmented_citizen100_v17/manifest.json')['videos']}
vocab={r['canonical_label']:r for r in read('active/v17/citizen100_manifest.json')['classes']}
rows=[];excluded=[];pinpaths=set(combined['source_manifests_sha256'])
pinpaths.update(['data/local/combined_dataset_v17_20260922/manifest.json','active/v17/phrase_segment_recipe_manifest_20261007.json','data/local/asllrp_segmented_citizen100_v17/manifest.json','data/local/asllrp_contiguous_phrases_v17/manifest.json','data/local/asllrp_other_ctc_v17/manifest.json'])
for r in combined['records']:
 source=r['source']
 if source not in ['asllrp_segmented','o5s5','stem','local_phrases']:continue
 if not TARGETS & set(r['target_sequence']):continue
 if source=='local_phrases' and r['video_sha256'] not in allowed:continue
 gloss=next(g for g in r['target_sequence'] if g in TARGETS)
 ident=r.get('source_item_id',r['feature_path'])
 if source=='o5s5':ident+=f":{r['ordinal']}"
 reason=None
 if r['video_sha256'] in held_hashes:reason='development/protected evaluation source hash'
 if r.get('parent_video') in held_parents:reason='development/protected evaluation parent utterance'
 if reason:excluded.append({'id':ident,'reason':reason,'source':source,'role':r['role']});continue
 assert sha(r['video_path'])==r['video_sha256'],r['video_path']
 assert sha(r['feature_path'])==r['feature_sha256'],r['feature_path']
 load_features(r,combined['label_to_index'])
 row={'id':ident,'source':source,'role':r['role'],'label':gloss,'signer_id':r['signer_id'],
  'citizen_asl_lex_code':vocab[gloss]['asl_lex_code'] if 'asl_lex_code' in vocab[gloss] else vocab[gloss].get('citizen_asl_lex_code'),
  'admission_manifest':r['origin_manifest'],'admitted_video_path':r['video_path'],'admitted_video_sha256':r['video_sha256'],
  'feature_path':r['feature_path'],'feature_sha256':r['feature_sha256'],
  'source_video_path':r['video_path'],'source_video_sha256':r['video_sha256'],
  'sequence_reference':r['target_sequence'],'known_background':False,'boundary_status':'inherited_verified_interval'}
 if source=='asllrp_segmented':
  m=segmented[r['video_sha256']];p=parents[r['parent_video']]
  assert sha(p['path'])==p['sha256']
  a=int(m['sign_start_frame'])-int(m['utterance_start_frame']);b=int(m['sign_end_frame'])-int(m['utterance_start_frame'])
  row.update(source_video_path=p['path'],source_video_sha256=p['sha256'],core_seconds=bounds(p['path'],a,b),
   source_start_frame=a,source_end_frame_inclusive=b,occurrence_key=f"asllrp:{m['master_video_filename']}:{m['sign_start_frame']}:{m['sign_end_frame']}",
   group_key=f"asllrp:{r['signer_id']}:{m['master_video_filename']}",parent_utterance=r['parent_video'],
   variant_evidence={k:m[k] for k in ['citizen_raw_gloss','citizen_asl_lex_code','entry_variant','occurrence','sign_type']})
 elif source=='stem':
  t=r['interval_sampling'];a=t['source_start_frame_inclusive'];b=t['source_end_frame_inclusive']
  row.update(core_seconds=bounds(r['video_path'],a,b),source_start_frame=a,source_end_frame_inclusive=b,
   group_key=f"stem:{r['signer_id']}:{r['video_sha256']}",occurrence_key=f"stem:{r['video_sha256']}:{a}:{b}")
 elif source=='o5s5':
  row.update(core_seconds=[r['start_seconds'],r['end_seconds']],group_key=f"o5s5:{r['signer_id']}:{r['video_sha256']}",
   occurrence_key=f"o5s5:{r['video_sha256']}:{r['ordinal']}")
 else:
  row.update(core_seconds=None,boundary_status='word_boundary_review_required',group_key=f"local:{r['signer_id']}:{r['video_sha256']}",occurrence_key=ident)
 row['duration_seconds']=float(pts(row['source_video_path'])[-1])
 row['native_inputs_ready']=False
 if row['core_seconds']:
  a,b=row['core_seconds'];t=pts(row['source_video_path']);assert 0<=a<b<=t[-1]+0.2
  row['native_context_seconds']=[max(0,a-.1),min(row['duration_seconds'],b+.1)]
  # Source-clock frame selection, exactly the production deadline recurrence.
  counts=[]
  for offset in [0,1/60,1/30]:
   deadline=offset;selected=[]
   for ts in t:
    if ts+1e-6<deadline:continue
    deadline=max(deadline+.05,ts);selected.append(float(ts))
   counts.append(sum(max(0,a-.1)-1e-9<=ts<=b+.1+1e-9 for ts in selected))
  row['context_frames_at_20hz_offsets']=counts
  row['source_timing_usable']=min(counts)>=4
  row['supervision']='positive_interval_only; surrounding motion remains unlabeled'
 else:row['source_timing_usable']=False;row['supervision']='reviewed sequence, unreviewed word timing'
 rows.append(row)
assert len({r['occurrence_key'] for r in rows})==len(rows)
groups=collections.defaultdict(set);hashroles=collections.defaultdict(set);signers=collections.defaultdict(set)
for r in rows:
 groups[r['group_key']].add(r['role']);hashroles[r['source_video_sha256']].add(r['role']);signers[(r['source'],r['signer_id'])].add(r['role'])
assert all(len(v)==1 for v in groups.values()),groups
assert all(len(v)==1 for v in hashroles.values()),hashroles
assert all(len(v)==1 for v in signers.values()),signers
summary={'events':len(rows),'unique_parent_videos':len(hashroles),'counts':dict(collections.Counter(f"{r['source']}:{r['role']}:{r['label']}" for r in rows)),
 'timing_usable':dict(collections.Counter(f"{r['role']}:{r['label']}" for r in rows if r['source_timing_usable'])),
 'short_contexts':[r['id'] for r in rows if r['core_seconds'] and not r['source_timing_usable']],
 'excluded':excluded,'cross_role_parent_overlap':0,'cross_role_group_overlap':0,'within_source_signer_overlap':0,
 'protected_or_tune_metadata_rows_used_for_exclusion':len(held),'protected_features_accessed':False}
(O/'source_inventory.json').write_text(json.dumps({'summary':summary,'records':rows,'pins':{p:sha(p) for p in sorted(pinpaths)}},indent=2)+'\n')
print(json.dumps(summary,indent=2))
