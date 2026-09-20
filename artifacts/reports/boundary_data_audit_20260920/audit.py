from pathlib import Path
import sys,json,collections,hashlib,runpy
import numpy as np
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/reports/boundary_data_audit_20260920'
from scripts.prepare_asllrp_continuous_citizen100_v17 import read_sentence_csv
from active.v17.live_transition_supervision_v17 import interior_gaps
from active.v17.stage1_window_v17 import window_end_times
p=json.load(open('artifacts/reports/o5s5_citizen100_v17/combined_supervision.json'))
prior=json.load(open('artifacts/reports/stage2_v17_revisable_v1/supervision.json'))['items']
csvrows,_=read_sentence_csv(ROOT/'data/local/dataset_metadata/asllrp_signbank/asllrp_sentence_signs_2025_06_28.csv')
original={(r['Utterance video filename'],int(r['Start frame of the sign video']),int(r['End frame of the sign video'])):r for r in csvrows if r.get('Hidden','F')!='T'}
spans={}
for source,path in [('asllrp_contiguous','data/local/asllrp_contiguous_phrases_v17/manifest.json'),('asllrp_other_ctc','data/local/asllrp_other_ctc_v17/manifest.json')]:
 for s in json.load(open(path))['spans']:
  key=(source,s['utterance_video_filename'],int(s['span_index_in_utterance']))
  spans[key]=s
stats={}; examples=[]; temporal=[]; perclass=collections.defaultdict(set)
for r in p['rows']:
 group=r['role']+'|'+r['source']; c=stats.setdefault(group,collections.Counter())
 with np.load(r['archive_path'],allow_pickle=False) as z:
  ts=z['timestamps_seconds']; meta=json.loads(str(z['metadata_json'].item()))
 events=r['intervals']; c['rows']+=1
 before=prior.get(r['source_item_id'],{})
 c['source_crop_marked_incomplete']+=before.get('annotation_crop_complete') is False
 known=[e for e in events if e['label']!='__OTHER__']
 c['known_events']+=len(known); c['other_events']+=len(events)-len(known)
 c['first_annotation_at_zero']+=min(e['start_seconds'] for e in events)<=1e-6
 c['last_annotation_after_last_observation']+=max(e['end_seconds'] for e in events)>ts[-1]+1e-6
 if r['source']!='o5s5':
  name=r['source_item_id'].split(':')[1]; name=name if name.endswith('.mp4') else name+'.mp4'
  span=spans[(r['source'],name,int(r['source_item_id'].split('span')[-1]))]
  c['source_sign_intervals_actually_clipped']=c.get('source_sign_intervals_actually_clipped',0)
  fps=float(__import__('fractions').Fraction(span['frame_rate']))
  for e in events:
   orig=original.get((name,e.get('annotation_start_frame_global'),e.get('annotation_end_frame_global')))
   if orig:
    typ=orig['Sign type']; c['type:'+('OTHER:' if e['label']=='__OTHER__' else 'KNOWN:')+typ]+=1
    g0=int(orig['Start frame of the containing utterance'])+int(span['crop_start_frame_local'])
    g1=int(orig['Start frame of the containing utterance'])+int(span['crop_end_frame_local'])
    clipped=int(orig['Start frame of the sign video'])<g0 or int(orig['End frame of the sign video'])>g1
    c['source_sign_intervals_actually_clipped']+=clipped
    if clipped: examples.append(dict(reason='source_crop_clips_annotation',item=r['source_item_id'],video=meta['video_path'],event=e,original=orig))
 for pos,e in enumerate(events):
  start,end=float(e['start_seconds']),float(e['end_seconds']); isknown=e['label']!='__OTHER__'
  if isknown:
   c['known_ends_before_first_online_window']+=end<ts[0]+.27
   c['known_observed_frames_le_2']+=np.count_nonzero((ts>=start)&(ts<=end))<=2
   if r['role']=='train': perclass[e['label']].add(r['signer_id'])
  if not isknown and not r['all_signs_annotated']:continue
  left,right=start+1/30,end-1/30
  ends=[(start+end)/2] if right<=left else sorted({(left+right)/2,right})
  used=0
  for t in ends:
   for d in (.27,.53):
    if t-d<ts[0]-1e-9 or t>ts[-1]+1e-9:continue
    used+=1; c['known_windows' if isknown else 'unknown_windows']+=1
    if isknown:
     retained=ts[(ts>=t-d-1e-9)&(ts<=t+1e-9)]
     covered=np.mean((retained>=start)&(retained<=end))
     c['known_windows_less_than_half_target']+=covered<.5
     c['known_windows_missing_sign_onset']+=t-d>start+1e-9
     c['known_windows_end_before_sign_end']+=t<end-1e-9
     c['known_windows_no_raw_target_frame']+=np.count_nonzero((retained>=start)&(retained<=end))==0
     others=[x for j,x in enumerate(events) if j!=pos and x['start_seconds']<=t<x['end_seconds']]
     c['known_windows_ambiguous_endpoint']+=bool(others)
     temporal.append(dict(role=r['role'],source=r['source'],item=r['source_item_id'],video=meta['video_path'],label=e['label'],start=start,end=end,window_start=t-d,window_end=t,duration=d,target_fraction=float(covered)))
  if isknown:c['known_events_with_no_training_window']+=used==0
 if r['all_signs_annotated']:
  for left,right in interior_gaps(events,1/30):
   t=(left+right)/2
   for d in (.27,.53):
    if t-d<ts[0]-1e-9:continue
    c['transition_windows']+=1
    c['transition_windows_overlap_annotated_sign']+=any(e['start_seconds']<t and e['end_seconds']>t-d for e in events)
  if not examples or r['source_item_id'] not in {x['item'] for x in examples}:
   if len([x for x in examples if x['reason']=='annotated_example'])<5:
    examples.append(dict(reason='annotated_example',item=r['source_item_id'],video=meta['video_path'],events=events))
result=dict(groups=stats,train_known_classes=len(perclass),train_class_signer_counts={k:len(v) for k,v in sorted(perclass.items())},source_clipping_examples=examples,notes=['Counts are manifest occurrences, not independent unique signs.','Interval coverage does not certify visual annotation correctness.','A transition endpoint window may intentionally contain preceding sign frames; that is not automatically a wrong label.'])
(OUT/'audit.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
(OUT/'window_details.json').write_text(json.dumps(temporal,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps({'groups':stats,'train_known_classes':len(perclass),'single_signer_classes':sum(len(x)==1 for x in perclass.values()),'clipped_examples':len([x for x in examples if x['reason']=='source_crop_clips_annotation'])},indent=2,default=lambda x:x.item()))
