"""Prepare a report-local recipe contract; does not enable any trainer."""
from pathlib import Path
import json,hashlib,collections,subprocess,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):(O/n).write_text(json.dumps(d,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
d=json.loads((O/'source_inventory.json').read_text());reviews={r['id']:r for r in json.loads((O/'my_visual_review.json').read_text())['records']}
for r in d['records']:
 if r['id'] not in reviews:continue
 v=reviews[r['id']];assert r['source_video_sha256']==v['video_sha256']
 r.update(core_seconds=v['core_seconds'],boundary_status='assistant_reviewed_conservative_core',visual_review=v,supervision='central MY positive only; adjacent NAME/context is not relabeled')
 f=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(R/r['source_video_path'])]))
 t=np.array([float(x['best_effort_timestamp_time']) for x in f['frames']]);t-=t[0]
 a,b=r['core_seconds'];counts=[];core=[]
 for off in [0,1/60,1/30]:
  deadline=off;ts=[]
  for s in t:
   if s+1e-6<deadline:continue
   deadline=max(deadline+.05,s);ts.append(s)
  counts.append(sum(a-.1-1e-9<=s<b+.1-1e-9 for s in ts));core.append(sum(a-1e-9<=s<b-1e-9 for s in ts))
 r.update(native_context_seconds=[max(0,a-.1),min(r['duration_seconds'],b+.1)],context_frames_at_20hz_offsets=counts,core_frames_at_20hz_offsets=core,source_timing_usable=min(counts)>=4)
summary=dict(d['summary']);summary['counts_by_role_label']=dict(collections.Counter(f"{r['role']}:{r['label']}" for r in d['records']))
summary['my_reviewed']=len(reviews);summary['my_source_local_signers']=len({r['signer_id'] for r in reviews.values()});summary['my_validation_events']=0
summary['my_single_frame_core_at_any_offset']=[r['id'] for r in d['records'] if r['label']=='MY' and min(r['core_frames_at_20hz_offsets'])<2]
summary['my_context_short_at_any_offset']=[r['id'] for r in d['records'] if r['label']=='MY' and not r['source_timing_usable']]
phone=json.loads((O/'phone_visual_review.json').read_text())['records']
pins=dict(d['pins'])
for n in ['source_inventory.json','my_visual_review.json','phone_visual_review.json','native_source_pins.json','pilot_cases.json','previous_my_alignment_audit.json']:
 p=O/n;pins[str(p.relative_to(R))]=sha(p)
for p in sorted((O/'native').glob('*.swift')):pins[str(p.relative_to(R))]=sha(p)
encoder=R/'artifacts/reports/clean_lineage_recovery_v17_20261011/resources/MobileCLIP2S0ImageEncoderV17FP16.mlpackage'
for p in sorted(encoder.rglob('*')):
 if p.is_file():pins[str(p.relative_to(R))]=sha(p)
for name in ['active/v17/approved_phrase_manifest_20260921_v2.json','active/v17/citizen100_manifest.json','artifacts/reports/clean_lineage_recovery_v17_20261011/adaptation_recipe.json']:
 pins[name]=sha(R/name)
contract=dict(format='slt_temporal_repair_contract_v17_v1',status='source_review_complete_native_pilot_only',training_ready=False,promotion_ready=False,
 scope='Proposed frozen-encoder temporal positive-head experiment; no background, boundary, encoder, or mixed historical trainer admission',
 summary=summary,records=d['records'],development_records=phone,pins=pins,
 sampling=dict(source_clock='actual presentation timestamps normalized to first frame',capture_hz=20,capture_offsets_seconds=[0,1/60,1/30],deadline='max(previous_deadline + 0.05, accepted_source_pts)',auxiliary_phases=list(range(8)),reset='New Vision/tracker state at source start for every phase/offset; never reset at sign core',source_geometry='aspect preserved; max side 1280; Vision detection max side 640; reject unhandled display transforms',context_views=dict(left_expansion_seconds=[.10,.15,.20],right_expansion_seconds=.10),inputs=dict(landmarks=[32,61,5],hand_embeddings=[16,3,512],hand_valid=[16,3],hand_boxes=[16,3,4]),eligibility='At least 4 context frames, 2 sampled core frames with detected hand evidence, and at least one valid crop position inside retained core after runtime trimming. Insufficient views are excluded, never relabeled background.',weighting='One source occurrence is one example group; average augmented-view losses per event so 72 views never count as 72 independent examples.'),
 data_contract=dict(roles='Preserve existing source train/validation roles; all windows, duplicate encodings, parent/session and within-source signer groups stay together.',identity_limit='Within-source signer separation verified. Cross-corpus signer identity is not established; no global signer-disjoint claim.',labels='One frozen Citizen raw-gloss/ASL-LEX variant per class. No synonym or numeric-variant merging.',boundaries='ASLLRP/O5S5/STEM retain reviewed source intervals. MY cores were assistant-reviewed against approved sequences, not expert-certified. Uncertain approach/release and context are unlabeled.',phone='Known failure recordings are development-only; mixed MY/LIKE region and aborted HAVE stay unlabeled; development recordings are not training or independent evaluation.',transition_limit='No admitted ASLLRP HAVE→BIG occurrence found. Seven reviewed phone HAVE→BIG occurrences are development evidence, not training examples.'),
 experiment=dict(arms=['incumbent model + incumbent runtime','adapted model + incumbent runtime','incumbent model + proposed runtime change','adapted model + proposed runtime change'],training='Frozen encoders; only fusion/word head adaptation; retain established 100-class isolated train replay with pinned provenance. No fragment/reject synthesis or unreviewed blank supervision.',model_selection='Declare optimizer, mixing weights, seeds, epoch bound and full cache hashes in executable recipe before first training. Include epoch 0; tune may select but is never described as independent.',runtime='Evaluate any contact preservation or continuous word-state change separately; do not rewrite labels from sentence expectations.'),
 evaluation=dict(retention='All 100 classes on existing validation splits; compare incumbent and candidate with identical inputs; require no aggregate loss and no per-class correct-count regression in this bounded experiment.',target_controls='Report MY↔LIKE and HAVE↔ANGRY/TIRED confusions, omissions, duplicates, and full sentence edit errors. Genuine LIKE/ANGRY/TIRED must remain recognized; output counts alone are not recall.',phase='All 8 auxiliary phases × 3 capture offsets on whole reviewed development sessions; report worst case and each case, not just averages. Source event/session remains the statistical unit.',export='Matched letter head; Python/Core ML comparison on identical tensors; test all word/NAME/letter modes and retained word order.',phone='New untouched normal shared-app sessions with intended MY NAME, MY FRIEND, I HAVE BIG and genuine LIKE/TIRED/ANGRY controls, multiple repetitions and signers. Freeze candidate and review labels before predictions; if used to tune, retire it from independent evaluation.',protected_test='Do not rerun Citizen official test or lab protected test. Existing metadata used only for exclusion.'),
 readiness_blockers=['Complete and verify native inputs for all admitted train/validation events, including retained-core hand/crop coverage; pilot does not satisfy full materialization.','Verify core hand/crop coverage on every MY event; all 28 pass timestamp checks and the six native MY training pilot events pass evidence checks.','Freeze executable optimizer/mixing/selection recipe and full 100-class retention input hashes.','Reserve and capture independent phone evaluation; existing capture pack remains pending.'])
audit=json.loads((O/'native_audit.json').read_text())
contract['native_pilot']={'training_loader_enabled':False,'unique_events':audit['unique_events'],'requested_views':audit['total_requested'],'constructed_views':audit['total_constructed'],'evidence_eligible_views':audit['total_evidence_eligible'],'limitation':'Includes development-only rows in shared arrays. Any future loader must select by pinned event membership and role, never load all array rows as training.'}
for name in ['native_audit.json','pilot_inputs_v2.json','recovered_my_inputs.json','recovered_my_cases.json']+[f'{p}.{i}.bin' for p in ['pilot_inputs_v2','recovered_my_inputs'] for i in range(4)]:
 q=O/name;contract['pins'][str(q.relative_to(R))]=sha(q)
contract['summary']['native_verified_training_events']=8
contract['summary']['native_verified_development_events']=2
save('contract.json',contract);print(json.dumps(summary,indent=2))
