from pathlib import Path
import json,hashlib
import numpy as np
D=Path(__file__).resolve().parent;P=D.parent/'temporal_repair_training_v17_20261011';R=D.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
summary=json.loads((D/'actual_feature_summary.json').read_text());assert summary['n']==89 and summary['parity_max_abs']<1e-5
metadata=json.loads((D/'actual_metadata.json').read_text());events={x['event'] for x in metadata};train=['MY_FRIEND_1','MY_NAME_1','MY_EAT'];assert set(train)<events
previous=json.loads((P/'isolated_feature_manifest.json').read_text());pins={}
for name in ['citizen_train','semlex_train','local_train','citizen_val','semlex_val','local_val']:
 p=P/'features'/(name+'.npz');digest=sha(p);assert previous['pins'][str(p)]==digest;pins[str(p)]=digest
for p in [R/'active/v17/approved_phrase_manifest_20260921_v2.json',Path(previous['base']),P/'isolated_feature_manifest.json',P/'isolated_role_check.json',P/'train.py',P/'features.py',D/'train_actual.py',D/'encode_actual.py',D/'actual_features.npz',D/'actual_metadata.json',D/'dump_cases.json',D/'actual_window_dump.json']+list((D/'actual_windows').glob('*.bin'))+list((D/'dump_native').glob('*.swift')):pins[str(p)]=sha(p)
video=D.parent/'sept30_fix_port_v17_20261011/phone/live_reel_videos/20261011_220739_03.mp4';pins[str(video)]=sha(video)
recipe=dict(format='slt_actual_runtime_window_recipe_v1',scope='actual_window_MY_LIKE_output_rows',ready=True,promotion_ready=False,canonical_training_ready=False,
 authorization='User authorizes sequential overnight recovery experiments and historical split reuse. This dedicated scoped recipe admits manually reviewed retained take03 MY cores as known phone training/development data; does not reuse a historical mixed trainer or change canonical phrase membership.',
 input_contract='Only emitted recognizer input arrays from real source-PTS full-engine replay, candidate covers the entire conservative MY interior; start within 300ms of core and end at most100ms after core. Preserve all source data. Two correlated phase views; 89 eligible candidates, all eight physical events represented in union. MY_FRIEND_2 has no qualifying phase0 candidate, explicitly excluded there. Three train events, five development events; no event appears in both.',
 training_events=train,development_events=sorted(events-set(train)),seeds=[27931,27932],learning_rates=[.0003,.001],phone_shares=[.2,.5],epochs=10,
 frozen='All encoders, normalization, gate and feature layers frozen; only final affine MY/LIKE rows updated, all other word rows bit-identical. Original frozen word head feeds existing letter head during export.',
 selection='Keep epoch0; among checkpoints retaining every represented class count in every existing isolated validation domain, maximize correct actual windows on the five other known-phone events, then minimum CE. Diagnostic checkpoint ignoring retention saved separately. This is development selection, never independent accuracy.',
 evaluation='Full native 570 streaming cases, 59 genuine controls, 72 historical schedules and eight retained-video phases after selection. All known phone video is development, including disjoint-event holdout. No official test access. Fresh independent live signing is required for generalization claims.',pins=pins)
(D/'actual_training_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n');print('Verified and pinned',len(pins),'inputs; eight bounded training arms')
