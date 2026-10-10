from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent; R=O.parents[2]
def save(n,d): (O/n).write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
inv=json.loads((O/'source_inventory.json').read_text())
reviews=json.loads((O/'my_visual_review.json').read_text())['records']
phone=[]
for label,path,cores,evidence in [
 ('MY','data/local/phone_session_pull_v17_20261011/20261011_102647_03.mp4',[(2.8,2.933333)],['my_first_detail.jpg']),
 ('HAVE','data/local/phone_session_pull_v17_20261009b/live_reel_videos/20261009_135024_11.mp4',[(1.4,1.633333),(4.2,4.533333),(7.0,7.333333),(11.6,11.933333),(21.0,21.333333),(24.9,25.133333),(28.6,28.933333)],['have_overview.jpg','have_transitions_0.jpg','have_transitions_1.jpg'])]:
 digest=sha(R/path)
 for i,core in enumerate(cores):
  phone.append(dict(id=f'phone_development:{label}:{i}',label=label,role='development_only',video_path=path,video_sha256=digest,core_seconds=core,following_sign='NAME' if label=='MY' else 'BIG',reviewer='Codex visual review; not signer/expert confirmation',boundary_precision='Conservative visible contact interior, not exact sign onset/offset',evidence=evidence,training_admitted=False,independent_evaluation=False))
save('phone_visual_review.json',dict(records=phone,unreviewed='Remaining mixed MY/LIKE footage and aborted HAVE attempt around 9–10 seconds remain unlabeled. No prediction-derived labels.'))
cases=[]
for r in reviews[:2]:
 cases.append(dict(id=r['id'],path=str(R/r['video_path']),limit=r['core_seconds'][1]+.2,events=[dict(id=r['id'],core=r['core_seconds'])]))
for suffix in ['sign_517744.mp4','sign_517422.mp4']:
 r=next(r for r in inv['records'] if r['id'].endswith(suffix));assert r['role']=='train'
 cases.append(dict(id=r['id'],path=str(R/r['source_video_path']),limit=r['core_seconds'][1]+.2,events=[dict(id=r['id'],core=r['core_seconds'])]))
for r in [phone[0],phone[1]]:
 cases.append(dict(id=r['id'],path=str(R/r['video_path']),limit=r['core_seconds'][1]+.2,events=[dict(id=r['id'],core=r['core_seconds'])]))
save('pilot_cases.json',cases)
print('Pilot:',len(cases),'events,',len(cases)*24,'independent stream initializations,',len(cases)*72,'requested context views')
