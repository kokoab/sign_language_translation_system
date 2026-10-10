"""Record the assistant's visual review of saved, timestamped frames.

These are conservative contact interiors with uncertain approach/release brackets,
not expert-confirmed frame-exact full-sign boundaries. No model scores are used.
"""
from pathlib import Path
import json
O=Path(__file__).resolve().parent
rows=json.load(open(O/'my_review_sources.json'))
cores=[(24,31),(7,8),(27,36),(31,39),(24,38),(35,43),(3,7),(7,8),(25,33),(5,6),(4,6),(28,37),(26,35),(7,11),(24,32),(6,9),(5,7),(7,9),(5,6),(30,40),(6,10),(5,8),(25,42),(24,33),(6,8),(28,37),(37,47),(0,4)]
review=[]
for i,(r,(a,b)) in enumerate(zip(rows,cores)):
 evidence=f'my_{i:02}_detail' if i==4 else f'my_{i:02}'
 frames=json.load(open(O/(evidence+'.json')))['frames'];times={f['index']:f['pts'] for f in frames}
 assert a in times and b in times,(i,a,b)
 prev=max([f['pts'] for f in frames if f['index']<a],default=times[a]);nxt=min([f['pts'] for f in frames if f['index']>b],default=times[b]+1/30)
 review.append({'id':r['source_item_id'],'label':'MY','role':'train','video_path':r['video_path'],'video_sha256':r['video_sha256'],'signer_id':r['signer_id'],
 'reviewer':'Codex visual review; not signer/expert confirmation','method':'Approved MY NAME sequence plus visual open-palm chest contact, independently of model predictions',
 'core_frame_start':a,'core_frame_end_inclusive':b,'core_seconds':[times[a],times[b]+1/30],
 'approach_boundary_bracket':[prev,times[a]],'release_boundary_bracket':[times[b],nxt],
 'onset_left_censored':a==0,'evidence':str(O/(evidence+'.jpg')),
 'boundary_precision':'conservative interior core; brackets are uncertain, not exact onset/offset',
 'training_target_scope':'central MY only; NAME remains adjacent context, not relabeled MY'})
assert len(review)==28
(O/'my_visual_review.json').write_text(json.dumps({'reviewed':len(review),'records':review},indent=2)+'\n')
print('Recorded',len(review),'visually reviewed MY contact cores')
