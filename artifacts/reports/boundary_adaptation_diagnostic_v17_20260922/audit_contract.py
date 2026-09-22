"""Reconstruct every cached target and check frozen supervision contracts."""
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from active.v17.temporal_boundary_v17 import boundary_targets,masked_boundary_loss
from active.v17.pretrained_boundary_v17 import load_pose,window_features,PretrainedBoundary,FPS,FRAMES,TARGET
from scripts.train_pretrained_boundary_v17 import PREPARED,BASE_CACHE
from scripts.train_temporal_boundary_v17 import atomic,COMBINED
from active.v17.approved_phrase_data_v17 import digest
out=Path(__file__).parent
prepared=json.loads(PREPARED.read_text());rows={r['item']:r for r in prepared['records']}
meta=json.loads((BASE_CACHE/'manifest.json').read_text())
labels={s:np.load(BASE_CACHE/f'{s}_y.npy',mmap_mode='r') for s in meta['counts']}
covered=Counter();counts=Counter();padding=Counter()
for span in meta['spans']:
 r=rows[span['item']];split=r['split'];ids=np.array(span['target_indices']);offset=span['offset']
 assert split==span['split'] and r['parent']==span['parent'] and offset==covered[split]
 n=int(np.floor((r['pose_frames']-1)/r['pose_fps']*FPS+1e-7))+1
 y=boundary_targets(np.arange(n)/FPS,r['intervals'],[],False)
 clock=np.arange(n)/FPS;inside=np.zeros(n,dtype=bool);near=np.zeros((n,2),dtype=bool)
 for start,end in r['intervals']:
  inside|=(clock>=start-1e-8)&(clock<=end+1e-8)
  near[:,0]|=np.abs(clock-start)<=.050001
  near[:,1]|=np.abs(clock-end)<=.050001
 assert not ((y==0)&~inside[:,None]).any(),r['item']
 assert not ((y==1)&~near).any(),r['item']
 assert np.array_equal(ids,np.flatnonzero((y>=0).any(1)))
 assert np.array_equal(y[ids],labels[split][offset:offset+len(ids)])
 covered[split]+=len(ids);counts[split+'_unknown_channels']+=int((y<0).sum())
 counts[split+'_positive_channels']+=int((y==1).sum());counts[split+'_negative_channels']+=int((y==0).sum())
 padding[split+'_eof_supervised_windows']+=int((ids+10>=n).sum())
assert dict(covered)==meta['counts']
z=torch.zeros((3,2),requires_grad=True);y=torch.tensor([[1.,-1.],[0.,1.],[-1.,-1.]])
masked_boundary_loss(z,y,torch.ones(2)).backward();assert torch.all(z.grad[y<0]==0)
combined=json.loads(COMBINED.read_text());by={r['source_item_id']:r for r in combined['records'] if 'source_item_id' in r}
cal=[r for r in rows.values() if r['split']=='calibration'];known=[]
for r in cal:
 s=by.get(r['item'])
 if s and '__OTHER__' not in s['target_sequence']:known.append(dict(item=r['item'],target_sequence=s['target_sequence'],video_sha256=r['video_sha256']))
model=PretrainedBoundary().eval();parity=[];times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32)
with torch.inference_mode():
 for split in meta['counts']:
  span=next(s for s in meta['spans'] if s['split']==split and s['count']>=3);r=rows[span['item']]
  assert digest(ROOT/r['pose_path'])==r['pose_sha256'];pose=load_pose(r)
  positions=np.array([0,span['count']//2,span['count']-1]);ids=np.array(span['target_indices'])[positions]
  x=torch.from_numpy(np.stack([window_features(pose,int(j))[0] for j in ids]));z=model.project(x).numpy()
  cached=np.load(BASE_CACHE/f'{split}_x.npy',mmap_mode='r')[span['offset']+positions]
  np.testing.assert_allclose(z,cached,rtol=2e-4,atol=2e-4)
  parity.append(dict(split=split,max_abs=float(np.abs(z-cached).max())))
result=dict(status='passed',counts=dict(counts),cached_rows_verified=dict(covered),padding=dict(padding),
 unknown_loss_gradient_zero=True,cache_parity=parity,calibration_records=len(cal),
 complete_locked_vocabulary_calibration=known,source_manifest_matches=digest(ROOT/prepared['source_manifest'])==prepared['source_sha256'],
 finding='Only two existing calibration videos have complete approved locked-vocabulary transcripts (four signs). Other boundary calibration records do not establish whole-video locked-vocabulary WER.')
assert result['source_manifest_matches']
atomic(out/'contract_audit.json',result);print(json.dumps(result))
