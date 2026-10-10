from pathlib import Path
import json,hashlib,collections,numpy as np
O=Path(__file__).resolve().parent
out={'classification_run':False,'groups':[],'runs':{},'note':'Input integrity and retained evidence only; no recognition accuracy measured.'}
for prefix in ['pilot_inputs_v2','recovered_my_inputs']:
 d=json.loads((O/f'{prefix}.json').read_text());rows=d['rows'];n=sum(r['valid'] for r in rows)
 assert len({(r['id'],r['phase'],r['capture_offset'],r['view']) for r in rows})==len(rows)
 arrays=[]
 for i,shape in enumerate(d['shapes']):
  a=np.fromfile(O/f'{prefix}.{i}.bin',dtype='<f4').reshape(-1,*shape);assert len(a)==n and np.isfinite(a).all();arrays.append(a)
 assert set(np.unique(arrays[2])).issubset({0.,1.})
 assert np.all((arrays[0][...,3:5]>=0)&(arrays[0][...,3:5]<=1))
 out['runs'][prefix]={'requested':len(rows),'constructed':n,'finite':True,'validity_binary':True}
 for ident in dict.fromkeys(r['id'] for r in rows):
  rr=[r for r in rows if r['id']==ident];assert len(rr)==72
  for r in rr:
   r['retained_core_frames']=sum(r['core'][0]-1e-9<=t<r['core'][1]-1e-9 for t in r.get('retained_seconds',[]))
   r['training_evidence_eligible']=r['valid'] and min(r['retained_core_frames'],r['core_hand_frames'])>=2 and r.get('core_crop_positions',0)>0
  def countdistinct(values):return len({hashlib.sha256(v.tobytes()).hexdigest() for v in values})
  groups=[]
  for offset in [0.,1/60,1/30]:
   sample=[r for r in rr if r['capture_offset']==offset and r['view']==0 and r['valid']]
   groups.append({'offset':offset,'distinct_landmark_phase_inputs':countdistinct([arrays[0][r['array_index']] for r in sample]),'distinct_embedding_phase_inputs':countdistinct([arrays[1][r['array_index']] for r in sample])})
  out['groups'].append(dict(id=ident,requested=72,constructed=sum(r['valid'] for r in rr),evidence_eligible=sum(r['training_evidence_eligible'] for r in rr),min_core_hand_frames=min(r['core_hand_frames'] for r in rr),min_retained_core_frames=min(r['retained_core_frames'] for r in rr),min_core_crop_positions=min(r.get('core_crop_positions',0) for r in rr),phase_variation=groups))
 # Audit is separate from raw native output; keep raw unchanged.
 (O/f'{prefix}_audit.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
out['total_requested']=sum(r['requested'] for r in out['runs'].values());out['total_constructed']=sum(r['constructed'] for r in out['runs'].values());out['total_evidence_eligible']=sum(r['evidence_eligible'] for r in out['groups']);out['unique_events']=len(out['groups'])
(O/'native_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='groups'},indent=2));print('groups',[(r['id'],r['constructed'],r['evidence_eligible']) for r in out['groups']])
