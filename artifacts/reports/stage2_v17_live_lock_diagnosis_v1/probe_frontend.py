"""Controlled landmark frontend interventions; keep cached hand evidence fixed."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import coremltools as ct
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import AppleVisionDetector, extract_frames_v17
from active.v17.schema_v17 import V17Config
from active.v17.export_stage2_frozen_encoder_coreml_v17 import paired_arrays
from active.v17.model_stage2_v17 import load_stage2_other_preserving
from active.v17.train_stage_2_v17 import collapse_ctc,edit_distance
OUT=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2)
 model,payload=load_stage2_other_preserving(ROOT/'artifacts/models/stage2_v17_transition_repair_v3/seed_1702.pth')
 encoder=ct.models.MLModel(str(ROOT/'artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage'),compute_units=ct.ComputeUnit.ALL)
 output=encoder.get_spec().description.output[0].name
 labels=[r['canonical_label'] for r in sorted(json.loads((ROOT/'active/v17/citizen100_manifest.json').read_text())['classes'],key=lambda r:r['class_index'])]
 def decode(features,mask):
  with torch.inference_mode(): logits,lengths=model(torch.from_numpy(features.copy()),torch.from_numpy(mask>.5))
  return [labels[i] for i in collapse_ctc(logits[0,:int(lengths[0])].argmax(-1).numpy()) if i<100]
 rows=json.loads((OUT/'frontend_probe.json').read_text())['rows'] if (OUT/'frontend_probe.json').exists() else []
 detector=AppleVisionDetector(.15)
 for item in json.loads((OUT/'matched_manifest.json').read_text()):
  if any(r['item_id']==item['item_id'] for r in rows):continue
  filename=item['item_id'].split(':')[1]
  paths=list((ROOT/'data/local/stage2_v17_multimodal/validation/asllrp_contiguous').glob('*'+filename+'*'))
  assert len(paths)==1
  rgb=paths[0]
  arrays,cached,n=paired_arrays(rgb,rgb_root=ROOT/'data/local/stage2_v17_multimodal',hand_root=ROOT/'data/local/stage2_v17_hand_mobileclip2',frozen_root=ROOT/'data/local/stage2_v17_frozen_features',maximum_windows=8)
  with np.load(rgb,allow_pickle=False) as z:
   ranges=z['window_source_ranges'].tolist();md=json.loads(str(z['metadata_json']))
  assert md['vision_coarse_rotation_clockwise']==0
  v=cv2.VideoCapture(str(ROOT/item['video']));fps=v.get(cv2.CAP_PROP_FPS);frames=[]
  while True:
   ok,f=v.read()
   if not ok:break
   frames.append(f)
  v.release()
  row=dict(item_id=item['item_id'],reference=item['reference'],cached=decode(cached,arrays[-1]),lanes={})
  names=['landmarks','hand_embeddings','hand_valid','hand_boxes','window_mask']
  encoded=np.asarray(encoder.predict(dict(zip(names,arrays)))[output]).reshape(cached.shape)
  row['cached_inputs_coreml']=decode(encoded,arrays[-1]);row['encoder_cache_max_abs']=float(np.max(np.abs(encoded-cached)))
  for lane,side,rate in [('fresh_1280_30',1280,30),('fresh_640_30',640,30),('fresh_1280_20',1280,20),('fresh_640_20',640,20)]:
   values=[a.copy() for a in arrays];diags=[];failure=None
   for wi,(start,end) in enumerate(ranges):
    window=frames[start:end]
    if rate==20:
     # Fixed cached window edges, only sampling changes: not an exact live replay.
     positions=np.unique(np.rint(np.arange(0,len(window),fps/rate)).astype(int));positions=positions[positions<len(window)]
     window=[window[i] for i in positions]
    result=extract_frames_v17(window,V17Config(target_frames=32,maximum_source_frames=32,maximum_image_side=side,trim_to_hand_activity=False),detector=detector)
    if result is None:
     failure=dict(window_index=wi,sampled_frames=len(window),reason='extractor_returned_no_features')
     break
    values[0][0,wi]=result.features
    diags.append(result.diagnostics)
   if failure is not None:
    row['lanes'][lane]=dict(prediction=None,unavailable=failure)
    continue
   ff=np.asarray(encoder.predict(dict(zip(names,values)))[output]).reshape(cached.shape)
   row['lanes'][lane]=dict(prediction=decode(ff,values[-1]),landmark_mean_abs=float(np.mean(np.abs(values[0][0,:n]-arrays[0][0,:n]))),diagnostics=diags)
  rows.append(row)
  (OUT/'frontend_probe.json').write_text(json.dumps(dict(rows=rows,disclosure='Diagnostic hybrid: cached hand embeddings/boxes and fixed window edges in all lanes. 20Hz sampling approximates sampling reduction, not exact live scheduling. Fresh detections reset per window as training does.'),indent=2)+'\n')
  print(item['item_id'], 'cached',row['cached'],'lanes',{k:v['prediction'] for k,v in row['lanes'].items()},flush=True)
 assert all(r['cached']==r['cached_inputs_coreml'] for r in rows),'cached encoder parity differs'
 for lane in ['cached','cached_inputs_coreml','fresh_1280_30','fresh_640_30','fresh_1280_20','fresh_640_20']:
  ps=[r[lane] if lane in r else r['lanes'][lane]['prediction'] for r in rows]
  print(lane,'evaluable',sum(p is not None for p in ps),'edits',sum(edit_distance(r['reference'],p) for r,p in zip(rows,ps) if p is not None),'exact',sum(r['reference']==p for r,p in zip(rows,ps)),flush=True)

if __name__=='__main__':main()
