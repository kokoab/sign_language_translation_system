"""Read-only matched validation check of the deployed recognizer export."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
import coremltools as ct
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model,load_pair

def main():
    torch.set_num_threads(1)
    checkpoint=ROOT/'artifacts/models/span_recognizer_v17_local_a/best_model.pth'
    package=ROOT/'artifacts/coreml/SpanRecognizerV17LocalALettersB8FP16.mlpackage'
    model,meta=load_model(checkpoint)
    runtime=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.ALL)
    outputs=[o.name for o in runtime.get_spec().description.output]
    output='word_logits' if 'word_logits' in outputs else outputs[0]
    base=ROOT/'data/local/citizen100_v17'
    paths=sorted((base/'landmarks/val').glob('*/*.v17.npz'))
    assert len(paths)>0 and all('test' not in p.parts for p in paths)
    rows=[]
    for offset in range(0,len(paths),8):
        batch=paths[offset:offset+8]
        arrays=[load_pair(p,base/'hand_mobileclip2_s0/val'/p.parent.name/(p.name.removesuffix('.v17.npz')+'.hand_mobileclip2_v17.npz')) for p in batch]
        feed=[np.concatenate([x[i] for x in arrays]) for i in range(4)]
        with torch.inference_mode():
            ref=model(*(torch.from_numpy(v) if i!=2 else torch.from_numpy(v)>.5 for i,v in enumerate(feed))).numpy()
        padded=[np.concatenate([v,np.repeat(v[-1:],8-len(batch),axis=0)]) if len(batch)<8 else v for v in feed]
        out=np.asarray(runtime.predict(dict(zip(['landmarks','hand_embeddings','hand_valid','hand_boxes'],padded)))[output]).reshape(8,-1)[:len(batch)]
        for j,p in enumerate(batch):
            target=int(meta['label_to_index'][p.parent.name])
            rows.append(dict(path=str(p.relative_to(ROOT)),target=target,base_top1=int(ref[j].argmax()),coreml_top1=int(out[j].argmax()),base_top5=target in np.argsort(ref[j])[-5:].tolist(),coreml_top5=target in np.argsort(out[j])[-5:].tolist()))
        if offset%80==0:print(f'Checked {len(rows)}/{len(paths)} validation clips',flush=True)
    n=len(rows)
    result=dict(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),package=str(package.relative_to(ROOT)),compute_units='ALL',validation_samples=n,test_accessed=False,final_partial_batch_padded=True,base_top1=100*sum(r['target']==r['base_top1'] for r in rows)/n,coreml_top1=100*sum(r['target']==r['coreml_top1'] for r in rows)/n,base_top5=100*sum(r['base_top5'] for r in rows)/n,coreml_top5=100*sum(r['coreml_top5'] for r in rows)/n,top1_agreement=100*sum(r['base_top1']==r['coreml_top1'] for r in rows)/n,changed_predictions=[r for r in rows if r['base_top1']!=r['coreml_top1']])
    dest=ROOT/'artifacts/reports/capstone_conversion_v17_20260930'
    (dest/'recognizer_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (dest/'recognizer_predictions.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
