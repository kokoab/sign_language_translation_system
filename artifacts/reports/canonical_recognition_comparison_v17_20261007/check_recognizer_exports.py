"""Mac conversion check of the 96.83-chain recognizer exports (same method as phone_precision_v17_20261007).

All 378 Citizen validation clips, identical cached landmark/hand inputs, fixed batch 8 with the
final partial batch padded. Compares FP32/FP16 Core ML word predictions with the PyTorch
checkpoint, and reports package storage. Validation only; never test.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'downstream_recipe'
sys.path.insert(0,str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model,load_pair,directory_bytes

def check(checkpoint,package):
    model,meta=load_model(checkpoint)
    runtime=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.ALL)
    outputs=[o.name for o in runtime.get_spec().description.output]
    output='word_logits' if 'word_logits' in outputs else outputs[0]
    base=ROOT/'data/local/citizen100_v17'
    paths=sorted((base/'landmarks/val').glob('*/*.v17.npz'))
    assert len(paths)==378 and all('test' not in p.parts for p in paths)
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
            rows.append(dict(target=target,base_top1=int(ref[j].argmax()),coreml_top1=int(out[j].argmax()),
                             base_top5=target in np.argsort(ref[j])[-5:].tolist(),coreml_top5=target in np.argsort(out[j])[-5:].tolist()))
    n=len(rows)
    return dict(package=str(package.relative_to(ROOT)),package_mb=directory_bytes(package)/1e6,samples=n,
                pytorch_top1_correct=sum(r['base_top1']==r['target'] for r in rows),
                coreml_top1_correct=sum(r['coreml_top1']==r['target'] for r in rows),
                coreml_top5_correct=sum(r['coreml_top5'] for r in rows),
                top1_agreement=sum(r['base_top1']==r['coreml_top1'] for r in rows))

def main():
    torch.set_num_threads(1)
    checkpoint=OUT/'chain_9683/span_recognizer/best_model.pth'
    result=dict(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                test_accessed=False,exports={})
    for tag in ('FP32','FP16'):
        result['exports'][tag]=check(checkpoint,OUT/'coreml'/f'SpanRecognizerV17Chain9683LettersB8{tag}.mlpackage')
        print(json.dumps({tag:result['exports'][tag]}),flush=True)
    august={tag:directory_bytes(ROOT/'artifacts/coreml'/f'SpanRecognizerV17LocalALettersB8{tag}.mlpackage')/1e6 for tag in ('FP32','FP16')}
    result['august_package_mb']=august
    (OUT/'export_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
