"""Compare actual candidate FP16 letter outputs against installed FP16 export."""
import json
import coremltools as ct
import numpy as np
from features import O, RAW

names=['landmarks','hand_embeddings','hand_valid','hand_boxes']
with np.load(RAW/'citizen_val.npz') as data:
    arrays=[data[k].astype(np.float32) for k in names]
outputs={}
for arm,filename,key in [('incumbent','SpanRecognizerV17FragmentRejectLettersB8FP16','var_2977'),('selected','SpanRecognizerV17TemporalSelectedB8FP16','letter_logits'),('diagnostic','SpanRecognizerV17TemporalDiagnosticB8FP16','letter_logits')]:
    model=ct.models.MLModel(str(O/'resources'/(filename+'.mlpackage')),compute_units=ct.ComputeUnit.ALL)
    results=[]
    for i in range(0,len(arrays[0]),8):
        feed={k:np.concatenate([a[i:i+8],np.repeat(a[-1:],max(0,i+8-len(a)),axis=0)]) for k,a in zip(names,arrays)}
        results.append(np.asarray(model.predict(feed)[key])[:min(8,len(arrays[0])-i)])
    outputs[arm]=np.concatenate(results)
base=outputs['incumbent'];results={}
for arm in ['selected','diagnostic']:
    value=outputs[arm]
    def prob(x):
        z=np.exp(x-x.max(1,keepdims=True));return z/z.sum(1,keepdims=True)
    bp=prob(base);vp=prob(value)
    results[arm]={'n':len(base),'logit_max_abs':float(np.abs(value-base).max()),'top1_agreement':int((value.argmax(1)==base.argmax(1)).sum()),'probability_max_abs':float(np.abs(vp-bp).max()),'max_probability_gate_disagreements':{str(t):int(((bp.max(1)>=t)!=(vp.max(1)>=t)).sum()) for t in [.5,.95]},'candidate_letters_equal':bool(np.array_equal(outputs['selected'],outputs['diagnostic']))}
(O/'actual_fp16_letter_preservation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
