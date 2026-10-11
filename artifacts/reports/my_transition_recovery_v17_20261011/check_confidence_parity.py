from pathlib import Path
import sys,json
import numpy as np,coremltools as ct
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'temporal_repair_training_v17_20261011'))
from features import RAW
keys=['landmarks','hand_embeddings','hand_valid','hand_boxes']
with np.load(RAW/'citizen_val.npz') as f:xx=[f[k].astype(np.float32) for k in keys]
outputs={}
for arm,name in [('incumbent','SpanRecognizerV17FragmentRejectLettersB8FP16'),('dual','SpanRecognizerV17ConfidenceDualB8FP16')]:
 m=ct.models.MLModel(str(D/'resources'/(name+'.mlpackage')),compute_units=ct.ComputeUnit.ALL)
 spec=[o.name for o in m.get_spec().description.output];letter=next(k for k in spec if k not in ['word_logits','original_word_logits']);word='original_word_logits' if arm=='dual' else 'word_logits';out={k:[] for k in [word,letter]}
 for i in range(0,len(xx[0]),8):
  feed={k:np.concatenate([a[i:i+8],np.repeat(a[-1:],max(0,i+8-len(a)),axis=0)]) for k,a in zip(keys,xx)};p=m.predict(feed)
  for k in out:out[k].append(np.asarray(p[k])[:min(8,len(xx[0])-i)])
 outputs[arm]={label:np.concatenate(out[k]) for label,k in [('word',word),('letter',letter)]}
report={}
for label in ['word','letter']:
 a=outputs['incumbent'][label];b=outputs['dual'][label];report[label]={'n':len(a),'bit_exact':bool(np.array_equal(a,b)),'max_abs':float(abs(a-b).max()),'top1_agreement':int((a.argmax(1)==b.argmax(1)).sum())}
(D/'confidence_fp16_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
