"""Export adapted words while preserving incumbent letter inputs and weights exactly."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.8');os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.4')
from pathlib import Path
import sys,copy,json,time
import numpy as np,torch,coremltools as ct
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'temporal_repair_training_v17_20261011'))
from features import R,BASE,RAW,sha
O=Path(__file__).resolve().parent
from active.v17.export_unified_multimodal_coreml_v17 import load_model,directory_bytes
from active.v17.export_stage1_coreml_v17 import replace_attention
from active.v17.letter_head_v17 import LetterHead,SpanWithLetters
HEAD=R/'artifacts/reports/go_split_root_cause_v17_20261009/deploy_reject/letter_head_a/model.pth'
class StableLetters(torch.nn.Module):
 def __init__(self,model,original,letters):
  super().__init__();self.model=model;self.original_word_head=original.fusion_head;self.letters=letters
 def forward(self,landmarks,hand_embeddings,hand_valid,hand_boxes):
  ll,lf=self.model.landmark_model(landmarks,return_embeddings=True);hf=self.model.hand_model.forward_features(hand_embeddings,hand_valid>.5,hand_boxes);hl=self.model.hand_model.classifier(hf)
  words=self.model.fusion_head(lf,hf,ll,hl);original_words=self.original_word_head(lf,hf,ll,hl)
  return words,self.letters(lf,hf,original_words)
def main():
 name=sys.argv[1] if len(sys.argv)>1 else 'diagnostic';checkpoint=O/f'{name}.pth';model,meta=load_model(checkpoint);original,base=load_model(BASE);payload=torch.load(HEAD,map_location='cpu',weights_only=False);letters=LetterHead(**payload['config']).eval();letters.load_state_dict(payload['state_dict']);wrapper=StableLetters(model,original,letters).eval();old=SpanWithLetters(original,letters).eval();torch.set_num_threads(2)
 for k in ['landmark_model_state_dict','hand_model_state_dict']:assert all(torch.equal(v,base[k][n]) for n,v in meta[k].items())
 with np.load(RAW/'citizen_val.npz') as d:xx=[d[k].astype(np.float32) for k in ['landmarks','hand_embeddings','hand_valid','hand_boxes']];targets=d['targets']
 refs=[];old_letters=[];letter_delta=0.
 with torch.inference_mode():
  for i in range(0,len(targets),32):
   x=tuple(torch.from_numpy(a[i:i+32]) for a in xx);w,l=wrapper(*x);_,ol=old(x[0],x[1],x[2]>.5,x[3]);letter_delta=max(letter_delta,(l-ol).abs().max().item());refs.append((w.numpy(),l.numpy()));old_letters.append(ol.numpy())
 assert letter_delta==0.,letter_delta
 export=copy.deepcopy(wrapper);replace_attention(export);sample=tuple(torch.from_numpy(a[:8]) for a in xx)
 invert=torch.Tensor.__invert__;torch.Tensor.__invert__=lambda self:torch.logical_not(self) if self.dtype==torch.bool else invert(self)
 try:traced=torch.jit.trace(export,sample,strict=False)
 finally:torch.Tensor.__invert__=invert
 package=O/('SpanRecognizerV17'+''.join(w.title() for w in name.split('_'))+'B8FP16.mlpackage');keys=['landmarks','hand_embeddings','hand_valid','hand_boxes']
 converted=ct.convert(traced,inputs=[ct.TensorType(name=k,shape=(8,)+a.shape[1:],dtype=np.float32) for k,a in zip(keys,xx)],outputs=[ct.TensorType(name='word_logits'),ct.TensorType(name='letter_logits')],convert_to='mlprogram',minimum_deployment_target=ct.target.iOS17,compute_precision=ct.precision.FLOAT16);converted.save(str(package));runtime=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.ALL)
 ref=[np.concatenate([r[i] for r in refs]) for i in range(2)];out=[[],[]]
 for i in range(0,len(targets),8):
  feed={k:np.concatenate([a[i:i+8],np.repeat(a[min(i+7,len(a)-1):min(i+7,len(a)-1)+1],max(0,i+8-len(a)),axis=0)]) for k,a in zip(keys,xx)};pred=runtime.predict(feed)
  for j,k in enumerate(['word_logits','letter_logits']):out[j].append(np.asarray(pred[k])[:min(8,len(targets)-i)])
 out=[np.concatenate(a) for a in out]
 result={'name':name,'checkpoint_sha256':sha(checkpoint),'incumbent_letter_head_sha256':sha(HEAD),'letter_reference_exact_max_abs':letter_delta,'letter_threshold':payload['threshold'],'samples':len(targets),'pytorch_correct':int((ref[0].argmax(1)==targets).sum()),'coreml_correct':int((out[0].argmax(1)==targets).sum()),'word_top1_agreement':int((out[0].argmax(1)==ref[0].argmax(1)).sum()),'letter_top1_agreement':int((out[1].argmax(1)==ref[1].argmax(1)).sum()),'max_abs':[float(abs(a-b).max()) for a,b in zip(ref,out)],'package':str(package),'package_mb':directory_bytes(package)/1e6,'note':'Original letter classifier still sees original fused word scores; only adapted word logits change. Not a new letter training run.'}
 (O/f'{name}_export.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
