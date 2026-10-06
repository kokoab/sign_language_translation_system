from pathlib import Path
import sys,copy,json
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.coreml_runtime_v17 import lightweight_imports
lightweight_imports()
import torch,numpy as np,coremltools as ct
from torch import nn
from scripts.benchmark_stage1_families_v17 import build_model,OUTPUT
from active.v17.train_stage_1_v17 import Citizen100V17Dataset,extractor_schema_fingerprint
from active.v17.export_stage1_coreml_v17 import ManualMHA,replace_attention,sha256_file,tree_sha256,directory_bytes
from active.v17.coreml_runtime_v17 import load,spec
OUT=Path(__file__).resolve().parent
RES=OUT/'phone/Resources';RES.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
data=Citizen100V17Dataset(ROOT/'data/local/citizen100_v17/landmarks','val',ROOT/'active/v17/citizen100_manifest.json',ROOT/'data/local/citizen100_v17/rejections.csv',expected_schema=extractor_schema_fingerprint('apple'))
assert all('val' in p.parts and 'test' not in p.parts for p in data.files)
x=torch.stack([data[i][0] for i in range(len(data))]);y=data.targets.numpy()
assert np.isfinite(x.numpy()).all()
x.numpy().astype('<f4').tofile(RES/'features.bin')
class ExportTransformer(nn.Module):
 def __init__(self,source):
  super().__init__();self.source=source
  self.attention=nn.ModuleList([ManualMHA(z.self_attn) for z in source.encoder.layers])
 def forward(self,features):
  s=self.source;value=s.local_conv(s.frame(features))+s.position
  for layer,attn in zip(s.encoder.layers,self.attention):
   v=layer.norm1(value);value=value+layer.dropout1(attn(v,v,v)[0])
   v=layer.norm2(value);value=value+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(v)))))
  if s.encoder.norm is not None:value=s.encoder.norm(value)
  active=features[:,:,:42,3].amax(dim=-1)>0.5
  empty=~active.any(dim=1)
  first=torch.arange(32,device=features.device).view(1,32)==0
  safe=torch.where(empty.unsqueeze(-1),first,active)
  scores=s.head.attention(value).squeeze(-1).masked_fill(~safe,torch.finfo(value.dtype).min)
  pooled=(value*scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
  return s.head.classifier(pooled*(~empty).unsqueeze(-1).to(value.dtype))
manifest={'shape':[1,32,61,5],'count':len(data),'targets':y.tolist(),'files':[str(p.relative_to(ROOT)) for p in data.files],'input_sha256':sha256_file(RES/'features.bin'),'models':{}}
summary={}
for family in ['transformer','squeezeformer']:
 path=OUTPUT/family/'best_model.pth';ck=torch.load(path,map_location='cpu',weights_only=False)
 original=build_model(family,data.num_classes);original.load_state_dict(ck['state_dict']);original.eval()
 exported=ExportTransformer(copy.deepcopy(original)) if family=='transformer' else copy.deepcopy(original)
 if family=='squeezeformer':replace_attention(exported)
 exported.eval()
 with torch.no_grad():
  ref=torch.cat([original(v[None]) for v in x]).numpy()
  wrapped=torch.cat([exported(v[None]) for v in x]).numpy()
  empty=torch.zeros_like(x[:1]);assert torch.allclose(original(empty),exported(empty),atol=1e-4,rtol=1e-4)
 assert np.isfinite(ref).all() and np.isfinite(wrapped).all()
 assert np.array_equal(ref.argmax(1),wrapped.argmax(1)), 'Wrapper prediction mismatch'
 assert np.max(np.abs(ref-wrapped))<1e-3
 expected=95.5026455026455 if family=='transformer' else 96.29629629629629
 assert abs(100*np.mean(ref.argmax(1)==y)-expected)<1e-8
 with torch.no_grad():traced=torch.jit.trace(exported,x[:1],strict=False)
 with torch.no_grad():assert torch.allclose(traced(empty),original(empty),atol=1e-4,rtol=1e-4)
 for precision in ['FP32','FP16']:
  name=family.title()+precision;package=RES/(name+'.mlpackage')
  if package.exists():raise FileExistsError(package)
  model=ct.convert(traced,inputs=[ct.TensorType(name='landmarks',shape=(1,32,61,5),dtype=np.float32)],outputs=[ct.TensorType(name='logits',dtype=np.float32)],convert_to='mlprogram',compute_precision=getattr(ct.precision,'FLOAT'+precision[2:]),minimum_deployment_target=ct.target.iOS15)
  model.save(str(package));runtime=load(package,'ALL');pred=[]
  for sample in x.numpy():pred.append(np.array(runtime.predict({'landmarks':sample[None]})['logits']).reshape(-1))
  pred=np.stack(pred);assert np.isfinite(pred).all()
  record={'checkpoint_sha256':sha256_file(path),'package_sha256':tree_sha256(package),'package_bytes':directory_bytes(package),'reference_top1':float(100*np.mean(ref.argmax(1)==y)),'top1':float(100*np.mean(pred.argmax(1)==y)),'top5':float(100*np.mean([target in row for target,row in zip(y,np.argsort(pred,axis=1)[:,-5:])])),'changed_predictions':int(np.sum(ref.argmax(1)!=pred.argmax(1))),'max_abs_logits':float(np.max(np.abs(ref-pred))),'wrapper_max_abs':float(np.max(np.abs(ref-wrapped)))}
  summary[name]=record
  manifest['models'][name]={'reference_top1':ref.argmax(1).tolist(),'mac_top1':pred.argmax(1).tolist()}
  print(name,json.dumps(record),flush=True)
  (OUT/'conversion_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  (RES/'manifest.json').write_text(json.dumps(manifest)+'\n')
  del runtime,model
print('PREPARATION COMPLETE',flush=True)
