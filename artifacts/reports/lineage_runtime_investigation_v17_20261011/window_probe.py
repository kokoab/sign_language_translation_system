from pathlib import Path
import subprocess,json,sys
O=Path(__file__).resolve().parent;R=O.parents[2];B=R/'artifacts/reports/my_like_regression_v17_20261011'
d=O/'window';d.mkdir(exist_ok=True)
s=(B/'dump_harness/main.swift').read_text().split('    let engine =')[0]
s+='''    var rows: [[String: Any]] = []
    var arrays = [[Float]](repeating: [], count: 4)
    for st in stride(from:44,through:60,by:2) {
      for en in stride(from:max(st+3,55),through:70,by:2) where en<observations.count {
        let ids=observations.indices.filter { observations[$0].seconds>=observations[st].seconds-0.1-1e-9 && observations[$0].seconds<=observations[en].seconds+0.1+1e-9 }
        if let x=LiveSpanRecognizer.inputs(ids.map{observations[$0]},ids.map{hands[$0]}) {
          rows.append(["start":st,"end":en]); arrays[0] += x.landmarks; arrays[1] += x.embeddings; arrays[2] += x.valid; arrays[3] += x.boxes
        }
      }
    }
    for (i,a) in arrays.enumerated() { try a.withUnsafeBufferPointer { Data(buffer:$0) }.write(to:URL(fileURLWithPath:out+".\\(i).bin")) }
    try writeJSON(rows,out+".json")
}
let a=CommandLine.arguments
try runTraceWords(a[1],a[2])
'''
(d/'main.swift').write_text(s)
cmd=['xcrun','swiftc','-O','-o',str(O/'window_probe')]+[str(O/'current'/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift']]+[str(d/'main.swift')]
with (O/'window_build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([str(O/'window_probe'),str(B/'first_miss_trace.json'),str(O/'windows')],check=True)
sys.path.insert(0,str(R))
import numpy as np,torch
from active.v17.export_unified_multimodal_coreml_v17 import load_model
torch.set_num_threads(2)
keys=json.load(open(O/'windows.json'));xx=[torch.tensor(np.fromfile(O/f'windows.{i}.bin',np.float32).reshape((-1,)+shape)) for i,shape in enumerate([(32,61,5),(16,3,512),(16,3),(16,3,4)])];xx[2]=xx[2]>.5
D=R/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe'
paths={'old_phone':R/'artifacts/models/span_recognizer_v17_local_a/best_model.pth','old_clean':D/'august/span_recognizer/best_model.pth','new_before_filter':D/'chain_9683/span_recognizer/best_model.pth','current':R/'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/best_model.pth'}
res={}
with torch.inference_mode():
 for name,path in paths.items():
  m,c=load_model(path);labels=sorted(c['label_to_index'],key=c['label_to_index'].get)
  pp=torch.cat([m(*[a[i:i+8] for a in xx]).softmax(-1) for i in range(0,len(keys),8)]).numpy()
  rows=[dict(k,top=labels[int(p.argmax())],my=float(p[labels.index('MY')]),like=float(p[labels.index('LIKE')])) for k,p in zip(keys,pp)]
  res[name]={'rows':rows,'my_top':sum(r['top']=='MY' for r in rows),'n':len(rows)}
  print(name,res[name]['my_top'],len(rows),flush=True)
  (O/'window_results.json').write_text(json.dumps(res,indent=2)+'\n')
