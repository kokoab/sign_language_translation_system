"""Matched-input before/after export timing on the development Mac; no test data."""
from pathlib import Path
import json,sys,time,platform,subprocess
import numpy as np
import torch
import coremltools as ct
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model,load_pair
from active.v17.letter_head_v17 import LetterHead,SpanWithLetters
from scripts.train_av_boundary_v17 import load as load_boundary
from scripts import segmental_lab_v17 as lab
from active.v17.av_boundary_v17 import windows
from active.v17.temporal_boundary_v17 import boundary_features


def measure(model,package,feeds,names):
    runtime=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.ALL)
    tensors=[tuple(torch.from_numpy(v) if names[i] not in ('valid','hand_valid') else torch.from_numpy(v)>.5 for i,v in enumerate(f)) for f in feeds]
    corefeeds=[dict(zip(names,f)) for f in feeds]
    def before(i):return model(*tensors[i])
    def after(i):return runtime.predict(corefeeds[i])
    times={'pytorch':[],'coreml':[]}
    with torch.inference_mode():
        for i in range(10):before(i%len(feeds));after(i%len(feeds))
        for repeat in range(5):
            for i in range(len(feeds)):
                order=[('pytorch',before),('coreml',after)]
                if (repeat+i)%2:order.reverse()
                for name,fn in order:
                    start=time.perf_counter();fn(i);times[name].append((time.perf_counter()-start)*1000)
    return {'inputs':len(feeds),'calls_per_backend':len(times['pytorch']),'median_ms':{k:float(np.median(v)) for k,v in times.items()},'p90_ms':{k:float(np.percentile(v,90)) for k,v in times.items()},'raw_ms':times}


def main():
    torch.set_num_threads(1)
    model,meta=load_model(ROOT/'artifacts/models/span_recognizer_v17_local_a/best_model.pth')
    payload=torch.load(ROOT/'artifacts/models/letter_head_v17_a/model.pth',map_location='cpu',weights_only=False)
    head=LetterHead(**payload['config']).eval();head.load_state_dict(payload['state_dict'])
    recognizer=SpanWithLetters(model,head).eval()
    paths=sorted((ROOT/'data/local/citizen100_v17/landmarks/val').glob('*/*.v17.npz'))
    # Sixteen full batches distributed through the validation list; identical batch8 inputs.
    offsets=np.linspace(0,len(paths)//8-1,16,dtype=int)*8
    feeds=[]
    for offset in offsets:
        arrays=[load_pair(p,ROOT/'data/local/citizen100_v17/hand_mobileclip2_s0/val'/p.parent.name/(p.name.removesuffix('.v17.npz')+'.hand_mobileclip2_v17.npz')) for p in paths[offset:offset+8]]
        feeds.append([np.concatenate([a[i] for a in arrays]) for i in range(4)])
    rec=measure(recognizer,ROOT/'artifacts/coreml/SpanRecognizerV17LocalALettersB8FP16.mlpackage',feeds,['landmarks','hand_embeddings','hand_valid','hand_boxes'])
    print('Recognizer timing complete',flush=True)
    boundary=load_boundary(ROOT/'artifacts/models/av_boundary_student_v17_l6_a/model.pth','cpu')
    bf=[];sources=[]
    for row in lab.rows_for('tune')[:4]:
        key=lab.key(row['source_item_id']);data=np.load(lab.CACHE/'av_raw'/(key+'.npz'))
        x,v=windows(boundary_features(data['raw'].astype(np.float32),data['times'].astype(np.float64),True),boundary.lookahead)
        for i in np.linspace(0,len(x)-1,8,dtype=int):bf.append([x[i:i+1],v[i:i+1].astype(np.float32)])
        sources.append(row['source_item_id'])
    bound=measure(boundary,ROOT/'artifacts/coreml/AVBoundaryStudentV17L6FP16.mlpackage',bf,['features','valid'])
    result={'hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'os':platform.platform(),'pytorch':torch.__version__,'coremltools':ct.__version__,'pytorch_device':'CPU, one thread','coreml_compute_units':'ALL','warmup_calls_per_backend':10,'repetitions':5,'order':'alternating backend order','preprocessing_included':False,'loading_compilation_included':False,'test_accessed':False,'recognizer_batch_size':8,'recognizer_full_graph_matched':True,'recognizer_sources':[str(p.relative_to(ROOT)) for o in offsets for p in paths[o:o+8]],'boundary_sources':sources,'recognizer':rec,'boundary':bound}
    dest=ROOT/'artifacts/reports/capstone_conversion_v17_20260930';(dest/'latency.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]['median_ms'] for k in ('recognizer','boundary')}),flush=True)
if __name__=='__main__':main()
