"""Pinned selected-checkpoint comparison; no training or protected-test access."""
from pathlib import Path
import sys, json, hashlib, time, statistics
ROOT = Path('/Volumes/secret/SLT/SLT'); sys.path.insert(0, str(ROOT))
from active.v17.coreml_runtime_v17 import lightweight_imports
lightweight_imports()
import torch
import numpy as np
from scripts.benchmark_stage1_families_v17 import build_model
from active.v17.train_stage_1_v17 import Citizen100V17Dataset, extractor_schema_fingerprint
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.export_unified_multimodal_coreml_v17 import load_model, load_pair, sha256_file

OUT = Path(__file__).resolve().parent
CHECKPOINTS = {
 'flat_transformer': ('family', 'artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark/transformer/best_model.pth'),
 'flat_squeezeformer': ('landmark', 'artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth'),
 'partwise_global_squeezeformer': ('landmark', 'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'),
 'adapted_landmark_branch': ('landmark', 'artifacts/models/stage1_v17_local_deep_clean_mouth_masked_replay_ft_v1/best_promotion_gate_model.pth'),
 'combined_recognizer': ('multimodal', 'artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth'),
 'phrase_activity_adapted': ('multimodal', 'artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/best_model.pth'),
 'interval_recognizer': ('multimodal', 'artifacts/models/span_recognizer_v17_local_a/best_model.pth'),
}
torch.set_num_threads(1)
data = Citizen100V17Dataset(ROOT/'data/local/citizen100_v17/landmarks', 'val',
    ROOT/'active/v17/citizen100_manifest.json', ROOT/'data/local/citizen100_v17/rejections.csv',
    expected_schema=extractor_schema_fingerprint('apple'))
assert all('val' in p.parts and 'test' not in p.parts for p in data.files)
inputs = []
input_manifest = []
for p in data.files:
    hand = ROOT/'data/local/citizen100_v17/hand_mobileclip2_s0/val'/p.parent.name/(p.name.removesuffix('.v17.npz')+'.hand_mobileclip2_v17.npz')
    arrays = load_pair(p, hand)
    inputs.append(tuple(torch.from_numpy(v) if i != 2 else torch.from_numpy(v) > .5 for i,v in enumerate(arrays)))
    input_manifest.append({'landmarks':str(p.relative_to(ROOT)), 'landmark_sha256':sha256_file(p),
                           'hand_features':str(hand.relative_to(ROOT)), 'hand_sha256':sha256_file(hand)})
(OUT/'inputs.json').write_text(json.dumps(input_manifest, indent=2)+'\n')
targets = data.targets.numpy()
models, records = {}, {}
for name,(kind,relative) in CHECKPOINTS.items():
    path = ROOT/relative
    meta = torch.load(path,map_location='cpu',weights_only=False)
    if kind == 'family':
        model = build_model('transformer',100); model.load_state_dict(meta['state_dict'],strict=True)
    elif kind == 'landmark':
        model = SLTStage1V17(Stage1V17Config(**meta['model_config']))
        model.load_state_dict(meta['model_state_dict'],strict=True)
    else:
        model,meta = load_model(path)
    if 'label_to_index' in meta: assert meta['label_to_index'] == data.label_to_index
    model.eval(); models[name] = model
    with torch.inference_mode():
        logits = torch.cat([model(*v) if kind == 'multimodal' else model(v[0]) for v in inputs]).numpy()
    assert logits.shape == (len(targets),100) and np.isfinite(logits).all()
    for source in meta.get('source_checkpoints',{}).values():
        assert sha256_file(ROOT/source['path']) == source['sha256']
    records[name] = {'checkpoint':relative, 'sha256':sha256_file(path),'kind':kind,
        'epoch':meta.get('epoch'), 'parameters':sum(p.numel() for p in model.parameters()),
        'top1':float(100*np.mean(logits.argmax(1)==targets)),
        'top5':float(100*np.mean((np.argsort(logits,axis=1)[:,-5:]==targets[:,None]).any(1))),
        'predictions':logits.argmax(1).tolist(),'source_checkpoints':meta.get('source_checkpoints',{}),
        'timing_runs':[]}
    print(name, records[name]['top1'], records[name]['top5'],flush=True)
# Complete real validation inputs in three orders; same CPU, precision and timing boundary.
names=list(models)
orders=[names,names[::-1],names[3:]+names[:3]]
with torch.inference_mode():
    for order in orders:
        for name in order:
            model=models[name]; multimodal=records[name]['kind']=='multimodal'
            for v in inputs[:30]: model(*v) if multimodal else model(v[0])
            times=[]
            for v in inputs:
                start=time.perf_counter_ns()
                y=model(*v) if multimodal else model(v[0])
                times.append((time.perf_counter_ns()-start)/1e6)
            records[name]['timing_runs'].append({'median_ms':statistics.median(times),'p90_ms':float(np.percentile(times,90)),'samples_ms':times})
            print('TIMED',name,statistics.median(times),flush=True)
for name,r in records.items():
    r['median_ms']=statistics.median(x['median_ms'] for x in r['timing_runs'])
    r['p90_ms']=statistics.median(x['p90_ms'] for x in r['timing_runs'])
result={'protocol':{'selection':'existing selected checkpoints; not matched from-scratch training',
 'runtime':'PyTorch CPU FP32, one thread, prepared input, batch one, 30 warm-ups, three ordered passes',
 'excludes':'video/camera, landmark extraction, image encoder, decoder, English and UI',
 'protected_test_accessed':False,'input_manifest_sha256':sha256_file(OUT/'inputs.json'),
 'torch_version':torch.__version__},'models':records}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPLETE',flush=True)
