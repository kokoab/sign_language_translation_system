"""Measure inference-weight and Core ML package storage without changing models."""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import os
os.chdir(ROOT)
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.letter_head_v17 import LetterHead,SpanWithLetters
from scripts.train_av_boundary_v17 import load as load_boundary
out=Path('artifacts/reports/capstone_conversion_v17_20260930');weights=out/'inference_weights';weights.mkdir(exist_ok=True)
m,_=load_model(Path('artifacts/models/span_recognizer_v17_local_a/best_model.pth'))
p=torch.load('artifacts/models/letter_head_v17_a/model.pth',map_location='cpu',weights_only=False);h=LetterHead(**p['config']);h.load_state_dict(p['state_dict']);m=SpanWithLetters(m,h).eval()
b=load_boundary(Path('artifacts/models/av_boundary_student_v17_l6_a/model.pth'))
results={}
for key,model,package in [('recognizer',m,Path('artifacts/coreml/SpanRecognizerV17LocalALettersB8FP16.mlpackage')),('boundary',b,Path('artifacts/coreml/AVBoundaryStudentV17L6FP16.mlpackage'))]:
 f=weights/(key+'_fp32_state.pt');torch.save(model.state_dict(),f)
 size=sum(x.stat().st_size for x in package.rglob('*') if x.is_file())
 results[key]={'before_file':str(f),'before_bytes':f.stat().st_size,'before_mb':f.stat().st_size/1e6,'after_package':str(package),'after_bytes':size,'after_mb':size/1e6}
results['definition']='Decimal MB=1,000,000 bytes. Before: inference state_dict serialization (no optimizer/training metadata). After: full source .mlpackage including graph, weights and manifest, not compiled app or resident memory.'
(out/'sizes.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
