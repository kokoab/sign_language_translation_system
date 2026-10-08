"""v2 deploy artifacts: head-A letter head on the v2 recognizer (NONE replay from the v2 raw cache), FP16 export, 378-clip check."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');E=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008';O=E/'deploy_v3'
REC=E/'run_v3/span_recognizer/best_model.pth';HEAD=O/'letter_head_a/model.pth';PKG=O/'SpanRecognizerV17CitizenVariantV3LettersB8FP16.mlpackage'
sys.path.insert(0,str(ROOT))
def letter():
    import scripts.train_letter_head_v17 as m
    m.ISO=E/'run_v3/raw_cache_citizen_variant'
    sys.argv=['x','--recognizer',str(REC),'--output',str(HEAD),'--no-corrections','--device','mps'];m.main()
if sys.argv[1:]==['letter']:letter();sys.exit()
py=str(ROOT/'venv/bin/python')
subprocess.run([py,__file__,'letter'],check=True,cwd=ROOT,stdout=open(O/'letter_head.log','w'),stderr=subprocess.STDOUT)
subprocess.run([py,'active/v17/export_span_recognizer_batched_coreml_v17.py',str(REC),str(PKG),'--precision','float16','--letter-head',str(HEAD),'--fixed-batch','8'],check=True,cwd=ROOT,stdout=open(O/'export_FP16.log','w'),stderr=subprocess.STDOUT)
sys.path.insert(0,str(ROOT/'artifacts/reports/canonical_recognition_comparison_v17_20261007'))
import check_recognizer_exports as c
r=c.check(REC,PKG);(O/'export_check.json').write_text(json.dumps(r,indent=1)+'\n');print(r)
import torch;h=torch.load(HEAD,map_location='cpu',weights_only=False);print('letter tau',h['threshold'])
(O/'done.json').write_text(json.dumps(dict(check=r,letter_threshold=h['threshold']))+'\n')
