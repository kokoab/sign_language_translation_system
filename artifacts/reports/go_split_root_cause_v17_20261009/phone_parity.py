"""Take 01-03 on the Mac through the phone path (Core ML FP16 boundary + installed v2 FP16 recognizer
package, letter gate 0.95, installed decoder settings) vs the phone's own committed words."""
import json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from active.v17.segmental_runtime_v17 import SegmentalRuntime,build_runtime
sys.path.insert(0,str(HERE));from eval_variants import restore_frames
E=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008'
cfg=json.loads((E/'user_take_replay/config_v2_installed.json').read_text())
cfg['coreml_recognizer']=str((E/'deploy_v2/SpanRecognizerV17CitizenVariantV2LettersB8FP16.mlpackage').relative_to(ROOT))
p=HERE/'config_phone_path.json';p.write_text(json.dumps(cfg,indent=1))
dual=build_runtime(p,backend='coreml');w=dual.words
rt=SegmentalRuntime(w.boundary.model,w.recognizer,w.config,mode='both',letter_gate=0.95)
phone=json.loads((HERE/'phone_words.json').read_text());out={}
for f in sorted(Path('/Users/frnzlo/slt_go_split_cache/takes').glob('*.pkl')):
    item=pickle.load(open(f,'rb'));restore_frames(item);rt.reset();ws=[]
    for o,h in zip(item['obs'],item['hands']):ws+=rt.observe(o,hands=h)
    ws+=rt.finish()
    mac=[(x['gloss'],x['start_frame'],x['end_frame'],round(x['score'],2)) for x in ws if not x['gloss'].startswith('FS_')]
    ph=[(x['gloss'],x['start_frame'],x['end_frame'],round(x['score'],2)) for x in phone[f.stem+'.mp4']]
    out[f.stem]=dict(mac_coreml=mac,phone=ph)
    print(f.stem,'\n  mac-coreml:',' '.join(g for g,*_ in mac),'\n  phone     :',' '.join(g for g,*_ in ph),flush=True)
(HERE/'phone_parity.json').write_text(json.dumps(out,indent=1))
