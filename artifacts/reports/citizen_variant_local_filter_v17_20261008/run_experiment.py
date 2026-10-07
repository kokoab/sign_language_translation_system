"""Experiment: 96.83 chain rebuilt with local clips restricted to the Citizen variant.

Same recipes as the audited Variant C chain (chain_9683_floor361_mildroll + downstream_recipe);
only the local train/validation manifests change (build_manifests.py). Stages: local replay ->
SemLex gate -> fusion (fresh cache) -> phrase-segment adaptation -> span recognizer. Then every
stage of both chains is evaluated on identical inputs: Citizen val, SemLex val and the filtered
local val. Validation only; lab held-out test untouched; no promotion or install.
"""
import json,os,shutil,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'run'
CMP=ROOT/'artifacts/reports/canonical_recognition_comparison_v17_20261007'
BASE_CHAIN=CMP/'chain_9683_floor361_mildroll';BASE_DOWN=CMP/'downstream_recipe/chain_9683'
PY=str(ROOT/'venv/bin/python');ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
TRAIN_M=HERE/'local_train_citizen_variant.json';VAL_M=HERE/'local_val_citizen_variant.json'
RECIPE=ROOT/'active/v17/phrase_segment_recipe_manifest_20261007.json';VIEW=ROOT/'data/local/phrase_segment_recipe_v17_20261007'
SPAN_CFG='{"lookahead":8,"span_mode":"both","alpha":0.7,"w_r":2,"w_a":0.25,"w_b":0.5,"c":-1,"log_theta":-1.6094}'
RAW_FULL=ROOT/'artifacts/generated/unfrozen_phrase_adapt_v17';RAW=OUT/'raw_cache_citizen_variant'
def status(**k):(OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**k},indent=2)+'\n')
def call(name,cmd,timeout):
    status(state='running',stage=name)
    with (OUT/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def filtered_raw_cache():
    RAW.mkdir()
    keep={}
    for split,m in (('local_train',TRAIN_M),('local_val',VAL_M)):
        keep[split]={f"{v['canonical_label']}/{v['item_id']}" for v in json.loads(m.read_text())['videos']}
    for f in sorted(RAW_FULL.glob('*.npz')):
        if f.stem not in keep:(RAW/f.name).symlink_to(f);continue
        with np.load(f,allow_pickle=False) as p:d={k:p[k] for k in p.files}
        mask=np.array([str(i) in keep[f.stem] for i in d['item_ids']])
        assert mask.sum()==len(keep[f.stem]),(f.stem,mask.sum(),len(keep[f.stem]))
        np.savez(RAW/f.name,**{k:v[mask] for k,v in d.items()})
def floors(domains):
    return {d:(v-1.0 if v-1.0>=90.0 else v) for d,v in ((d,domains[d]['top1']) for d in ('citizen','semlex','local'))}
EVAL=r'''
import sys,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,'.')
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.train_unfrozen_phrase_adapt_v17 import isolated_raw,tensors,logits_for
from active.v17.model_v17 import SLTStage1V17,Stage1V17Config
raw=isolated_raw(Path(sys.argv[2]));dev=torch.device('mps');res={}
for name,(kind,path) in json.loads(sys.argv[1]).items():
    if kind=='lm':
        c=torch.load(path,map_location='cpu',weights_only=False);net=SLTStage1V17(Stage1V17Config(**c['model_config']));net.load_state_dict(c['model_state_dict'])
        m=type('L',(torch.nn.Module,),{'forward':lambda s,l,*a:s.net(l)})();m.net=net;lab=c['label_to_index']
    else:
        m,meta=load_model(Path(path));lab=meta['label_to_index']
    m.to(dev).eval();res[name]={}
    for split in ('citizen_val','semlex_val','local_val'):
        v=tensors(raw[split]);p=logits_for(m,v,dev).argmax(1).numpy();t=v[4].numpy()
        cm=np.zeros((100,100),int);np.add.at(cm,(t,p),1);res[name][split]=cm.tolist()
    m.cpu();res['labels']=sorted(lab,key=lab.get)
Path(sys.argv[3]).write_text(json.dumps(res)+'\n');print('ok')
'''
def run():
    resume=(OUT/'local_replay/best_promotion_gate_model.pth').is_file() and (OUT/'semlex_gate/summary.json').is_file()
    OUT.mkdir(exist_ok=resume)
    base_cmd=json.loads((BASE_CHAIN/'recipe.json').read_text())['local_replay_command']
    cmd=list(base_cmd)
    for flag,val in (('--local-manifest',TRAIN_M),('--local-validation-manifest',VAL_M),('--output',OUT/'local_replay')):cmd[cmd.index(flag)+1]=str(val)
    counts={'local_train':len(json.loads(TRAIN_M.read_text())['videos']),'local_val':len(json.loads(VAL_M.read_text())['videos'])}
    if not resume:(OUT/'recipe.json').write_text(json.dumps({'local_replay_command':cmd,'expected_record_counts':counts,'baseline_chain':str(BASE_CHAIN),
        'baseline_downstream':str(BASE_DOWN),'span_cfg':SPAN_CFG,'test_accessed':False},indent=2)+'\n')
    try:
        if not resume:call('local_replay_train',cmd,4*3600)
        branch=OUT/'local_replay/best_promotion_gate_model.pth'
        if not resume:call('semlex_gate',[PY,'scripts/evaluate_semlex_citizen100_train_audit.py',str(branch),'--feature-root','data/local/semlex_citizen100_val_audit/landmarks_v17',
             '--provenance','data/local/semlex_citizen100_val_audit/download_provenance.json','--expected-split','val','--output-dir',str(OUT/'semlex_gate')],1800)
        import torch
        ck=torch.load(branch,map_location='cpu',weights_only=False);citizen=int(round(ck['validation_metrics']['top1']*3.78))
        call('fusion',[PY,'-m','active.v17.train_unified_multimodal_student_v17','--rebuild-cache','--cache-dir',str(OUT/'fusion_cache'),
             '--output',str(OUT/'fusion'),'--landmark-checkpoint',str(branch),'--citizen-floor-correct',str(citizen),
             '--local-train-manifest',str(TRAIN_M),'--local-val-manifest',str(VAL_M),'--expected-record-counts',json.dumps(counts)],7200)
        filtered_raw_cache()
        call('phrase_adapt',[PY,'-m','active.v17.train_unified_phrase_adapt_v17','--base',str(OUT/'fusion/best_model.pth'),
             '--isolated-cache',str(OUT/'fusion_cache'),'--phrase-rgb',str(VIEW/'multimodal'),'--phrase-hand',str(VIEW/'hand'),
             '--selection-key','reel_v2','--pair-loss-weight','0','--pair-sampling-multiplier','1','--output-dir',str(OUT/'phrase_adapt')],3600)
        sel=json.loads((OUT/'phrase_adapt/result.json').read_text())['selected']['domains']
        call('span_recognizer',[PY,'scripts/train_span_recognizer_v17.py','--output-dir',str(OUT/'span_recognizer'),'--base',str(OUT/'phrase_adapt/best_model.pth'),
             '--epochs','8','--seed','27927','--sets','local_train','--exclude-prefix-spans','--recipe-manifest',str(RECIPE),
             '--raw-cache',str(RAW),'--floors',json.dumps(floors(sel)),'--cfg',SPAN_CFG],4*3600)
        models={'base_local':('lm',str(BASE_CHAIN/'local_replay/best_promotion_gate_model.pth')),'new_local':('lm',str(branch)),
                'base_fusion':('uni',str(BASE_CHAIN/'fusion/best_model.pth')),'new_fusion':('uni',str(OUT/'fusion/best_model.pth')),
                'base_phrase':('uni',str(BASE_DOWN/'phrase_adapt/best_model.pth')),'new_phrase':('uni',str(OUT/'phrase_adapt/best_model.pth')),
                'base_recognizer':('uni',str(BASE_DOWN/'span_recognizer/best_model.pth')),'new_recognizer':('uni',str(OUT/'span_recognizer/best_model.pth')),
                'phone_recognizer':('uni',str(ROOT/'artifacts/models/span_recognizer_v17_local_a/best_model.pth'))}
        call('evaluate',[PY,'-c',EVAL,json.dumps(models),str(RAW),str(OUT/'confusions.json')],3600)
        status(state='complete',promotion='none; experiment')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "Citizen-variant local experiment finished or stopped." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
