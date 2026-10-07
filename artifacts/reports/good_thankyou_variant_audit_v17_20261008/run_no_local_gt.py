"""Experiment: Variant C local replay from 96.83 without local GOOD/THANKYOU training clips.

Identical to chain_9683_floor361_mildroll/local_replay (full roll 0, floor 361, patience 80,
80 epochs, seed 1701, 34/33/33 replay, local lip mask) except --local-manifest is
local_train_no_good_thankyou.json (226 local GOOD/THANKYOU train clips removed; local
validation unchanged). Then evaluates GOOD/THANKYOU confusion and per-dataset accuracy for the
new branch and the Variant C baseline on identical inputs. Validation only; no test access.
"""
import json,os,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'no_local_good_thankyou'
CHAIN=ROOT/'artifacts/reports/canonical_recognition_comparison_v17_20261007/chain_9683_floor361_mildroll'
PY=str(ROOT/'venv/bin/python');ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
def status(**k):(OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**k},indent=2)+'\n')
def call(name,cmd,timeout):
    status(state='running',stage=name)
    with (OUT/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
EVAL='''
import sys,json,torch
from pathlib import Path
sys.path.insert(0,'.')
from active.v17.train_unfrozen_phrase_adapt_v17 import isolated_raw,tensors,logits_for
from active.v17.model_v17 import SLTStage1V17,Stage1V17Config
raw=isolated_raw(Path('artifacts/generated/unfrozen_phrase_adapt_v17'));dev=torch.device('mps')
result={}
for name,path in json.loads(sys.argv[1]).items():
    c=torch.load(path,map_location='cpu',weights_only=False);net=SLTStage1V17(Stage1V17Config(**c['model_config']));net.load_state_dict(c['model_state_dict'])
    m=type('L',(torch.nn.Module,),{'forward':lambda s,l,*a:s.net(l)})();m.net=net;m.to(dev).eval()
    lab=c['label_to_index'];G,T=lab['GOOD'],lab['THANKYOU'];row=dict(epoch=c['epoch'])
    for split in ('citizen_val','semlex_val','local_val'):
        v=tensors(raw[split]);pred=logits_for(m,v,dev).argmax(1);t=v[4];g=t==G;th=t==T
        row[split]=dict(correct=int((pred==t).sum()),n=len(t),good=int((pred[g]==G).sum()),good_n=int(g.sum()),good_to_ty=int((pred[g]==T).sum()),
                        ty=int((pred[th]==T).sum()),ty_n=int(th.sum()),ty_to_good=int((pred[th]==G).sum()))
    result[name]=row;m.cpu()
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result,indent=1))
'''
def run():
    OUT.mkdir(exist_ok=False)
    recipe=json.loads((CHAIN/'recipe.json').read_text())['local_replay_command']
    cmd=list(recipe);cmd[cmd.index('--local-manifest')+1]=str(HERE/'local_train_no_good_thankyou.json')
    cmd[cmd.index('--output')+1]=str(OUT/'local_replay')
    (OUT/'recipe.json').write_text(json.dumps({'command':cmd,'baseline':str(CHAIN/'local_replay'),'test_accessed':False},indent=2)+'\n')
    try:
        call('local_replay_train',cmd,4*3600)
        for name in ('promotion_gate','citizen_best'):
            f='best_promotion_gate_model.pth' if name=='promotion_gate' else 'best_model.pth'
            call('semlex_'+name,[PY,'scripts/evaluate_semlex_citizen100_train_audit.py',str(OUT/'local_replay'/f),
                 '--feature-root','data/local/semlex_citizen100_val_audit/landmarks_v17','--provenance',
                 'data/local/semlex_citizen100_val_audit/download_provenance.json','--expected-split','val',
                 '--output-dir',str(OUT/('semlex_'+name))],1800)
        targets={'variant_c_baseline_e55':str(CHAIN/'local_replay/best_promotion_gate_model.pth'),
                 'no_local_gt_promotion_gate':str(OUT/'local_replay/best_promotion_gate_model.pth'),
                 'no_local_gt_citizen_best':str(OUT/'local_replay/best_model.pth'),
                 '96.83_isolated':str(ROOT/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth')}
        call('evaluate_good_thankyou',[PY,'-c',EVAL,json.dumps(targets),str(OUT/'good_thankyou_eval.json')],3600)
        status(state='complete',promotion='none; experiment')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "No-local GOOD/THANKYOU experiment finished or stopped." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
